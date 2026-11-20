# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unfuses BertSelfAttention's QKV: the fused output goes to HBM and Q, K, V each re-read it all."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.models.bert import BertAttention, BertSelfAttention

from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod, spyre_linear_t
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    _encoder_rect_kernel,
    encoder_grid,
    encoder_inline_active,
)

logger = init_logger(__name__)

_NAMES = ("q", "k", "v")


def _param(t: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(t.contiguous(), requires_grad=False)


def _split_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    q, k, v = (
        spyre_linear_t(hidden_states, getattr(self, f"{n}_weight_t"), getattr(self, f"{n}_bias"))
        for n in _NAMES
    )
    return self.attn(q, k, v)


def split_bert_qkv(model: nn.Module) -> None:
    """Runs on CPU, before the device move."""
    count = 0
    for module in model.modules():
        if not isinstance(module, BertSelfAttention):
            continue
        proj = module.qkv_proj
        if type(proj.quant_method) is not SpyreUnquantizedLinearMethod:
            continue
        if proj.weight.device.type != "cpu":
            continue
        sizes = [module.q_size, module.kv_size, module.kv_size]
        weight = cast(torch.Tensor, proj.weight).data  # stored transposed: [in, q + 2kv]
        weights = torch.split(weight, sizes, dim=1)
        biases = torch.split(proj.bias.data, sizes) if proj.bias is not None else (None,) * 3
        for n, w, b in zip(_NAMES, weights, biases):
            module.register_parameter(f"{n}_weight_t", _param(w))
            module.register_parameter(f"{n}_bias", None if b is None else _param(b))
        del module.qkv_proj
        module.forward = _split_forward.__get__(module)  # ty: ignore[invalid-assignment]
        count += 1
    if count:
        logger.info("Split the fused QKV projection of %d BertSelfAttention layers.", count)
        _group_heads(model)


# Head groups: where torch-spyre's SDPA planner KV-tiles a rectangle's attention but runs half the
# heads untiled, two groups with their own Q/K/V matmuls and out-proj rows keep Q, K, V out of HBM.
_GROUPS = 2
_PATCHED_ATTR = "_spyre_head_groups_patched"


def _kv_tiled(width: int, extent: int, heads: int, kv_heads: int, head_dim: int) -> bool:
    from torch_spyre._inductor import config, decompositions

    q, kv = width * heads * extent * head_dim, width * kv_heads * extent * head_dim
    plan = decompositions._select_sdpa_tiling(
        batch_size=width,
        num_heads=heads,
        num_kvheads=kv_heads,
        max_seqlen_q=extent,
        max_seqlen_kv=extent,
        head_dim=head_dim,
        element_size=2,
        num_cores=config.sencores,
        lx_budget_bytes=decompositions._sdpa_lx_budget_bytes(),
        mask_shapes=((width, 1, 1, extent),),
        # The view(B, L, H, D).transpose(1, 2) head axis is never dense: head tiles stage via HBM.
        head_tile_staging_bytes=2 * (q + 2 * kv) * 2,
    )
    return plan.num_kv_blocks * plan.num_q_tiles * plan.num_head_tiles * plan.num_batch_tiles > 1


def grouped_rectangles(
    rectangles, heads: int, kv_heads: int, head_dim: int
) -> frozenset[tuple[int, int]]:
    """The ``(width, extent)`` of each ``(extent, width)`` rectangle that head groups untile."""
    if heads % _GROUPS or kv_heads % _GROUPS:
        return frozenset()
    g = _GROUPS
    return frozenset(
        (width, extent)
        for extent, width in rectangles
        if _kv_tiled(width, extent, heads, kv_heads, head_dim)
        and not _kv_tiled(width, extent, heads // g, kv_heads // g, head_dim)
    )


def install_bert_head_groups(model_config) -> None:
    """Record on each BertAttention, at construction, the rectangles to run as head groups."""
    if getattr(model_config, "runner_type", None) != "pooling":
        return
    if getattr(BertAttention, _PATCHED_ATTR, False):
        return
    original_init = BertAttention.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.spyre_group_rects = frozenset()
        if get_tensor_model_parallel_world_size() > 1:  # the grouped out-proj skips the all-reduce
            return
        sa = cast(BertSelfAttention, self.self)
        try:
            from spyre_inference.v1.worker.spyre_shape_bucketer import encoder_rectangles

            rects = encoder_rectangles(get_current_vllm_config())
            self.spyre_group_rects = grouped_rectangles(
                rects, sa.num_heads, sa.num_kv_heads, sa.head_dim
            )
        except Exception as e:
            logger.warning_once("Encoder head groups disabled: %r", e)

    BertAttention.__init__ = patched_init  # ty: ignore[invalid-assignment]
    setattr(BertAttention, _PATCHED_ATTR, True)


def _grouped_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    sa, so = self.self, self.output
    mask = encoder_grid().mask
    if (
        mask is None
        or not encoder_inline_active()
        or (mask.shape[0], mask.shape[-1]) not in self.spyre_group_rects
    ):
        return so(sa(hidden_states), hidden_states)
    heads, kv_heads, d = sa.num_heads // _GROUPS, sa.num_kv_heads // _GROUPS, sa.head_dim
    out = so.dense.bias
    for g in range(_GROUPS):
        q, k, v = (
            spyre_linear_t(
                hidden_states, getattr(sa, f"{n}_weight_t_{g}"), getattr(sa, f"{n}_bias_{g}")
            )
            for n in _NAMES
        )
        a = _encoder_rect_kernel(
            q.view(-1, heads, d),
            k.view(-1, kv_heads, d),
            v.view(-1, kv_heads, d),
            mask,
            sa.attn.impl.scale,
            mask.shape[0],
            mask.shape[-1],
            heads,
            kv_heads,
            d,
        )
        out = torch.matmul(a.reshape(-1, heads * d), getattr(so, f"wo_t_{g}")) + out
    return so.LayerNorm(out + hidden_states)


def _group_heads(model: nn.Module) -> None:
    """Adds per-group weights next to the split ones (in-graph weight slices are slower)."""
    count, rects = 0, set[tuple[int, int]]()
    for module in model.modules():
        if not isinstance(module, BertAttention) or not getattr(module, "spyre_group_rects", None):
            continue
        sa, so = cast(BertSelfAttention, module.self), module.output
        if (
            not hasattr(sa, "q_weight_t")
            or type(so.dense.quant_method) is not SpyreUnquantizedLinearMethod
        ):
            continue
        for n in _NAMES:
            w, b = getattr(sa, f"{n}_weight_t"), getattr(sa, f"{n}_bias")
            width = w.shape[1] // _GROUPS
            for g in range(_GROUPS):
                cols = slice(g * width, (g + 1) * width)
                sa.register_parameter(f"{n}_weight_t_{g}", _param(w.data[:, cols]))
                sa.register_parameter(f"{n}_bias_{g}", None if b is None else _param(b.data[cols]))
        wo = cast(torch.Tensor, so.dense.weight).data  # stored transposed: [q_size, hidden]
        rows = wo.shape[0] // _GROUPS
        for g in range(_GROUPS):
            so.register_parameter(f"wo_t_{g}", _param(wo[g * rows : (g + 1) * rows]))
        module.forward = _grouped_forward.__get__(module)  # ty: ignore[invalid-assignment]
        rects.update(cast(frozenset, module.spyre_group_rects))
        count += 1
    if count:
        logger.info(
            "Encoder attention of %d BertAttention layers runs as %d head groups on rectangles %s.",
            count,
            _GROUPS,
            sorted(rects),
        )
