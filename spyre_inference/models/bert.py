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

"""Spyre adaptations for vLLM BERT-family pooling models.

These classes route ``token_type_ids`` around vLLM's bit-pack transport;
the embedding subclass also fuses its prologue. See ``spyre_inference.models._token_type``.
They also split each ``BertSelfAttention``'s fused QKV projection; see ``split_bert_qkv``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from vllm.config import CompilationMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.models.bert import (
    BertAttention,
    BertEmbedding,
    BertEmbeddingModel,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
    BertSelfAttention,
    BertSpladeSparseEmbeddingModel,
)

from spyre_inference.custom_ops.lazy_compile import CompileOutermost
from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod, spyre_linear_t
from spyre_inference.models._token_type import (
    SpyreTokenTypeEmbedding,
    SpyreTokenTypeModel,
)
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    _encoder_rect_kernel,
    encoder_grid,
    encoder_inline_active,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_QKV = ("q", "k", "v")


def _split_qkv_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    proj = self.qkv_proj
    q, k, v = (
        spyre_linear_t(hidden_states, getattr(proj, f"{n}_weight_t"), getattr(proj, f"{n}_bias"))
        for n in _QKV
    )
    return self.attn(q, k, v)


class _SplitQKVLinearMethod(SpyreUnquantizedLinearMethod):
    def __init__(self, sizes: list[int]) -> None:
        super().__init__()
        self.sizes = sizes
        self.grouped = False

    def build_weight_t(self, layer: nn.Module, w: torch.Tensor) -> None:
        # Reload restores the original fused weight, then copies these processed
        # parameters into their existing storage, including compiled graph references.
        # The retained bias can still be on Spyre while the restored weight is on CPU.
        bias = cast(torch.Tensor, layer.bias).data.to(w.device) if layer.bias is not None else None
        biases = bias.split(self.sizes) if bias is not None else (None,) * 3
        for n, shard, b in zip(_QKV, w.split(self.sizes, dim=0), biases):
            layer.register_parameter(f"{n}_weight_t", _frozen(shard.t()))
            layer.register_parameter(f"{n}_bias", None if b is None else _frozen(b))
        del layer.weight
        if self.grouped:
            _register_qkv_groups(layer)


def split_bert_qkv(model: nn.Module) -> None:
    """Replace each ``BertSelfAttention``'s fused QKV projection with three matmuls.

    The fused output has three consumers needing different layouts, so torch-spyre
    writes it to HBM and Q, K and V each read the whole buffer back
    (https://github.com/torch-spyre/torch-spyre/issues/6566). Runs on the host, on
    the transposed weights, before the move to Spyre.
    """
    count = 0
    for module in model.modules():
        if not isinstance(module, BertSelfAttention):
            continue
        proj = getattr(module, "qkv_proj", None)
        if proj is None or type(proj.quant_method) is not SpyreUnquantizedLinearMethod:
            continue
        if proj.weight.device.type != "cpu":
            continue
        sizes = [module.q_size, module.kv_size, module.kv_size]
        proj.quant_method = _SplitQKVLinearMethod(sizes)
        # Initial loading already transposed the weight. Retain the projection
        # and its bias so vLLM's checkpoint mapper and reload metadata stay valid.
        proj.quant_method.build_weight_t(proj, cast(torch.Tensor, proj.weight).data.t())
        module.forward = _split_qkv_forward.__get__(module)  # ty: ignore[invalid-assignment]
        count += 1
    if count:
        logger.info("Split the fused QKV projection of %d BertSelfAttention layers.", count)
        _group_heads(model)


# Head groups: where torch-spyre's SDPA planner KV-tiles a rectangle's attention but runs half the
# heads untiled, two groups with their own Q/K/V matmuls and out-proj rows keep Q, K, V out of HBM.
_GROUPS = 2


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


def _plan_head_groups(model: nn.Module, vllm_config: VllmConfig) -> None:
    """Record on each BertAttention the rectangles to run as head groups."""
    # The grouped out-proj skips the all-reduce, and eager never inlines encoder attention.
    if (
        get_tensor_model_parallel_world_size() > 1
        or vllm_config.compilation_config.mode != CompilationMode.STOCK_TORCH_COMPILE
    ):
        return
    from spyre_inference.v1.worker.spyre_shape_bucketer import encoder_rectangles

    try:
        rects = encoder_rectangles(vllm_config)
        for module in model.modules():
            if isinstance(module, BertAttention):
                sa = cast(BertSelfAttention, module.self)
                module.spyre_group_rects = grouped_rectangles(
                    rects, sa.num_heads, sa.num_kv_heads, sa.head_dim
                )
    except Exception as e:
        logger.warning_once("Encoder head groups disabled: %r", e)


def _register_qkv_groups(proj: nn.Module) -> None:
    for n in _QKV:
        w, b = getattr(proj, f"{n}_weight_t"), getattr(proj, f"{n}_bias")
        width = w.shape[1] // _GROUPS
        for g in range(_GROUPS):
            cols = slice(g * width, (g + 1) * width)
            proj.register_parameter(f"{n}_weight_t_{g}", _frozen(w.data[:, cols]))
            proj.register_parameter(f"{n}_bias_{g}", None if b is None else _frozen(b.data[cols]))


def _register_row_groups(dense: nn.Module) -> None:
    wo = cast(torch.Tensor, dense.weight).data  # stored transposed: [q_size, hidden]
    rows = wo.shape[0] // _GROUPS
    for g in range(_GROUPS):
        dense.register_parameter(f"weight_t_{g}", _frozen(wo[g * rows : (g + 1) * rows]))


class _GroupedRowsLinearMethod(SpyreUnquantizedLinearMethod):
    def build_weight_t(self, layer: nn.Module, w: torch.Tensor) -> None:
        super().build_weight_t(layer, w)
        _register_row_groups(layer)


def _grouped_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    sa, so = self.self, self.output
    mask = encoder_grid().mask
    if (
        mask is None
        or not encoder_inline_active()
        or (mask.shape[0], mask.shape[-1]) not in self.spyre_group_rects
    ):
        return so(sa(hidden_states), hidden_states)
    proj = sa.qkv_proj
    heads, kv_heads, d = sa.num_heads // _GROUPS, sa.num_kv_heads // _GROUPS, sa.head_dim
    out = so.dense.bias
    for g in range(_GROUPS):
        q, k, v = (
            spyre_linear_t(
                hidden_states, getattr(proj, f"{n}_weight_t_{g}"), getattr(proj, f"{n}_bias_{g}")
            )
            for n in _QKV
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
        out = torch.matmul(a.reshape(-1, heads * d), getattr(so.dense, f"weight_t_{g}")) + out
    return so.LayerNorm(out + hidden_states)


def _group_heads(model: nn.Module) -> None:
    """Adds per-group weights next to the split ones (in-graph weight slices are slower).

    The groups are rebuilt by each projection's quant method, so in-place reload refreshes them.
    """
    count, rects = 0, set[tuple[int, int]]()
    for module in model.modules():
        if not isinstance(module, BertAttention) or not getattr(module, "spyre_group_rects", None):
            continue
        proj, dense = cast(BertSelfAttention, module.self).qkv_proj, module.output.dense
        if (
            type(proj.quant_method) is not _SplitQKVLinearMethod
            or type(dense.quant_method) is not SpyreUnquantizedLinearMethod
        ):
            continue
        proj.quant_method.grouped = True
        _register_qkv_groups(proj)
        dense.quant_method = _GroupedRowsLinearMethod()
        _register_row_groups(dense)
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


def _frozen(t: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(t.contiguous(), requires_grad=False)


class SpyreSplitQKVMixin:
    """Run ``split_bert_qkv`` from vLLM's model-level post-load hook, which follows the
    per-layer weight transpose and precedes the runner's move to Spyre."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)  # ty: ignore[unknown-argument]
        _plan_head_groups(cast(nn.Module, self), vllm_config)

    def process_weights_after_loading(self) -> None:
        split_bert_qkv(cast(nn.Module, self))

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        if loaded is not None:
            for prefix, module in cast(nn.Module, self).named_modules():
                method = getattr(module, "quant_method", None)
                if isinstance(method, _SplitQKVLinearMethod | _GroupedRowsLinearMethod):
                    loaded.update(
                        f"{prefix}.{name}"
                        for name, _ in module.named_parameters(recurse=False)
                        if f"{prefix}.{'bias' if 'bias' in name else 'weight'}" in loaded
                    )
        return loaded


class SpyreBertEmbedding(CompileOutermost, SpyreTokenTypeEmbedding, BertEmbedding):
    """``BertEmbedding`` reading segment ids from the side buffer."""


class SpyreBertEmbeddingMixin:
    """Inject the Spyre embedding through ``BertEmbeddingModel._build_model``."""

    def _build_model(self, vllm_config: VllmConfig, prefix: str = "") -> BertModel:
        return BertModel(vllm_config=vllm_config, prefix=prefix, embedding_class=SpyreBertEmbedding)


class SpyreBertEmbeddingModel(SpyreSplitQKVMixin, SpyreBertEmbeddingMixin, BertEmbeddingModel):
    pass


class SpyreBertSpladeSparseEmbeddingModel(
    SpyreSplitQKVMixin, SpyreBertEmbeddingMixin, BertSpladeSparseEmbeddingModel
):
    pass


class SpyreBertForSequenceClassification(
    SpyreSplitQKVMixin, SpyreTokenTypeModel, BertForSequenceClassification
):
    spyre_embedding_class = SpyreBertEmbedding


class SpyreBertForTokenClassification(
    SpyreSplitQKVMixin, SpyreTokenTypeModel, BertForTokenClassification
):
    spyre_embedding_class = SpyreBertEmbedding


class SpyreBertForMaskedLM(SpyreSplitQKVMixin, SpyreTokenTypeModel, BertForMaskedLM):
    spyre_embedding_class = SpyreBertEmbedding
