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

"""SigLIP vision-encoder workarounds for Spyre."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert
from spyre_inference.multimodal.utils import STICK, align_up

logger = init_logger(__name__)


def _pad_qkv_weight(w: torch.Tensor, num_heads: int, orig_d: int, pad_d: int) -> torch.Tensor:
    """Zero-extend each head block of the Spyre-transposed fused QKV weight.

    SpyreTransposedWeightMethod stores weight as Wᵀ with shape [hidden, 3*H*D]
    (transposed from the original [3*H*D, hidden]).  We pad across the column
    axis — each head's D columns become pad_d columns — and keep the result in
    the same transposed layout [hidden, 3*H*D_pad] so the `x @ Wᵀ` forward
    in SpyreTransposedWeightMethod.apply still works unchanged.
    """
    w = convert(w.detach(), device="cpu")
    hidden = w.shape[0]
    out = torch.zeros(hidden, 3 * num_heads * pad_d, dtype=w.dtype)
    for b in range(3):  # Q, K, V blocks
        for h in range(num_heads):
            src = b * num_heads * orig_d + h * orig_d
            dst = b * num_heads * pad_d + h * pad_d
            out[:, dst : dst + orig_d] = w[:, src : src + orig_d]
    return out


def _pad_qkv_bias(b: torch.Tensor, num_heads: int, orig_d: int, pad_d: int) -> torch.Tensor:
    """Zero-extend each head block of a fused QKV bias [3*H*D] -> [3*H*D_pad]."""
    b = convert(b.detach(), device="cpu")
    out = torch.zeros(3 * num_heads * pad_d, dtype=b.dtype)
    for blk in range(3):
        for h in range(num_heads):
            src = blk * num_heads * orig_d + h * orig_d
            dst = blk * num_heads * pad_d + h * pad_d
            out[dst : dst + orig_d] = b[src : src + orig_d]
    return out


def _pad_out_weight(w: torch.Tensor, num_heads: int, orig_d: int, pad_d: int) -> torch.Tensor:
    """Zero-extend each head block of the Spyre-transposed out_proj weight.

    SpyreTransposedWeightMethod stores out_proj weight as Wᵀ with shape
    [H*D, hidden] (transposed from the original [hidden, H*D]).  We pad
    across the row axis — each head's orig_d rows become pad_d rows — and
    keep the result in the same transposed layout [H*D_pad, hidden] so the
    `x @ Wᵀ` forward still works unchanged.
    """
    w = convert(w.detach(), device="cpu")
    hidden = w.shape[1]
    out = torch.zeros(num_heads * pad_d, hidden, dtype=w.dtype)
    for h in range(num_heads):
        src = h * orig_d
        dst = h * pad_d
        out[dst : dst + orig_d, :] = w[src : src + orig_d, :]
    return out


def patch_siglip_attention(model: torch.nn.Module) -> None:
    """Pad SiglipAttention Q/K/V/O projections from head_dim=72 to 128 for MMEncoderAttention.

    SigLIP so400m has head_dim=72 (not stick-aligned). Any on-device op that
    materialises a D=72 tensor triggers a restickify whose store index contains
    a fractional coefficient (9*c2/8), which torch-spyre's inductor cannot lower
    (torch-spyre#1353). Fix: zero-extend each head block of qkv_proj and out_proj
    from 72→128 at load time (mirroring hf-adapters' pad_attention_heads_linear),
    and update module.attn.head_size to 128 so MMEncoderAttention (SpyreMMEncoderAttention)
    handles attention natively on-device.

    No-op for models already stick-aligned (head_dim % 64 == 0, e.g. D=64).
    """
    try:
        from vllm.model_executor.models.siglip import SiglipAttention
    except ImportError:
        return

    # Discover orig_head_dim from the first SiglipAttention instance.
    orig_head_dim: int | None = None
    for module in model.modules():
        if isinstance(module, SiglipAttention):
            orig_head_dim = module.head_dim
            break
    if orig_head_dim is None or align_up(orig_head_dim, STICK) == orig_head_dim:
        return  # no SiglipAttention found, or already stick-aligned

    pad_head_dim = align_up(orig_head_dim, STICK)

    # Pad weights on every SiglipAttention instance.  Each weight is pulled
    # to CPU before the slice-assignments (Spyre slice-assign is silently wrong);
    # .to(dev) moves the padded result back.
    for module in model.modules():
        if not isinstance(module, SiglipAttention):
            continue
        if getattr(module, "_spyre_head_dim_padded", False):
            continue
        num_heads = module.num_heads_per_partition

        qkv = module.qkv_proj
        qkv_w = cast(torch.Tensor, qkv.weight)
        dev = qkv_w.device
        qkv.weight = nn.Parameter(
            convert(
                _pad_qkv_weight(qkv_w.data, num_heads, orig_head_dim, pad_head_dim),
                device=dev,
            ),
            requires_grad=False,
        )
        if qkv.bias is not None:
            qkv_b = cast(torch.Tensor, qkv.bias)
            qkv.bias = nn.Parameter(
                convert(
                    _pad_qkv_bias(qkv_b.data, num_heads, orig_head_dim, pad_head_dim),
                    device=dev,
                ),
                requires_grad=False,
            )

        out = module.out_proj
        out_w = cast(torch.Tensor, out.weight)
        out.weight = nn.Parameter(
            convert(
                _pad_out_weight(out_w.data, num_heads, orig_head_dim, pad_head_dim),
                device=dev,
            ),
            requires_grad=False,
        )
        # out_proj bias is [hidden] — output dim, unchanged.

        if hasattr(module, "attn") and module.attn is not None:
            module.attn.head_size = pad_head_dim

        module._spyre_head_dim_padded = True

    logger.info(
        "Spyre: padded SiglipAttention head_dim %d -> %d and updated MMEncoderAttention head_size.",
        orig_head_dim,
        pad_head_dim,
    )


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Apply SigLIP vision workarounds."""
    patch_siglip_attention(model)
