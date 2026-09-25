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
from spyre_inference.multimodal.utils import STICK, align_up, padded_sdpa

logger = init_logger(__name__)


def _host(t: torch.Tensor) -> torch.Tensor:
    """Detach a weight onto the host before reshaping or padding it.

    Strided slice-assignments lower silently wrong on Spyre-resident tensors.
    vLLM moves the model before our patches run, so every padding helper pulls
    its source to CPU here; the caller moves the finished weight back.
    Mirrors gemma4_vision._host.
    """
    return convert(t.detach(), device="cpu")


def _pad_qkv_weight(w: torch.Tensor, num_heads: int, orig_d: int, pad_d: int) -> torch.Tensor:
    """Zero-extend each head block of the Spyre-transposed fused QKV weight.

    SpyreTransposedWeightMethod stores weight as Wᵀ with shape [hidden, 3*H*D]
    (transposed from the original [3*H*D, hidden]).  We pad across the column
    axis — each head's D columns become pad_d columns — and keep the result in
    the same transposed layout [hidden, 3*H*D_pad] so the `x @ Wᵀ` forward
    in SpyreTransposedWeightMethod.apply still works unchanged.
    """
    w = _host(w)
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
    b = _host(b)
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
    w = _host(w)
    hidden = w.shape[1]
    out = torch.zeros(num_heads * pad_d, hidden, dtype=w.dtype)
    for h in range(num_heads):
        src = h * orig_d
        dst = h * pad_d
        out[dst : dst + orig_d, :] = w[src : src + orig_d, :]
    return out


def patch_siglip_vision_embeddings(model: torch.nn.Module, device: torch.device) -> None:
    """Patch SiglipVisionEmbeddings.forward to run the position embedding on CPU.

    aten.embedding(position_embedding.weight, position_ids) called eagerly on
    Spyre tensors hits torch-spyre's compile_once eager kernel, causing Dynamo
    re-entrancy / RecursionError when tracing. Keeping the position embedding lookup
    on CPU avoids this.
    """
    try:
        from vllm.model_executor.models.siglip import SiglipVisionEmbeddings
    except ImportError:
        return

    if getattr(SiglipVisionEmbeddings.forward, "_spyre_patched", False):
        return

    def _siglip_embeddings_forward(
        self: SiglipVisionEmbeddings,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        # Download once: position_embedding and position_ids are pinned to CPU
        # (see module setup below), and aten.embedding / aten.add called eagerly
        # on Spyre tensors hit compile_once, causing Dynamo re-entrancy.
        embeddings_cpu = convert(embeddings, device="cpu")
        if interpolate_pos_encoding:
            pos_emb = self.interpolate_pos_encoding(embeddings_cpu, height, width)
        else:
            pos_emb = self.position_embedding(self.position_ids)
        return convert(embeddings_cpu + pos_emb, device=device)

    _siglip_embeddings_forward._spyre_patched = True  # type: ignore[attr-defined]
    SiglipVisionEmbeddings.forward = _siglip_embeddings_forward  # type: ignore[method-assign]

    # Pin weights and buffers to CPU on every existing instance so the forward
    # replacement can perform the embedding lookup there without a device mismatch.
    for module in model.modules():
        if isinstance(module, SiglipVisionEmbeddings):
            module.position_embedding.to("cpu")
            module.register_buffer(
                "position_ids",
                module.position_ids.to("cpu"),  # ty: ignore[invalid-argument-type]
                persistent=False,
            )

    logger.info(
        "Spyre: patched SiglipVisionEmbeddings.forward to run the position "
        "embedding lookup on CPU (aten.embedding not traceable on Spyre)."
    )


def patch_siglip_attention(model: torch.nn.Module) -> None:
    """Pad SiglipAttention Q/K/V/O projections from head_dim=72 to 128 and replace forward.

    SigLIP so400m has head_dim=72 (not stick-aligned). Any on-device op that
    materialises a D=72 tensor triggers a restickify whose store index contains
    a fractional coefficient (9*c2/8), which torch-spyre's inductor cannot lower
    (torch-spyre#1353). Fix: zero-extend each head block of qkv_proj and out_proj
    from 72→128 at load time (mirroring hf-adapters' pad_attention_heads_linear),
    then replace SiglipAttention.forward to bypass MMEncoderAttention and call
    padded_sdpa directly. Scale is fixed to the original head_dim so the softmax
    temperature is unchanged.

    No-op for models already stick-aligned (head_dim % 64 == 0, e.g. D=64).
    """
    try:
        from vllm.model_executor.models.siglip import SiglipAttention
    except ImportError:
        return

    if getattr(SiglipAttention.forward, "_spyre_patched", False):
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
    scale = orig_head_dim**-0.5

    # Pad weights on every SiglipAttention instance.  _host() pulls each weight
    # to CPU before the slice-assignments (Spyre slice-assign is silently wrong);
    # .to(dev) moves the padded result back.  Mirrors gemma4_vision._prepare_attention.
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

        module._spyre_head_dim_padded = True

    def _siglip_attention_forward(
        self: SiglipAttention,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        bsz, seq_len, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        # qkv: [B, S, 3*H*pad_head_dim]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads_per_partition, pad_head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads_per_partition, pad_head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads_per_partition, pad_head_dim).transpose(1, 2)

        # padded_sdpa handles seq-alignment; scale fixed to orig head_dim.
        # Use a full-attend mask: SigLIP is bidirectional with no real mask.
        from spyre_inference.custom_ops.vit_attn import _full_attend_mask

        attn_out = padded_sdpa(q, k, v, _full_attend_mask(seq_len), scale=scale)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_output, _ = self.out_proj(attn_out)
        return attn_output, None

    _siglip_attention_forward._spyre_patched = True  # type: ignore[attr-defined]
    SiglipAttention.forward = _siglip_attention_forward  # type: ignore[method-assign]
    logger.info(
        "Spyre: padded SiglipAttention head_dim %d -> %d and replaced forward "
        "to use padded_sdpa on-device (torch-spyre#1353).",
        orig_head_dim,
        pad_head_dim,
    )


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Apply SigLIP vision workarounds."""
    patch_siglip_vision_embeddings(model, device)
    patch_siglip_attention(model)
