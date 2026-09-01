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

"""Encoder-only (bidirectional) self-attention for Spyre without a KV cache.

Selected by ``TorchSpyrePlatform.get_attn_backend_cls`` for ENCODER/ENCODER_ONLY
layers. Operates on direct Q/K/V tensors rather than the paged KV-cache path.

Ragged→dense packing uses host-built indices + ``index_select`` (gather).
Pad slots gather a trailing zero row so the dense batch stays zeros in the
padding region.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.v1.attention.backend import AttentionLayer

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadata,
    SpyrePagedKVCache,
)
from spyre_inference.v1.pool import select_rows

# Pad seq length *and* head dim to the Spyre stick (64 fp16 elements).
# L-aligned keeps P·V's K stick-aligned; D-aligned keeps QKᵀ's K stick-aligned
# so Inductor never enters insert_bmm_padding (torch-spyre KeyError: 'val' on
# FX nodes missing meta["val"] when padding MiniLM's head_size=32).
ENCODER_SEQ_ALIGNMENT = 64


def _align_up(n: int, align: int = ENCODER_SEQ_ALIGNMENT) -> int:
    return max(align, (n + align - 1) // align * align)


def host_pack_indices(
    q_starts: list[int],
    lengths: list[int],
    aligned_len: int,
    pad_row: int,
) -> torch.Tensor:
    """Build ``[B, L]`` int64 row indices; pad slots point at ``pad_row``."""
    batch = len(q_starts)
    indices = torch.full((batch, aligned_len), pad_row, dtype=torch.int64)
    for s, (start, length) in enumerate(zip(q_starts, lengths)):
        if length > 0:
            indices[s, :length] = torch.arange(start, start + length, dtype=torch.int64)
    return indices


def host_unpack_indices(
    q_starts: list[int],
    query_lens: list[int],
    aligned_len: int,
    num_tokens: int,
) -> torch.Tensor:
    """Build ``[T]`` int64 indices from flat padded ``[B*L]`` back to tokens."""
    indices = torch.empty(num_tokens, dtype=torch.int64)
    for s, (start, length) in enumerate(zip(q_starts, query_lens)):
        if length > 0:
            base = s * aligned_len
            indices[start : start + length] = torch.arange(base, base + length, dtype=torch.int64)
    return indices


def _pad_head_dim_to_stick(flat: torch.Tensor, head_size_padded: int) -> torch.Tensor:
    """Pad last dim to a stick. MiniLM ``[T,H,32]`` cannot ``F.pad`` on Spyre."""
    head_size = flat.shape[-1]
    if head_size == head_size_padded:
        return flat
    device = flat.device
    if device.type == "spyre":
        flat = convert(flat, "cpu")
    flat = F.pad(flat, (0, head_size_padded - head_size))
    if device.type == "spyre":
        flat = convert(flat, device)
    return flat


def gather_pack(
    flat: torch.Tensor,
    pack_indices: torch.Tensor,
    head_size_padded: int,
) -> torch.Tensor:
    """Pack varlen ``[T, H, D]`` → padded ``[B, H, L, Dp]`` via ``index_select``.

    ``pack_indices`` is host ``[B, L]`` int64. Head dim is padded to the stick
    (CPU for MiniLM D=32), then a zero token row is ``F.pad``'d on-device.
    """
    batch, aligned_len = pack_indices.shape
    _t, num_heads, _d = flat.shape
    flat = _pad_head_dim_to_stick(flat, head_size_padded)
    flat_ext = F.pad(flat, (0, 0, 0, 0, 0, 1))
    gathered = select_rows(flat_ext, pack_indices)  # [B*L, H, Dp]
    packed = gathered.view(batch, aligned_len, num_heads, head_size_padded)
    return packed.permute(0, 2, 1, 3).contiguous()


def gather_unpack(
    attn_out: torch.Tensor,
    unpack_indices: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Unpack padded ``[B, H, L, Dp]`` → flat ``[T, H, D]`` via ``index_select``."""
    batch, num_heads, aligned_len, head_size_padded = attn_out.shape
    tokens = attn_out.permute(0, 2, 1, 3).contiguous()
    flat_padded = tokens.reshape(batch * aligned_len, num_heads, head_size_padded)
    gathered = select_rows(flat_padded, unpack_indices)
    if gathered.shape[-1] == head_size:
        return gathered
    # Crop is a slice, not pad. D=32 is half a stick; do it on CPU.
    if gathered.device.type == "spyre":
        gathered = convert(gathered, "cpu")
    return gathered[..., :head_size].contiguous()


def build_attention_mask(
    num_seqs: int,
    aligned_len: int,
    query_lens: list[int],
    kv_lens: list[int],
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Additive mask ``[B, 1, L, L]``: 0 where attend, ``-inf`` elsewhere.

    Built on the host (vectorized ``lt`` + nested ``where``), then ``convert``'d.
    On-device materialization is not stick-safe: Spyre cannot produce bool from
    int32 ``lt``, and cannot broadcast ``where`` of ``[B,1,L,1]`` × ``[B,1,1,L]``
    into ``[B,1,L,L]`` (no stick-scatter).
    """
    if device is None:
        device = torch.device("cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if num_seqs != len(query_lens):
        raise ValueError(f"num_seqs={num_seqs} != len(query_lens)={len(query_lens)}")

    q_len = torch.tensor(query_lens, dtype=torch.int32)
    kv_len = torch.tensor(
        [min(q, k) for q, k in zip(query_lens, kv_lens)],
        dtype=torch.int32,
    )
    q_pos = torch.arange(aligned_len, dtype=torch.int32)
    kv_pos = torch.arange(aligned_len, dtype=torch.int32)
    zeros = torch.zeros((), dtype=dtype)
    neg_inf = torch.tensor(torch.finfo(dtype).min, dtype=dtype)

    q_ok = (q_pos.unsqueeze(0) < q_len.unsqueeze(1)).unsqueeze(1).unsqueeze(-1)
    k_ok = (kv_pos.unsqueeze(0) < kv_len.unsqueeze(1)).unsqueeze(1).unsqueeze(2)
    mask = torch.where(q_ok, torch.where(k_ok, zeros, neg_inf), neg_inf)
    if device.type == "spyre":
        return convert(mask, device)
    return mask.to(device)


class SpyreEncoderAttentionImpl(SpyreAttentionImpl):
    """Bidirectional encoder self-attention (no KV cache).

    The platform selects this impl for ENCODER/ENCODER_ONLY layers (see
    ``TorchSpyrePlatform.get_attn_backend_cls``), so there is no per-call
    ``attn_type`` branch. Setup is shared with the paged decoder impl; forward
    packs with gather (``index_select``), runs one batched SDPA on Spyre, then
    unpacks with gather.
    """

    def forward(  # ty: ignore[invalid-method-override]
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: SpyrePagedKVCache,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,  # [num_tokens, num_heads, head_size]
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, kv_cache, output_scale, output_block_scale
        if attn_metadata is None:
            return output

        n = attn_metadata.num_actual_tokens
        query = query[:n]
        key = key[:n]
        value = value[:n]
        output = output[:n]

        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        num_seqs = attn_metadata.num_seqs
        scale = self.scale

        qsl = query_start_loc.cpu()
        # query_start_loc is a cumulative offset array of length num_seqs + 1;
        # diff() yields the num_seqs per-sequence query lengths in one pass.
        q_starts = qsl[:-1].tolist()
        query_lens = torch.diff(qsl).tolist()
        kv_lens = seq_lens.cpu().tolist()

        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        head_size = query.shape[2]
        # Pad D to the stick when models use a smaller head dim (MiniLM=32).
        # Zero-pad is exact for SDPA: padded Q/K slots add 0 to scores; cropped
        # output drops the zero V channels. Keeps self.scale = 1/sqrt(real D).
        head_size_padded = _align_up(head_size)

        max_len = max(query_lens, default=0)
        aligned_len = _align_up(max_len)

        target_device = output.device
        # Keep activations on the SDPA device; pack/unpack via index_select.
        if query.device.type != target_device.type:
            query = convert(query, target_device.type)
            key = convert(key, target_device.type)
            value = convert(value, target_device.type)

        pad_row = n  # index of the appended zero row in gather_pack
        q_pack_idx = host_pack_indices(q_starts, query_lens, aligned_len, pad_row)
        # K/V may be shorter than Q when seq_lens < query_lens; still use q_starts.
        kv_pack_lens = [min(q, k) for q, k in zip(query_lens, kv_lens)]
        kv_pack_idx = host_pack_indices(q_starts, kv_pack_lens, aligned_len, pad_row)
        unpack_idx = host_unpack_indices(q_starts, query_lens, aligned_len, n)

        q_batched = gather_pack(query, q_pack_idx, head_size_padded)
        k_batched = gather_pack(key, kv_pack_idx, head_size_padded)
        v_batched = gather_pack(value, kv_pack_idx, head_size_padded)

        mask = build_attention_mask(
            num_seqs,
            aligned_len,
            query_lens,
            kv_lens,
            dtype=query.dtype,
            device=target_device,
        )

        sdpa_kwargs: dict = {"is_causal": False, "scale": scale}
        if num_kv_heads != num_heads:
            sdpa_kwargs["enable_gqa"] = True

        # Single on-device SDPA: [num_seqs, H, L_aligned, D_padded].
        attn_out = F.scaled_dot_product_attention(
            q_batched, k_batched, v_batched, attn_mask=mask, **sdpa_kwargs
        )

        result = gather_unpack(attn_out, unpack_idx, head_size)
        if result.dtype != output.dtype:
            result = convert(result, dtype=output.dtype)

        # MiniLM D=32: flatten to [T, H*D] (384 = 6 sticks) so the write is aligned.
        use_flat_write = target_device.type == "spyre" and head_size % ENCODER_SEQ_ALIGNMENT != 0
        if use_flat_write:
            if result.device.type == "spyre":
                result = convert(result, "cpu")
            src = convert(result.reshape(n, -1).contiguous(), target_device.type, output.dtype)
            output.reshape(n, -1).copy_(src)
        else:
            if result.device.type != output.device.type:
                result = convert(result, output.device)
            output.copy_(result)

        return output


class SpyreEncoderAttentionBackend(SpyreAttentionBackend):
    """Encoder-only (no KV cache) variant of the Spyre backend."""

    # These layers have no KV cache, but vLLM still hands encoder-only specs a
    # zero-filled slot mapping, so upstream must skip `unified_kv_cache_update` entirely.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_impl_cls() -> type[SpyreEncoderAttentionImpl]:
        return SpyreEncoderAttentionImpl
