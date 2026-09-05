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

Ragged→dense packing uses compiled ``index_copy_`` on Spyre into a
slot-major workspace (decoder KV / torch-spyre#3705) so ``view(B, L, …)``
is address-order-preserving. Default-layout ``view`` after ``index_copy_``
scrambles B>1 (real-slot cosine ~0.07). Body-pad dests write an extra
dummy row (not slot 0 — that is CLS). Unpack is ``index_select``. ``B=1``
with ``T == L`` and a full prompt compiles permute+SDPA (no ``attn_mask``).
Any live pad uses packed QK: compile matmul only, eager pad add, compile P·V
(Inductor ``matmul + mask`` → ``F.sdpa`` drops the mask; BGE cosine
~0.46). Dest/mask use ``min(qsl, seq_lens, num_actual_tokens)``.
Dest/unpack stay on host when ``T == L``. Pack tensors are built once
per step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import AttentionLayer

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadata,
    SpyrePagedKVCache,
    slot_major_kv_layout,
)
from spyre_inference.v1.pool import select_rows
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    default_encoder_len_buckets,
    pick_encoder_attention_shape,
    pooling_warmup_shapes,
)

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


def _content_query_lens(
    qsl_lens: list[int],
    kv_lens: list[int],
    num_actual_tokens: int | None = None,
) -> list[int]:
    """Per-seq counts for dest/mask. ``query_start_loc`` can include 1D body pad.

    When qsl and ``seq_lens`` are both the body bucket, ``num_actual_tokens``
    (unpadded scheduled count) still marks pad. B=1 is ``min(qsl, seq, n)``.
    B>1 shrinks the tail: runner pad is appended after the last sequence.
    """
    out = [min(int(q), int(k)) for q, k in zip(qsl_lens, kv_lens)]
    if not out or num_actual_tokens is None:
        return out
    n = int(num_actual_tokens)
    if len(out) == 1:
        out[0] = min(out[0], n)
        return out
    excess = sum(out) - n
    if excess <= 0:
        return out
    for i in range(len(out) - 1, -1, -1):
        take = min(out[i], excess)
        out[i] -= take
        excess -= take
        if excess == 0:
            break
    return out


def dummy_pack_row(lengths: list[int], aligned_len: int) -> int:
    """First pad slot in the ``[B, L]`` grid, or ``0`` when the grid is full.

    Tests only. Serve writes body-pad to an extra workspace row (``B×L``),
    never ``0``: a full grid plus a 1D body bucket ``> B×L`` would otherwise
    overwrite CLS.
    """
    for s, length in enumerate(lengths):
        if length < aligned_len:
            return s * aligned_len + length
    return 0


def host_scatter_pack_dest(
    q_starts: list[int],
    lengths: list[int],
    aligned_len: int,
    num_src_rows: int,
    dummy_row: int,
) -> torch.Tensor:
    """``[num_src_rows]`` packed-row dest for each source token.

    Real tokens write ``s * L + pos``. Body-pad rows write ``dummy_row``
    (serve: extra workspace row ``B×L``, not a real token).
    """
    dest = torch.full((num_src_rows,), dummy_row, dtype=torch.int64)
    for s, (start, length) in enumerate(zip(q_starts, lengths)):
        if length > 0:
            dest[start : start + length] = torch.arange(
                s * aligned_len, s * aligned_len + length, dtype=torch.int64
            )
    return dest


def host_unpack_indices(
    q_starts: list[int],
    query_lens: list[int],
    aligned_len: int,
    num_tokens: int,
) -> torch.Tensor:
    """Build ``[T]`` int64 indices from flat padded ``[B*L]`` back to tokens.

    ``num_tokens`` may exceed the real count; unfilled entries stay ``0``
    (a safe row to read — nothing downstream reads those output rows).
    """
    indices = torch.zeros(num_tokens, dtype=torch.int64)
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


def _is_identity_row_map(indices: torch.Tensor, num_rows: int) -> bool:
    """True when ``indices`` is ``0 .. num_rows-1`` (no pad gather). Host only."""
    if indices.numel() != num_rows or indices.device.type != "cpu":
        return False
    flat = indices.reshape(-1)
    return bool(torch.equal(flat, torch.arange(num_rows, dtype=flat.dtype)))


def _is_b1_dense_body(batch: int, num_src: int, aligned_len: int) -> bool:
    """Single sequence already shaped ``[L, H, D]`` (token bucket == SDPA L)."""
    return batch == 1 and num_src == aligned_len


def _is_b1_fused_sdpa(
    batch: int, padded_tokens: int, aligned_len: int, real_len: int
) -> bool:
    """Compiled permute+SDPA is only legal when the mask is dense (no live pad)."""
    return _is_b1_dense_body(batch, padded_tokens, aligned_len) and real_len == aligned_len


def _index_copy_kernel(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Tiny mutation, compiled alone — do not fuse with SDPA."""
    dst.index_copy_(0, index, src)
    return dst


_compiled_index_copy: object | None = None
_compiled_b1_sdpa: dict[bool, object] = {}
_compiled_packed_qk: dict[bool, object] = {}
_compiled_packed_pv: dict[bool, object] = {}
# Cached on fused B=1 so later layers skip dest/mask build. ``numel()==0`` is
# the forward sentinel — do not convert or H2D this.
_FUSED_NO_MASK = torch.empty(0)


def _compile_if_spyre(cache: dict[bool, object], kernel, enable_gqa: bool, device_type: str):
    """Compile one kernel on Spyre. QK and P·V must not share a graph (SDPA fusion)."""
    if device_type != "spyre":
        return kernel
    compiled = cache.get(enable_gqa)
    if compiled is None:
        compiled = torch.compile(kernel, dynamic=False)
        cache[enable_gqa] = compiled
    return compiled


def _b1_sdpa_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """``[T,H,D]`` → SDPA ``[1,H,T,D]`` → ``[T,H,D]``. No mask, no eager ``contiguous``.

    Full-bucket fused path only (``real_len == L``). A zeros ``attn_mask`` is
    still an H2D + per-layer add; Spyre compiled SDPA also drops a live pad
    mask, so pad stays on the packed path.
    """
    q = query.unsqueeze(0).transpose(1, 2)
    k = key.unsqueeze(0).transpose(1, 2)
    v = value.unsqueeze(0).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)
    t, h, d = query.shape
    return out.transpose(1, 2).reshape(t, h, d)


def _b1_sdpa_kernel_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    q = query.unsqueeze(0).transpose(1, 2)
    k = key.unsqueeze(0).transpose(1, 2)
    v = value.unsqueeze(0).transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=scale, enable_gqa=True
    )
    t, h, d = query.shape
    return out.transpose(1, 2).reshape(t, h, d)


def host_key_pad_mask(mask: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """CPU ``[B,1,L,L]`` → dense ``[B*KV, 1, L, L]``. Spyre cannot broadcast key-pad."""
    key = mask[:, :, :1, :]
    batch, _, _, length = key.shape
    return (
        key.expand(batch, num_kv_heads, length, length)
        .reshape(batch * num_kv_heads, 1, length, length)
        .contiguous()
    )


def _packed_qk_matmul(query: torch.Tensor, key: torch.Tensor, scale: float) -> torch.Tensor:
    """``[B, H, L, D]`` → scores ``[B*Hkv, G, L, L]``. Mask stays out of this graph."""
    batch, hq, length, dim = query.shape
    hkv = key.shape[1]
    g = hq // hkv
    q = query.reshape(batch, hkv, g, length, dim).reshape(batch * hkv, g, length, dim)
    k = key.reshape(batch * hkv, 1, length, dim)
    return torch.matmul(q, k.transpose(-2, -1)) * scale


def _packed_pv(scores: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    batch, hkv, length, dim = value.shape
    g = scores.shape[1]
    v = value.reshape(batch * hkv, 1, length, dim)
    scores_max = torch.amax(scores, dim=-1, keepdim=True)
    probs = torch.exp(scores - scores_max)
    out = torch.matmul(probs, v) / probs.sum(dim=-1, keepdim=True)
    return out.reshape(batch, hkv * g, length, dim)


def _packed_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    enable_gqa: bool,
) -> torch.Tensor:
    """Scatter-path attention. Compile QK and P·V separately; pad add is eager.

    Compiling ``matmul + mask`` lets Inductor rewrite to ``F.sdpa``, which drops
    ``attn_mask`` on Spyre (BGE cosine ~0.46).
    """
    device_type = query.device.type
    qk = _compile_if_spyre(_compiled_packed_qk, _packed_qk_matmul, enable_gqa, device_type)
    pv = _compile_if_spyre(_compiled_packed_pv, _packed_pv, enable_gqa, device_type)
    scores = qk(query, key, scale)
    if mask.shape != scores.shape:
        mask = mask.expand_as(scores).contiguous()
    return pv(scores + mask, value)


def _b1_dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    enable_gqa: bool,
    head_size_padded: int,
    head_size: int,
) -> torch.Tensor:
    """B=1 ``T==L`` full-bucket attention. Dest/unpack/mask unused. Compiled permute+SDPA."""
    query = _pad_head_dim_to_stick(query, head_size_padded)
    key = _pad_head_dim_to_stick(key, head_size_padded)
    value = _pad_head_dim_to_stick(value, head_size_padded)
    kernel = _b1_sdpa_kernel_gqa if enable_gqa else _b1_sdpa_kernel
    kernel = _compile_if_spyre(_compiled_b1_sdpa, kernel, enable_gqa, query.device.type)
    result = kernel(query, key, value, scale)
    if result.shape[-1] == head_size:
        return result
    if result.device.type == "spyre":
        result = convert(result, "cpu")
    return result[..., :head_size].contiguous()


def _index_copy(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Eager on CPU; compiled ``index_copy_`` on Spyre (eager falls back / rejects)."""
    global _compiled_index_copy
    if dst.device.type != "spyre":
        return _index_copy_kernel(dst, index, src)
    if _compiled_index_copy is None:
        _compiled_index_copy = torch.compile(_index_copy_kernel, dynamic=False)
    return _compiled_index_copy(dst, index, src)


def _dest_on_flat_device(dest_idx: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
    """Spyre compiled index_copy_ wants int32; eager CPU wants int64."""
    if dest_idx.device.type == "spyre":
        return dest_idx
    if flat.device.type == "spyre":
        return convert(dest_idx.to(torch.int32), flat.device)
    return dest_idx.to(device=flat.device, dtype=torch.int64)


def _zeros_slot_major(
    rows: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """``[rows, H, D]`` zeros. Spyre pins rows at device position 0.

    ``empty`` + ``zero_`` stays on device. Pass size as one tuple —
    ``empty_with_layout`` rejects ``empty(B, L, H, D, device_layout=)``.
    """
    if device.type != "spyre":
        return torch.zeros(rows, num_heads, head_size, dtype=dtype, device=device)
    layout = slot_major_kv_layout(rows, num_heads, head_size, dtype)
    return torch.empty(  # ty: ignore[no-matching-overload]
        (rows, num_heads, head_size),
        dtype=dtype,
        device=device,
        device_layout=layout,
    ).zero_()


def gather_pack(
    flat: torch.Tensor,
    pack_indices: torch.Tensor,
    head_size_padded: int,
) -> torch.Tensor:
    """Pack via ``F.pad`` zero row + ``index_select`` of ``B×L``.

    Reference / tests. Serve uses ``scatter_pack``. ``B=1`` identity skips
    ``index_select``.
    """
    batch, aligned_len = pack_indices.shape
    _t, num_heads, _d = flat.shape
    flat = _pad_head_dim_to_stick(flat, head_size_padded)
    if batch == 1 and _is_identity_row_map(pack_indices, flat.shape[0]):
        packed = flat.unsqueeze(0)
        return packed.permute(0, 2, 1, 3).contiguous()
    flat_ext = F.pad(flat, (0, 0, 0, 0, 0, 1))
    gathered = select_rows(flat_ext, pack_indices)  # [B*L, H, Dp]
    packed = gathered.view(batch, aligned_len, num_heads, head_size_padded)
    return packed.permute(0, 2, 1, 3).contiguous()


def scatter_pack(
    flat: torch.Tensor,
    dest_idx: torch.Tensor,
    batch: int,
    aligned_len: int,
    head_size_padded: int,
) -> torch.Tensor:
    """Pack varlen ``[T, H, D]`` → ``[B, H, L, Dp]`` via compiled ``index_copy_``.

    ``dest_idx`` is ``[T]`` packed-row ids (host or device). Pad slots stay
    zeros. Body-pad dests write an extra dummy row (``B×L``), not CLS.

    ``B=1`` with ``T == L`` skips ``index_copy_``: the runner already padded
    the body to the SDPA length. The attention mask hides leftover pad tokens.

    Spyre workspace is slot-major so ``view(B, L, …)`` splits an outermost
    dim (decoder KV). Default-layout ``view`` after ``index_copy_`` scrambles
    ``B>1``.
    """
    _t, num_heads, _d = flat.shape
    flat = _pad_head_dim_to_stick(flat, head_size_padded)
    # Fused QKV views are strided (BGE ``stride=(2304, 64, 1)``). Compiled
    # ``index_copy_`` from that layout writes the wrong rows.
    if not flat.is_contiguous():
        flat = flat.contiguous()
    if _is_b1_dense_body(batch, flat.shape[0], aligned_len):
        packed = flat.unsqueeze(0)
        return packed.permute(0, 2, 1, 3).contiguous()
    packed_rows = batch * aligned_len
    # Extra row is the dummy dest. Slot-major prefix view is 2c on hardware.
    workspace = _zeros_slot_major(
        packed_rows + 1,
        num_heads,
        head_size_padded,
        flat.dtype,
        flat.device,
    )
    _index_copy(workspace, _dest_on_flat_device(dest_idx, flat), flat)
    packed = workspace[:packed_rows].view(batch, aligned_len, num_heads, head_size_padded)
    return packed.permute(0, 2, 1, 3).contiguous()


def gather_unpack(
    attn_out: torch.Tensor,
    unpack_indices: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Unpack padded ``[B, H, L, Dp]`` → flat ``[T, H, D]`` via ``index_select``.

    Identity ``B=1`` (``T == B×L``) is a reshape; pad / multi-seq still gather.
    """
    batch, num_heads, aligned_len, head_size_padded = attn_out.shape
    tokens = attn_out.permute(0, 2, 1, 3).contiguous()
    flat_padded = tokens.reshape(batch * aligned_len, num_heads, head_size_padded)
    if _is_identity_row_map(unpack_indices, flat_padded.shape[0]) or _is_b1_dense_body(
        batch, unpack_indices.shape[0], aligned_len
    ):
        gathered = flat_padded
    else:
        gathered = select_rows(flat_padded, unpack_indices)
    if gathered.shape[-1] == head_size:
        return gathered
    if gathered.device.type == "spyre":
        gathered = convert(gathered, "cpu")
    return gathered[..., :head_size].contiguous()


def _indices_for_device(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move unpack indices onto ``device`` once. Spyre index_select is int32."""
    if device.type == "spyre":
        cpu = indices if indices.device.type == "cpu" else indices.cpu()
        return convert(cpu.to(torch.int32), device)
    return indices.to(device=device, dtype=torch.long)


def _dest_for_device(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move scatter dest onto ``device`` once. Spyre compiled index_copy_ is int32."""
    if device.type == "spyre":
        cpu = indices if indices.device.type == "cpu" else indices.cpu()
        return convert(cpu.to(torch.int32), device)
    return indices.to(device=device, dtype=torch.long)


def _ensure_encoder_pack(
    attn_metadata: SpyreAttentionMetadata,
    *,
    padded_tokens: int,
    n: int,
    query: torch.Tensor,
    num_kv_heads: int,
    target_device: torch.device,
    cached_encoder_shapes: list[tuple[int, int]],
    cached_max_num_seqs: int,
    cached_max_model_len: int,
    cached_max_num_batched_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build scatter dest + unpack + mask once per step; later layers reuse them."""
    if attn_metadata.encoder_q_pack_idx is not None:
        assert attn_metadata.encoder_kv_pack_idx is not None
        assert attn_metadata.encoder_unpack_idx is not None
        assert attn_metadata.encoder_attn_mask is not None
        assert attn_metadata.encoder_key_pad_mask is not None
        return (
            attn_metadata.encoder_q_pack_idx,
            attn_metadata.encoder_kv_pack_idx,
            attn_metadata.encoder_unpack_idx,
            attn_metadata.encoder_attn_mask,
            attn_metadata.encoder_key_pad_mask,
        )

    qsl = attn_metadata.query_start_loc.cpu()
    q_starts = qsl[:-1].tolist()
    qsl_lens = torch.diff(qsl).tolist()
    kv_lens = attn_metadata.seq_lens.cpu().tolist()
    num_seqs = attn_metadata.num_seqs
    q_starts = q_starts[:num_seqs]
    qsl_lens = qsl_lens[:num_seqs]
    kv_lens = kv_lens[:num_seqs]
    # Pick (B, L) from padded qsl/seq so identity T==L still matches the body
    # bucket. Cap dest/mask with num_actual_tokens after that (BGE: both
    # cu_seqlens and seq_lens can be the bucket).
    max_len = max(_content_query_lens(qsl_lens, kv_lens), default=0)
    pair = pick_encoder_attention_shape(
        num_seqs,
        max_len,
        cached_encoder_shapes,
        cached_max_num_seqs,
        cached_max_model_len,
        cached_max_num_batched_tokens,
    )
    batch_bucket, aligned_len = pair if pair is not None else (num_seqs, _align_up(max_len))
    query_lens = _content_query_lens(qsl_lens, kv_lens, num_actual_tokens=n)
    orig_q_starts = q_starts
    orig_query_lens = query_lens
    fused = _is_b1_fused_sdpa(
        batch_bucket,
        padded_tokens,
        aligned_len,
        orig_query_lens[0] if orig_query_lens else 0,
    )
    if fused:
        # No dest, unpack, or L×L mask. Compiled SDPA has no attn_mask.
        empty = torch.empty(0, dtype=torch.int64)
        attn_metadata.encoder_q_pack_idx = empty
        attn_metadata.encoder_kv_pack_idx = empty
        attn_metadata.encoder_unpack_idx = empty
        attn_metadata.encoder_attn_mask = _FUSED_NO_MASK
        attn_metadata.encoder_key_pad_mask = _FUSED_NO_MASK
        return (
            attn_metadata.encoder_q_pack_idx,
            attn_metadata.encoder_kv_pack_idx,
            attn_metadata.encoder_unpack_idx,
            attn_metadata.encoder_attn_mask,
            attn_metadata.encoder_key_pad_mask,
        )
    if batch_bucket > num_seqs:
        q_starts = q_starts + [n] * (batch_bucket - num_seqs)
        query_lens = query_lens + [0] * (batch_bucket - num_seqs)
        kv_lens = kv_lens + [0] * (batch_bucket - num_seqs)

    dummy_row = batch_bucket * aligned_len
    kv_pack_lens = [min(q, k) for q, k in zip(query_lens, kv_lens)]
    q_dest = host_scatter_pack_dest(q_starts, query_lens, aligned_len, padded_tokens, dummy_row)
    kv_dest = host_scatter_pack_dest(q_starts, kv_pack_lens, aligned_len, padded_tokens, dummy_row)
    unpack_idx = host_unpack_indices(orig_q_starts, orig_query_lens, aligned_len, padded_tokens)
    mask_cpu = build_attention_mask(
        batch_bucket,
        aligned_len,
        query_lens,
        kv_lens,
        dtype=query.dtype,
        device=torch.device("cpu"),
    )
    key_pad = host_key_pad_mask(mask_cpu, num_kv_heads)
    if target_device.type == "spyre":
        mask = convert(mask_cpu, target_device)
        key_pad = convert(key_pad, target_device)
    else:
        mask = mask_cpu.to(target_device)
        key_pad = key_pad.to(target_device)

    # B=1 T==L with live pad still identity-packs; dest/unpack stay on host.
    b1_dense = _is_b1_dense_body(batch_bucket, padded_tokens, aligned_len)
    if b1_dense:
        attn_metadata.encoder_q_pack_idx = q_dest
        attn_metadata.encoder_kv_pack_idx = kv_dest
        attn_metadata.encoder_unpack_idx = unpack_idx
    else:
        attn_metadata.encoder_q_pack_idx = _dest_for_device(q_dest, target_device)
        attn_metadata.encoder_kv_pack_idx = _dest_for_device(kv_dest, target_device)
        attn_metadata.encoder_unpack_idx = _indices_for_device(unpack_idx, target_device)
    attn_metadata.encoder_attn_mask = mask
    attn_metadata.encoder_key_pad_mask = key_pad
    return (
        attn_metadata.encoder_q_pack_idx,
        attn_metadata.encoder_kv_pack_idx,
        attn_metadata.encoder_unpack_idx,
        attn_metadata.encoder_attn_mask,
        attn_metadata.encoder_key_pad_mask,
    )


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
    packs Q/K/V with scatter into a slot-major workspace (decoder KV
    layout; extra dummy row for body-pad), then attention and gather-unpack.
    ``B=1`` ``T == L`` with a full prompt compiles permute+SDPA; live pad
    uses packed QK (matmul and P·V compiled apart so Inductor cannot fuse
    to ``F.sdpa``).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # get_current_vllm_config() only works at construction time; forward()
        # runs through a custom-op boundary that loses the context.
        cfg = get_current_vllm_config()
        self._cached_max_num_seqs = cfg.scheduler_config.max_num_seqs
        self._cached_max_model_len = cfg.model_config.max_model_len
        self._cached_max_num_batched_tokens = cfg.scheduler_config.max_num_batched_tokens
        self._cached_encoder_shapes = pooling_warmup_shapes(
            max_num_seqs=self._cached_max_num_seqs,
            max_model_len=self._cached_max_model_len,
            max_num_batched_tokens=self._cached_max_num_batched_tokens,
            len_bucket=default_encoder_len_buckets(self._cached_max_model_len),
        )

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

        # query/key/value/output are padded to the runner's warmed body-bucket
        # size, not num_actual_tokens. Keep that shape or index_select
        # recompiles per request.
        n = attn_metadata.num_actual_tokens
        padded_tokens = query.shape[0]
        target_device = output.device
        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        head_size = query.shape[2]
        head_size_padded = _align_up(head_size)
        scale = self.scale

        pack = _ensure_encoder_pack(
            attn_metadata,
            padded_tokens=padded_tokens,
            n=n,
            query=query,
            num_kv_heads=num_kv_heads,
            target_device=target_device,
            cached_encoder_shapes=self._cached_encoder_shapes,
            cached_max_num_seqs=self._cached_max_num_seqs,
            cached_max_model_len=self._cached_max_model_len,
            cached_max_num_batched_tokens=self._cached_max_num_batched_tokens,
        )

        if query.device.type != target_device.type:
            query = convert(query, target_device.type)
            key = convert(key, target_device.type)
            value = convert(value, target_device.type)

        q_pack, kv_pack, unpack_idx, mask, key_pad_mask = pack
        if mask.numel() == 0:
            result = _b1_dense_attention(
                query,
                key,
                value,
                scale,
                num_kv_heads != num_heads,
                head_size_padded,
                head_size,
            )
        else:
            batch, _, aligned_len, _ = mask.shape
            q_batched = scatter_pack(query, q_pack, batch, aligned_len, head_size_padded)
            k_batched = scatter_pack(key, kv_pack, batch, aligned_len, head_size_padded)
            v_batched = scatter_pack(value, kv_pack, batch, aligned_len, head_size_padded)
            attn_out = _packed_masked_attention(
                q_batched,
                k_batched,
                v_batched,
                key_pad_mask,
                scale,
                num_kv_heads != num_heads,
            )
            result = gather_unpack(attn_out, unpack_idx, head_size)
        if result.dtype != output.dtype:
            result = convert(result, dtype=output.dtype)

        # MiniLM D=32: flatten to [T, H*D] (384 = 6 sticks) so the write is aligned.
        use_flat_write = target_device.type == "spyre" and head_size % ENCODER_SEQ_ALIGNMENT != 0
        if use_flat_write:
            if result.device.type == "spyre":
                result = convert(result, "cpu")
            src = convert(
                result.reshape(padded_tokens, -1).contiguous(), target_device.type, output.dtype
            )
            output.reshape(padded_tokens, -1).copy_(src)
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
