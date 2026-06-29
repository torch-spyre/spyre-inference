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

Used by ``SpyreEncoderAttentionImpl``, which the platform selects for
ENCODER/ENCODER_ONLY layers (see ``TorchSpyrePlatform.get_attn_backend_cls``).
Operates on direct Q/K/V tensors rather than the paged KV-cache path.
"""

import torch
import torch.nn.functional as F

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import _overwrite

# Pad each sequence's KV length to the Spyre stick (64 fp16 elements) so the
# cache-less encoder SDPA's P·V matmul gets a valid hardware layout.
ENCODER_SEQ_ALIGNMENT = 64


def forward_encoder_attention(
    query: torch.Tensor,  # [num_tokens, num_heads, head_size]
    key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
    output: torch.Tensor,  # [num_tokens, num_heads, head_size]
    *,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_seqs: int,
    scale: float,
) -> torch.Tensor:
    """Bidirectional self-attention for encoder layers (no KV cache), on Spyre.

    Gathers/pads the flat varlen Q/K/V into one dense batch on CPU, runs a single
    masked SDPA on ``output.device``, and scatters the unpadded results back. The
    batched, stick-aligned shape is required for Spyre layout propagation.
    """
    qsl = query_start_loc.cpu()
    q_starts = qsl[:-1].tolist()
    q_ends = qsl[1:].tolist()
    kv_lens = seq_lens.cpu().tolist()

    num_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    head_size = query.shape[2]

    # Gather host-side (Q/K/V usually arrive on CPU; convert defensively).
    query_cpu = convert(query, "cpu")
    key_cpu = convert(key, "cpu")
    value_cpu = convert(value, "cpu")
    dtype = query_cpu.dtype

    query_lens = [q_ends[s] - q_starts[s] for s in range(num_seqs)]
    max_len = max(query_lens, default=0)
    aligned_len = (
        (max_len + ENCODER_SEQ_ALIGNMENT - 1) // ENCODER_SEQ_ALIGNMENT * ENCODER_SEQ_ALIGNMENT
    )
    aligned_len = max(aligned_len, ENCODER_SEQ_ALIGNMENT)

    # Dense padded batch: [num_seqs, H, L_aligned, D]. Padding rows stay zero.
    q_batched = torch.zeros(num_seqs, num_heads, aligned_len, head_size, dtype=dtype)
    k_batched = torch.zeros(num_seqs, num_kv_heads, aligned_len, head_size, dtype=dtype)
    v_batched = torch.zeros(num_seqs, num_kv_heads, aligned_len, head_size, dtype=dtype)

    # Additive mask: 0 where a (query, kv) pair may attend, -inf elsewhere.
    neg_inf = torch.finfo(dtype).min
    mask = torch.full((num_seqs, 1, aligned_len, aligned_len), neg_inf, dtype=dtype)

    for s in range(num_seqs):
        q_start, q_end = q_starts[s], q_ends[s]
        q_len = q_end - q_start
        kv_len = min(q_len, kv_lens[s])

        # [L, H, D] -> [H, L, D]
        q_batched[s, :, :q_len, :] = query_cpu[q_start:q_end].transpose(0, 1)
        k_batched[s, :, :kv_len, :] = key_cpu[q_start : q_start + kv_len].transpose(0, 1)
        v_batched[s, :, :kv_len, :] = value_cpu[q_start : q_start + kv_len].transpose(0, 1)
        mask[s, 0, :q_len, :kv_len] = 0.0

    sdpa_kwargs: dict = {"is_causal": False, "scale": scale}
    if num_kv_heads != num_heads:
        sdpa_kwargs["enable_gqa"] = True

    target_device = output.device
    q_dev = convert(q_batched, target_device.type)
    k_dev = convert(k_batched, target_device.type)
    v_dev = convert(v_batched, target_device.type)
    mask_dev = convert(mask, target_device.type)

    # Single on-device SDPA over the whole batch: [num_seqs, H, L_aligned, D].
    attn_out = F.scaled_dot_product_attention(
        q_dev, k_dev, v_dev, attn_mask=mask_dev, **sdpa_kwargs
    )

    # Scatter unpadded results back into the flat output. Pull to CPU first
    # (Spyre slicing corrupts memory); _overwrite handles the per-token write
    # for both Spyre and CPU targets.
    attn_out_cpu = convert(attn_out, "cpu", output.dtype)
    for s in range(num_seqs):
        q_start, q_end = q_starts[s], q_ends[s]
        q_len = q_end - q_start
        # [H, L_aligned, D] -> [q_len, H, D]
        seq_out = attn_out_cpu[s, :, :q_len, :].transpose(0, 1)
        for i in range(q_len):
            tok = convert(seq_out[i : i + 1].contiguous(), target_device.type, output.dtype)
            _overwrite(tok, output, [0], [q_start + i])

    return output
