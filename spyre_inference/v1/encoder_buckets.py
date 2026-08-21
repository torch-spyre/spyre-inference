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

"""Encoder compile-shape buckets.

Spyre ``torch.compile(dynamic=False)`` specializes both the encoder SDPA batch
``[B, H, L, D]`` and the body (embed / Linear / LN) on flat ``[T, …]``. Without
a ladder, every new ``max_len`` or ``num_seqs`` compiles a new graph (~60s).

Pad each sequence to the next length bucket ``L`` and the batch to ``B`` so
``T = B × L``. A 30-token, 3-seq request then reuses the warmed ``(4, 64)``
graph for attention **and** Linear/LN. Attention still masks to the real
lengths so pad tokens do not mix into embeddings.

Env:
    SPYRE_ENCODER_BUCKET_LENS          CSV of prompt-length buckets
                                       (default ``64,128,256,512,1024,2048``).
                                       Each value is rounded up to a multiple
                                       of 64 (Spyre stick).
    SPYRE_ENCODER_BUCKET_BATCH_SIZES   CSV of batch buckets. Default: ``1, 2,
                                       4, …, max_num_seqs``.
"""

from __future__ import annotations

import os
from typing import NamedTuple

# Spyre stick (64 fp16 elements). Length buckets and MiniLM head-dim padding
# both align to this so Inductor never enters insert_bmm_padding.
ENCODER_SEQ_ALIGNMENT = 64

_DEFAULT_LEN_BUCKETS = (64, 128, 256, 512, 1024, 2048)


def parse_csv_ints(env_name: str, default: list[int]) -> list[int]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return list(default)
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return values or list(default)


def _align_up(n: int, align: int = ENCODER_SEQ_ALIGNMENT) -> int:
    return max(align, (n + align - 1) // align * align)


def next_bucket(n: int, buckets: list[int]) -> int:
    """Smallest bucket ``>= n``. If ``n`` exceeds the ladder, stick-align ``n``."""
    if n < 1:
        n = 1
    ordered = sorted({b for b in buckets if b > 0})
    for bucket in ordered:
        if bucket >= n:
            return bucket
    return _align_up(n)


def len_buckets() -> list[int]:
    """Configured length ladder, each entry stick-aligned, strictly increasing."""
    raw = parse_csv_ints("SPYRE_ENCODER_BUCKET_LENS", list(_DEFAULT_LEN_BUCKETS))
    aligned = sorted({_align_up(v) for v in raw if v > 0})
    return aligned or [_DEFAULT_LEN_BUCKETS[0]]


def batch_buckets(max_num_seqs: int) -> list[int]:
    """Configured batch ladder, clipped to ``[1, max_num_seqs]``."""
    cap = max(1, max_num_seqs)
    env = parse_csv_ints("SPYRE_ENCODER_BUCKET_BATCH_SIZES", [])
    if env:
        values = sorted({b for b in env if 1 <= b <= cap})
        return values or [cap]
    out: list[int] = []
    size = 1
    while size < cap:
        out.append(size)
        size *= 2
    if cap not in out:
        out.append(cap)
    return out


def encoder_len_bucket(max_len: int) -> int:
    """Nearest length bucket for encoder SDPA ``L`` (always ≥ stick size)."""
    return next_bucket(max(max_len, 1), len_buckets())


def encoder_batch_bucket(num_seqs: int, max_num_seqs: int) -> int:
    """Nearest batch bucket for encoder SDPA ``B`` (≤ ``max_num_seqs``)."""
    cap = max(1, max_num_seqs)
    n = min(max(num_seqs, 1), cap)
    return min(next_bucket(n, batch_buckets(cap)), cap)


def pooling_warmup_pad_query_lens(prompt_len: int) -> list[int]:
    """Unpadded dummy lengths (``L-2``, ``L-1``) so pad leftovers compile.

    vLLM ``--random-input-len L`` samples ``L`` minus tokenizer specials (often 2).
    """
    lens = {max(1, prompt_len - 1)}
    if prompt_len > 2:
        lens.add(prompt_len - 2)
    return sorted(length for length in lens if length < prompt_len)


def pooling_warmup_shapes(
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
) -> list[tuple[int, int]]:
    """``(batch_size, prompt_len)`` pairs to dummy at serve start."""
    shapes: list[tuple[int, int]] = []
    for batch_size in batch_buckets(max_num_seqs):
        for prompt_len in len_buckets():
            if prompt_len > max_model_len:
                continue
            if batch_size * prompt_len > max_num_batched_tokens:
                continue
            shapes.append((batch_size, prompt_len))
    return shapes


class EncoderBucketPad(NamedTuple):
    """Runtime pad of a pooling batch onto a warmed ``(B, L)`` shape."""

    batch_bucket: int
    len_bucket: int
    orig_query_lens: list[int]
    orig_num_tokens: int
    orig_num_reqs: int

    @property
    def num_tokens(self) -> int:
        return self.batch_bucket * self.len_bucket


def runtime_encoder_bucket(
    num_seqs: int,
    max_query_len: int,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
) -> tuple[int, int] | None:
    """``(B, L)`` to pad onto, or ``None`` if that shape does not fit the budget.

    ``None`` means leave the batch unpadded (a new compile, same as before).
    """
    if num_seqs < 1 or max_query_len < 1:
        return None
    batch_bucket = encoder_batch_bucket(num_seqs, max_num_seqs)
    len_bucket = encoder_len_bucket(max_query_len)
    if len_bucket > max_model_len:
        return None
    if batch_bucket * len_bucket > max_num_batched_tokens:
        return None
    return batch_bucket, len_bucket


def expand_packed_to_encoder_bucket(
    input_ids: list[int],
    positions: list[int],
    query_lens: list[int],
    batch_bucket: int,
    len_bucket: int,
    pad_token_id: int = 0,
) -> tuple[list[int], list[int]]:
    """Pad each sequence to ``L`` and the batch to ``B``; return ``[B*L]`` lists.

    Real pad tokens continue positions from the true length. Dummy sequences
    (batch pad) are ``pad_token_id`` with positions ``0 .. L-1``.
    """
    if len(query_lens) > batch_bucket:
        raise ValueError(f"num_seqs={len(query_lens)} exceeds batch_bucket={batch_bucket}")
    if any(length > len_bucket for length in query_lens):
        raise ValueError(f"a query length exceeds len_bucket={len_bucket}: {query_lens}")

    total = batch_bucket * len_bucket
    padded_ids = [int(pad_token_id)] * total
    padded_pos = [0] * total
    src = 0
    for seq_idx, length in enumerate(query_lens):
        dst = seq_idx * len_bucket
        padded_ids[dst : dst + length] = list(input_ids[src : src + length])
        padded_pos[dst : dst + length] = list(positions[src : src + length])
        for offset in range(length, len_bucket):
            padded_pos[dst + offset] = offset
        src += length
    for seq_idx in range(len(query_lens), batch_bucket):
        dst = seq_idx * len_bucket
        for offset in range(len_bucket):
            padded_pos[dst + offset] = offset
    return padded_ids, padded_pos


def encoder_bucket_valid_row_indices(
    orig_query_lens: list[int],
    len_bucket: int,
) -> list[int]:
    """Row indices of real tokens inside a ``B×L`` packed hidden state."""
    indices: list[int] = []
    for seq_idx, length in enumerate(orig_query_lens):
        start = seq_idx * len_bucket
        indices.extend(range(start, start + length))
    return indices
