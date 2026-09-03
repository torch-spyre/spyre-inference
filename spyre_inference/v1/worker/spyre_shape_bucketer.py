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

"""Spyre shape bucketer for compilation warmup and runtime dispatch.

Body (1D, decoder and pooling): sorted ``compile_sizes`` token counts; pad the
packed batch to the nearest bucket ``>=`` actual ``num_tokens``. Linear / LN
compile on ``[T, …]``.

Attention (encoder only): warmed ``(B, L)`` cells for SDPA ``[B, H, L, D]``.
The attention backend gathers rows into that grid; the body is not rewritten
to ``T = B × L``. ``L`` is the ``max_model_len`` ladder, not ``compile_sizes``.
``B`` is powers of two up to ``--max-num-seqs``, same as decoder attention.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# Spyre stick (64 fp16 elements). Length buckets and MiniLM head-dim padding
# both align to this so Inductor never enters insert_bmm_padding.
ENCODER_SEQ_ALIGNMENT = 64


def default_encoder_len_buckets(max_model_len: int) -> list[int]:
    """Stick-aligned prompt-length buckets from 64 up to ``max_model_len``.

    Powers of two through the last value that still fits, then ``max_model_len``
    rounded down to a stick if that is not already on the ladder.
    """
    cap = max(1, int(max_model_len))
    buckets: list[int] = []
    size = ENCODER_SEQ_ALIGNMENT
    while size < cap:
        buckets.append(size)
        size *= 2
    aligned_cap = max(ENCODER_SEQ_ALIGNMENT, (cap // ENCODER_SEQ_ALIGNMENT) * ENCODER_SEQ_ALIGNMENT)
    if aligned_cap <= cap and aligned_cap not in buckets:
        buckets.append(aligned_cap)
    return buckets or [ENCODER_SEQ_ALIGNMENT]


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


def len_buckets(
    max_model_len: int,
    compile_sizes: Sequence[int] | None = None,
) -> list[int]:
    """Attention ``L`` ladder from ``max_model_len``.

    Optional ``compile_sizes`` overrides ``L`` in tests. Platform
    ``compile_sizes`` are body token counts and must not be passed here.
    """
    if compile_sizes:
        aligned = sorted({_align_up(int(v)) for v in compile_sizes if int(v) > 0})
        fitted = [v for v in aligned if v <= max_model_len]
        if fitted:
            return fitted
    return default_encoder_len_buckets(max_model_len)


def batch_buckets(max_num_seqs: int) -> list[int]:
    """Powers of two in ``[1, max_num_seqs]``, plus ``max_num_seqs`` itself.

    Same ladder as decoder attention (``_powers_of_two_up_to``): clip with
    ``--max-num-seqs``, no extra env var.
    """
    cap = max(1, max_num_seqs)
    out: list[int] = []
    size = 1
    while size < cap:
        out.append(size)
        size *= 2
    if cap not in out:
        out.append(cap)
    return out


def encoder_len_bucket(max_len: int, buckets: list[int] | None = None) -> int:
    """Nearest length bucket for encoder SDPA ``L`` (always ≥ stick size)."""
    return next_bucket(max(max_len, 1), buckets or [])


def pick_encoder_attention_shape(
    num_seqs: int,
    max_query_len: int,
    encoder_shapes: Sequence[tuple[int, int]],
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
) -> tuple[int, int] | None:
    """Smallest warmed ``(B, L)`` covering the batch, or None.

    Prefer smallest ``T = B × L``, then smallest ``B``, then smallest ``L``.
    """
    if num_seqs < 1 or max_query_len < 1 or not encoder_shapes:
        return None
    candidates = [
        (batch, length)
        for batch, length in encoder_shapes
        if batch >= num_seqs
        and length >= max_query_len
        and batch <= max_num_seqs
        and length <= max_model_len
        and batch * length <= max_num_batched_tokens
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda pair: (pair[0] * pair[1], pair[0], pair[1]))


def encoder_batch_bucket(num_seqs: int, max_num_seqs: int) -> int:
    """Nearest batch bucket for encoder SDPA ``B`` (≤ ``max_num_seqs``)."""
    cap = max(1, max_num_seqs)
    n = min(max(num_seqs, 1), cap)
    return min(next_bucket(n, batch_buckets(cap)), cap)


def pooling_warmup_shapes(
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    len_ladder: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    """``(batch_size, prompt_len)`` pairs to dummy at serve start."""
    shapes: list[tuple[int, int]] = []
    for batch_size in batch_buckets(max_num_seqs):
        for prompt_len in len_buckets(max_model_len, len_ladder):
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


@dataclass(frozen=True)
class SpyreBucketDescriptor:
    """Descriptor for a 1D (decoder) compilation bucket."""

    actual_num_tokens: int
    padded_num_tokens: int


@dataclass(frozen=True)
class EncoderBucketDescriptor:
    """Descriptor for a 2D encoder ``(B, L)`` compilation bucket."""

    batch_bucket: int
    len_bucket: int
    actual_num_seqs: int
    actual_max_len: int

    @property
    def padded_num_tokens(self) -> int:
        return self.batch_bucket * self.len_bucket


class SpyreShapeBucketer:
    """Dispatches runtime batches to pre-compiled bucket sizes.

    1D (``compile_sizes``): body token count ``>=`` actual ``num_tokens``.
    2D (``encoder_shapes``): attention ``(B, L)`` covering the batch.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        encoder_shapes: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        if encoder_shapes is not None:
            self._encoder_shapes: list[tuple[int, int]] = list(encoder_shapes)
            self._bucket_sizes: list[int] = sorted(
                {batch * length for batch, length in self._encoder_shapes}
            )
        else:
            self._encoder_shapes = []
            compilation_config = vllm_config.compilation_config
            sizes: list[int] = [int(s) for s in (compilation_config.compile_sizes or [])]
            self._bucket_sizes = sorted(sizes)
        self._max_bucket_size = self._bucket_sizes[-1] if self._bucket_sizes else 0
        self._is_warmed_up = False

        if self._encoder_shapes:
            logger.info(
                "SpyreShapeBucketer initialized with %d encoder (B, L) shapes: %s",
                len(self._encoder_shapes),
                self._encoder_shapes,
            )
        else:
            logger.info(
                "SpyreShapeBucketer initialized with %d bucket sizes: min=%d, max=%d",
                len(self._bucket_sizes),
                self._bucket_sizes[0] if self._bucket_sizes else 0,
                self._max_bucket_size,
            )

    @classmethod
    def for_pooling(cls, vllm_config: VllmConfig) -> SpyreShapeBucketer | None:
        """Pooling bucketer: 1D body ``compile_sizes`` plus attention ``(B, L)``."""
        model_config = vllm_config.model_config
        if getattr(model_config, "runner_type", None) != "pooling":
            return None
        scheduler = vllm_config.scheduler_config
        compile_sizes = [int(s) for s in (vllm_config.compilation_config.compile_sizes or [])]
        shapes = pooling_warmup_shapes(
            max_num_seqs=scheduler.max_num_seqs,
            max_model_len=model_config.max_model_len,
            max_num_batched_tokens=scheduler.max_num_batched_tokens,
            len_ladder=default_encoder_len_buckets(model_config.max_model_len),
        )
        if not shapes and not compile_sizes:
            return None
        inst = cls(vllm_config, encoder_shapes=shapes or None)
        if compile_sizes:
            inst._bucket_sizes = sorted(set(compile_sizes))
            inst._max_bucket_size = inst._bucket_sizes[-1] if inst._bucket_sizes else 0
        return inst

    @property
    def bucket_sizes(self) -> list[int]:
        return self._bucket_sizes

    @property
    def encoder_shapes(self) -> list[tuple[int, int]]:
        return list(self._encoder_shapes)

    @property
    def max_bucket_size(self) -> int:
        return self._max_bucket_size

    @property
    def is_warmed_up(self) -> bool:
        return self._is_warmed_up

    def mark_warmed_up(self) -> None:
        self._is_warmed_up = True

    def find_bucket(self, num_tokens: int) -> int | None:
        """Find the smallest 1D bucket size >= num_tokens.

        Returns None if num_tokens exceeds the largest compiled bucket.
        The caller (execute_model) handles the None case by running the
        forward pass without bucket padding, which may trigger Dynamo
        recompilation for the unseen shape.
        """
        idx = bisect.bisect_left(self._bucket_sizes, num_tokens)
        if idx < len(self._bucket_sizes):
            return self._bucket_sizes[idx]
        return None

    def dispatch(self, num_tokens: int) -> SpyreBucketDescriptor | None:
        """Compute padded batch descriptor for the given token count.

        Returns None if no suitable bucket exists.
        """
        padded = self.find_bucket(num_tokens)
        if padded is None:
            return None
        return SpyreBucketDescriptor(
            actual_num_tokens=num_tokens,
            padded_num_tokens=padded,
        )

    def find_encoder_bucket(
        self,
        num_seqs: int,
        max_query_len: int,
        max_num_seqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
    ) -> tuple[int, int] | None:
        """Smallest warmed attention ``(B, L)`` that covers the batch, or None."""
        return pick_encoder_attention_shape(
            num_seqs,
            max_query_len,
            self._encoder_shapes,
            max_num_seqs,
            max_model_len,
            max_num_batched_tokens,
        )

    def dispatch_encoder(
        self,
        num_seqs: int,
        max_query_len: int,
        max_num_seqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
    ) -> EncoderBucketDescriptor | None:
        """Pad descriptor for encoder SDPA, or None if no warmed cell fits."""
        pair = self.find_encoder_bucket(
            num_seqs,
            max_query_len,
            max_num_seqs,
            max_model_len,
            max_num_batched_tokens,
        )
        if pair is None:
            return None
        batch_bucket, len_bucket = pair
        return EncoderBucketDescriptor(
            batch_bucket=batch_bucket,
            len_bucket=len_bucket,
            actual_num_seqs=num_seqs,
            actual_max_len=max_query_len,
        )
