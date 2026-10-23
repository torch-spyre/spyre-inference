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

"""Bucketer for the attention kernel's compiled variants.

Dynamo specializes the attention kernel on ``(num_blocks, padded_query_len)``,
compiling on first use, which would put a full Inductor compile in the serving
path. This module enumerates the pairs a run can reach so warmup can record them
all up front.

Separate from ``SpyreShapeBucketer``, which dispatches a single ``num_tokens``
int for the model graph; a per-sequence attention variant is 2-D (kv_len and
query_len buckets).

The batched decode kernel specializes on its own key,
``(num_seqs, blocks_per_chunk, num_chunks)``, hence a second bucket type and
enumerator.

Vocabulary: a *bucket* is one padded size a runtime length rounds up onto; the
sorted list of them for one axis is that axis's *buckets*; the spacing between
consecutive buckets is the *bucket step*.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.logger import init_logger

from spyre_inference import envs
from spyre_inference.v1.worker.spyre_shape_bucketer import _align_up_pow2

logger = init_logger(__name__)

# Spacing of the default query buckets above the decode bucket, capped against
# max_num_batched_tokens. Every non-decode batch pads its query length up to a
# multiple of this.
_DEFAULT_QUERY_BUCKET_STEP = 512

# Cores available to split a gather's entry axis across.
_SPYRE_CORE_COUNT = 32


def batched_decode_chunking(b_seqs: int, b_blocks: int) -> tuple[int, int]:
    """``(blocks_per_chunk, num_chunks)`` for a bucketed ``(num_seqs, num_blocks)`` pair.

    ``entries = b_seqs * blocks_per_chunk`` targets the cores: fewer under-fills
    them, more than one stick's worth hits a backend axis-merge limit. The block
    axis pads up to a whole chunk, so ``blocks_per_chunk * num_chunks >= b_blocks``.
    """
    blocks_per_chunk = max(1, min(_SPYRE_CORE_COUNT // b_seqs, b_blocks))
    num_chunks = (b_blocks + blocks_per_chunk - 1) // blocks_per_chunk
    return blocks_per_chunk, num_chunks


@dataclass(frozen=True)
class SpyreAttnBucket:
    """One recordable per-sequence attention kernel variant.

    Fields are the values the kernel specializes on, so a recorded bucket and a
    runtime dispatch reach the same Dynamo entry.
    """

    num_blocks: int
    padded_query_len: int


@dataclass(frozen=True)
class SpyreAttnBatchedDecodeBucket:
    """One recordable batched decode kernel variant.

    The kernel specializes on ``num_seqs``, ``blocks_per_chunk`` and
    ``num_chunks`` (derived from the logical block-axis tensor extent).
    ``num_blocks`` is the bucket they were derived from, kept so the recorder can
    skip a bucket that outruns the KV allocation.
    """

    num_seqs: int
    num_blocks: int
    blocks_per_chunk: int
    num_chunks: int


def _parse_buckets(raw: str | None) -> list[int] | None:
    """Parse a comma-separated env-var bucket list, or None when unset/empty."""
    if not raw:
        return None
    values = sorted({int(part) for part in raw.split(",") if part.strip()})
    if not values or values[0] < 1:
        raise ValueError(f"bucket entries must be >= 1, got {raw!r}")
    return values


def _powers_of_two_up_to(n: int, start: int = 1) -> tuple[int, ...]:
    """Powers of 2 in [start, n] (start rounded up to a power of 2), plus n itself."""
    if n < 1:
        return ()
    v = 1
    while v < start:
        v *= 2
    result = []
    while v < n:
        result.append(v)
        v *= 2
    result.append(n)
    return tuple(result)


def _uniform_union_up_to(n: int, block_size: int, step_div: int, knee_div: int) -> list[int]:
    step = max(block_size, (n // step_div) // block_size * block_size)
    knee = max(block_size, (n // knee_div) // block_size * block_size)
    extra = range(knee + step, n, step)
    return sorted(set(_powers_of_two_up_to(n, start=block_size)) | set(extra) | {n})


# Grain chosen up to here, not up to max_model_len, so the context never moves lower buckets.
_KV_LADDER_GRAIN_HORIZON = 32768


def _geometric_union_up_to(n: int, block_size: int, numerator: int, denominator: int) -> list[int]:
    def _series(grain: int, limit: int) -> list[int]:
        out = []
        v = grain
        while (v := v * numerator // denominator) < limit:
            if (r := v // grain * grain) > block_size:
                out.append(r)
        return out

    # A 64-token grain leaves some buckets off whole blocks from block_size 128 upward.
    grain = 64
    if any(r % block_size for r in _series(grain, _KV_LADDER_GRAIN_HORIZON)):
        grain = block_size
    series = {r // block_size * block_size for r in _series(grain, n)}
    return sorted(set(_powers_of_two_up_to(n, start=block_size)) | series | {n})


# Every preset keeps the powers of two, so a KV length's bucket never moves up. Batched
# decode pads blocks to whole chunks, so at one sequence some buckets save nothing there.
_KV_LADDERS: dict[str, Callable[[int, int], list[int]]] = {
    "pow2": lambda n, bs: list(_powers_of_two_up_to(n, start=bs)),
    "8_5": lambda n, bs: _geometric_union_up_to(n, bs, 8, 5),
    "4_3": lambda n, bs: _geometric_union_up_to(n, bs, 4, 3),
    "9_8": lambda n, bs: _geometric_union_up_to(n, bs, 9, 8),
    "uniform": lambda n, bs: _uniform_union_up_to(n, bs, step_div=8, knee_div=16),
}


def kv_ladder(name: str, var: str = "SPYRE_ATTN_KV_LADDER") -> Callable[[int, int], list[int]]:
    try:
        return _KV_LADDERS[name]
    except KeyError:
        raise ValueError(
            f"{var}={name!r} is not one of {sorted(_KV_LADDERS)}. "
            "SPYRE_ATTN_KV_BUCKETS overrides the ladder entirely if none of these fit."
        ) from None


def large_batch_min_seqs() -> int:
    raw = envs.SPYRE_ATTN_LARGE_BATCH_MIN_SEQS
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"SPYRE_ATTN_LARGE_BATCH_MIN_SEQS must be an integer >= 1, got {raw!r}")
    return value


def validate_kv_ladder_env() -> None:
    # SPYRE_ATTN_KV_BUCKETS replaces both ladders, so neither name is read then.
    large_batch_min_seqs()
    if envs.SPYRE_ATTN_KV_BUCKETS:
        return
    kv_ladder(envs.SPYRE_ATTN_KV_LADDER)
    if envs.SPYRE_ATTN_KV_LADDER_LARGE_BATCH:
        kv_ladder(envs.SPYRE_ATTN_KV_LADDER_LARGE_BATCH, "SPYRE_ATTN_KV_LADDER_LARGE_BATCH")


def _min_num_kv_heads(vllm_config: VllmConfig) -> int:
    """The smallest KV head count any layer carries, after the TP split.

    ``get_num_kv_heads`` reports one model-wide number that a heterogeneous config can
    hide a smaller layer behind: gemma-4 reports 4 at TP2, its full-attention layers 1.
    """
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    arch = model_config.model_arch_config
    model_wide = model_config.get_num_kv_heads(parallel_config)
    overrides = arch.per_layer_overrides or ()
    return min(
        (model_config.get_num_kv_heads(parallel_config, arch[i]) for i in range(len(overrides))),
        default=model_wide,
    )


def _resolve_buckets(
    raw: str | None, limit: int, name: str, default: Callable[[], list[int]]
) -> list[int]:
    """One axis's buckets: the env override clamped to ``limit``, else ``default()``.

    ``limit`` bounds the lengths the engine can schedule (max_model_len for kv,
    max_num_batched_tokens for query, max_num_seqs for the batch axis). Entries
    above it are unreachable, so they are dropped, and ``limit`` itself is added
    when missing.
    """
    buckets = _parse_buckets(raw)
    if buckets is None:
        return default()
    kept = [b for b in buckets if b <= limit]
    if len(kept) != len(buckets):
        logger.warning(
            "%s lists %s above the %d it must cover; dropping them as unreachable.",
            name,
            [b for b in buckets if b > limit],
            limit,
        )
    if not kept or kept[-1] < limit:
        logger.warning(
            "%s does not cover %d; appending it. Lengths in (%d, %d] would otherwise "
            "have no recorded bucket and would compile an attention kernel in the "
            "serving path.",
            name,
            limit,
            kept[-1] if kept else 0,
            limit,
        )
        kept.append(limit)
    return kept


class SpyreAttnBucketer:
    """Enumerates the attention variants to record, and rounds lengths onto them.

    Both axes round *up*: a runtime length lands on the smallest recorded
    bucket that fits it, matching ``SpyreShapeBucketer.find_bucket``. Over-max
    returns None, and the caller falls back to compiling on demand.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        block_size = vllm_config.cache_config.block_size
        self.block_size = block_size
        max_model_len = vllm_config.model_config.max_model_len
        max_batched = vllm_config.scheduler_config.max_num_batched_tokens

        # A pooling request's query_len is its own context_len, so it can't
        # exceed max_model_len even when max_num_batched_tokens is larger (unlike
        # a decoder's chunked-prefill step). Without this cap, warmup could record
        # a query bucket with no matching num_blocks bucket, crashing with
        # "num_blocks=N exceeds the largest recorded bucket". Stick-aligned via
        # _align_up_pow2 to match the encoder shape ladder's padding (CLIP: 77 -> 128).
        if vllm_config.model_config.runner_type == "pooling":
            max_batched = min(max_batched, _align_up_pow2(max_model_len))

        if block_size & (block_size - 1):
            # Not fatal: _powers_of_two_up_to rounds the start up to a power of
            # two, just coarser at the bottom. The platform rejects such a block size
            # (TorchSpyrePlatform._align_block_size), so only a config built without
            # it gets here.
            logger.warning(
                "block_size=%d is not a power of two; the smallest KV bucket is the next "
                "power of two instead, making it larger than one block. Prefer a "
                "power-of-two block_size.",
                block_size,
            )

        # Default: powers of two up to max_num_seqs, the batch sizes the batched
        # decode kernel can be asked for.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self._num_seqs_buckets: list[int] = _resolve_buckets(
            envs.SPYRE_ATTN_NUM_SEQS_BUCKETS,
            max_num_seqs,
            "SPYRE_ATTN_NUM_SEQS_BUCKETS",
            lambda: list(_powers_of_two_up_to(max_num_seqs)),
        )

        # Default: [1] (the decode-only batch, exempt from query padding by
        # build()) then multiples of a step up to max_num_batched_tokens, the
        # query lengths a prefill pads up to.
        step = min(_DEFAULT_QUERY_BUCKET_STEP, max_batched)
        self._query_buckets: list[int] = _resolve_buckets(
            envs.SPYRE_ATTN_QUERY_BUCKETS,
            max_batched,
            "SPYRE_ATTN_QUERY_BUCKETS",
            lambda: sorted({1, *range(step, max_batched + 1, step), max_batched}),
        )

        # Default: the SPYRE_ATTN_KV_LADDER preset. Geometric because the recorded set
        # is a product of both axes; the extra padding each bucket costs is absorbed by the mask.
        self._kv_buckets: list[int] = _resolve_buckets(
            envs.SPYRE_ATTN_KV_BUCKETS,
            max_model_len,
            "SPYRE_ATTN_KV_BUCKETS",
            lambda: kv_ladder(envs.SPYRE_ATTN_KV_LADDER)(max_model_len, block_size),
        )

        # num_blocks is what the kernel specializes on. Derived from the kv
        # buckets, one block count per kv bucket, rather than enumerating every
        # integer up to max_model_len / block_size.
        self._num_blocks_buckets: list[int] = sorted(
            {(kv + block_size - 1) // block_size for kv in self._kv_buckets}
        )

        # One KV head on one page leaves torch-spyre's codegen no spare dim for the stick
        # dim, so the head-major decode kernel will not compile there. A second, fully
        # masked page does, and it lifts the SWA bound _max_active_blocks derives from it.
        if envs.SPYRE_ATTN_KV_LAYOUT == "head_major" and _min_num_kv_heads(vllm_config) == 1:
            self._num_blocks_buckets = sorted({max(2, n) for n in self._num_blocks_buckets})

        # Batched decode only: below the threshold the chunking already pads to whole chunks,
        # so dense buckets there mostly compile duplicate kernels. Unioned, so none rounds up.
        self._large_batch_min_seqs = large_batch_min_seqs()
        self._large_batch_blocks_buckets = self._num_blocks_buckets
        large = envs.SPYRE_ATTN_KV_LADDER_LARGE_BATCH
        if large and envs.SPYRE_ATTN_KV_BUCKETS:
            logger.warning(
                "SPYRE_ATTN_KV_LADDER_LARGE_BATCH=%s is ignored: SPYRE_ATTN_KV_BUCKETS is set.",
                large,
            )
        elif large and self._large_batch_min_seqs > self._num_seqs_buckets[-1]:
            logger.warning(
                "SPYRE_ATTN_KV_LADDER_LARGE_BATCH=%s never applies: "
                "SPYRE_ATTN_LARGE_BATCH_MIN_SEQS=%d is above the largest num_seqs bucket %d.",
                large,
                self._large_batch_min_seqs,
                self._num_seqs_buckets[-1],
            )
        elif large:
            dense = kv_ladder(large, "SPYRE_ATTN_KV_LADDER_LARGE_BATCH")(max_model_len, block_size)
            self._large_batch_blocks_buckets = sorted(
                set(self._num_blocks_buckets)
                | {
                    max(self._num_blocks_buckets[0], (kv + block_size - 1) // block_size)
                    for kv in dense
                }
            )
            logger.info(
                "SpyreAttnBucketer: batched decode at >= %d seqs uses %d num_blocks buckets "
                "(%s) instead of %d.",
                self._large_batch_min_seqs,
                len(self._large_batch_blocks_buckets),
                large,
                len(self._num_blocks_buckets),
            )

        logger.info(
            "SpyreAttnBucketer: %d kv buckets [%d..%d], %d query buckets [%d..%d], "
            "max num_blocks=%d",
            len(self._kv_buckets),
            self._kv_buckets[0],
            self._kv_buckets[-1],
            len(self._query_buckets),
            self._query_buckets[0],
            self._query_buckets[-1],
            self._num_blocks_buckets[-1],
        )

    @property
    def kv_buckets(self) -> list[int]:
        return self._kv_buckets

    @property
    def query_buckets(self) -> list[int]:
        return self._query_buckets

    @property
    def num_blocks_buckets(self) -> list[int]:
        return self._num_blocks_buckets

    @property
    def num_seqs_buckets(self) -> list[int]:
        return self._num_seqs_buckets

    def find_kv_bucket(self, kv_len: int) -> int | None:
        return self._round_up(kv_len, self._kv_buckets)

    def find_query_bucket(self, query_len: int) -> int | None:
        return self._round_up(query_len, self._query_buckets)

    def find_sequence_bucket(self, num_seqs: int) -> int | None:
        return self._round_up(num_seqs, self._num_seqs_buckets)

    def find_blocks_bucket(self, num_blocks: int) -> int | None:
        return self._round_up(num_blocks, self._num_blocks_buckets)

    def batched_blocks_buckets(self, num_seqs_bucket: int) -> list[int]:
        """The num_blocks buckets batched decode records and dispatches onto at this batch size."""
        if num_seqs_bucket >= self._large_batch_min_seqs:
            return self._large_batch_blocks_buckets
        return self._num_blocks_buckets

    def find_batched_blocks_bucket(self, num_blocks: int, num_seqs_bucket: int) -> int | None:
        return self._round_up(num_blocks, self.batched_blocks_buckets(num_seqs_bucket))

    def min_real_query_len(self, padded_query_len: int) -> int:
        """Smallest runtime query_len that rounds up onto ``padded_query_len``."""
        idx = bisect.bisect_left(self._query_buckets, padded_query_len)
        return self._query_buckets[idx - 1] + 1 if idx else 1

    @staticmethod
    def _round_up(n: int, buckets: list[int]) -> int | None:
        idx = bisect.bisect_left(buckets, n)
        return buckets[idx] if idx < len(buckets) else None

    def variants(self) -> list[SpyreAttnBucket]:
        """Every variant worth recording, largest first.

        The two size axes aren't independent: ``kv_len >= query_len`` always, so
        a query bucket only pairs with block counts that can hold it -- the full
        cross product would record many unreachable variants at a long context.
        Requires the backend to round each sequence's own query_len, so the bound
        holds per sequence and not against a batch max.
        The bound is on the *smallest real* query_len that reaches a bucket, not
        the bucket itself, since a 2-token query on a 1-block sequence still
        dispatches to a large padded bucket; bounding by the bucket would prune
        that variant and put a compile back in the serving path.
        """
        out: list[SpyreAttnBucket] = []
        for num_blocks in sorted(self._num_blocks_buckets, reverse=True):
            max_query_here = num_blocks * self.block_size
            for padded_query_len in sorted(self._query_buckets, reverse=True):
                if self.min_real_query_len(padded_query_len) > max_query_here:
                    continue
                out.append(
                    SpyreAttnBucket(num_blocks=num_blocks, padded_query_len=padded_query_len)
                )
        return out

    def batched_decode_variants(self) -> list[SpyreAttnBatchedDecodeBucket]:
        """Every batched decode variant worth recording, largest first.

        Each num_seqs bucket against its own ``batched_blocks_buckets``: unlike
        ``variants()`` there is no inter-axis bound to exploit, since a decode
        batch of any size can sit at any context length.
        """
        if not envs.SPYRE_BATCHED_DECODE:
            return []
        out: list[SpyreAttnBatchedDecodeBucket] = []
        pairs = sorted(
            ((nb, ns) for ns in self._num_seqs_buckets for nb in self.batched_blocks_buckets(ns)),
            reverse=True,
        )
        for num_blocks, num_seqs in pairs:
            blocks_per_chunk, num_chunks = batched_decode_chunking(num_seqs, num_blocks)
            out.append(
                SpyreAttnBatchedDecodeBucket(
                    num_seqs=num_seqs,
                    num_blocks=num_blocks,
                    blocks_per_chunk=blocks_per_chunk,
                    num_chunks=num_chunks,
                )
            )
        return out
