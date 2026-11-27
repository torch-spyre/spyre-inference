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

The batched decode kernel runs all decode sequences of a step in one call and
specializes on its own key, ``(num_seqs, blocks_per_chunk, num_chunks)``, hence a
second bucket type and enumerator. Up to ``shared_slot_seqs`` sequences run on the
shared-slot kernels instead, whose key leaves ``num_seqs`` out.

Vocabulary:

- A *bucket* is one padded size a runtime length rounds up onto; the sorted list of
  them for one axis is that axis's *buckets*; the spacing between consecutive buckets
  is the *bucket step*.
- A *page* (also *block*) is one KV cache block: ``block_size`` tokens of one sequence.
- A *chunk* is what the batched decode kernel reads in one loop step: the next
  ``blocks_per_chunk`` pages of every sequence in the batch. ``_chunk_width`` picks
  ``blocks_per_chunk`` so that a chunk holds about 32 pages, one per core, and at least
  one page of each sequence. A batch runs as many chunks as its longest sequence needs;
  shorter sequences are masked past their end.
- A *block slot* is one of a chunk's pages. Each block slot keeps its own running
  softmax across chunks, and a sequence's block slots merge after the last one. The
  per-bucket kernels give every sequence ``blocks_per_chunk`` of them. The *shared-slot*
  kernels instead share ``shared_slots`` of them (64, or 32 for a layer group whose KV row is
  too wide for two pages per core) among up to ``shared_slot_seqs`` sequences, each step and
  by length, so they do not depend on the batch size. The per-sequence prefill kernel's
  counterpart is its *page group*, ``page_group`` pages of one sequence per loop step.
  Neither is related to vLLM's chunked prefill, which splits a prompt's tokens across
  scheduler steps.
- A *dynamic* kernel is compiled for a range of sizes (a ``DynamicRange``) instead of
  one, so one compiled kernel runs every page count or chunk count in that range.
"""

from __future__ import annotations

import bisect
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, TypedDict

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

# Block slots per shared-slot chunk, as whole pages per core: an uneven split costs a full
# chunk for a partial gather. Each chunk has a fixed cost, so two pages per core halve a long
# context's chunks, as long as both of a core's K and V pages fit on chip.
_MAX_SLOTS_PER_CORE = 2
# Head elements (num_kv_heads * head_size, per page) one core may gather per chunk before a
# second page per core turns slower than one: the largest row measured faster at two.
_CORE_ROW_BUDGET = 2 * 8 * 128


def shared_block_slots(num_kv_heads: int, head_size: int) -> int:
    """Block slots per shared-slot chunk for a layer group of this per-card KV shape: two
    pages per core while both fit the core's budget, one otherwise."""
    per_core = _MAX_SLOTS_PER_CORE
    while per_core > 1 and per_core * num_kv_heads * head_size > _CORE_ROW_BUDGET:
        per_core //= 2
    return per_core * _SPYRE_CORE_COUNT


class DynamicRange(TypedDict):
    """The sizes one dynamic kernel is compiled for: ``min`` to ``max`` in steps of
    ``granularity``. A dict, since it is passed as is to torch-spyre's
    ``tensor.to("spyre", dynamic={dim: ...})``."""

    min: int
    max: int
    granularity: int


def _chunk_width(b_seqs: int) -> int:
    """The pages per sequence in one chunk: ``32 // b_seqs``, so that a chunk gives each
    of the 32 cores one page, but at least 1."""
    return max(1, _SPYRE_CORE_COUNT // b_seqs)


def batched_decode_chunking(b_seqs: int, b_blocks: int) -> tuple[int, int]:
    """``(blocks_per_chunk, num_chunks)`` for a bucketed ``(num_seqs, num_blocks)`` pair.

    ``entries = b_seqs * blocks_per_chunk`` targets the cores: fewer under-fills
    them, more than one stick's worth hits a backend axis-merge limit. The block
    axis pads up to a whole chunk, so ``blocks_per_chunk * num_chunks >= b_blocks``.
    """
    blocks_per_chunk = min(_chunk_width(b_seqs), b_blocks)
    num_chunks = (b_blocks + blocks_per_chunk - 1) // blocks_per_chunk
    return blocks_per_chunk, num_chunks


def share_block_slots(blocks: list[int], slots: int) -> tuple[int, list[int]]:
    """The fewest chunks C with ``sum(ceil(b / C)) <= slots``, and each sequence's block slots.

    Sequence i takes ``ceil(blocks[i] / C)`` consecutive block slots, so each of them reads at
    least one real block. Every sequence needs at least one block and one block slot.
    """
    assert blocks and min(blocks) >= 1 and len(blocks) <= slots, (blocks, slots)
    lo, hi = -(-sum(blocks) // slots), max(blocks)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(-(-b // mid) for b in blocks) <= slots:
            hi = mid
        else:
            lo = mid + 1
    return lo, [-(-b // lo) for b in blocks]


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
    ``num_chunks`` (derived from the logical block-axis tensor extent). A dynamic kernel
    (``dynamic_chunks``) runs every chunk count in its range; ``num_chunks`` is then the
    count warmup records it at.
    ``num_blocks`` is the bucket they were derived from, kept so the recorder can
    skip a bucket that outruns the KV allocation.
    """

    num_seqs: int
    num_blocks: int
    blocks_per_chunk: int
    num_chunks: int
    dynamic_chunks: bool = False
    # A shared-slot kernel: num_seqs is its output row count, blocks_per_chunk its block slots,
    # and it serves every batch of up to num_seqs sequences.
    shared_slots: bool = False


class BatchedDecodePlan(NamedTuple):
    """How ``build()`` splits one batched decode call into chunks.

    ``num_blocks`` is the page count the call covers: the KV bucket for a kernel with a
    fixed chunk count, the longest sequence's own page count for a dynamic one. The call
    runs ``num_chunks`` chunks of ``blocks_per_chunk`` pages. ``chunk_range`` is the
    range of chunk counts a dynamic kernel runs, None for a fixed one.
    """

    num_blocks: int
    blocks_per_chunk: int
    num_chunks: int
    chunk_range: DynamicRange | None


class SharedSlotPlan(NamedTuple):
    """How ``build()`` splits one shared-slot call into chunks: its chunk count, the chunk
    range of the dynamic kernel that runs it, and each sequence's block slots."""

    num_chunks: int
    chunk_range: DynamicRange
    slots: list[int]


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
        # A seed this small maps onto itself under v * numerator // denominator, so the loop
        # below would never end. Only a block_size the platform rejects gets here.
        if grain * (numerator - denominator) < denominator:
            raise ValueError(
                f"block_size={block_size} is too small for the {numerator}_{denominator} KV "
                "ladder: its series would never grow."
            )
        out = []
        v = grain
        while (v := v * numerator // denominator) < limit:
            if (r := v // grain * grain) > block_size:
                out.append(r)
        return out

    # The 64-token grain survives only if all its buckets up to the horizon are whole blocks;
    # otherwise the series is reseeded at block_size, a different series, not a coarser one.
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


def validate_kv_ladder_env() -> None:
    min_seqs = envs.SPYRE_ATTN_LARGE_BATCH_MIN_SEQS
    if min_seqs < 1:
        raise ValueError(f"SPYRE_ATTN_LARGE_BATCH_MIN_SEQS must be >= 1, got {min_seqs}")
    large = envs.SPYRE_ATTN_KV_LADDER_LARGE_BATCH
    if envs.SPYRE_ATTN_KV_BUCKETS:
        # The explicit buckets replace the base ladder, so its name is not read.
        if large:
            raise ValueError(
                f"SPYRE_ATTN_KV_LADDER_LARGE_BATCH={large!r} cannot be combined with "
                "SPYRE_ATTN_KV_BUCKETS, which replaces the ladder it extends. Unset one of them."
            )
        return
    kv_ladder(envs.SPYRE_ATTN_KV_LADDER)
    if large:
        kv_ladder(large, "SPYRE_ATTN_KV_LADDER_LARGE_BATCH")


def _declared_page_range(max_pages: int) -> DynamicRange | None:
    """Return the range of page counts one per-sequence kernel is compiled for.

    Returns None, and logs why, when ``max_pages`` does not fit torch-spyre's limits; the
    kernels then use the KV buckets instead.
    """
    from torch_spyre._inductor import config as ts_config
    from torch_spyre._inductor import pass_utils

    if not hasattr(pass_utils, "declared_dims"):
        raise RuntimeError(
            "This torch-spyre cannot run for_each_tile over a dynamic number of tiles "
            "(torch-spyre#6066), which the per-sequence attention kernels need. "
            "Upgrade torch-spyre."
        )
    # The page count is the only dynamic size, so it sets these torch-spyre limits unless
    # the user exported them.
    if "MIN_RESERVED_DIM_SIZE" not in os.environ:
        ts_config.min_reserved_dim_size = 1
    if "MAX_BUCKETS" not in os.environ:
        ts_config.max_buckets = max(ts_config.max_buckets, max_pages)
    min_pages = ts_config.min_reserved_dim_size
    # With min == max the size is a constant, which fails to trace.
    if not min_pages < max_pages <= ts_config.max_buckets:
        logger.warning_once(
            "A dynamic page count needs the %d pages per sequence to be above "
            "MIN_RESERVED_DIM_SIZE=%d and at most MAX_BUCKETS=%d, so the per-sequence "
            "attention kernels compile one variant per KV bucket instead.",
            max_pages,
            min_pages,
            ts_config.max_buckets,
        )
        return None
    return {"min": min_pages, "max": max_pages, "granularity": 1}


# Prefill reads this many pages per loop step, so its page counts are multiples of it.
PREFILL_PAGE_GRANULARITY = 8
# A prefill walk of one page lets that page's unused slots into the output
# (tests/e2e/test_kv_cache_determinism.py), so prefill covers at least two pages.
MIN_PREFILL_PAGES = 2


def _prefill_page_range(page_range: DynamicRange) -> DynamicRange:
    """Return ``page_range`` with its bounds rounded up to multiples of
    ``PREFILL_PAGE_GRANULARITY``.

    Returns ``page_range`` unchanged when the rounded range holds a single page count or is
    above an exported ``MAX_BUCKETS``.
    """
    from torch_spyre._inductor import config as ts_config

    g = PREFILL_PAGE_GRANULARITY
    min_pages = -(-page_range["min"] // g) * g
    max_pages = -(-page_range["max"] // g) * g
    if "MAX_BUCKETS" not in os.environ:
        ts_config.max_buckets = max(ts_config.max_buckets, max_pages)
    if not min_pages < max_pages <= ts_config.max_buckets:
        return page_range
    return {"min": min_pages, "max": max_pages, "granularity": g}


def _declared_chunk_ranges(
    page_range: DynamicRange, num_seqs_buckets: list[int]
) -> dict[int, DynamicRange]:
    """For each num_seqs bucket, the chunk counts its dynamic batched decode kernel runs:
    from the chunks ``page_range["min"]`` pages need up to the count ``max_model_len``
    needs.

    A bucket that never needs more than its minimum gets no entry and keeps its KV
    buckets.
    """
    from torch_spyre._inductor import config as ts_config

    ranges: dict[int, DynamicRange] = {}
    for num_seqs in num_seqs_buckets:
        width = _chunk_width(num_seqs)
        min_chunks = max(-(-page_range["min"] // width), ts_config.min_reserved_dim_size)
        max_chunks = -(-page_range["max"] // width)
        # A range needs at least two sizes; a range with min == max does not compile.
        if min_chunks < max_chunks:
            ranges[num_seqs] = {"min": min_chunks, "max": max_chunks, "granularity": 1}
    return ranges


def _shared_slot_chunk_range(
    page_range: DynamicRange, num_seqs: int, slots: int
) -> DynamicRange | None:
    """The chunk counts the dynamic shared-slot kernel runs: from ``MIN_RESERVED_DIM_SIZE``,
    one chunk unless exported, up to what ``num_seqs`` sequences of ``page_range["max"]``
    pages need, so one kernel serves every such batch.

    None, and logged, when the range would hold one count; batched decode then keeps one
    kernel per num_seqs bucket.
    """
    from torch_spyre._inductor import config as ts_config

    min_chunks = ts_config.min_reserved_dim_size
    max_chunks = -(-page_range["max"] // (slots // num_seqs))
    if not min_chunks < max_chunks:
        logger.info_once(
            "No shared-slot batched decode: %d sequences of %d pages fit in %d chunks, not "
            "above the minimum of %d.",
            num_seqs,
            page_range["max"],
            max_chunks,
            min_chunks,
        )
        return None
    return {"min": min_chunks, "max": max_chunks, "granularity": 1}


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

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        declare_pages: bool = False,
        kv_row: tuple[int, int] | None = None,
    ) -> None:
        """``kv_row`` is the layer group's per-card ``(num_kv_heads, head_size)``, from which
        ``shared_block_slots`` sets the shared-slot kernels' block slots. A page range is what
        gives those kernels, so ``declare_pages`` needs it."""
        assert kv_row is not None or not declare_pages, "declare_pages needs the KV row"
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

        # Default: the SPYRE_ATTN_KV_LADDER preset. Sparse because the recorded set is a
        # product of both axes; the extra padding each bucket costs is absorbed by the mask.
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

        # With a page range, the per-sequence kernels take the page count at run time, so
        # they compile once per query bucket rather than once per (query, KV) bucket pair.
        self._page_range: DynamicRange | None = None
        self._prefill_page_range: DynamicRange | None = None
        if declare_pages:
            self._page_range = _declared_page_range(self._num_blocks_buckets[-1])
        if self._page_range is not None:
            self._prefill_page_range = _prefill_page_range(self._page_range)

        # Batched decode gets one dynamic kernel per num_seqs bucket, for every chunk count.
        self._chunk_ranges: dict[int, DynamicRange] = {}
        if self._page_range is not None:
            self._chunk_ranges = _declared_chunk_ranges(self._page_range, self._num_seqs_buckets)

        # Up to this many decode sequences share a chunk's block slots by length, so the one
        # kernel that runs them does not depend on the batch size. Per layer group, so the
        # builder that dispatches and the recorder that walks this instance agree on every
        # group's chunks.
        self._shared_slots: int | None = None
        self._shared_slot_seqs = 0
        self._shared_slot_chunk_range: DynamicRange | None = None
        if self._page_range is not None:
            assert kv_row is not None
            self._shared_slots = shared_block_slots(*kv_row)
            self._shared_slot_seqs = min(self._shared_slots, self._num_seqs_buckets[-1])
            self._shared_slot_chunk_range = _shared_slot_chunk_range(
                self._page_range, self._shared_slot_seqs, self._shared_slots
            )
        if self._shared_slot_chunk_range is not None:
            logger.info(
                "SpyreAttnBucketer: batched decode of up to %d sequences shares the %d block "
                "slots of %d to %d chunks per call.",
                self._shared_slot_seqs,
                self._shared_slots,
                self._shared_slot_chunk_range["min"],
                self._shared_slot_chunk_range["max"],
            )

        validate_kv_ladder_env()

        # Batched decode only: below the threshold the dense buckets cost warmup without a
        # measured gain. Unioned, so none rounds up.
        self._large_batch_min_seqs = envs.SPYRE_ATTN_LARGE_BATCH_MIN_SEQS
        self._large_batch_blocks_buckets = self._num_blocks_buckets
        large = envs.SPYRE_ATTN_KV_LADDER_LARGE_BATCH
        if large and self._large_batch_min_seqs > self._num_seqs_buckets[-1]:
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
                | {(kv + block_size - 1) // block_size for kv in dense}
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

    @property
    def page_range(self) -> DynamicRange | None:
        """The ``min``, ``max`` and ``granularity`` of the page count one per-sequence
        kernel runs. None when page counts are rounded up to ``num_blocks_buckets``."""
        return self._page_range

    @property
    def prefill_page_range(self) -> DynamicRange | None:
        """``page_range`` for a prefill sequence (query width > 1), whose page counts are
        multiples of ``PREFILL_PAGE_GRANULARITY``."""
        return self._prefill_page_range

    def find_declared_pages(self, num_blocks: int, prefill: bool = False) -> int:
        """The page count a sequence of ``num_blocks`` pages runs at: its own, rounded up
        to the range granularity, or the range minimum if that is larger. A prefill runs at
        least ``MIN_PREFILL_PAGES``."""
        page_range = self._prefill_page_range if prefill else self._page_range
        assert page_range is not None
        g = page_range["granularity"]
        floor = max(page_range["min"], MIN_PREFILL_PAGES) if prefill else page_range["min"]
        return max(-(-num_blocks // g) * g, floor)

    def batched_blocks_buckets(self, num_seqs_bucket: int) -> list[int]:
        """The num_blocks buckets batched decode records and dispatches onto at this batch size.

        The default ``SPYRE_ATTN_LARGE_BATCH_MIN_SEQS`` is tuned to ``batched_decode_chunking``;
        revisit it if that chunking changes.
        """
        if num_seqs_bucket >= self._large_batch_min_seqs:
            return self._large_batch_blocks_buckets
        return self._num_blocks_buckets

    def find_batched_blocks_bucket(self, num_blocks: int, num_seqs_bucket: int) -> int | None:
        return self._round_up(num_blocks, self.batched_blocks_buckets(num_seqs_bucket))

    def batched_decode_plan(
        self, num_seqs_bucket: int, num_blocks: int
    ) -> BatchedDecodePlan | None:
        """How to run a batched decode call whose longest sequence has ``num_blocks`` pages.

        - If this bucket has no chunk range, the call is padded to its KV bucket and runs
          the kernel compiled for that bucket.
        - Otherwise it runs the bucket's dynamic kernel with just enough chunks for
          ``num_blocks``, and at least ``chunk_range["min"]``.

        Returns None if no KV bucket holds ``num_blocks``.
        """
        b_blocks = self.find_batched_blocks_bucket(num_blocks, num_seqs_bucket)
        if b_blocks is None:
            return None
        chunk_range = self._chunk_ranges.get(num_seqs_bucket)
        if chunk_range is None:
            return BatchedDecodePlan(
                b_blocks, *batched_decode_chunking(num_seqs_bucket, b_blocks), None
            )
        width = _chunk_width(num_seqs_bucket)
        num_chunks = max(-(-num_blocks // width), chunk_range["min"])
        return BatchedDecodePlan(num_blocks, width, num_chunks, chunk_range)

    @property
    def shared_slots(self) -> int:
        """The block slots of one shared-slot chunk; only a bucketer with a page range has
        them."""
        assert self._shared_slots is not None, "no shared-slot kernels without a page range"
        return self._shared_slots

    @property
    def shared_slot_seqs(self) -> int | None:
        """The most decode sequences one shared-slot call serves, None without the kernels."""
        return None if self._shared_slot_chunk_range is None else self._shared_slot_seqs

    @property
    def shared_slot_chunk_range(self) -> DynamicRange | None:
        """The ``min``, ``max`` and ``granularity`` of the dynamic shared-slot kernel's chunks."""
        return self._shared_slot_chunk_range

    def shared_slot_plan(self, blocks: list[int]) -> SharedSlotPlan | None:
        """The layout of a shared-slot call over sequences of ``blocks`` blocks each, or None
        without shared-slot kernels or when the call has more than ``shared_slot_seqs``
        sequences.

        Every call takes the dynamic kernel, at no fewer chunks than its range minimum.
        """
        chunk_range = self._shared_slot_chunk_range
        if chunk_range is None or len(blocks) > self._shared_slot_seqs:
            return None
        # A sequence with no blocks still takes a block slot, fully masked, so its row is
        # finite.
        num_chunks, slots = share_block_slots([max(1, b) for b in blocks], self.shared_slots)
        # The range covers shared_slot_seqs sequences at max_model_len.
        assert num_chunks <= chunk_range["max"], (num_chunks, chunk_range)
        return SharedSlotPlan(max(num_chunks, chunk_range["min"]), chunk_range, slots)

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
        # With a page range, one kernel per query bucket runs every page count in it.
        blocks_buckets = self._num_blocks_buckets
        if self._page_range is not None:
            blocks_buckets = [blocks_buckets[-1]]
        for num_blocks in sorted(blocks_buckets, reverse=True):
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
        batch of any size can sit at any context length. A num_seqs bucket with a chunk
        range records only its dynamic kernel, at the chunk count of max_model_len. With the
        shared-slot kernels, only the buckets above ``shared_slot_seqs``.
        """
        if not envs.SPYRE_BATCHED_DECODE:
            return []
        shared_slot_seqs = self.shared_slot_seqs or 0
        pairs = []
        for num_seqs in self._num_seqs_buckets:
            if num_seqs <= shared_slot_seqs:
                continue
            blocks_buckets = self.batched_blocks_buckets(num_seqs)
            if num_seqs in self._chunk_ranges:
                blocks_buckets = [self._num_blocks_buckets[-1]]
            pairs.extend((nb, num_seqs) for nb in blocks_buckets)
        out: list[SpyreAttnBatchedDecodeBucket] = []
        for num_blocks, num_seqs in sorted(pairs, reverse=True):
            plan = self.batched_decode_plan(num_seqs, num_blocks)
            assert plan is not None
            out.append(
                SpyreAttnBatchedDecodeBucket(
                    num_seqs=num_seqs,
                    num_blocks=plan.num_blocks,
                    blocks_per_chunk=plan.blocks_per_chunk,
                    num_chunks=plan.num_chunks,
                    dynamic_chunks=plan.chunk_range is not None,
                )
            )
        return out

    def shared_slot_variants(self) -> list[SpyreAttnBatchedDecodeBucket]:
        """The dynamic shared-slot kernel, recorded at ``shared_slot_seqs`` sequences of
        max_model_len; empty without shared-slot kernels."""
        num_seqs = self.shared_slot_seqs
        if not envs.SPYRE_BATCHED_DECODE or num_seqs is None:
            return []
        assert self._page_range is not None
        num_blocks = self._page_range["max"]
        plan = self.shared_slot_plan([num_blocks] * num_seqs)
        assert plan is not None
        return [
            SpyreAttnBatchedDecodeBucket(
                num_seqs=num_seqs,
                num_blocks=num_blocks,
                blocks_per_chunk=self.shared_slots,
                num_chunks=plan.num_chunks,
                dynamic_chunks=True,
                shared_slots=True,
            )
        ]
