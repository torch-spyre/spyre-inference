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

Dynamo specializes the per-sequence attention kernel on its padded query length,
compiling on first use, which would put a full Inductor compile in the serving
path. This module enumerates the query buckets a run can reach so warmup can record
them all up front. The page count is not bucketed: each sequence walks its own pages,
copied to the device with a declared page range, so one kernel per query bucket runs
every page count up to the context.

Separate from ``SpyreShapeBucketer``, which dispatches a single ``num_tokens``
int for the model graph.

The batched decode kernel runs all decode sequences of a step in one call and
specializes on its own key, ``(num_seqs, blocks_per_chunk, num_chunks)``, hence a
second bucket type and enumerator.

Vocabulary:

- A *bucket* is one padded size a runtime length rounds up onto; the sorted list of
  them for one axis is that axis's *buckets*; the spacing between consecutive buckets
  is the *bucket step*.
- A *page* (also *block*) is one KV cache block: ``block_size`` tokens of one sequence.
- A *chunk* is what the batched decode kernel reads in one loop step: the next
  ``blocks_per_chunk`` pages of every sequence in the batch. ``_chunk_width`` picks
  ``blocks_per_chunk`` so that a chunk holds about 32 pages, one per core, and at least
  one page of each sequence. A batch runs as many chunks as its longest sequence needs;
  shorter sequences are masked past their end. The per-sequence prefill kernel's
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
from vllm.utils.math_utils import cdiv

from spyre_inference import envs
from spyre_inference.v1.worker.spyre_shape_bucketer import _align_up_pow2

logger = init_logger(__name__)

# Spacing of the default query buckets above the decode bucket, capped against
# max_num_batched_tokens. Every non-decode batch pads its query length up to a
# multiple of this.
_DEFAULT_QUERY_BUCKET_STEP = 512

# Cores available to split a gather's entry axis across.
_SPYRE_CORE_COUNT = 32


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
    count warmup records it at. ``num_blocks`` is the page count the variant is recorded
    at.
    """

    num_seqs: int
    num_blocks: int
    blocks_per_chunk: int
    num_chunks: int
    dynamic_chunks: bool = False


class BatchedDecodePlan(NamedTuple):
    """How ``build()`` splits one batched decode call into chunks.

    ``num_blocks`` is the page count the call covers: the longest sequence's own page
    count for a dynamic kernel, otherwise ``num_chunks`` whole chunks. The call runs
    ``num_chunks`` chunks of ``blocks_per_chunk`` pages. ``chunk_range`` is the range of
    chunk counts a dynamic kernel runs, None for a fixed one.
    """

    num_blocks: int
    blocks_per_chunk: int
    num_chunks: int
    chunk_range: DynamicRange | None


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


def _declared_page_range(max_pages: int) -> DynamicRange | None:
    """Return the range of page counts one per-sequence kernel is compiled for.

    Returns None when ``max_pages`` is not above ``MIN_RESERVED_DIM_SIZE``: the range would
    hold a single count, and every sequence then runs at ``max_pages``.

    Raises:
        ValueError: ``MAX_BUCKETS`` was exported below ``max_pages``.
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
    # the user exported them. The minimum is two pages because torch.compile compiles a
    # size of one separately (torch-spyre#6018); a one-page sequence gets a masked copy
    # of its page as the second one.
    if "MIN_RESERVED_DIM_SIZE" not in os.environ:
        ts_config.min_reserved_dim_size = 2
    if "MAX_BUCKETS" not in os.environ:
        ts_config.max_buckets = max(ts_config.max_buckets, max_pages)
    min_pages = ts_config.min_reserved_dim_size
    # With min == max the size is a constant, which fails to trace.
    if max_pages <= min_pages:
        logger.info_once(
            "%d pages per sequence is not above MIN_RESERVED_DIM_SIZE=%d, so every sequence "
            "pads to %d pages and the per-sequence attention kernels use a static page count.",
            max_pages,
            min_pages,
            max_pages,
        )
        return None
    if max_pages > ts_config.max_buckets:
        raise ValueError(
            f"MAX_BUCKETS={ts_config.max_buckets} is below the {max_pages} pages one sequence "
            "can span, which one per-sequence attention kernel walks as many loop steps. "
            f"Unset MAX_BUCKETS or raise it to at least {max_pages}."
        )
    return {"min": min_pages, "max": max_pages, "granularity": 1}


# Prefill reads this many pages per loop step, so its page counts are multiples of it.
PREFILL_PAGE_GRANULARITY = 8


def _prefill_page_range(page_range: DynamicRange) -> DynamicRange:
    """Return ``page_range`` with its bounds rounded up to multiples of
    ``PREFILL_PAGE_GRANULARITY``.

    The minimum is two loop steps, since torch.compile compiles a single loop step
    separately, as it does a single page. Returns ``page_range`` unchanged when the
    rounded range holds a single page count or is above an exported ``MAX_BUCKETS``.
    """
    from torch_spyre._inductor import config as ts_config

    g = PREFILL_PAGE_GRANULARITY
    min_pages = max(-(-page_range["min"] // g) * g, 2 * g)
    max_pages = -(-page_range["max"] // g) * g
    if "MAX_BUCKETS" not in os.environ:
        ts_config.max_buckets = max(ts_config.max_buckets, max_pages)
    if not min_pages < max_pages <= ts_config.max_buckets:
        return page_range
    return {"min": min_pages, "max": max_pages, "granularity": g}


def _multi_chunk_plans(
    max_chunks_by_seqs: dict[int, int], page_range: DynamicRange | None
) -> tuple[dict[int, DynamicRange], dict[int, int]]:
    """How each num_seqs bucket runs a batched decode call longer than one chunk.

    ``max_chunks_by_seqs`` is each bucket's chunk count at the longest context, for the
    buckets that can need more than one chunk. Returns ``(chunk_ranges, static_chunks)``:
    a bucket in ``chunk_ranges`` takes its chunk count at run time, from
    ``page_range["min"]`` up; one in ``static_chunks`` pads every such call to that count,
    because a range with min == max does not compile (or, without a ``page_range``,
    nothing is dynamic).
    """
    ranges: dict[int, DynamicRange] = {}
    static: dict[int, int] = {}
    for num_seqs, max_chunks in max_chunks_by_seqs.items():
        if page_range is None or max_chunks <= page_range["min"]:
            static[num_seqs] = max_chunks
        else:
            ranges[num_seqs] = {"min": page_range["min"], "max": max_chunks, "granularity": 1}
    return ranges, static


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

    Query lengths and batch sizes round *up* onto their buckets, matching
    ``SpyreShapeBucketer.find_bucket``; over-max returns None, and the caller falls back to
    compiling on demand. Page counts are not bucketed: a sequence runs at its own count,
    within the declared ``page_range``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        declare_pages: bool = False,
        sliding_window: int | None = None,
    ) -> None:
        block_size = vllm_config.cache_config.block_size
        self.block_size = block_size
        max_model_len = vllm_config.model_config.max_model_len
        max_batched = vllm_config.scheduler_config.max_num_batched_tokens

        # A pooling request's query_len is its own context_len, so it can't
        # exceed max_model_len even when max_num_batched_tokens is larger (unlike
        # a decoder's chunked-prefill step). Without this cap, warmup could record
        # a query bucket no sequence can reach. Stick-aligned via _align_up_pow2 to match
        # the encoder shape ladder's padding (CLIP: 77 -> 128).
        if vllm_config.model_config.runner_type == "pooling":
            max_batched = min(max_batched, _align_up_pow2(max_model_len))

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

        # The block table holds every page of a max_model_len sequence. A sliding window
        # walks only the pages its widest query's windows touch: ceil(span / B) + 1 for a
        # span of W + query_len - 1 positions that need not start on a page boundary.
        self._context_pages = cdiv(max_model_len, block_size)
        max_pages = self._context_pages
        if sliding_window is not None:
            span = sliding_window + self._query_buckets[-1] - 1
            max_pages = min(max_pages, cdiv(span, block_size) + 1)

        # One KV head on one page leaves torch-spyre's codegen no spare dim for the stick
        # dim, so the decode kernel will not compile there. A second, fully masked page
        # does.
        self._min_blocks = 2 if _min_num_kv_heads(vllm_config) == 1 else 1
        self._max_pages = max(max_pages, self._min_blocks)

        # The per-sequence kernels take the page count at run time, so they compile once
        # per query bucket. Without a range every sequence pads to max_pages.
        self._page_range: DynamicRange | None = None
        self._prefill_page_range: DynamicRange | None = None
        if declare_pages:
            self._page_range = _declared_page_range(self._max_pages)
        if self._page_range is not None:
            self._prefill_page_range = _prefill_page_range(self._page_range)

        # Batched decode runs a call within one chunk as that one chunk, and a longer one at
        # a chunk count taken at run time.
        max_chunks_by_seqs = {}
        for num_seqs in self._num_seqs_buckets:
            max_chunks = self._chunks_needed(num_seqs, self._max_pages)
            if max_chunks > 1:
                max_chunks_by_seqs[num_seqs] = max_chunks
        self._chunk_ranges, self._static_chunks = _multi_chunk_plans(
            max_chunks_by_seqs, self._page_range
        )

        logger.info(
            "SpyreAttnBucketer: %d query buckets [%d..%d], %d num_seqs buckets, pages %s",
            len(self._query_buckets),
            self._query_buckets[0],
            self._query_buckets[-1],
            len(self._num_seqs_buckets),
            self._page_range or f"static at {self._max_pages}",
        )

    @property
    def query_buckets(self) -> list[int]:
        return self._query_buckets

    @property
    def num_seqs_buckets(self) -> list[int]:
        return self._num_seqs_buckets

    @property
    def max_pages(self) -> int:
        """The most pages one sequence's kernel walks."""
        return self._max_pages

    def find_query_bucket(self, query_len: int) -> int | None:
        return self._round_up(query_len, self._query_buckets)

    def find_sequence_bucket(self, num_seqs: int) -> int | None:
        return self._round_up(num_seqs, self._num_seqs_buckets)

    @property
    def page_range(self) -> DynamicRange | None:
        """The ``min``, ``max`` and ``granularity`` of the page count one per-sequence
        kernel runs. None when every sequence runs at ``max_pages``."""
        return self._page_range

    @property
    def prefill_page_range(self) -> DynamicRange | None:
        """``page_range`` for a prefill sequence (query width > 1), whose page counts are
        multiples of ``PREFILL_PAGE_GRANULARITY``."""
        return self._prefill_page_range

    def find_pages(self, num_blocks: int, prefill: bool = False) -> int:
        """The page count a sequence of ``num_blocks`` pages runs at: its own, rounded up
        to the range granularity, or the range minimum if that is larger; ``max_pages``
        without a range."""
        assert num_blocks <= self._max_pages, (
            f"num_blocks={num_blocks} exceeds the {self._max_pages} pages a sequence can span"
        )
        page_range = self._prefill_page_range if prefill else self._page_range
        if page_range is None:
            return self._max_pages
        g = page_range["granularity"]
        return max(-(-num_blocks // g) * g, page_range["min"])

    def chunk_width(self, num_seqs_bucket: int) -> int:
        """Pages per sequence in one batched decode chunk: ``_chunk_width``, even past the
        longest context, since a narrower chunk is a gather torch-spyre may not lay out."""
        return _chunk_width(num_seqs_bucket)

    def _chunks_needed(self, num_seqs_bucket: int, num_blocks: int) -> int:
        """Chunks a call of ``num_blocks`` blocks needs. A one-block chunk takes a second,
        masked one where one KV head needs two pages."""
        width = self.chunk_width(num_seqs_bucket)
        return max(cdiv(num_blocks, width), self._min_blocks if width < self._min_blocks else 1)

    def batched_decode_plan(
        self, num_seqs_bucket: int, num_blocks: int
    ) -> BatchedDecodePlan | None:
        """How to run a batched decode call whose longest sequence has ``num_blocks`` pages.

        - If the call fits in one chunk, it runs the one-chunk kernel.
        - If this bucket has a static chunk count, the call pads to it.
        - Otherwise it runs the bucket's dynamic kernel with just enough chunks for
          ``num_blocks``, and at least ``chunk_range["min"]``.

        Returns None past ``max_pages``.
        """
        if num_blocks > self._max_pages:
            return None
        width = self.chunk_width(num_seqs_bucket)
        needed = self._chunks_needed(num_seqs_bucket, num_blocks)
        if needed == 1:
            return BatchedDecodePlan(width, width, 1, None)
        static = self._static_chunks.get(num_seqs_bucket)
        if static is not None:
            return BatchedDecodePlan(static * width, width, static, None)
        chunk_range = self._chunk_ranges[num_seqs_bucket]
        return BatchedDecodePlan(num_blocks, width, max(needed, chunk_range["min"]), chunk_range)

    def min_real_query_len(self, padded_query_len: int) -> int:
        """Smallest runtime query_len that rounds up onto ``padded_query_len``."""
        idx = bisect.bisect_left(self._query_buckets, padded_query_len)
        return self._query_buckets[idx - 1] + 1 if idx else 1

    @staticmethod
    def _round_up(n: int, buckets: list[int]) -> int | None:
        idx = bisect.bisect_left(buckets, n)
        return buckets[idx] if idx < len(buckets) else None

    def variants(self) -> list[SpyreAttnBucket]:
        """Every variant worth recording, largest first: one per query bucket.

        Each is requested at ``max_pages``, or at the context its smallest real query
        needs if that is longer, and ``build()`` realizes its own page count: rounded into
        the range, or under a sliding window the pages the window touches. A query bucket
        no context up to max_model_len can hold is left out. The bound is on the
        *smallest real* query_len that reaches a bucket, not the bucket itself, since a
        2-token query on a 1-block sequence still dispatches to a large padded bucket;
        bounding by the bucket would prune that variant and put a compile back in the
        serving path.
        """
        out = []
        for padded_query_len in sorted(self._query_buckets, reverse=True):
            needed = cdiv(self.min_real_query_len(padded_query_len), self.block_size)
            if needed > self._context_pages:
                continue
            num_blocks = max(self._max_pages, needed)
            out.append(SpyreAttnBucket(num_blocks=num_blocks, padded_query_len=padded_query_len))
        return out

    def batched_decode_variants(self) -> list[SpyreAttnBatchedDecodeBucket]:
        """Every batched decode variant worth recording, largest first.

        Per num_seqs bucket, the one-chunk kernel and the one past it: its dynamic kernel,
        recorded at ``max_pages``, or its static chunk count.
        """
        if not envs.SPYRE_BATCHED_DECODE:
            return []
        out: dict[tuple[int, int, int, bool], SpyreAttnBatchedDecodeBucket] = {}
        for num_seqs in self._num_seqs_buckets:
            for num_blocks in (self.chunk_width(num_seqs), self._max_pages):
                plan = self.batched_decode_plan(num_seqs, num_blocks)
                if plan is None:
                    continue
                bucket = SpyreAttnBatchedDecodeBucket(
                    num_seqs=num_seqs,
                    # Recorded at a context a sequence can have; the chunks still pad past it.
                    num_blocks=min(plan.num_blocks, self._max_pages),
                    blocks_per_chunk=plan.blocks_per_chunk,
                    num_chunks=plan.num_chunks,
                    dynamic_chunks=plan.chunk_range is not None,
                )
                key = (num_seqs, plan.blocks_per_chunk, plan.num_chunks, bucket.dynamic_chunks)
                out.setdefault(key, bucket)
        return sorted(out.values(), key=lambda b: (b.num_blocks, b.num_seqs), reverse=True)
