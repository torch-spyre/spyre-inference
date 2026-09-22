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
import functools
from collections.abc import Callable
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.logger import init_logger

from spyre_inference import envs

logger = init_logger(__name__)

# Spacing of the default query buckets above the decode bucket, capped against
# max_num_batched_tokens. Every non-decode batch pads its query length up to a
# multiple of this.
_DEFAULT_QUERY_BUCKET_STEP = 512

# Batches below this fall back to the per-seq loop: the batched matmul's
# padded-row overhead exceeds the per-seq cost at small N. So the num_seqs ladder
# starts here -- smaller batches never dispatch to a batched variant.
_MIN_BATCHED_SEQS = 4

# Cores available to split a gather's entry axis across.
_SPYRE_CORE_COUNT = 32


def sliding_active_blocks(
    kv_len: int, context_len: int, sliding_window: int, block_size: int
) -> range:
    """The block indices a windowed sequence attends to.

    The count is what the per-sequence kernel specializes on, so the enumerator needs
    this alone; ``sliding_tile_plan`` builds on it for the per-block mask content.
    """
    # Bound by the EARLIEST query's window, not the latest (``kv_len - W``): early queries
    # in a prefill batch have earlier windows, and the latest would drop blocks they need.
    first_active = max(0, context_len - sliding_window + 1) // block_size
    return range(first_active, (kv_len + block_size - 1) // block_size)


def sliding_tile_plan(
    kv_len: int,
    query_len: int,
    context_len: int,
    sliding_window: int,
    block_size: int,
    apply_causal_mask: bool,
) -> tuple[list[int], tuple[bool, ...]]:
    """``(active_block_indices, needs_real_tile_per_block)`` for a sliding-window sequence.

    Active blocks are those whose mask contributes to at least one query's attention, i.e.
    inside the earliest query's window. Classification, by absolute block index ``b``:

      - ``[0, first_active)``:
            entirely outside every query's window; dropped.
      - ``[first_active, last_lower_boundary]``:
            lower-boundary blocks -- the window cutoff falls inside them for at least one
            query. Real tile, with per-query-row cutoffs. Collapses to a single block in
            decode (``query_len == 1``).
      - ``(last_lower_boundary, last_causal_interior]``:
            interior blocks -- fully inside every query's window AND fully below the
            earliest query's causal limit. Mask is all-zero, so no real tile.
      - ``(last_causal_interior, last_block)``:
            causal-boundary blocks -- inside every window, but early queries have causal
            cutoffs falling inside them (prefill only). Real tile.
      - ``last_block``:
            upper-boundary block -- always has KV padding, plus causal cutoffs during
            prefill. Real tile.

    When those ranges overlap (short ``kv_len``, single-block sequence) the union gets real
    tiles -- never a zero tile where content is needed.
    """
    num_blocks = (kv_len + block_size - 1) // block_size
    # Blocks up to the latest query's window start can hold a per-query cutoff.
    last_lower_boundary = max(0, kv_len - sliding_window) // block_size
    # Fully below the earliest query's causal limit iff
    # (b + 1) * block_size - 1 <= context_len. Decode applies no causal mask.
    last_causal_interior = (
        (context_len + 1) // block_size - 1 if apply_causal_mask else num_blocks - 1
    )
    last_block = num_blocks - 1
    active = list(sliding_active_blocks(kv_len, context_len, sliding_window, block_size))
    needs_real_tile: list[bool] = []
    for b in active:
        is_lower_boundary = b <= last_lower_boundary
        is_upper_boundary = (b == last_block) and not is_lower_boundary
        is_causal_boundary = apply_causal_mask and b > last_causal_interior and b != last_block
        needs_real_tile.append(is_lower_boundary or is_upper_boundary or is_causal_boundary)
    return active, tuple(needs_real_tile)


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
    ``num_chunks`` (the per-chunk index list it unrolls at trace time).
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

    def __init__(self, vllm_config: VllmConfig, sliding_window: int | None = None) -> None:
        block_size = vllm_config.cache_config.block_size
        self.block_size = block_size
        # This group's window, not the model's: gemma-4 runs windowed and full-attention
        # layers side by side, and each attention group gets its own bucketer.
        self._sliding_window = sliding_window
        max_model_len = vllm_config.model_config.max_model_len
        max_batched = vllm_config.scheduler_config.max_num_batched_tokens

        # A pooling request's query_len is its own context_len, so it can't
        # exceed max_model_len even when max_num_batched_tokens is larger (unlike
        # a decoder's chunked-prefill step). Without this cap, warmup could record
        # a query bucket with no matching num_blocks bucket, crashing with
        # "num_blocks=N exceeds the largest recorded bucket".
        if vllm_config.model_config.runner_type == "pooling":
            max_batched = min(max_batched, max_model_len)

        if block_size & (block_size - 1):
            # Not fatal: _powers_of_two_up_to rounds the start up to a power of
            # two, just coarser at the bottom. Reachable because the platform
            # only forces a multiple of 64 (SpyrePlatform.check_and_update_config).
            logger.warning(
                "block_size=%d is not a power of two; the smallest KV bucket is the next "
                "power of two instead, making it larger than one block. Prefer a "
                "power-of-two block_size.",
                block_size,
            )

        # Default: powers of two from _MIN_BATCHED_SEQS up to max_num_seqs, the
        # batch sizes the batched decode kernel can be asked for.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self._num_seqs_buckets: list[int] = (
            _resolve_buckets(
                envs.SPYRE_ATTN_NUM_SEQS_BUCKETS,
                max_num_seqs,
                "SPYRE_ATTN_NUM_SEQS_BUCKETS",
                lambda: list(_powers_of_two_up_to(max_num_seqs, start=_MIN_BATCHED_SEQS)),
            )
            if max_num_seqs >= _MIN_BATCHED_SEQS
            else []
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

        # Default: powers of two from block_size up to max_model_len. Geometric
        # because the recorded set is a product of both axes; the extra padding
        # each bucket costs is absorbed by the mask.
        self._kv_buckets: list[int] = _resolve_buckets(
            envs.SPYRE_ATTN_KV_BUCKETS,
            max_model_len,
            "SPYRE_ATTN_KV_BUCKETS",
            lambda: list(_powers_of_two_up_to(max_model_len, start=block_size)),
        )

        # num_blocks is what the kernel specializes on. Derived from the kv
        # buckets, one block count per kv bucket, rather than enumerating every
        # integer up to max_model_len / block_size.
        self._num_blocks_buckets: list[int] = sorted(
            {(kv + block_size - 1) // block_size for kv in self._kv_buckets}
        )

        # How far `_sliding_variants` scans. One window plus one query bucket puts the
        # window's lower boundary anywhere in the sequence; two further blocks walk it
        # through every offset inside a block, after which the plan repeats with period
        # block_size. Pinned by test_every_length_dispatches_to_an_enumerated_variant, which
        # scans all the way to max_model_len and finds nothing new.
        self._scan_limit = min(
            max_model_len, (sliding_window or 0) + self._query_buckets[-1] + 2 * block_size
        )

        # Deliberately does not report the variant count: under a window that would force
        # the `_sliding_variants` scan here, wasting it when recording is off. The recorder
        # logs the count it is about to record anyway.
        logger.info(
            "SpyreAttnBucketer: %d kv buckets [%d..%d], %d query buckets [%d..%d], "
            "max num_blocks=%d, sliding_window=%s",
            len(self._kv_buckets),
            self._kv_buckets[0],
            self._kv_buckets[-1],
            len(self._query_buckets),
            self._query_buckets[0],
            self._query_buckets[-1],
            self._num_blocks_buckets[-1],
            self._sliding_window,
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

    def witness(self, bucket: SpyreAttnBucket) -> tuple[int, int]:
        """A ``(kv_len, query_len)`` one sequence can have that dispatches to ``bucket``.

        Warmup needs a concrete sequence per variant. Without a window the bucket already
        is one, since ``build()`` pads the block count up onto this ladder: ``num_blocks``
        whole blocks holding the shortest query reaching this width dispatch there.
        Under a window nothing pads, so the witness comes from the scan that found the
        variant in the first place.
        """
        if self._sliding_window is not None:
            return self._sliding_variants[bucket]
        kv_len = bucket.num_blocks * self.block_size
        query_len = self.min_real_query_len(bucket.padded_query_len)
        assert query_len <= kv_len, f"{bucket} pairs a query length no sequence can reach"
        return kv_len, query_len

    def decode_witness_kv_len(self, num_blocks: int) -> int | None:
        """A ``kv_len`` whose one-token decode step pads its block count onto ``num_blocks``.

        ``None`` when no sequence can ask for that bucket, so the recorder skips it.

        Without a window a block-aligned ``kv_len`` realizes exactly ``num_blocks``, so the
        bucket is its own witness. Under one it is the *worst* possible choice: the window's
        lower boundary lands on a block edge, so every block-aligned length leaves exactly
        ``ceil(W / block_size)`` blocks active and every bucket above that realizes the same
        one. A real decode reaches one block more (its ``kv_len`` is almost never
        block-aligned), which rounds onto the next bucket up -- and that bucket then gets no
        witness at any ``max_model_len``. Hence the search over realizable decode counts.
        """
        if self._sliding_window is None:
            return num_blocks * self.block_size
        counts = sorted(b.num_blocks for b in self._sliding_variants if b.padded_query_len == 1)
        for count in counts:
            if self.find_blocks_bucket(count) == num_blocks:
                return self._sliding_variants[SpyreAttnBucket(count, 1)][0]
        return None

    @functools.cached_property
    def _sliding_variants(self) -> dict[SpyreAttnBucket, tuple[int, int]]:
        """Each windowed variant a sequence can reach, and the shortest sequence reaching it.

        Enumerated by scanning ``(query_len, kv_len)``, not solved. Two reasons not to
        reach for arithmetic here: the active count saws as the window's lower boundary
        moves through a block, so it inverts to no closed form; and a closed *bound* on
        the count is the kind of thing that is wrong by one in a corner (a window just
        past a block multiple) and then silently drops a kernel -- which is the bug class
        this whole path exists to fix. Scanning the definition cannot be off by one.

        Runs once per bucketer -- shared by every layer in the attention group, while
        ``record_graphs`` runs per layer -- and costs ~0.6s at gemma-4's shapes against the
        minutes of Inductor compiles it sizes.
        """
        assert self._sliding_window is not None
        out: dict[SpyreAttnBucket, tuple[int, int]] = {}
        for query_len in range(1, min(self._query_buckets[-1], self._scan_limit) + 1):
            width = 1 if query_len == 1 else self.find_query_bucket(query_len)
            if width is None:
                continue
            for kv_len in range(query_len, self._scan_limit + 1):
                active = sliding_active_blocks(
                    kv_len, kv_len - query_len, self._sliding_window, self.block_size
                )
                if active:
                    bucket = SpyreAttnBucket(num_blocks=len(active), padded_query_len=width)
                    out.setdefault(bucket, (kv_len, query_len))
        return out

    def find_kv_bucket(self, kv_len: int) -> int | None:
        return self._round_up(kv_len, self._kv_buckets)

    def find_query_bucket(self, query_len: int) -> int | None:
        return self._round_up(query_len, self._query_buckets)

    def find_sequence_bucket(self, num_seqs: int) -> int | None:
        return self._round_up(num_seqs, self._num_seqs_buckets)

    def find_blocks_bucket(self, num_blocks: int) -> int | None:
        return self._round_up(num_blocks, self._num_blocks_buckets)

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

        Under a window the block count is the ACTIVE count, which no kv ladder can reach:
        for a block-aligned ``kv_len`` the window's lower boundary lands on a block edge,
        so every kv bucket at or above the window leaves the same blocks active. The
        reachable set is scanned instead -- see ``_sliding_variants``.

        Without one, the two axes are the kv-derived block ladder and the query ladder,
        bounded against each other: ``kv_len >= query_len`` always, so a query bucket only
        pairs with block counts that can hold it, and the full cross product would record
        many variants nothing reaches (each an Inductor compile). The bound is on the
        *smallest real* query_len reaching a bucket, not the bucket itself, since a 2-token
        query on a 1-block sequence still pads up to a large bucket; bounding by the bucket
        would prune that variant and put a compile back in the serving path.
        """
        if self._sliding_window is not None:
            return sorted(
                self._sliding_variants,
                key=lambda b: (b.num_blocks, b.padded_query_len),
                reverse=True,
            )
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

        The full ``num_seqs_buckets x num_blocks_buckets`` grid, minus the block counts a
        windowed decode cannot ask for: unlike ``variants()`` there is no inter-axis bound
        to exploit, since a decode batch of any size can sit at any context length. Both
        axes are geometric, so the grid stays small.

        Unlike the per-sequence axis this one still *pads* -- ``build()`` rounds the decode
        block count onto it -- so a coarse ladder is only a throughput cost, not a
        correctness one. What it cannot do is leave a reachable entry without a witness;
        see ``decode_witness_kv_len``.
        """
        if not envs.SPYRE_BATCHED_DECODE:
            return []
        out: list[SpyreAttnBatchedDecodeBucket] = []
        for num_blocks in sorted(self._num_blocks_buckets, reverse=True):
            if self.decode_witness_kv_len(num_blocks) is None:
                # Windowed: the window caps the active count, so the upper ladder entries
                # are unreachable. Recording them wastes a compile each and, worse, they
                # all realize the same low bucket and look like coverage.
                continue
            for num_seqs in sorted(self._num_seqs_buckets, reverse=True):
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
