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

    The earliest query (q_pos=0) has window [max(0, context_len - W + 1), context_len];
    the latest (q_pos=query_len-1) has [max(0, kv_len - W), kv_len - 1]. A block is fully
    outside every query's window when its highest KV position is below the EARLIEST
    query's window start, which is what this bounds by.

    Using the earliest query's window rather than the latest (``kv_len - W``) is required
    for prefill correctness: in a batch with query_len > 1 the early queries have earlier
    windows, and bounding by the latest would drop blocks they need. For decode
    (query_len == 1) the two formulas coincide.

    Shared with the builder deliberately. ``build()`` pads this count onto the ladder and
    the bucketer sizes the ladder from it, so the two agreeing is what makes the recorded
    set the reachable set; a second copy of the arithmetic is how that guarantee is lost.
    """
    first_active = max(0, context_len - sliding_window + 1) // block_size
    return range(first_active, (kv_len + block_size - 1) // block_size)


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
        # layers side by side, and each attention group gets its own builder and bucketer.
        self._sliding_window = sliding_window
        max_model_len = vllm_config.model_config.max_model_len
        self._max_model_len = max_model_len
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
        geometric = sorted({(kv + block_size - 1) // block_size for kv in self._kv_buckets})
        if sliding_window is None:
            self._num_blocks_buckets: list[int] = geometric
        else:
            # Under a window the kernel's block axis is the ACTIVE count, which the kv
            # ladder cannot reach: for a block-aligned kv_len the window's lower boundary
            # lands on a block edge, so every kv bucket at or above the window leaves the
            # same blocks active. The window instead caps the count, tightly enough to
            # name the top rung exactly -- one per query bucket, since a longer query
            # reaches further back. Exact at the top is what keeps padding near-free:
            # decode sits at its own cap on almost every step.
            caps = sorted({self._max_active_blocks(q) for q in self._query_buckets})
            self._num_blocks_buckets = sorted({r for r in geometric if r < caps[-1]} | set(caps))

        logger.info(
            "SpyreAttnBucketer: %d kv buckets [%d..%d], %d query buckets [%d..%d], "
            "num_blocks buckets %s, sliding_window=%s",
            len(self._kv_buckets),
            self._kv_buckets[0],
            self._kv_buckets[-1],
            len(self._query_buckets),
            self._query_buckets[0],
            self._query_buckets[-1],
            self._num_blocks_buckets,
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

    def _max_active_blocks(self, query_len: int) -> int:
        """Most blocks a windowed sequence with this ``query_len`` can leave active.

        Monotone in ``query_len`` -- a longer query reaches further back, so its
        earliest window starts earlier -- so a query bucket's own width bounds every
        real length that rounds onto it.

        Scanned, not solved: the count saws as the window's lower boundary crosses a
        block, and a closed form for its peak is exactly the kind of thing that is
        wrong by one for a window just past a block multiple. Scanning one window plus
        two blocks past the first clipped length covers every offset inside a block,
        after which the pattern repeats with period ``block_size``.

        Deliberately separate from ``_sliding_plan``, which could yield these as a
        by-product: the ladder is read by ``build()`` on every step so it has to exist
        before serving, while the plan is only needed to record, and this scan is a few
        thousand iterations against the plan's few hundred thousand.
        """
        assert self._sliding_window is not None
        query_len = min(query_len, self._max_model_len)
        limit = min(self._max_model_len, query_len + self._sliding_window + 2 * self.block_size)
        return max(
            len(
                sliding_active_blocks(
                    kv_len, kv_len - query_len, self._sliding_window, self.block_size
                )
            )
            for kv_len in range(query_len, limit + 1)
        )

    @functools.cached_property
    def _sliding_plan(self) -> dict[SpyreAttnBucket, tuple[int, int]]:
        """Each windowed variant a sequence can reach, and the shortest one reaching it.

        Warmup needs a concrete sequence per variant, and under a window the bucket is
        not one: the block count it carries is an ACTIVE count, and the length that
        realizes it is neither block-aligned nor derivable from the count alone. So the
        reachable ``(block bucket, query bucket)`` pairs and their witnesses come from
        one scan of the definition -- which also means a pair is recorded only if some
        sequence dispatches to it, and none that does is missed.

        Runs once per bucketer, shared by every layer in the attention group while
        ``record_graphs`` runs per layer. ~0.2s at gemma-4's shapes, against the minutes
        of Inductor compiles it sizes.
        """
        assert self._sliding_window is not None
        block_size = self.block_size
        out: dict[SpyreAttnBucket, tuple[int, int]] = {}
        for width in self._query_buckets:
            for query_len in range(
                self.min_real_query_len(width), min(width, self._max_model_len) + 1
            ):
                limit = min(
                    self._max_model_len,
                    query_len + self._sliding_window + 2 * block_size,
                )
                for kv_len in range(query_len, limit + 1):
                    first = max(0, kv_len - query_len - self._sliding_window + 1) // block_size
                    count = (kv_len + block_size - 1) // block_size - first
                    rung = self.find_blocks_bucket(count)
                    assert rung is not None, (
                        f"active count {count} exceeds the top block bucket "
                        f"{self._num_blocks_buckets[-1]}, which the window caps"
                    )
                    out.setdefault(SpyreAttnBucket(rung, width), (kv_len, query_len))
        return out

    def witness(self, bucket: SpyreAttnBucket) -> tuple[int, int]:
        """A ``(kv_len, query_len)`` one sequence can have that dispatches to ``bucket``.

        Without a window the bucket already is one: ``build()`` pads the block count up
        onto this ladder, so ``num_blocks`` whole blocks holding the shortest query that
        reaches this width dispatch there. Under a window the length comes from the scan
        that found the variant.
        """
        if self._sliding_window is not None:
            return self._sliding_plan[bucket]
        kv_len = bucket.num_blocks * self.block_size
        query_len = self.min_real_query_len(bucket.padded_query_len)
        assert query_len <= kv_len, f"{bucket} pairs a query length no sequence can reach"
        return kv_len, query_len

    def decode_witness_kv_len(self, num_blocks: int) -> int | None:
        """A ``kv_len`` whose one-token decode step pads its block count onto ``num_blocks``.

        ``None`` when no decode step can ask for that bucket, so the recorder skips it.

        Without a window a block-aligned ``kv_len`` realizes exactly ``num_blocks``, so
        the bucket is its own witness. Under one it is the *worst* possible choice: the
        window's lower boundary lands on a block edge, so every block-aligned length
        leaves exactly ``ceil(W / block_size)`` blocks active and every bucket above
        that realizes the same one -- while a real decode reaches one block more (its
        ``kv_len`` is almost never block-aligned) and rounds onto the next bucket up,
        which then gets no witness at any ``max_model_len``.
        """
        if self._sliding_window is None:
            return num_blocks * self.block_size
        decode = self._sliding_plan.get(SpyreAttnBucket(num_blocks, 1))
        return None if decode is None else decode[0]

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

        The two size axes aren't independent: ``kv_len >= query_len`` always, so
        a query bucket only pairs with block counts that can hold it -- the full
        cross product would record many unreachable variants at a long context.
        Requires the backend to round each sequence's own query_len, so the bound
        holds per sequence and not against a batch max.
        The bound is on the *smallest real* query_len that reaches a bucket, not
        the bucket itself, since a 2-token query on a 1-block sequence still
        dispatches to a large padded bucket; bounding by the bucket would prune
        that variant and put a compile back in the serving path.

        Under a window the same bound would be wrong in both directions -- the block
        axis is an active count the window caps well below ``kv_len / block_size``, and
        a wide query reaches counts a narrow one cannot -- so the pairs come from the
        scan in ``_sliding_plan``, which is the reachable set by construction.
        """
        if self._sliding_window is not None:
            return sorted(
                self._sliding_plan,
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

        The full ``num_seqs_buckets x num_blocks_buckets`` grid, minus the block buckets
        no decode step can reach: unlike ``variants()`` there is no inter-axis bound to
        exploit, since a decode batch of any size can sit at any context length. Both
        axes are geometric, so the grid stays small.
        """
        if not envs.SPYRE_BATCHED_DECODE:
            return []
        out: list[SpyreAttnBatchedDecodeBucket] = []
        for num_blocks in sorted(self._num_blocks_buckets, reverse=True):
            if self.decode_witness_kv_len(num_blocks) is None:
                # A windowed ladder's top rungs belong to wide prefill chunks; no decode
                # step reaches them, so there is no sequence to record one with.
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
