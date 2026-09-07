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

"""Bucketer for the attention kernel's compile cache.

``SpyreAttentionImpl`` compiles one kernel per
``(num_blocks, padded_query_len, store_mode, needs_gather)`` key, lazily on
first use, which puts a full Inductor compile in the serving path. This module
enumerates the keys a run can reach so warmup can record them all up front.

Separate from ``SpyreShapeBucketer``, which dispatches a single ``num_tokens``
int for the model graph; an attention variant is 2-D (kv_len and query_len
buckets) plus two discrete flags.

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

logger = init_logger(__name__)

# Store modes the kernel factory accepts, in the order forward() prefers them.
STORE_MODES = ("none", "copy", "index")

# Spacing of the default query buckets above the decode bucket, capped against
# max_num_batched_tokens. Every non-decode batch pads its query length up to a
# multiple of this.
_DEFAULT_QUERY_BUCKET_STEP = 512


@dataclass(frozen=True)
class SpyreAttnBucket:
    """One recordable attention kernel variant.

    Fields mirror ``SpyreAttentionImpl._get_attn_fn``'s cache key exactly, so a
    recorded bucket and a runtime dispatch are the same tuple.
    """

    num_blocks: int
    padded_query_len: int
    store_mode: str
    needs_gather: bool

    @property
    def key(self) -> tuple[int, int, str, bool]:
        return (self.num_blocks, self.padded_query_len, self.store_mode, self.needs_gather)


def _parse_buckets(raw: str | None) -> list[int] | None:
    """Parse a comma-separated env-var bucket list, or None when unset/empty."""
    if not raw:
        return None
    values = sorted({int(part) for part in raw.split(",") if part.strip()})
    if not values or values[0] < 1:
        raise ValueError(f"bucket entries must be >= 1, got {raw!r}")
    return values


def _resolve_buckets(
    raw: str | None, limit: int, name: str, default: Callable[[], list[int]]
) -> list[int]:
    """One axis's buckets: the env override topped up to ``limit``, else ``default()``.

    ``limit`` bounds the lengths the engine can schedule (max_model_len for kv,
    max_num_batched_tokens for query); an override topping out below it would
    leave that range with no bucket, missing the lookup for batches warmup was
    meant to cover. Only an override can be short (defaults already end at their
    limit). Entries above the limit are left alone -- unreachable, not wrong.
    """
    buckets = _parse_buckets(raw)
    if buckets is None:
        return default()
    if buckets[-1] < limit:
        logger.warning(
            "%s tops out at %d, below the %d it must cover; appending %d. Lengths in "
            "(%d, %d] would otherwise have no recorded bucket and would compile an "
            "attention kernel in the serving path.",
            name,
            buckets[-1],
            limit,
            limit,
            buckets[-1],
            limit,
        )
        buckets = [*buckets, limit]
    return buckets


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

        # Imported at call time, not module scope: spyre_attn imports this
        # module, so a top-level import back into it would be circular.
        from spyre_inference.v1.attention.backends.spyre_attn import _powers_of_two_up_to

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

        # Default: powers of two from block_size up to max_model_len. The
        # recorded set is a product of both axes, so a bucket per KV token at a
        # 32k context would be tens of thousands of variants; doubling keeps it
        # affordable, with each bucket's extra padding absorbed by the mask.
        # Starting at block_size rather than 1 skips buckets that would dedupe
        # away anyway, since num_blocks = ceil(kv / block_size).
        self._kv_buckets: list[int] = _resolve_buckets(
            envs.SPYRE_ATTN_KV_BUCKETS,
            max_model_len,
            "SPYRE_ATTN_KV_BUCKETS",
            lambda: list(_powers_of_two_up_to(max_model_len, start=block_size)),
        )

        # Default: [1] (the decode-only batch, exempt from query padding by
        # build()) then multiples of a step up to max_num_batched_tokens. Coarse
        # bucketing: a prefill pays padding up to the next bucket, which the mask
        # discards. The step is capped at 512 so a large max_num_batched_tokens
        # doesn't make the one non-decode bucket enormous.
        step = min(_DEFAULT_QUERY_BUCKET_STEP, max_batched)
        self._query_buckets: list[int] = _resolve_buckets(
            envs.SPYRE_ATTN_QUERY_BUCKETS,
            max_batched,
            "SPYRE_ATTN_QUERY_BUCKETS",
            lambda: sorted({1, *range(step, max_batched + 1, step), max_batched}),
        )

        # num_blocks is what the kernel specializes on. Derived from the kv
        # buckets, one block count per kv bucket, rather than enumerating every
        # integer up to max_model_len / block_size.
        self._num_blocks_buckets: list[int] = sorted(
            {(kv + block_size - 1) // block_size for kv in self._kv_buckets}
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

    def find_kv_bucket(self, kv_len: int) -> int | None:
        return self._round_up(kv_len, self._kv_buckets)

    def find_query_bucket(self, query_len: int) -> int | None:
        return self._round_up(query_len, self._query_buckets)

    @staticmethod
    def _round_up(n: int, buckets: list[int]) -> int | None:
        idx = bisect.bisect_left(buckets, n)
        return buckets[idx] if idx < len(buckets) else None

    def variants(self) -> list[SpyreAttnBucket]:
        """Every variant worth recording, largest first.

        The two size axes aren't independent: ``kv_len >= query_len`` always, so
        a query bucket only pairs with block counts that can hold it -- the full
        cross product would record many unreachable variants at a long context.
        The bound is on the *smallest real* query_len that reaches a bucket, not
        the bucket itself, since a 2-token query on a 1-block sequence still
        dispatches to a large padded bucket; bounding by the bucket would prune
        that variant and put a compile back in the serving path.

        The flags aren't enumerated directly -- each bucket's inputs are fed
        through ``forward``'s own resolvers, so the recorded set follows the
        backend by construction. Above the decode bucket the two flags vary
        independently: ``store_mode`` follows batch width (one-token -> "copy",
        wider -> "index"); ``needs_gather`` follows whether one sequence owns the
        query buffer whole from row 0. ``"copy"`` is thus only reachable at
        ``padded_query_len == 1``, where ``build()`` exempts the batch from query
        padding. ``store_mode="none"`` (the un-fused fallback) pairs with either
        gather setting.
        """
        # Imported at call time; see __init__ for why module scope would be circular.
        from spyre_inference.v1.attention.backends.spyre_attn import (
            resolve_needs_gather,
            resolve_store_mode,
        )

        # Smallest real query_len that rounds up to each bucket: one past the
        # bucket below (1 for the smallest).
        ascending = sorted(self._query_buckets)
        min_real_query = {
            bucket: (ascending[i - 1] + 1 if i else 1) for i, bucket in enumerate(ascending)
        }
        out: list[SpyreAttnBucket] = []
        for num_blocks in sorted(self._num_blocks_buckets, reverse=True):
            max_query_here = num_blocks * self.block_size
            for padded_query_len in sorted(self._query_buckets, reverse=True):
                if min_real_query[padded_query_len] > max_query_here:
                    continue
                # `output` and `query` share one row count (query.shape[0]), the
                # only input both resolvers read. At the decode bucket that count
                # can be 1 (lone sequence owning row 0) or more (several one-token
                # sequences also padding to aligned_max_query_len == 1).
                row_counts = (1, 2) if padded_query_len == 1 else (padded_query_len,)
                flag_pairs = {
                    (
                        resolve_store_mode(fused_store_ok, rows),
                        resolve_needs_gather(q_start, padded_query_len, padded_query_len, rows),
                    )
                    for fused_store_ok in (True, False)
                    for rows in row_counts
                    for q_start in (0, 1)
                    # A sequence cannot start past the buffer it lives in.
                    if q_start < rows
                }
                for store_mode, needs_gather in sorted(flag_pairs):
                    out.append(
                        SpyreAttnBucket(
                            num_blocks=num_blocks,
                            padded_query_len=padded_query_len,
                            store_mode=store_mode,
                            needs_gather=needs_gather,
                        )
                    )
        return out
