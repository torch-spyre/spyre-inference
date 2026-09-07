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

"""Tests for the attention graph recorder.

CPU-only: these check that recording populates the kernel cache with exactly
the keys the bucketer enumerates, and that a subsequent dispatch reuses them
instead of growing the cache. The kernels run eagerly here (no Spyre), which
is enough to exercise dummy-arg construction and cache bookkeeping.
"""

from unittest.mock import MagicMock

import pytest
import torch

from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyrePagedKVCache,
)
from spyre_inference.v1.attention.spyre_attn_bucketer import SpyreAttnBucketer

pytestmark = pytest.mark.attention

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_SIZE = 64
BLOCK_SIZE = 64
NUM_PAGES = 8


@pytest.fixture()
def impl(default_vllm_config):
    impl = SpyreAttentionImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        scale=1.0 / (HEAD_SIZE**0.5),
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
    )
    # The fixture's bare CompilationConfig leaves mode unset, which resolves to
    # eager. Recording is a no-op there by design, so force the compiled path;
    # _maybe_compile is what actually decides whether Inductor is invoked.
    impl._compile_attn = True
    return impl


@pytest.fixture()
def kv_cache():
    shape = (NUM_PAGES, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    return SpyrePagedKVCache(
        k_pages=torch.zeros(shape, dtype=torch.float16),
        v_pages=torch.zeros(shape, dtype=torch.float16),
    )


def _runtime_flag_pairs(bucketer, padded_query_len: int):
    """The (store_mode, needs_gather) pairs a batch at this query bucket can hit.

    Read off the bucketer rather than hardcoded, since it derives them from the
    backend's own resolvers. "copy" needs a one-row batch, so it only appears
    at the decode bucket.
    """
    pairs = {
        (v.store_mode, v.needs_gather)
        for v in bucketer.variants()
        if v.padded_query_len == padded_query_len
    }
    assert pairs, f"no recorded variant at query bucket {padded_query_len}"
    return pairs


def make_bucketer(max_model_len=256, max_num_batched_tokens=64):
    config = MagicMock()
    config.cache_config.block_size = BLOCK_SIZE
    config.model_config.max_model_len = max_model_len
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    return SpyreAttnBucketer(config)


class TestRecordGraphs:
    def test_populates_cache_with_enumerated_keys(self, impl, kv_cache):
        bucketer = make_bucketer()
        assert impl._attn_fns == {}

        recorded = impl.record_graphs(torch.device("cpu"), bucketer, kv_cache)

        assert recorded > 0
        expected = {v.key for v in bucketer.variants() if v.num_blocks <= NUM_PAGES}
        assert set(impl._attn_fns) == expected

    def test_dispatch_after_recording_does_not_grow_the_cache(self, impl, kv_cache):
        """The acceptance criterion: no request compiles a new variant.

        Rounds sizes and resolves flags the way production does, so a drift
        between the two copies of the rules shows up here.
        """
        bucketer = make_bucketer()
        impl.record_graphs(torch.device("cpu"), bucketer, kv_cache)
        snapshot = len(impl._attn_fns)

        for kv_len in (1, 60, 64, 200, 256):
            for query_len in (1, 5, 32, 64):
                if query_len > kv_len:
                    continue
                padded_query_len = bucketer.find_query_bucket(query_len)
                num_blocks = bucketer._round_up(
                    (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE, bucketer.num_blocks_buckets
                )
                assert padded_query_len is not None and num_blocks is not None
                if num_blocks > NUM_PAGES:
                    continue
                for store_mode, needs_gather in _runtime_flag_pairs(bucketer, padded_query_len):
                    impl._get_attn_fn(
                        num_blocks,
                        padded_query_len,
                        store_mode=store_mode,
                        needs_gather=needs_gather,
                    )

        assert len(impl._attn_fns) == snapshot

    def test_is_idempotent(self, impl, kv_cache):
        bucketer = make_bucketer()
        impl.record_graphs(torch.device("cpu"), bucketer, kv_cache)
        after_first = len(impl._attn_fns)

        assert impl.record_graphs(torch.device("cpu"), bucketer, kv_cache) == 0
        assert len(impl._attn_fns) == after_first

    def test_skips_variants_exceeding_the_page_allocation(self, impl, kv_cache):
        """Buckets sized from max_model_len can outrun a small KV cache."""
        bucketer = make_bucketer(max_model_len=4096)
        impl.record_graphs(torch.device("cpu"), bucketer, kv_cache)

        assert impl._attn_fns
        assert all(key[0] <= NUM_PAGES for key in impl._attn_fns)

    def test_real_metadata_dispatch_does_not_grow_the_cache(self, impl, kv_cache, monkeypatch):
        """The acceptance criterion, driven from real builder metadata.

        Unlike ``test_dispatch_after_recording_does_not_grow_the_cache``, this
        builds metadata for unbucketed kv_lens through
        ``SpyreAttentionMetadataBuilder`` and dispatches on the block counts
        ``build()`` actually produced.
        """
        from vllm.config import get_current_vllm_config

        from tests.attention.test_spyre_attn import _padded_mask_metadata

        # Built from the live config, not make_bucketer's narrower stand-in, so
        # this bucketer and the builder's derive from the same config.
        vllm_config = get_current_vllm_config()
        bucketer = SpyreAttnBucketer(vllm_config)
        impl.record_graphs(torch.device("cpu"), bucketer, kv_cache)
        snapshot = len(impl._attn_fns)
        assert snapshot > 0

        for query_len, kv_len in [(1, 1), (1, 65), (1, 200), (7, 65), (32, 300), (33, 300)]:
            metadata = _padded_mask_metadata(
                [(query_len, kv_len)],
                block_size=BLOCK_SIZE,
                num_query_heads=NUM_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_size=HEAD_SIZE,
                max_num_blocks=NUM_PAGES,
            )
            assert metadata.padded_num_blocks is not None
            num_blocks = metadata.padded_num_blocks[0]
            assert num_blocks in bucketer.num_blocks_buckets, (
                f"kv_len={kv_len} produced an unrecorded block count {num_blocks}"
            )
            for store_mode, needs_gather in _runtime_flag_pairs(
                bucketer, metadata.aligned_max_query_len
            ):
                impl._get_attn_fn(
                    num_blocks,
                    metadata.aligned_max_query_len,
                    store_mode=store_mode,
                    needs_gather=needs_gather,
                )

        assert len(impl._attn_fns) == snapshot

    def test_eager_records_nothing(self, impl, kv_cache):
        impl._compile_attn = False
        assert impl.record_graphs(torch.device("cpu"), make_bucketer(), kv_cache) == 0
        assert impl._attn_fns == {}

    def test_a_failing_variant_does_not_abort_the_pass(self, impl, kv_cache, monkeypatch):
        """One bad variant must not take down engine startup."""
        bucketer = make_bucketer()
        calls = {"n": 0}
        real = impl._record_one

        def flaky(bucket, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("synthetic lowering failure")
            return real(bucket, *args, **kwargs)

        monkeypatch.setattr(impl, "_record_one", flaky)
        recorded = impl.record_graphs(torch.device("cpu"), bucketer, kv_cache)

        assert recorded == calls["n"] - 1
        # The failed key is left uncached, so it can still compile on first use.
        assert bucketer.variants()[0].key not in impl._attn_fns


class TestRecompileLimit:
    def test_limit_is_raised_during_recording_and_restored(self, impl, kv_cache):
        """Dynamo's accumulated limit is global, so more buckets than it allows would
        otherwise stop compiling partway through and fall back to eager."""
        bucketer = make_bucketer()
        before = torch._dynamo.config.accumulated_recompile_limit
        seen = []

        real = impl._record_one

        def spy(*args, **kwargs):
            seen.append(torch._dynamo.config.accumulated_recompile_limit)
            return real(*args, **kwargs)

        impl._record_one = spy
        impl.record_graphs(torch.device("cpu"), bucketer, kv_cache)

        assert seen and min(seen) >= len(bucketer.variants())
        assert torch._dynamo.config.accumulated_recompile_limit == before

    def test_limit_is_restored_even_when_recording_raises(self, impl, kv_cache, monkeypatch):
        before = torch._dynamo.config.accumulated_recompile_limit
        monkeypatch.setattr(
            impl, "_record_all", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        with pytest.raises(RuntimeError):
            impl.record_graphs(torch.device("cpu"), make_bucketer(), kv_cache)
        assert torch._dynamo.config.accumulated_recompile_limit == before
