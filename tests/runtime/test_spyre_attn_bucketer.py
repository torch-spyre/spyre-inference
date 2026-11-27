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

"""Unit tests for SpyreAttnBucketer. No hardware required."""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from spyre_inference import envs
from spyre_inference.v1.attention.spyre_attn_bucketer import (
    SpyreAttnBucketer,
    _parse_buckets,
    batched_decode_chunking,
)

BLOCK_SIZE = 64


def make_config(
    max_model_len=2048,
    max_num_batched_tokens=512,
    block_size=BLOCK_SIZE,
    max_num_seqs=8,
    runner_type="generate",
):
    config = MagicMock()
    config.cache_config.block_size = block_size
    config.model_config.max_model_len = max_model_len
    config.model_config.runner_type = runner_type
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    config.scheduler_config.max_num_seqs = max_num_seqs
    return config


@pytest.fixture()
def bucketer():
    return SpyreAttnBucketer(make_config())


@pytest.fixture(autouse=True)
def _clear_env_cache(monkeypatch):
    """envs caches on first read, so each test must start from a clean slate."""
    envs.clear_env_cache()
    yield
    envs.clear_env_cache()


class TestBuckets:
    def test_query_buckets_lead_with_decode_case(self, bucketer):
        assert bucketer.query_buckets[0] == 1
        assert bucketer.query_buckets == [1, 512]

    def test_query_buckets_are_multiples_of_the_step(self):
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=2048))
        assert b.query_buckets == [1, 512, 1024, 1536, 2048]

    def test_query_bucket_step_capped_by_max_num_batched_tokens(self):
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=300))
        assert b.query_buckets == [1, 300]

    def test_buckets_include_non_power_of_two_limit(self):
        b = SpyreAttnBucketer(make_config(max_model_len=3000, max_num_batched_tokens=100))
        assert b.num_blocks_buckets == [-(-3000 // BLOCK_SIZE)]
        assert b.query_buckets == [1, 100]

    def test_largest_bucket_is_always_the_limit(self):
        for limit in (1, 2, 3, 64, 100, 4096, 32768):
            b = SpyreAttnBucketer(make_config(max_model_len=limit, max_num_batched_tokens=limit))
            assert b.num_blocks_buckets == [-(-limit // BLOCK_SIZE)]
            assert b.query_buckets[-1] == limit

    def test_buckets_have_no_duplicates(self):
        for limit in (1, 2, 3, 64, 100, 512, 513, 4096, 32768):
            b = SpyreAttnBucketer(make_config(max_model_len=limit, max_num_batched_tokens=limit))
            assert b.query_buckets == sorted(set(b.query_buckets))


class TestPoolingQueryBucketCap:
    """Pooling's query_len can't exceed max_model_len; without this cap, warmup
    could record a query bucket with no matching num_blocks bucket, crashing
    with "num_blocks=N exceeds the largest recorded bucket" (CLIP's text tower:
    max_model_len=77, max_num_batched_tokens much larger).

    Stick-aligned the same way the encoder shape ladder pads a request
    (_align_up_pow2): CLIP's max_model_len=77 pads to 128, not 77."""

    def test_pooling_caps_query_buckets_at_stick_aligned_max_model_len(self):
        b = SpyreAttnBucketer(
            make_config(max_model_len=77, max_num_batched_tokens=2048, runner_type="pooling")
        )
        assert b.query_buckets[-1] == 128
        # The largest recorded query bucket must round onto a real num_blocks
        # bucket -- this is what crashed for CLIP.
        largest_query_blocks = -(-b.query_buckets[-1] // b.block_size)
        assert b.find_blocks_bucket(largest_query_blocks) is not None

    def test_pooling_query_bucket_covers_clip_text_tower_padded_length(self):
        """CLIP's max_model_len=77 pads to query_len=128; that must round onto a bucket."""
        b = SpyreAttnBucketer(
            make_config(max_model_len=77, max_num_batched_tokens=512, runner_type="pooling")
        )
        assert b.find_query_bucket(128) == 128

    def test_generate_is_unaffected(self):
        b = SpyreAttnBucketer(
            make_config(max_model_len=77, max_num_batched_tokens=2048, runner_type="generate")
        )
        assert b.query_buckets[-1] == 2048

    def test_pooling_is_a_noop_when_max_batched_is_already_smaller(self):
        b = SpyreAttnBucketer(
            make_config(max_model_len=2048, max_num_batched_tokens=512, runner_type="pooling")
        )
        assert b.query_buckets[-1] == 512


class TestFindBucket:
    def test_exact_match(self, bucketer):
        assert bucketer.find_query_bucket(512) == 512

    def test_rounds_up(self, bucketer):
        assert bucketer.find_query_bucket(33) == 512

    def test_query_len_one_maps_to_decode_bucket(self, bucketer):
        assert bucketer.find_query_bucket(1) == 1

    def test_query_above_decode_rounds_to_the_step(self, bucketer):
        """Only two buckets by default, so every non-decode query pads to the step."""
        for query_len in (2, 33, 129, 511, 512):
            assert bucketer.find_query_bucket(query_len) == 512

    def test_every_block_count_pads_to_max_model_len(self, bucketer):
        assert {bucketer.find_blocks_bucket(n) for n in range(1, 33)} == {2048 // BLOCK_SIZE}

    def test_exceeds_max_returns_none(self, bucketer):
        assert bucketer.find_blocks_bucket(2048 // BLOCK_SIZE + 1) is None
        assert bucketer.find_query_bucket(513) is None

    def test_min_real_query_len_is_one_past_the_bucket_below(self, bucketer):
        """The length the recorder builds its synthetic sequence at, and the one
        ``variants()`` prunes against."""
        assert bucketer.min_real_query_len(1) == 1
        assert bucketer.min_real_query_len(512) == 2
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=2048))
        assert [b.min_real_query_len(q) for q in b.query_buckets] == [1, 2, 513, 1025, 1537]

    def test_min_real_query_len_rounds_back_onto_its_own_bucket(self, bucketer):
        for bucket in bucketer.query_buckets:
            assert bucketer.find_query_bucket(bucketer.min_real_query_len(bucket)) == bucket


class TestVariants:
    def test_no_duplicates(self, bucketer):
        variants = bucketer.variants()
        assert len(variants) == len(set(variants))

    def test_stable_across_calls(self, bucketer):
        assert bucketer.variants() == bucketer.variants()

    def test_largest_first(self, bucketer):
        variants = bucketer.variants()
        assert variants[0].num_blocks == max(v.num_blocks for v in variants)

    def test_descriptor_is_frozen(self, bucketer):
        with pytest.raises(FrozenInstanceError):
            bucketer.variants()[0].num_blocks = 10

    def test_query_buckets_stay_narrower_than_the_query_buffer(self):
        """``SpyreAttentionImpl`` gathers unconditionally, which faults if a
        sequence is the whole buffer, so every bucket must fit with a row spare."""
        max_batched = 512
        bucketer = SpyreAttnBucketer(make_config(max_num_batched_tokens=max_batched))
        for bucket in bucketer.query_buckets:
            assert bucket < max_batched + 1

    def test_prunes_query_buckets_no_real_query_len_can_reach(self, bucketer):
        """Pruning bounds the *smallest real* query_len that reaches a bucket, not
        the bucket itself: a 2-token query on a 1-block sequence still legitimately
        pads to 512."""
        ascending = sorted(bucketer.query_buckets)
        min_real = {b: (ascending[i - 1] + 1 if i else 1) for i, b in enumerate(ascending)}
        for v in bucketer.variants():
            assert min_real[v.padded_query_len] <= v.num_blocks * BLOCK_SIZE

    @pytest.mark.parametrize("kv_len", [1, 256, 257, 2048])
    @pytest.mark.parametrize("query_len", [1, 32, 33, 512])
    def test_every_rounded_size_lands_on_a_recorded_variant(self, bucketer, kv_len, query_len):
        """The whole point of recording: no runtime batch may miss the cache.

        Drives the two lookups production uses -- find_query_bucket, and
        _round_up onto num_blocks_buckets (what _pad_num_blocks calls)."""
        if query_len > kv_len:
            pytest.skip("a sequence cannot have more new tokens than total KV")
        padded_query_len = bucketer.find_query_bucket(query_len)
        num_blocks = bucketer._round_up(
            (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE, bucketer.num_blocks_buckets
        )
        assert padded_query_len is not None and num_blocks is not None
        sizes = {(v.num_blocks, v.padded_query_len) for v in bucketer.variants()}
        assert (num_blocks, padded_query_len) in sizes

    def test_count_stays_tractable_at_long_context(self):
        """Dense buckets here would be tens of thousands of Inductor compiles."""
        b = SpyreAttnBucketer(make_config(32768, 2048))
        assert len(b.variants()) < 500

    def test_num_seqs_buckets_ladder_from_one_to_max_num_seqs(self):
        """The ladder starts at 1 and tops out at max_num_seqs."""
        assert SpyreAttnBucketer(make_config(max_num_seqs=8)).num_seqs_buckets == [1, 2, 4, 8]
        assert SpyreAttnBucketer(make_config(max_num_seqs=6)).num_seqs_buckets == [1, 2, 4, 6]

    @pytest.mark.parametrize("max_num_seqs", [1, 2, 3])
    def test_num_seqs_buckets_cover_small_batches(self, max_num_seqs, monkeypatch):
        """Small batches dispatch to the batched kernel too, so each has a bucket."""
        b = SpyreAttnBucketer(make_config(max_num_seqs=max_num_seqs))
        assert b.num_seqs_buckets == list(range(1, max_num_seqs + 1))
        assert b.batched_decode_variants()
        assert b.find_sequence_bucket(max_num_seqs) == max_num_seqs

        monkeypatch.setenv("SPYRE_ATTN_NUM_SEQS_BUCKETS", str(max_num_seqs))
        envs.clear_env_cache()
        assert SpyreAttnBucketer(make_config(max_num_seqs=max_num_seqs)).num_seqs_buckets == [
            max_num_seqs
        ]


class TestEnvOverride:
    def test_query_buckets_override_is_sorted_and_deduped(self, monkeypatch):
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "64,1,16,64")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=64))
        assert b.query_buckets == [1, 16, 64]

    def test_truncated_query_override_is_topped_up_to_max_batched(self, monkeypatch):
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "1,16")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=512))
        assert b.query_buckets == [1, 16, 512]
        assert b.find_query_bucket(512) == 512

    def test_override_above_the_limit_is_dropped(self, monkeypatch):
        """Entries past the limit are unreachable; drop them and cover the limit."""
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "16,8192")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=512))
        assert b.query_buckets == [16, 512]

    def test_override_entirely_above_the_limit_keeps_only_the_limit(self, monkeypatch):
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "4096,8192")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=512))
        assert b.query_buckets == [512]

    def test_covers_every_in_contract_length_under_a_short_override(self, monkeypatch):
        """The point of the top-up: no in-contract batch falls off either axis."""
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "1")
        envs.clear_env_cache()
        max_model_len, max_batched = 1024, 256
        b = SpyreAttnBucketer(make_config(max_model_len, max_batched))
        for kv_len in (1, 129, 500, max_model_len):
            for query_len in (1, 2, 200, max_batched):
                if query_len > kv_len:
                    continue
                assert b.find_blocks_bucket(-(-kv_len // BLOCK_SIZE)) is not None
                assert b.find_query_bucket(query_len) is not None

    def test_a_length_outside_the_contract_has_no_bucket(self, monkeypatch):
        """Past max_model_len there is no bucket by design."""
        b = SpyreAttnBucketer(make_config(max_model_len=2048))
        assert b.find_blocks_bucket(2048 // BLOCK_SIZE + 1) is None

    def test_parse_buckets_rejects_non_positive(self):
        with pytest.raises(ValueError):
            _parse_buckets("0,32")

    def test_parse_buckets_empty_is_none(self):
        assert _parse_buckets("") is None
        assert _parse_buckets(None) is None


class TestBatchedDecodeVariants:
    """The batched decode enumeration, keyed on (num_seqs, blocks_per_chunk, num_chunks)."""

    @pytest.fixture()
    def enabled(self):
        return SpyreAttnBucketer(make_config())

    def test_covers_the_full_num_seqs_by_num_blocks_grid(self, enabled):
        assert {(v.num_seqs, v.num_blocks) for v in enabled.batched_decode_variants()} == {
            (s, n) for n in enabled.num_blocks_buckets for s in enabled.num_seqs_buckets
        }

    def test_no_duplicates(self, enabled):
        variants = enabled.batched_decode_variants()
        assert len(set(variants)) == len(variants)

    def test_stable_across_calls(self, enabled):
        assert enabled.batched_decode_variants() == enabled.batched_decode_variants()

    def test_largest_first(self, enabled):
        blocks = [v.num_blocks for v in enabled.batched_decode_variants()]
        assert blocks == sorted(blocks, reverse=True)

    def test_descriptor_is_frozen(self, enabled):
        with pytest.raises(FrozenInstanceError):
            enabled.batched_decode_variants()[0].num_seqs = 1  # ty: ignore[invalid-assignment]

    def test_chunking_matches_the_shared_helper(self, enabled):
        for v in enabled.batched_decode_variants():
            assert batched_decode_chunking(v.num_seqs, v.num_blocks) == (
                v.blocks_per_chunk,
                v.num_chunks,
            )
            # The block axis pads up to a whole chunk, never truncates.
            assert v.blocks_per_chunk * v.num_chunks >= v.num_blocks

    def test_chunking_pads_when_the_ladder_is_not_a_power_of_two(self):
        """With power-of-two buckets the padding vanishes, so a drifting copy of the
        chunking rule would look correct."""
        assert batched_decode_chunking(8, 8) == (4, 2)  # 4*2 == 8, no padding
        assert batched_decode_chunking(6, 8) == (5, 2)  # 5*2 == 10, padded

    def test_count_stays_tractable_at_long_context(self, monkeypatch):
        b = SpyreAttnBucketer(make_config(32768, 2048, max_num_seqs=64))
        assert len(b.batched_decode_variants()) < 100

    def test_a_sliding_window_pads_batched_decode_to_its_window(self):
        """A one-token query's window touches at most ceil(window / block_size) + 1 pages."""
        b = SpyreAttnBucketer(make_config(), sliding_window=128)
        assert b.num_blocks_buckets == [1, 2, 4, 8, 16, 32]
        assert b.batched_blocks_buckets(8) == [1, 2, 3]
        assert {v.num_blocks for v in b.batched_decode_variants()} == {1, 2, 3}
        wide = SpyreAttnBucketer(make_config(), sliding_window=100_000)
        assert max(v.num_blocks for v in wide.batched_decode_variants()) == 2048 // BLOCK_SIZE


class TestDeclaredPages:
    """The per-sequence kernels take a page range instead of the num_blocks bucket when it fits."""

    @pytest.fixture(autouse=True)
    def _restore_ts_limits(self):
        from torch_spyre._inductor import config as ts_config

        # Restores the limits the bucketer sets. Not monkeypatch: its undo deletes a
        # config-module entry, hiding it for later tests.
        with ts_config.patch(min_reserved_dim_size=8, max_buckets=32):
            yield

    def test_one_variant_per_query_bucket(self):
        b = SpyreAttnBucketer(make_config(max_model_len=2048), declare_pages=True)
        variants = b.variants()
        assert {v.num_blocks for v in variants} == {b.num_blocks_buckets[-1]}
        assert [v.padded_query_len for v in variants] == sorted(b.query_buckets, reverse=True)

    def test_page_range_spans_one_page_to_max_model_len(self):
        b = SpyreAttnBucketer(make_config(max_model_len=2048), declare_pages=True)
        assert b.page_range == {"min": 1, "max": 2048 // BLOCK_SIZE, "granularity": 1}
        assert [b.find_declared_pages(n) for n in (1, 2, 7, 32)] == [1, 2, 7, 32]

    def test_prefill_page_counts_are_multiples_of_eight(self):
        from torch_spyre._inductor import config as ts_config

        b = SpyreAttnBucketer(make_config(max_model_len=31 * BLOCK_SIZE), declare_pages=True)
        assert b.page_range is not None
        assert b.page_range["max"] == 31
        assert b.prefill_page_range == {"min": 8, "max": 32, "granularity": 8}
        assert ts_config.max_buckets >= 32
        assert [b.find_declared_pages(n, prefill=True) for n in (1, 8, 9, 17, 32)] == [
            8,
            8,
            16,
            24,
            32,
        ]

    def test_a_prefill_on_the_per_sequence_range_covers_two_pages(self):
        """Too short for multiples of eight, prefill takes the per-sequence range, but a
        one-page prefill still runs over two: one page lets its unused slots through."""
        b = SpyreAttnBucketer(make_config(max_model_len=2 * BLOCK_SIZE), declare_pages=True)
        assert b.page_range == {"min": 1, "max": 2, "granularity": 1}
        assert b.prefill_page_range == b.page_range
        assert b.find_declared_pages(1, prefill=True) == 2
        assert b.find_declared_pages(1) == 1

    def test_a_one_page_context_keeps_the_bucket(self):
        b = SpyreAttnBucketer(make_config(max_model_len=BLOCK_SIZE), declare_pages=True)
        assert b.page_range is None
        assert {v.num_blocks for v in b.variants()} == {1}

    def test_max_buckets_grows_to_cover_max_model_len(self):
        from torch_spyre._inductor import config as ts_config

        b = SpyreAttnBucketer(make_config(max_model_len=32768), declare_pages=True)
        assert b.page_range == {"min": 1, "max": 32768 // BLOCK_SIZE, "granularity": 1}
        assert ts_config.max_buckets == 32768 // BLOCK_SIZE

    def test_exported_torch_spyre_limits_leave_the_ranges_unchanged(self, monkeypatch):
        """The plugin sets the floor and the cap itself, exported or not."""
        from torch_spyre._inductor import config as ts_config

        monkeypatch.setenv("MIN_RESERVED_DIM_SIZE", "8")
        monkeypatch.setenv("MAX_BUCKETS", "32")
        b = SpyreAttnBucketer(make_config(max_model_len=33 * BLOCK_SIZE), declare_pages=True)
        assert b.page_range == {"min": 1, "max": 33, "granularity": 1}
        assert b.prefill_page_range == {"min": 8, "max": 40, "granularity": 8}
        assert b._chunk_ranges and all(r["min"] == 1 for r in b._chunk_ranges.values())
        assert (ts_config.min_reserved_dim_size, ts_config.max_buckets) == (1, 40)

    def test_without_declare_pages_keeps_the_buckets(self):
        b = SpyreAttnBucketer(make_config(max_model_len=2048))
        assert b.page_range is None


class TestDynamicChunks:
    """Batched decode takes a chunk range instead of the num_blocks bucket, one chunk included."""

    @pytest.fixture(autouse=True)
    def _declarable(self):
        from torch_spyre._inductor import config as ts_config

        # Restores the limits the bucketer sets. Not monkeypatch: its undo deletes a
        # config-module entry, hiding it for later tests.
        with ts_config.patch(min_reserved_dim_size=8, max_buckets=32):
            yield

    def test_one_variant_covers_every_context_a_single_chunk_included(self):
        b = SpyreAttnBucketer(make_config(max_model_len=8192), declare_pages=True)
        max_pages = 8192 // BLOCK_SIZE
        for num_seqs in b.num_seqs_buckets:
            width = 32 // num_seqs
            (variant,) = [v for v in b.batched_decode_variants() if v.num_seqs == num_seqs]
            assert variant.dynamic_chunks
            assert (variant.blocks_per_chunk, variant.num_chunks) == (width, max_pages // width)
            for nb in (1, width, width + 1, 3 * width - 1, max_pages):
                plan = b.batched_decode_plan(num_seqs, nb)
                assert plan is not None
                assert plan.chunk_range == {"min": 1, "max": max_pages // width, "granularity": 1}
                assert plan.blocks_per_chunk == width
                assert (plan.num_chunks - 1) * width < nb <= plan.num_chunks * width

    def test_a_batch_size_whose_context_fits_one_chunk_keeps_the_buckets(self):
        # 32 pages: one full chunk at one sequence, two at two sequences.
        b = SpyreAttnBucketer(make_config(max_model_len=2048), declare_pages=True)
        variants = [v for v in b.batched_decode_variants() if v.num_seqs == 1]
        assert not any(v.dynamic_chunks for v in variants)
        assert {v.num_blocks for v in variants} == set(b.num_blocks_buckets)
        assert any(v.dynamic_chunks for v in b.batched_decode_variants() if v.num_seqs == 2)

    def test_without_a_page_range_batched_decode_keeps_the_buckets(self):
        b = SpyreAttnBucketer(make_config(max_model_len=8192))
        assert b.page_range is None
        assert not any(v.dynamic_chunks for v in b.batched_decode_variants())
        plan = b.batched_decode_plan(4, 100)
        assert plan is not None and plan.chunk_range is None
