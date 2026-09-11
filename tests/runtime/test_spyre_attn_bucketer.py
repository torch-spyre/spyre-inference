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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from spyre_inference import envs
from spyre_inference.v1.attention.backends.spyre_attn import (
    _powers_of_two_up_to,
    _token_buckets_up_to,
)
from spyre_inference.v1.attention.spyre_attn_bucketer import (
    _KV_DENSE_LADDER_CAP,
    MIN_BATCHED_SEQS,
    SpyreAttnBucketer,
    _parse_buckets,
)

BLOCK_SIZE = 64
MAX_NUM_SEQS = 8


def make_config(
    max_model_len=2048,
    max_num_batched_tokens=512,
    block_size=BLOCK_SIZE,
    max_num_seqs=MAX_NUM_SEQS,
):
    config = MagicMock()
    config.cache_config.block_size = block_size
    config.model_config.max_model_len = max_model_len
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    config.scheduler_config.max_num_seqs = max_num_seqs
    return config


def _list_pow2(limit: int, start: int = 1) -> list[int]:
    """[start, 2*start, ..., limit]."""
    return list(_powers_of_two_up_to(limit, start=start))


def _list_default_kv(limit: int, anchor: int = BLOCK_SIZE) -> list[int]:
    """The buckets the kv axis defaults to: 4/3-spaced up to _KV_DENSE_LADDER_CAP,
    powers-of-two above."""
    cap = min(limit, _KV_DENSE_LADDER_CAP)
    dense = list(_token_buckets_up_to(cap, anchor=anchor))
    if cap < limit:
        coarse = list(_powers_of_two_up_to(limit, start=cap))
        dense_set = set(dense)
        dense = dense + [b for b in coarse if b not in dense_set]
    return dense


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
    def test_kv_buckets_bound_round_up_to_max_model_len(self, bucketer):
        assert bucketer.kv_buckets == _list_default_kv(2048)
        assert bucketer.kv_buckets[-1] == 2048
        # Asserted on block counts, which is what the kernel pads to: token steps are
        # quantized to the anchor and so are coarser at the bottom.
        blocks = bucketer.num_blocks_buckets
        for prev, nxt in zip(blocks, blocks[1:]):
            if prev >= 4:
                assert (nxt - (prev + 1)) / nxt <= 1 / 4

    def test_kv_buckets_start_at_block_size(self, bucketer):
        """Buckets below block_size all collapse to num_blocks == 1, so the
        smallest bucket is block_size rather than 1."""
        assert bucketer.kv_buckets[0] == BLOCK_SIZE

    @pytest.mark.parametrize("block_size", [64, 128, 256])
    def test_kv_buckets_are_block_size_independent(self, block_size):
        """Buckets are token counts, so block_size only re-quantizes the derived block
        counts; it does not move where the token boundaries fall."""
        b = SpyreAttnBucketer(make_config(max_model_len=4096, block_size=block_size))
        assert b.kv_buckets == _list_default_kv(4096)

    def test_kv_buckets_unaffected_by_non_power_of_two_block_size(self):
        """The platform only forces block_size to a multiple of 64, so a
        non-power-of-two value is reachable. Token buckets do not depend on it;
        only the derived block counts are coarser."""
        b = SpyreAttnBucketer(make_config(max_model_len=4096, block_size=192))
        assert b.kv_buckets == _list_default_kv(4096)

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
        assert b.kv_buckets == _list_default_kv(3000)
        assert b.kv_buckets[-1] == 3000
        assert b.query_buckets == [1, 100]

    def test_largest_bucket_is_always_the_limit(self):
        for limit in (1, 2, 3, 64, 100, 4096, 32768):
            b = SpyreAttnBucketer(make_config(max_model_len=limit, max_num_batched_tokens=limit))
            assert b.kv_buckets[-1] == limit
            assert b.query_buckets[-1] == limit

    def test_buckets_have_no_duplicates(self):
        for limit in (1, 2, 3, 64, 100, 512, 513, 4096, 32768):
            b = SpyreAttnBucketer(make_config(max_model_len=limit, max_num_batched_tokens=limit))
            assert b.kv_buckets == sorted(set(b.kv_buckets))
            assert b.query_buckets == sorted(set(b.query_buckets))


class TestFindBucket:
    def test_exact_match(self, bucketer):
        assert bucketer.find_kv_bucket(512) == 512
        assert bucketer.find_query_bucket(512) == 512

    def test_rounds_up(self, bucketer):
        # Derived: the kv bucket set is a tunable default (SPYRE_ATTN_KV_BUCKETS).
        assert bucketer.find_kv_bucket(257) == next(b for b in bucketer.kv_buckets if b >= 257)
        assert bucketer.find_query_bucket(33) == 512

    def test_query_len_one_maps_to_decode_bucket(self, bucketer):
        assert bucketer.find_query_bucket(1) == 1

    def test_query_above_decode_rounds_to_the_step(self, bucketer):
        """Only two buckets by default, so every non-decode query pads to the step."""
        for query_len in (2, 33, 129, 511, 512):
            assert bucketer.find_query_bucket(query_len) == 512

    def test_kv_below_block_size_rounds_to_block_size(self, bucketer):
        assert bucketer.find_kv_bucket(1) == BLOCK_SIZE
        assert bucketer.find_kv_bucket(BLOCK_SIZE) == BLOCK_SIZE

    def test_exceeds_max_returns_none(self, bucketer):
        assert bucketer.find_kv_bucket(2049) is None
        assert bucketer.find_query_bucket(513) is None


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

    @pytest.mark.parametrize("num_seqs", [4, 5, 7, 8])
    @pytest.mark.parametrize("kv_len", [64, 300, 1025, 2048])
    def test_every_batched_size_lands_on_a_recorded_variant(self, bucketer, num_seqs, kv_len):
        """Same guarantee for the batched kernel: no runtime batch may miss the cache.

        Drives the two lookups the batched dispatch uses -- _round_up onto
        num_seqs_buckets and onto num_blocks_buckets.
        """
        b_seqs = bucketer._round_up(num_seqs, bucketer.num_seqs_buckets)
        b_blocks = bucketer._round_up(
            (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE, bucketer.num_blocks_buckets
        )
        assert b_seqs is not None and b_blocks is not None
        sizes = {(v.num_seqs, v.num_blocks) for v in bucketer.batched_variants()}
        assert (b_seqs, b_blocks) in sizes

    def test_batched_variants_cover_both_flag_states(self, bucketer):
        """needs_gather varies per step, so both values must be recorded at every size.

        store_out additionally needs a fused-store-eligible output buffer, and the
        dispatch only takes it when needs_gather is False.
        """
        by_size: dict[tuple[int, int], set[tuple[bool, bool]]] = {}
        for v in bucketer.batched_variants():
            by_size.setdefault((v.num_seqs, v.num_blocks), set()).add((v.needs_gather, v.store_out))
        from spyre_inference.v1.attention.spyre_attn_bucketer import MIN_BATCHED_SEQS

        assert by_size
        for (num_seqs, _), flags in by_size.items():
            assert (False, False) in flags
            assert (False, True) in flags
            # A gather needs a query buffer narrower than the bucket, which cannot happen
            # at the smallest one: build() only enters there and it rounds to itself.
            assert ((True, False) in flags) == (num_seqs > MIN_BATCHED_SEQS)
            # store_out with a gather is unreachable: the dispatch requires not needs_gather.
            assert (True, True) not in flags

    def test_num_seqs_buckets_are_all_reachable(self):
        """num_seqs_buckets is pre-filtered: every entry is a valid batched-kernel input."""
        b = SpyreAttnBucketer(make_config(max_num_seqs=16))
        assert all(n >= MIN_BATCHED_SEQS for n in b.num_seqs_buckets)
        assert {v.num_seqs for v in b.batched_variants()} == set(b.num_seqs_buckets)

    def test_count_stays_tractable_at_long_context(self):
        """Dense buckets here would be tens of thousands of Inductor compiles."""
        b = SpyreAttnBucketer(make_config(32768, 2048))
        assert len(b.variants()) < 500


class TestEnvOverride:
    def test_kv_buckets_override(self, monkeypatch):
        """Kept verbatim: the top entry already covers max_model_len=2048."""
        monkeypatch.setenv("SPYRE_ATTN_KV_BUCKETS", "128,512,4096")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config())
        assert b.kv_buckets == [128, 512, 4096]

    def test_query_buckets_override_is_sorted_and_deduped(self, monkeypatch):
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "64,1,16,64")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=64))
        assert b.query_buckets == [1, 16, 64]

    def test_truncated_kv_override_is_topped_up_to_max_model_len(self, monkeypatch):
        """A short override would otherwise leave (512, 2048] with no bucket."""
        monkeypatch.setenv("SPYRE_ATTN_KV_BUCKETS", "128,512")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(max_model_len=2048))
        assert b.kv_buckets == [128, 512, 2048]
        assert b.find_kv_bucket(2048) == 2048

    def test_truncated_query_override_is_topped_up_to_max_batched(self, monkeypatch):
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "1,16")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=512))
        assert b.query_buckets == [1, 16, 512]
        assert b.find_query_bucket(512) == 512

    def test_override_above_the_limit_is_left_alone(self, monkeypatch):
        """Entries past the limit are unreachable, not wrong; don't prune them."""
        monkeypatch.setenv("SPYRE_ATTN_KV_BUCKETS", "128,8192")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(max_model_len=2048))
        assert b.kv_buckets == [128, 8192]

    def test_covers_every_in_contract_length_under_a_short_override(self, monkeypatch):
        """The point of the top-up: no in-contract batch falls off either axis."""
        monkeypatch.setenv("SPYRE_ATTN_KV_BUCKETS", "128")
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "1")
        envs.clear_env_cache()
        max_model_len, max_batched = 1024, 256
        b = SpyreAttnBucketer(make_config(max_model_len, max_batched))
        for kv_len in (1, 129, 500, max_model_len):
            for query_len in (1, 2, 200, max_batched):
                if query_len > kv_len:
                    continue
                assert b.find_kv_bucket(kv_len) is not None
                assert b.find_query_bucket(query_len) is not None

    def test_a_length_outside_the_contract_has_no_bucket(self, monkeypatch):
        """Past max_model_len there is no bucket by design."""
        b = SpyreAttnBucketer(make_config(max_model_len=2048))
        assert b.find_kv_bucket(2049) is None

    def test_parse_buckets_rejects_non_positive(self):
        with pytest.raises(ValueError):
            _parse_buckets("0,32")

    def test_parse_buckets_empty_is_none(self):
        assert _parse_buckets("") is None
        assert _parse_buckets(None) is None


class TestBuilderAttnBucketer:
    """The recorder takes the builders' bucketer instead of building its own."""

    @staticmethod
    def _runner(*group_bucketers):
        """A bare runner whose attention groups hold the given bucketers.

        One argument per group, each a list of per-ubatch bucketers (Nones
        stand in for a builder that exposes none).
        """
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
        runner.attn_groups = [
            [
                SimpleNamespace(
                    metadata_builders=[
                        SimpleNamespace(_attn_bucketer=b) if b is not None else SimpleNamespace()
                        for b in builders
                    ]
                )
                for builders in group_bucketers
            ]
        ]
        return runner

    def test_returns_the_builders_instance(self):
        bucketer = SpyreAttnBucketer(make_config())
        runner = self._runner([bucketer])
        assert runner._resolve_builder_attn_bucketer() is bucketer

    def test_none_when_no_builder_exposes_one(self):
        assert self._runner([None])._resolve_builder_attn_bucketer() is None
        assert self._runner()._resolve_builder_attn_bucketer() is None

    def test_agreeing_builders_are_accepted(self):
        """Two groups, separately constructed from the same config: same buckets."""
        first = SpyreAttnBucketer(make_config())
        second = SpyreAttnBucketer(make_config())
        runner = self._runner([first], [second])
        assert runner._resolve_builder_attn_bucketer() is first

    def test_diverging_builders_raise(self):
        first = SpyreAttnBucketer(make_config(max_model_len=2048))
        second = SpyreAttnBucketer(make_config(max_model_len=8192))
        runner = self._runner([first], [second])
        with pytest.raises(AssertionError, match="diverge between metadata builders"):
            runner._resolve_builder_attn_bucketer()

    def test_skips_builders_without_a_bucketer(self):
        bucketer = SpyreAttnBucketer(make_config())
        runner = self._runner([None, bucketer])
        assert runner._resolve_builder_attn_bucketer() is bucketer
