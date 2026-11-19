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
from spyre_inference.v1.attention.spyre_attn_bucketer import (
    SpyreAttnBucketer,
    _parse_buckets,
)

BLOCK_SIZE = 64


def make_config(
    max_model_len=2048,
    max_num_batched_tokens=512,
    block_size=BLOCK_SIZE,
    max_num_seqs=8,
    runner_type="generate",
    num_kv_heads=8,
    per_layer_kv_heads=None,
):
    config = MagicMock()
    config.cache_config.block_size = block_size
    config.model_config.max_model_len = max_model_len
    config.model_config.runner_type = runner_type
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    config.scheduler_config.max_num_seqs = max_num_seqs

    arch = config.model_config.model_arch_config
    if per_layer_kv_heads is None:
        arch.per_layer_overrides = None
        config.model_config.get_num_kv_heads.return_value = num_kv_heads
    else:
        # A heterogeneous config, whose model-wide number hides the smallest layer.
        arch.per_layer_overrides = [{} for _ in per_layer_kv_heads]
        layers = []
        for n in per_layer_kv_heads:
            layer = MagicMock()
            layer.kv = n
            layers.append(layer)
        arch.__getitem__.side_effect = layers.__getitem__
        config.model_config.get_num_kv_heads.side_effect = lambda _pc, arch_config=None: (
            num_kv_heads if arch_config is None else arch_config.kv
        )
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


@pytest.fixture()
def restore_ts_limits():
    """Restores the torch-spyre limits a declaring bucketer sets. Not monkeypatch: its
    undo deletes a config-module entry, hiding it for later tests."""
    from torch_spyre._inductor import config as ts_config

    with ts_config.patch(min_reserved_dim_size=8, max_buckets=32):
        yield


class TestQueryBuckets:
    def test_query_buckets_lead_with_decode_case(self, bucketer):
        assert bucketer.query_buckets[0] == 1
        assert bucketer.query_buckets == [1, 512]

    def test_query_buckets_are_multiples_of_the_step(self):
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=2048))
        assert b.query_buckets == [1, 512, 1024, 1536, 2048]

    def test_query_bucket_step_capped_by_max_num_batched_tokens(self):
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=300))
        assert b.query_buckets == [1, 300]

    def test_largest_bucket_is_always_the_limit(self):
        for limit in (1, 2, 3, 64, 100, 4096, 32768):
            b = SpyreAttnBucketer(make_config(max_model_len=limit, max_num_batched_tokens=limit))
            assert b.query_buckets[-1] == limit

    def test_buckets_have_no_duplicates(self):
        for limit in (1, 2, 3, 64, 100, 512, 513, 4096, 32768):
            b = SpyreAttnBucketer(make_config(max_model_len=limit, max_num_batched_tokens=limit))
            assert b.query_buckets == sorted(set(b.query_buckets))


class TestPoolingQueryBucketCap:
    """Pooling's query_len can't exceed max_model_len; without this cap, warmup
    could record a query bucket no sequence reaches (CLIP's text tower:
    max_model_len=77, max_num_batched_tokens much larger).

    Stick-aligned the same way the encoder shape ladder pads a request
    (_align_up_pow2): CLIP's max_model_len=77 pads to 128, not 77."""

    def test_pooling_caps_query_buckets_at_stick_aligned_max_model_len(self):
        b = SpyreAttnBucketer(
            make_config(max_model_len=77, max_num_batched_tokens=2048, runner_type="pooling")
        )
        assert b.query_buckets[-1] == 128
        # The largest query bucket is recorded: this is what crashed for CLIP.
        assert 128 in {v.padded_query_len for v in b.variants()}

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

    def test_exceeds_max_returns_none(self, bucketer):
        assert bucketer.find_query_bucket(513) is None

    def test_without_a_page_range_every_count_pads_to_the_context(self, bucketer):
        assert bucketer.page_range is None
        assert bucketer.max_pages == 2048 // BLOCK_SIZE
        assert {bucketer.find_pages(n) for n in (1, 2, 17, 32)} == {32}

    def test_a_count_past_the_context_is_out_of_contract(self, bucketer):
        with pytest.raises(AssertionError, match="exceeds"):
            bucketer.find_pages(33)

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
        queries = [v.padded_query_len for v in bucketer.variants()]
        assert queries == sorted(queries, reverse=True)

    def test_descriptor_is_frozen(self, bucketer):
        with pytest.raises(FrozenInstanceError):
            bucketer.variants()[0].num_blocks = 10

    def test_one_variant_per_query_bucket_at_the_context(self, bucketer):
        variants = bucketer.variants()
        assert {v.num_blocks for v in variants} == {bucketer.max_pages}
        assert [v.padded_query_len for v in variants] == sorted(
            bucketer.query_buckets, reverse=True
        )

    def test_query_buckets_stay_narrower_than_the_query_buffer(self):
        """``SpyreAttentionImpl`` gathers unconditionally, which faults if a
        sequence is the whole buffer, so every bucket must fit with a row spare."""
        max_batched = 512
        bucketer = SpyreAttnBucketer(make_config(max_num_batched_tokens=max_batched))
        for bucket in bucketer.query_buckets:
            assert bucket < max_batched + 1

    def test_prunes_query_buckets_no_context_can_hold(self):
        """Pruning bounds the *smallest real* query_len that reaches a bucket, not
        the bucket itself: a 2-token query on a 1-block sequence still legitimately
        pads to 512."""
        b = SpyreAttnBucketer(make_config(max_model_len=1024, max_num_batched_tokens=4096))
        assert [v.padded_query_len for v in b.variants()] == [1024, 512, 1]
        for v in b.variants():
            assert b.min_real_query_len(v.padded_query_len) <= v.num_blocks * BLOCK_SIZE

    @pytest.mark.parametrize("kv_len", [1, 256, 257, 2048])
    @pytest.mark.parametrize("query_len", [1, 32, 33, 512])
    def test_every_rounded_size_lands_on_a_recorded_variant(self, bucketer, kv_len, query_len):
        """The whole point of recording: no runtime batch may miss the cache.

        Drives the two lookups production uses, find_query_bucket and find_pages
        (what _pad_num_blocks calls)."""
        if query_len > kv_len:
            pytest.skip("a sequence cannot have more new tokens than total KV")
        padded_query_len = bucketer.find_query_bucket(query_len)
        assert padded_query_len is not None
        num_blocks = bucketer.find_pages((kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
        sizes = {(v.num_blocks, v.padded_query_len) for v in bucketer.variants()}
        assert (num_blocks, padded_query_len) in sizes

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

    def test_parse_buckets_rejects_non_positive(self):
        with pytest.raises(ValueError):
            _parse_buckets("0,32")

    def test_parse_buckets_empty_is_none(self):
        assert _parse_buckets("") is None
        assert _parse_buckets(None) is None


class TestRecorderBuilders:
    """The recorder records through the builder each layer dispatches against."""

    @staticmethod
    def _builder():
        """A stand-in that still satisfies the runner's isinstance check."""
        from spyre_inference.v1.attention.backends.spyre_attn import (
            SpyreAttentionMetadataBuilder,
        )

        return MagicMock(spec=SpyreAttentionMetadataBuilder)

    @staticmethod
    def _runner(*groups):
        """A bare runner with one ``(layer_names, builders)`` argument per attention
        group; ``builders`` is that group's per-ubatch list."""
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
        runner.attn_groups = [
            [
                SimpleNamespace(layer_names=list(layer_names), metadata_builders=list(builders))
                for layer_names, builders in groups
            ]
        ]
        return runner

    def test_every_layer_maps_to_its_own_groups_builder(self):
        builder = self._builder()
        runner = self._runner((["layers.0.self_attn", "layers.1.self_attn"], [builder]))
        assert runner._attn_metadata_builders() == {
            "layers.0.self_attn": builder,
            "layers.1.self_attn": builder,
        }

    def test_groups_keep_their_own_builders(self):
        """Groups exist because their KV specs differ (block size, sliding window),
        so one group's builder must not stand in for another's layers."""
        first, second = self._builder(), self._builder()
        runner = self._runner((["layers.0.self_attn"], [first]), (["layers.1.self_attn"], [second]))
        builders = runner._attn_metadata_builders()
        assert builders["layers.0.self_attn"] is first
        assert builders["layers.1.self_attn"] is second

    def test_first_ubatch_builder_stands_for_the_group(self):
        """Ubatch builders share a spec, so either records the same variants."""
        first, second = self._builder(), self._builder()
        runner = self._runner((["layers.0.self_attn"], [first, second]))
        assert runner._attn_metadata_builders() == {"layers.0.self_attn": first}

    def test_skips_groups_without_a_spyre_builder(self):
        """A foreign or empty group's layers are left out, so the recorder logs them
        rather than recording against a builder that cannot enumerate variants."""
        builder = self._builder()
        runner = self._runner(
            (["encoder.0.attn"], [MagicMock()]),
            (["layers.0.self_attn"], [builder]),
            (["layers.1.self_attn"], []),
        )
        assert runner._attn_metadata_builders() == {"layers.0.self_attn": builder}

    def test_batched_decode_dispatches_onto_a_recorded_key(
        self, monkeypatch, default_vllm_config, restore_ts_limits
    ):
        """The regression this guards: ``build()`` and warmup must agree."""
        from spyre_testing_plugin.attn_helpers import _padded_mask_metadata
        from vllm.config import get_current_vllm_config

        monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
        envs.clear_env_cache()

        # block_size pinned to match what _padded_mask_metadata builds with.
        vllm_config = get_current_vllm_config()
        vllm_config.cache_config.block_size = BLOCK_SIZE
        bucketer = SpyreAttnBucketer(vllm_config, declare_pages=True)
        keys = {
            (v.num_seqs, v.blocks_per_chunk, v.num_chunks)
            for v in bucketer.batched_decode_variants()
        }

        # 4 blocks of real KV.
        metadata = _padded_mask_metadata(
            [(1, 4 * BLOCK_SIZE)] * 4, max_num_blocks=bucketer.max_pages
        )

        assert metadata.padded_num_seqs in bucketer.num_seqs_buckets
        assert metadata.blocks_per_chunk is not None and metadata.padded_batch_blocks is not None
        num_chunks = metadata.padded_batch_blocks // metadata.blocks_per_chunk
        assert (metadata.padded_num_seqs, metadata.blocks_per_chunk, num_chunks) in keys


class TestBatchedDecodeVariants:
    """The batched decode enumeration, keyed on (num_seqs, blocks_per_chunk, num_chunks)."""

    @pytest.fixture()
    def enabled(self, monkeypatch):
        monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
        envs.clear_env_cache()
        return SpyreAttnBucketer(make_config())

    def test_empty_when_the_path_is_disabled(self, monkeypatch):
        monkeypatch.setenv("SPYRE_BATCHED_DECODE", "0")
        envs.clear_env_cache()
        assert SpyreAttnBucketer(make_config()).batched_decode_variants() == []

    def test_one_chunk_and_the_longest_context_per_batch_size(self, enabled):
        """Without a page range, a call past one chunk pads to the context's chunk count."""
        for num_seqs in enabled.num_seqs_buckets:
            width = enabled.chunk_width(num_seqs)
            got = {
                (v.blocks_per_chunk, v.num_chunks)
                for v in enabled.batched_decode_variants()
                if v.num_seqs == num_seqs
            }
            assert got == {(width, 1), (width, -(-enabled.max_pages // width))}

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

    def test_chunks_cover_the_blocks(self, enabled):
        for v in enabled.batched_decode_variants():
            assert v.blocks_per_chunk == enabled.chunk_width(v.num_seqs)
            # The block axis pads up to a whole chunk, never truncates.
            assert v.blocks_per_chunk * v.num_chunks >= v.num_blocks

    def test_chunk_width_fills_the_cores_past_a_short_context(self):
        b = SpyreAttnBucketer(make_config(max_model_len=8 * BLOCK_SIZE))
        assert [b.chunk_width(n) for n in (1, 2, 4, 8)] == [32, 16, 8, 4]
        # Up to 4 sequences, every call fits in one chunk, recorded at the context.
        narrow = [v for v in b.batched_decode_variants() if v.num_seqs <= 4]
        assert {(v.num_chunks, v.num_blocks) for v in narrow} == {(1, 8)}

    def test_a_call_past_the_context_has_no_kernel(self, enabled):
        assert enabled.batched_decode_plan(1, enabled.max_pages + 1) is None

    def test_count_stays_tractable_at_long_context(self, monkeypatch):
        monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(32768, 2048, max_num_seqs=64))
        assert len(b.batched_decode_variants()) <= 2 * len(b.num_seqs_buckets)


class TestSingleKvHeadBlockFloor:
    """A lone KV head on a lone page will not compile, so a walk takes at least two pages
    -- see the note at the floor."""

    def test_a_short_context_still_walks_two_pages(self):
        """max_model_len <= block_size otherwise admits only a single-page walk, which
        is the shape both failing CI configs had."""
        b = SpyreAttnBucketer(make_config(max_model_len=BLOCK_SIZE, num_kv_heads=1))
        assert b.max_pages == 2
        assert {v.num_blocks for v in b.variants()} == {2}

    def test_a_single_kv_head_layer_floors_a_heterogeneous_model(self):
        """The gemma-4 shape: the model-wide count is 8, one layer type carries 1."""
        b = SpyreAttnBucketer(
            make_config(max_model_len=BLOCK_SIZE, num_kv_heads=8, per_layer_kv_heads=[8, 8, 1])
        )
        assert b.max_pages == 2

    def test_no_floor_when_every_layer_has_several_kv_heads(self):
        b = SpyreAttnBucketer(
            make_config(max_model_len=BLOCK_SIZE, num_kv_heads=8, per_layer_kv_heads=[8, 8, 4])
        )
        assert b.max_pages == 1

    def test_an_empty_override_list_falls_back_to_the_model_wide_count(self):
        """A config reporting no per-layer overrides is homogeneous, so the model-wide
        count decides; min() over no layers must not raise."""
        b = SpyreAttnBucketer(
            make_config(max_model_len=BLOCK_SIZE, num_kv_heads=1, per_layer_kv_heads=[])
        )
        assert b.max_pages == 2

    @pytest.mark.parametrize(("num_kv_heads", "num_chunks"), [(1, 2), (2, 1)])
    def test_a_one_block_chunk_takes_a_masked_second_one(self, num_kv_heads, num_chunks):
        b = SpyreAttnBucketer(
            make_config(max_model_len=2 * BLOCK_SIZE, max_num_seqs=32, num_kv_heads=num_kv_heads)
        )
        assert b.chunk_width(32) == 1
        plan = b.batched_decode_plan(32, 1)
        assert plan is not None and plan.num_chunks == num_chunks


@pytest.mark.usefixtures("restore_ts_limits")
class TestDeclaredPages:
    """The per-sequence kernels take a page range: one kernel per query bucket."""

    def test_page_range_spans_two_pages_to_max_model_len(self):
        b = SpyreAttnBucketer(make_config(max_model_len=2048), declare_pages=True)
        assert b.page_range == {"min": 2, "max": 2048 // BLOCK_SIZE, "granularity": 1}
        assert [b.find_pages(n) for n in (1, 2, 7, 32)] == [2, 2, 7, 32]

    def test_one_variant_per_query_bucket(self):
        b = SpyreAttnBucketer(make_config(max_model_len=2048), declare_pages=True)
        variants = b.variants()
        assert {v.num_blocks for v in variants} == {b.max_pages}
        assert [v.padded_query_len for v in variants] == sorted(b.query_buckets, reverse=True)

    def test_prefill_page_counts_are_multiples_of_eight_from_sixteen(self):
        from torch_spyre._inductor import config as ts_config

        b = SpyreAttnBucketer(make_config(max_model_len=31 * BLOCK_SIZE), declare_pages=True)
        assert b.page_range is not None
        assert b.page_range["max"] == 31
        assert b.prefill_page_range == {"min": 16, "max": 32, "granularity": 8}
        assert ts_config.max_buckets >= 32
        assert [b.find_pages(n, prefill=True) for n in (1, 8, 16, 17, 31)] == [
            16,
            16,
            16,
            24,
            32,
        ]

    def test_prefill_keeps_single_pages_above_the_exported_max_buckets(self, monkeypatch):
        from torch_spyre._inductor import config as ts_config

        monkeypatch.setenv("MAX_BUCKETS", "31")
        ts_config.max_buckets = 31
        b = SpyreAttnBucketer(make_config(max_model_len=31 * BLOCK_SIZE), declare_pages=True)
        assert b.page_range is not None
        assert b.prefill_page_range == b.page_range

    def test_an_exported_min_reserved_dim_size_raises_the_minimum_page_count(self, monkeypatch):
        monkeypatch.setenv("MIN_RESERVED_DIM_SIZE", "8")
        b = SpyreAttnBucketer(make_config(max_model_len=2048), declare_pages=True)
        assert b.page_range is not None
        assert b.page_range["min"] == 8
        assert b.find_pages(3) == 8

    def test_a_context_at_an_exported_min_reserved_dim_size_runs_at_one_count(self, monkeypatch):
        monkeypatch.setenv("MIN_RESERVED_DIM_SIZE", "8")
        b = SpyreAttnBucketer(make_config(max_model_len=8 * BLOCK_SIZE), declare_pages=True)
        assert b.page_range is None
        assert {v.num_blocks for v in b.variants()} == {8}
        assert b.find_pages(3) == 8

    def test_max_buckets_grows_to_cover_max_model_len(self):
        from torch_spyre._inductor import config as ts_config

        b = SpyreAttnBucketer(make_config(max_model_len=32768), declare_pages=True)
        assert b.page_range == {"min": 2, "max": 32768 // BLOCK_SIZE, "granularity": 1}
        assert ts_config.max_buckets == 32768 // BLOCK_SIZE

    def test_a_context_above_the_exported_max_buckets_is_refused(self, monkeypatch):
        monkeypatch.setenv("MAX_BUCKETS", "32")
        with pytest.raises(ValueError, match="MAX_BUCKETS=32 is below the 65 pages"):
            SpyreAttnBucketer(make_config(max_model_len=4096 + BLOCK_SIZE), declare_pages=True)

    def test_without_declare_pages_every_sequence_runs_at_the_context(self):
        b = SpyreAttnBucketer(make_config(max_model_len=2048))
        assert b.page_range is None
        assert b.find_pages(3) == b.max_pages == 2048 // BLOCK_SIZE


@pytest.mark.usefixtures("restore_ts_limits")
class TestSlidingWindowPages:
    """A sliding window bounds the pages a sequence walks, and with them the range."""

    def test_the_window_and_the_widest_query_bound_the_pages(self):
        # A 128 + 512 - 1 position span over 64-token pages, plus one for its offset.
        b = SpyreAttnBucketer(
            make_config(max_model_len=8192, max_num_batched_tokens=512),
            declare_pages=True,
            sliding_window=128,
        )
        assert b.max_pages == 11
        assert b.page_range == {"min": 2, "max": 11, "granularity": 1}
        # Rounded to multiples of 8 from 16, the prefill range would hold one count.
        assert b.prefill_page_range == b.page_range

    def test_a_window_wider_than_the_context_walks_the_context(self):
        b = SpyreAttnBucketer(
            make_config(max_model_len=2048), declare_pages=True, sliding_window=100_000
        )
        assert b.max_pages == 2048 // BLOCK_SIZE

    def test_variants_ask_for_the_context_their_query_needs(self):
        b = SpyreAttnBucketer(
            make_config(max_model_len=8192, max_num_batched_tokens=2048),
            declare_pages=True,
            sliding_window=128,
        )
        assert b.max_pages == 35
        for v in b.variants():
            assert b.min_real_query_len(v.padded_query_len) <= v.num_blocks * BLOCK_SIZE
            assert v.num_blocks <= 8192 // BLOCK_SIZE
        assert {v.num_blocks for v in b.variants()} == {35}


@pytest.mark.usefixtures("restore_ts_limits")
class TestDynamicChunks:
    """Batched decode takes a chunk range past one chunk."""

    def test_one_variant_past_one_chunk_covers_every_longer_context(self):
        b = SpyreAttnBucketer(make_config(max_model_len=8192), declare_pages=True)
        max_pages = 8192 // BLOCK_SIZE
        for num_seqs in b.num_seqs_buckets:
            width = 32 // num_seqs
            variants = [v for v in b.batched_decode_variants() if v.num_seqs == num_seqs]
            (dynamic,) = [v for v in variants if v.dynamic_chunks]
            assert (dynamic.blocks_per_chunk, dynamic.num_chunks) == (width, max_pages // width)
            assert all(v.num_blocks <= width for v in variants if not v.dynamic_chunks)
            for nb in (width + 1, 3 * width - 1, max_pages):
                plan = b.batched_decode_plan(num_seqs, nb)
                assert plan is not None
                assert plan.chunk_range == {"min": 2, "max": max_pages // width, "granularity": 1}
                assert plan.blocks_per_chunk == width
                assert (plan.num_chunks - 1) * width < nb <= plan.num_chunks * width
            plan = b.batched_decode_plan(num_seqs, width)
            assert plan is not None
            assert (plan.chunk_range, plan.num_chunks) == (None, 1)

    def test_one_chunk_kernel_serves_every_call_within_one_chunk(self):
        b = SpyreAttnBucketer(make_config(max_model_len=8192), declare_pages=True)
        for num_seqs in b.num_seqs_buckets:
            width = 32 // num_seqs
            variants = [v for v in b.batched_decode_variants() if v.num_seqs == num_seqs]
            assert [
                (v.blocks_per_chunk, v.num_chunks) for v in variants if not v.dynamic_chunks
            ] == [(width, 1)]
            for nb in range(1, width + 1):
                assert b.batched_decode_plan(num_seqs, nb) == (width, width, 1, None)

    def test_a_batch_size_whose_context_fits_two_chunks_pads_to_two(self):
        # 32 pages: one chunk at one sequence, two at two sequences.
        b = SpyreAttnBucketer(make_config(max_model_len=2048), declare_pages=True)

        def keys(num_seqs):
            return {
                (v.blocks_per_chunk, v.num_chunks, v.dynamic_chunks)
                for v in b.batched_decode_variants()
                if v.num_seqs == num_seqs
            }

        assert keys(1) == {(32, 1, False)}
        assert keys(2) == {(16, 1, False), (16, 2, False)}
        assert b.batched_decode_plan(2, 17) == (32, 16, 2, None)
        assert any(v.dynamic_chunks for v in b.batched_decode_variants() if v.num_seqs == 4)

    def test_without_a_page_range_a_longer_call_pads_to_the_context(self):
        b = SpyreAttnBucketer(make_config(max_model_len=8192))
        assert b.page_range is None
        assert not any(v.dynamic_chunks for v in b.batched_decode_variants())
        assert b.batched_decode_plan(4, 100) == (128, 8, 16, None)
