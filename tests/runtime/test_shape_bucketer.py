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

"""Unit tests for SpyreShapeBucketer."""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from spyre_inference.v1.worker.spyre_shape_bucketer import (
    EncoderBucketDescriptor,
    SpyreShapeBucketer,
    batch_buckets,
    default_encoder_len_buckets,
    encoder_batch_bucket,
    encoder_bucket_valid_row_indices,
    encoder_len_bucket,
    expand_packed_to_encoder_bucket,
    len_buckets,
    next_bucket,
    pooling_warmup_shapes,
)


@pytest.fixture()
def mock_vllm_config():
    """Create a minimal VllmConfig mock with compile_sizes."""
    config = MagicMock()
    config.compilation_config.compile_sizes = [1, 2, 4, 8, 16]
    return config


@pytest.fixture()
def bucketer(mock_vllm_config):
    return SpyreShapeBucketer(mock_vllm_config)


class TestFindBucket:
    def test_exact_match(self, bucketer):
        assert bucketer.find_bucket(8) == 8

    def test_rounds_up_to_next_bucket(self, bucketer):
        assert bucketer.find_bucket(3) == 4
        assert bucketer.find_bucket(5) == 8
        assert bucketer.find_bucket(9) == 16

    def test_smallest_token_count(self, bucketer):
        assert bucketer.find_bucket(1) == 1

    def test_exceeds_max_returns_none(self, bucketer):
        assert bucketer.find_bucket(17) is None
        assert bucketer.find_bucket(100) is None

    def test_zero_tokens(self, bucketer):
        assert bucketer.find_bucket(0) == 1


class TestDispatch:
    def test_returns_descriptor_with_padding(self, bucketer):
        desc = bucketer.dispatch(5)
        assert desc is not None
        assert desc.actual_num_tokens == 5
        assert desc.padded_num_tokens == 8

    def test_exact_match_no_padding(self, bucketer):
        desc = bucketer.dispatch(4)
        assert desc is not None
        assert desc.actual_num_tokens == 4
        assert desc.padded_num_tokens == 4

    def test_exceeds_max_returns_none(self, bucketer):
        assert bucketer.dispatch(20) is None

    def test_descriptor_is_frozen(self, bucketer):
        desc = bucketer.dispatch(3)
        with pytest.raises(FrozenInstanceError):
            desc.actual_num_tokens = 10


class TestBucketerState:
    def test_initial_state_not_warmed_up(self, bucketer):
        assert not bucketer.is_warmed_up

    def test_mark_warmed_up(self, bucketer):
        bucketer.mark_warmed_up()
        assert bucketer.is_warmed_up

    def test_bucket_sizes_sorted(self, bucketer):
        assert bucketer.bucket_sizes == [1, 2, 4, 8, 16]

    def test_max_bucket_size(self, bucketer):
        assert bucketer.max_bucket_size == 16


class TestEdgeCases:
    def test_empty_compile_sizes(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = []
        b = SpyreShapeBucketer(config)
        assert b.bucket_sizes == []
        assert b.max_bucket_size == 0
        assert b.find_bucket(1) is None
        assert b.dispatch(1) is None

    def test_single_bucket(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = [8]
        b = SpyreShapeBucketer(config)
        assert b.find_bucket(1) == 8
        assert b.find_bucket(8) == 8
        assert b.find_bucket(9) is None

    def test_unsorted_input_gets_sorted(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = [16, 2, 8, 1, 4]
        b = SpyreShapeBucketer(config)
        assert b.bucket_sizes == [1, 2, 4, 8, 16]


def _pooling_vllm_config(
    *,
    max_num_seqs: int = 4,
    max_model_len: int = 128,
    max_num_batched_tokens: int = 512,
    runner_type: str = "pooling",
) -> MagicMock:
    config = MagicMock()
    config.model_config.runner_type = runner_type
    config.model_config.max_model_len = max_model_len
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    config.compilation_config.compile_sizes = []
    return config


class TestEncoderDispatch:
    def test_for_pooling_loads_warmup_shapes(self):
        cfg = _pooling_vllm_config()
        cfg.compilation_config.compile_sizes = [64, 128]
        b = SpyreShapeBucketer.for_pooling(cfg)
        assert b is not None
        assert b.encoder_shapes == [
            (1, 64),
            (1, 128),
            (2, 64),
            (2, 128),
            (4, 64),
            (4, 128),
        ]
        assert b.bucket_sizes == [64, 128]

    def test_for_pooling_skips_non_pooling(self):
        assert SpyreShapeBucketer.for_pooling(_pooling_vllm_config(runner_type="generate")) is None

    def test_for_pooling_none_when_no_shapes(self):
        cfg = _pooling_vllm_config(max_model_len=64, max_num_seqs=1, max_num_batched_tokens=32)
        cfg.compilation_config.compile_sizes = []
        assert SpyreShapeBucketer.for_pooling(cfg) is None

    def test_for_pooling_1d_only_when_no_attention_shapes(self):
        cfg = _pooling_vllm_config(max_model_len=64, max_num_seqs=1, max_num_batched_tokens=32)
        cfg.compilation_config.compile_sizes = [256]
        b = SpyreShapeBucketer.for_pooling(cfg)
        assert b is not None
        assert b.encoder_shapes == []
        assert b.bucket_sizes == [256]

    def test_dispatch_encoder_pads_to_warmed_cell(self):
        cfg = _pooling_vllm_config()
        cfg.compilation_config.compile_sizes = [64, 128]
        b = SpyreShapeBucketer.for_pooling(cfg)
        assert b is not None
        desc = b.dispatch_encoder(
            num_seqs=3,
            max_query_len=30,
            max_num_seqs=4,
            max_model_len=128,
            max_num_batched_tokens=512,
        )
        assert desc is not None
        assert (desc.batch_bucket, desc.len_bucket) == (4, 64)
        assert desc.padded_num_tokens == 256
        assert desc.actual_num_seqs == 3
        assert desc.actual_max_len == 30

    def test_pooling_2d_not_1d_token_ladder(self):
        """Body 1D pad and attention (B, L) are independent.

        3 seqs × 30 tokens is 90 packed tokens. Body picks T=128; SDPA
        still gathers to (4, 64).
        """
        cfg = _pooling_vllm_config()
        cfg.compilation_config.compile_sizes = [64, 128]
        b = SpyreShapeBucketer.for_pooling(cfg)
        assert b is not None
        one_d = b.dispatch(90)
        assert one_d is not None
        assert one_d.padded_num_tokens == 128
        two_d = b.dispatch_encoder(
            num_seqs=3,
            max_query_len=30,
            max_num_seqs=4,
            max_model_len=128,
            max_num_batched_tokens=512,
        )
        assert two_d is not None
        assert (two_d.batch_bucket, two_d.len_bucket) == (4, 64)
        assert two_d.padded_num_tokens == 256

    def test_dispatch_encoder_stays_on_warmed_shapes(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = []
        b = SpyreShapeBucketer(config, encoder_shapes=[(4, 64)])
        desc = b.dispatch_encoder(
            num_seqs=1,
            max_query_len=30,
            max_num_seqs=4,
            max_model_len=128,
            max_num_batched_tokens=512,
        )
        assert desc is not None
        assert (desc.batch_bucket, desc.len_bucket) == (4, 64)

    def test_dispatch_encoder_none_when_over_token_budget(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = []
        b = SpyreShapeBucketer(config, encoder_shapes=[(4, 64)])
        assert (
            b.dispatch_encoder(
                num_seqs=3,
                max_query_len=30,
                max_num_seqs=4,
                max_model_len=2048,
                max_num_batched_tokens=200,
            )
            is None
        )

    def test_dispatch_encoder_none_on_1d_bucketer(self, bucketer):
        assert (
            bucketer.dispatch_encoder(
                num_seqs=1,
                max_query_len=8,
                max_num_seqs=4,
                max_model_len=128,
                max_num_batched_tokens=512,
            )
            is None
        )

    def test_encoder_descriptor_is_frozen(self):
        desc = EncoderBucketDescriptor(
            batch_bucket=4, len_bucket=64, actual_num_seqs=3, actual_max_len=30
        )
        with pytest.raises(FrozenInstanceError):
            desc.batch_bucket = 1


class TestEncoderBuckets:
    def test_next_bucket_picks_smallest_fit(self):
        assert next_bucket(30, [64, 128, 256]) == 64
        assert next_bucket(64, [64, 128, 256]) == 64
        assert next_bucket(65, [64, 128, 256]) == 128

    def test_next_bucket_overflow_stick_aligns(self):
        assert next_bucket(3000, [64, 128]) == 3008  # 3000 → 47*64 = 3008

    def test_len_bucket_stick_aligns_when_ladder_unset(self):
        assert encoder_len_bucket(1) == 64
        assert encoder_len_bucket(32) == 64
        assert encoder_len_bucket(65) == 128

    def test_len_buckets_from_max_model_len(self):
        assert len_buckets(512) == [64, 128, 256, 512]
        assert default_encoder_len_buckets(2048) == [64, 128, 256, 512, 1024, 2048]
        assert len_buckets(100) == [64]
        assert len_buckets(768) == [64, 128, 256, 512, 768]

    def test_len_buckets_prefer_compile_sizes(self):
        assert len_buckets(512, compile_sizes=[128, 512]) == [128, 512]
        assert encoder_len_bucket(30, [128, 512]) == 128
        assert encoder_len_bucket(200, [128, 512]) == 512

    def test_len_buckets_stick_align_compile_sizes(self):
        assert len_buckets(512, compile_sizes=[100, 200]) == [128, 256]

    def test_default_batch_buckets_are_powers_of_two(self):
        assert batch_buckets(4) == [1, 2, 4]
        assert batch_buckets(3) == [1, 2, 3]

    def test_batch_bucket_pads_to_next_power(self):
        assert encoder_batch_bucket(1, 4) == 1
        assert encoder_batch_bucket(3, 4) == 4
        assert encoder_batch_bucket(4, 4) == 4

    def test_warmup_shapes_use_max_model_len(self):
        assert pooling_warmup_shapes(
            max_num_seqs=4,
            max_model_len=128,
            max_num_batched_tokens=512,
        ) == [
            (1, 64),
            (1, 128),
            (2, 64),
            (2, 128),
            (4, 64),
            (4, 128),
        ]

    def test_warmup_shapes_skip_over_token_budget(self):
        # 4*256 = 1024 and 2*256 = 512 both exceed 300; 4*64 = 256 still fits.
        assert pooling_warmup_shapes(
            max_num_seqs=4,
            max_model_len=2048,
            max_num_batched_tokens=300,
            len_ladder=[64, 256],
        ) == [(1, 64), (1, 256), (2, 64), (4, 64)]

    def test_expand_packed_to_encoder_bucket_pads_seq_and_batch(self):
        padded_ids, padded_pos = expand_packed_to_encoder_bucket(
            input_ids=[1, 2, 3, 4, 5],
            positions=[0, 1, 2, 0, 1],
            query_lens=[3, 2],
            batch_bucket=4,
            len_bucket=4,
            pad_token_id=9,
        )
        # seq0, seq1, then two dummy rows
        assert padded_ids == [1, 2, 3, 9, 4, 5, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
        assert padded_pos == list(range(4)) * 4

    def test_encoder_bucket_valid_row_indices_skips_pads(self):
        indices = encoder_bucket_valid_row_indices([3, 2], len_bucket=4)
        assert indices == [0, 1, 2, 4, 5]
