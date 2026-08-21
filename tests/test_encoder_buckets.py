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

"""CPU tests for encoder compile-shape buckets. No Spyre device required."""

from spyre_inference.v1.encoder_buckets import (
    batch_buckets,
    encoder_batch_bucket,
    encoder_bucket_valid_row_indices,
    encoder_len_bucket,
    expand_packed_to_encoder_bucket,
    len_buckets,
    next_bucket,
    pooling_warmup_pad_query_lens,
    pooling_warmup_shapes,
    runtime_encoder_bucket,
)


def test_next_bucket_picks_smallest_fit():
    assert next_bucket(30, [64, 128, 256]) == 64
    assert next_bucket(64, [64, 128, 256]) == 64
    assert next_bucket(65, [64, 128, 256]) == 128


def test_next_bucket_overflow_stick_aligns():
    assert next_bucket(3000, [64, 128]) == 3008  # 3000 → 47*64 = 3008


def test_default_len_bucket_reuses_64_for_short_prompts():
    assert encoder_len_bucket(1) == 64
    assert encoder_len_bucket(32) == 64
    assert encoder_len_bucket(65) == 128


def test_custom_len_buckets(monkeypatch):
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_LENS", "128,512")
    assert encoder_len_bucket(30) == 128
    assert encoder_len_bucket(200) == 512
    assert len_buckets() == [128, 512]


def test_len_buckets_stick_align(monkeypatch):
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_LENS", "100,200")
    assert len_buckets() == [128, 256]


def test_default_batch_buckets_are_powers_of_two():
    assert batch_buckets(4) == [1, 2, 4]
    assert batch_buckets(3) == [1, 2, 3]


def test_batch_bucket_pads_to_next_power():
    assert encoder_batch_bucket(1, 4) == 1
    assert encoder_batch_bucket(3, 4) == 4
    assert encoder_batch_bucket(4, 4) == 4


def test_custom_batch_buckets(monkeypatch):
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_BATCH_SIZES", "1,4")
    assert encoder_batch_bucket(2, 4) == 4
    assert batch_buckets(4) == [1, 4]


def test_warmup_pad_query_lens_include_random_dataset_specials():
    assert pooling_warmup_pad_query_lens(64) == [62, 63]
    assert pooling_warmup_pad_query_lens(2) == [1]
    assert pooling_warmup_pad_query_lens(1) == []


def test_warmup_shapes_are_bucket_cartesian(monkeypatch):
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_LENS", "64,128")
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_BATCH_SIZES", "1,4")
    assert pooling_warmup_shapes(
        max_num_seqs=4,
        max_model_len=128,
        max_num_batched_tokens=512,
    ) == [(1, 64), (1, 128), (4, 64), (4, 128)]


def test_warmup_shapes_skip_over_token_budget(monkeypatch):
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_LENS", "64,256")
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_BATCH_SIZES", "4")
    # 4*256 = 1024 > 300 tokens; 4*64 = 256 still fits.
    assert pooling_warmup_shapes(
        max_num_seqs=4,
        max_model_len=2048,
        max_num_batched_tokens=300,
    ) == [(4, 64)]


def test_runtime_bucket_matches_warmup_shape(monkeypatch):
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_LENS", "64,128")
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_BATCH_SIZES", "1,4")
    # 3×30 → pad to the warmed (4, 64) so T = 256.
    assert runtime_encoder_bucket(
        num_seqs=3,
        max_query_len=30,
        max_num_seqs=4,
        max_model_len=128,
        max_num_batched_tokens=512,
    ) == (4, 64)


def test_runtime_bucket_none_when_over_token_budget(monkeypatch):
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_LENS", "64")
    monkeypatch.setenv("SPYRE_ENCODER_BUCKET_BATCH_SIZES", "4")
    assert (
        runtime_encoder_bucket(
            num_seqs=3,
            max_query_len=30,
            max_num_seqs=4,
            max_model_len=2048,
            max_num_batched_tokens=200,
        )
        is None
    )


def test_expand_packed_to_encoder_bucket_pads_seq_and_batch():
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


def test_encoder_bucket_valid_row_indices_skips_pads():
    indices = encoder_bucket_valid_row_indices([3, 2], len_bucket=4)
    assert indices == [0, 1, 2, 4, 5]
