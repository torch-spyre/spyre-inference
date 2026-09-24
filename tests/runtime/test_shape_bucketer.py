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
    SpyreShapeBucketer,
    encoder_budget_rows,
    encoder_group_shapes,
    encoder_group_width_caps,
    encoder_len_ladder,
    encoder_rectangle_for_batch,
    encoder_rectangles,
    encoder_shape_tables,
    encoder_width_for,
    logits_row_buckets,
    next_bucket,
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


class TestForPooling:
    """Pooling gets the same 1D bucketer as decode, with a single entry."""

    def test_skips_non_pooling(self):
        assert SpyreShapeBucketer.for_pooling(_pooling_vllm_config(runner_type="generate")) is None

    def test_none_when_no_compile_sizes(self):
        cfg = _pooling_vllm_config()
        cfg.compilation_config.compile_sizes = []
        assert SpyreShapeBucketer.for_pooling(cfg) is None

    def test_uses_1d_compile_sizes(self):
        cfg = _pooling_vllm_config()
        cfg.compilation_config.compile_sizes = [64, 128]
        b = SpyreShapeBucketer.for_pooling(cfg)
        assert b is not None
        assert b.bucket_sizes == [64, 128]


class TestNextBucket:
    """Still shared with the pooler's own bucketed-row trimming (``spyre_pooler.py``)."""

    def test_next_bucket_picks_smallest_fit(self):
        assert next_bucket(30, [64, 128, 256]) == 64
        assert next_bucket(64, [64, 128, 256]) == 64
        assert next_bucket(65, [64, 128, 256]) == 128

    def test_next_bucket_overflow_stick_aligns(self):
        assert next_bucket(3000, [64, 128]) == 3008  # 3000 → 47*64 = 3008


class TestEncoderLenLadder:
    def test_powers_of_two_up_to_max_model_len(self):
        cfg = _pooling_vllm_config(max_model_len=512, max_num_batched_tokens=2048)
        assert encoder_len_ladder(cfg) == [64, 128, 256, 512]

    def test_top_entry_is_the_power_of_two_stick_count_above_max_model_len(self):
        """Rounded *up*: an extent is a matmul dimension, and rounding down would
        leave the longest requests with no covering bucket. To a power-of-two stick
        count, not just a whole stick, so every entry divides the budget."""
        cfg = _pooling_vllm_config(max_model_len=500, max_num_batched_tokens=2048)
        assert encoder_len_ladder(cfg) == [64, 128, 256, 512]
        # 320 is a whole number of sticks (64*5) but not a power-of-two count.
        cfg = _pooling_vllm_config(max_model_len=320, max_num_batched_tokens=2048)
        assert encoder_len_ladder(cfg) == [64, 128, 256, 512]

    def test_override_entries_are_power_of_two_stick_counts(self, monkeypatch):
        """Same rounding as the derived ladder, or the documented knob would put the
        non-dividing entry back."""
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "320")
        cfg = _pooling_vllm_config(max_model_len=512, max_num_batched_tokens=2048)
        assert encoder_len_ladder(cfg) == [512]

    def test_short_model_gets_one_bucket(self):
        cfg = _pooling_vllm_config(max_model_len=100, max_num_batched_tokens=2048)
        assert encoder_len_ladder(cfg) == [64, 128]

    def test_query_bucket_override_is_clamped_and_appended(self, monkeypatch):
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "100,4096")
        cfg = _pooling_vllm_config(max_model_len=512, max_num_batched_tokens=2048)
        # 100 aligns up to 128; 4096 is unreachable and dropped; 512 is appended.
        assert encoder_len_ladder(cfg) == [128, 512]


class TestEncoderBudget:
    def test_floored_at_the_top_of_the_ladder(self):
        """Encoder prefill cannot be chunked, so a budget under max_model_len would
        head-of-line block forever -- and no rectangle would hold one sequence."""
        assert encoder_budget_rows(512, 256, 32) == 512
        assert encoder_budget_rows(500, 256, 32) == 512
        assert encoder_budget_rows(512, 2048, 32) == 2048

    def test_floored_to_a_multiple_of_the_longest_length(self):
        """Every declared length divides the budget, so no rectangle truncates. An
        awkward --max-num-batched-tokens costs rows, not correctness."""
        assert encoder_budget_rows(512, 1000, 32) == 512  # 1000 -> 1 * 512
        assert encoder_budget_rows(512, 1600, 32) == 1536  # 1600 -> 3 * 512
        assert encoder_budget_rows(320, 2048, 32) == 2048  # longest 512, 2048 = 4 * 512

    def test_capped_at_what_max_num_seqs_can_carry(self):
        """A rectangle is always the whole buffer, so a narrow engine would otherwise
        run the body on budget rows for one short request."""
        assert encoder_budget_rows(64, 2048, 1) == 64
        assert encoder_budget_rows(512, 2048, 2) == 1024
        assert encoder_budget_rows(512, 2048, 4) == 2048


class TestEncoderRectangles:
    def test_worked_example(self):
        cfg = _pooling_vllm_config(max_model_len=512, max_num_seqs=32, max_num_batched_tokens=2048)
        assert encoder_rectangles(cfg) == [(64, 32), (128, 16), (256, 8), (512, 4)]

    @pytest.mark.parametrize("max_num_seqs", [1, 4, 32])
    @pytest.mark.parametrize("max_model_len", [64, 100, 320, 384, 512, 576, 1024, 4096])
    @pytest.mark.parametrize("max_num_batched_tokens", [512, 1000, 2048, 8192])
    def test_every_rectangle_is_exactly_the_budget(
        self, max_num_seqs, max_model_len, max_num_batched_tokens
    ):
        """The single body shape depends on it: a rectangle reinterprets the body buffer
        with ``view``, so a ``budget // length`` that truncates would not cover it.

        Parametrised past the power-of-two cases on purpose. A ``max_model_len`` that is
        not a power-of-two stick count, or a ``max_num_batched_tokens`` that is not a
        multiple of the longest length, both used to leave rectangles short of the body.
        """
        cfg = _pooling_vllm_config(
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        budget = encoder_shape_tables(cfg).budget
        assert {length * batch for length, batch in encoder_rectangles(cfg)} == {budget}

    @pytest.mark.parametrize("max_model_len", [64, 100, 320, 384, 512, 576, 1024])
    def test_ladder_declares_every_extent_a_request_can_take(self, max_model_len):
        """``_alignment_units_for`` rounds a request's extent to a power-of-two stick
        count, so the ladder has to contain exactly those -- an extent it omits is a
        group width of 1 that warmup never traced, and one it adds is warmed for nothing.
        """
        from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
            ENCODER_LEN_ALIGNMENT,
            _alignment_units_for,
        )

        cfg = _pooling_vllm_config(max_model_len=max_model_len, max_num_batched_tokens=2048)
        ladder = set(encoder_len_ladder(cfg))
        assignable = {
            _alignment_units_for(n) * ENCODER_LEN_ALIGNMENT for n in range(1, max_model_len + 1)
        }
        assert assignable <= ladder, f"undeclared extents: {sorted(assignable - ladder)}"

    def test_width_for_is_capped_by_max_num_seqs(self):
        cfg = _pooling_vllm_config(max_model_len=512, max_num_seqs=4, max_num_batched_tokens=2048)
        assert encoder_width_for(64, cfg) == 4  # 2048 // 64 = 32, capped
        assert encoder_width_for(512, cfg) == 4


class TestEncoderGroupShapes:
    def test_worked_example_is_eighteen_pairs(self):
        cfg = _pooling_vllm_config(max_model_len=512, max_num_seqs=32, max_num_batched_tokens=2048)
        groups = encoder_group_shapes(cfg)
        assert len(groups) == 18
        assert encoder_group_width_caps(cfg) == {64: 32, 128: 16, 256: 8, 512: 4}
        # Powers of two only, never wider than the cap.
        for width, extent in groups:
            assert width & (width - 1) == 0
            assert width <= encoder_width_for(extent, cfg)

    def test_empty_when_the_rectangle_cannot_miss(self):
        """``max_num_seqs`` at or below ``R // longest length`` means every batch fits a
        rectangle, so the group family is unreachable and warming it is pure cost."""
        cfg = _pooling_vllm_config(max_model_len=512, max_num_seqs=4, max_num_batched_tokens=2048)
        assert encoder_group_shapes(cfg) == []

    def test_declared_shape_count_matches_the_plan(self):
        cfg = _pooling_vllm_config(max_model_len=512, max_num_seqs=32, max_num_batched_tokens=2048)
        assert 1 + len(encoder_rectangles(cfg)) + len(encoder_group_shapes(cfg)) == 23


class TestEncoderDispatch:
    @pytest.fixture()
    def rectangles(self):
        return encoder_rectangles(
            _pooling_vllm_config(max_model_len=512, max_num_seqs=32, max_num_batched_tokens=2048)
        )

    @pytest.mark.parametrize(
        ("num_seqs", "max_len", "expected"),
        [
            (1, 10, (64, 32)),
            (32, 64, (64, 32)),
            (16, 128, (128, 16)),
            (8, 256, (256, 8)),
            (4, 512, (512, 4)),
            # One too wide for the length it needs: ragged path, not an error.
            (33, 64, None),
            (17, 128, None),
            (9, 256, None),
            (5, 512, None),
            # The plan's go/no-go case: ~100-token prompts fill to num_seqs 20 while
            # B(128) is 16.
            (20, 100, None),
        ],
    )
    def test_selection(self, rectangles, num_seqs, max_len, expected):
        assert encoder_rectangle_for_batch(num_seqs, max_len, rectangles) == expected

    def test_empty_batch_returns_none(self, rectangles):
        assert encoder_rectangle_for_batch(0, 0, rectangles) is None

    def test_no_rectangles_is_the_ragged_path(self):
        assert encoder_rectangle_for_batch(1, 64, []) is None


class TestLogitsRowBuckets:
    def test_clips_prefill_bucket_to_max_num_reqs(self):
        # The 512-token prefill bucket samples at most max_num_seqs rows.
        assert logits_row_buckets([1, 2, 4, 8, 512], max_num_reqs=8) == [1, 2, 4, 8]

    def test_keeps_a_non_power_of_two_max(self):
        assert logits_row_buckets([1, 2, 4, 6, 512], max_num_reqs=6) == [1, 2, 4, 6]

    def test_ignores_non_positive_sizes(self):
        assert logits_row_buckets([0, -1, 4], max_num_reqs=8) == [4]
