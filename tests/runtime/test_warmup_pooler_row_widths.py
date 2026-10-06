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

"""The pooler row sweep covers the width the poolers round up to, not just the limit."""

from __future__ import annotations

import types

import pytest
import torch

from spyre_inference.v1.worker import spyre_model_runner
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

ROWS = 128
HIDDEN = 8


def _swept_shapes(
    monkeypatch, max_num_seqs: int, len_ladder: list[int] | None = None
) -> list[tuple[int, int]]:
    runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
    runner._pooling_on_spyre = True
    runner.scheduler_config = types.SimpleNamespace(max_num_seqs=max_num_seqs)
    runner._encoder_len_ladder = len_ladder or []

    shapes: list[tuple[int, int]] = []

    def record(hidden_states, row_indices):
        shapes.append((hidden_states.shape[0], int(row_indices.numel())))
        return torch.zeros(row_indices.numel(), HIDDEN, dtype=torch.float16)

    monkeypatch.setattr(
        spyre_model_runner,
        "select_rows",
        record,
    )
    TorchSpyreModelRunner._warm_pooler_row_widths(
        runner, torch.zeros(ROWS, HIDDEN, dtype=torch.float16)
    )
    return shapes


@pytest.mark.parametrize(
    ("max_num_seqs", "expected"),
    [
        pytest.param(6, [1, 2, 4, 8], id="six_rounds_up_to_eight"),
        pytest.param(24, [1, 2, 4, 8, 16, 32], id="twenty_four_rounds_up_to_thirty_two"),
        pytest.param(4, [1, 2, 4], id="a_power_of_two_is_unchanged"),
        pytest.param(1, [1], id="one_sequence"),
    ],
)
def test_sweep_reaches_the_rounded_up_width(monkeypatch, max_num_seqs, expected):
    shapes = set(_swept_shapes(monkeypatch, max_num_seqs))
    assert {(ROWS, width) for width in expected} <= shapes


def test_sweep_never_exceeds_the_available_rows(monkeypatch):
    """A body smaller than the rounded-up width has nothing to gather from."""
    assert max(width for _, width in _swept_shapes(monkeypatch, ROWS * 4)) <= ROWS


def test_sweep_covers_token_lengths_from_the_fixed_body(monkeypatch):
    shapes = set(_swept_shapes(monkeypatch, max_num_seqs=2, len_ladder=[64, 128]))

    assert (ROWS, 1) in shapes
    assert (ROWS, 2) in shapes
    assert (ROWS, 64) in shapes
    assert (ROWS, 128) in shapes


def test_sweep_rejects_a_token_length_larger_than_the_fixed_body(monkeypatch):
    with pytest.raises(AssertionError, match="pooler gather width exceeds"):
        _swept_shapes(monkeypatch, max_num_seqs=2, len_ladder=[64, ROWS * 2])
