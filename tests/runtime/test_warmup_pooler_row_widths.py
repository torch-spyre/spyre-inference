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

import pytest
import torch

from spyre_inference.v1.pool import warmup
from spyre_inference.v1.pool.warmup import warm_pooler_row_widths

ROWS = 64
HIDDEN = 8


def _swept_widths(monkeypatch, max_num_seqs: int) -> list[int]:
    widths: list[int] = []
    monkeypatch.setattr(
        warmup,
        "select_rows",
        lambda hidden_states, row_indices: widths.append(int(row_indices.numel())),
    )
    # No classifier/embed head, so the real-width sweep (which needs one to warm)
    # stays a no-op and only the plain gather below runs.
    warm_pooler_row_widths(
        torch.nn.Module(), torch.zeros(ROWS, HIDDEN, dtype=torch.float16), [], max_num_seqs
    )
    return widths


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
    assert _swept_widths(monkeypatch, max_num_seqs) == expected


def test_sweep_never_exceeds_the_available_rows(monkeypatch):
    """A body smaller than the rounded-up width has nothing to gather from."""
    assert max(_swept_widths(monkeypatch, ROWS * 4)) <= ROWS
