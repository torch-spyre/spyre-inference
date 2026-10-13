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

"""Test select_rows row-gather correctness, including the Spyre view/_base
handling that lets it avoid a full-tensor clone before index_select."""

import sys

import pytest
import torch

from spyre_inference.v1.pool.spyre_pooler import pad_row_count_to_bucket

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize(
    "n_rows,expected",
    [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (7, 8), (8, 8), (9, 16), (33, 64)],
)
def test_pad_row_count_rounds_to_a_power_of_two(n_rows: int, expected: int) -> None:
    """The pooler's row gather specializes on the exact index length, and serving
    pools one row per request -- any count up to max_num_seqs. Rounding to a power
    of two caps the reachable widths at a handful that warmup can sweep, instead of
    compiling mid-request the first time each new request count appears.
    """
    idx = torch.arange(n_rows, dtype=torch.int64)
    padded, real = pad_row_count_to_bucket(idx)

    assert real == n_rows, "the real row count must survive for the caller's trim"
    assert padded.numel() == expected
    assert torch.equal(padded[:n_rows], idx), "real rows must be untouched"
    if n_rows:
        # Pads duplicate the last real row, so a duplicate-index gather is harmless.
        assert bool((padded[n_rows:] == idx[-1]).all())
