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

"""Spyre MEAN pooling: per-lane fp32 sum on a rectangle, else packed D2H then ``MeanPool``.

Destagger of a device fp32 sum is garbage; see
``test_spyre_fp32_reduce_d2h_with_destagger``.
"""

from __future__ import annotations

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.v1.pool.spyre_pooler import SpyreMeanPool


@pytest.fixture()
def spyre_device():
    if not spyre_available():
        pytest.skip("Spyre device not available")
    return torch.device("spyre")


def _mean_cursor(lens: torch.Tensor, grid_extent: int | None = None):
    class _Cursor:
        prompt_lens_cpu = lens
        spyre_grid_extent = grid_extent

        def is_partial_prefill(self) -> bool:
            return False

    class _Meta:
        def get_pooling_cursor(self):
            return _Cursor()

    return _Meta()


def test_spyre_mean_pool_crops_trailing_pad_on_host():
    """Pad crop is a host ``index_select``; CPU CI must still cover it.

    Device arithmetic sits behind ``spyre_device``. This is the crop itself:
    trailing pad, empty batch, and a zero-length sequence (both of the last
    two gather with ``arange(0)``).
    """
    hidden = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [99.0, 99.0],
        ],
        dtype=torch.float16,
    )
    out = SpyreMeanPool().forward(hidden, _mean_cursor(torch.tensor([2], dtype=torch.int64)))
    expected = hidden[:2].to(torch.float32).mean(0, keepdim=True)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)

    empty = SpyreMeanPool().forward(hidden, _mean_cursor(torch.tensor([], dtype=torch.int64)))
    assert empty.shape == (0, 2) and empty.dtype == torch.float32

    zero = SpyreMeanPool().forward(hidden, _mean_cursor(torch.tensor([0], dtype=torch.int64)))
    assert zero.shape == (1, 2) and zero.dtype == torch.float32
    assert bool(torch.isnan(zero).all())


def test_spyre_mean_pool_varlen_matches_cpu_fp32(spyre_device):
    """Two sequences of lengths 3 and 2 match a host fp32 mean."""
    hidden_cpu = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        dtype=torch.float16,
    )
    hidden = hidden_cpu.to(spyre_device)
    out = SpyreMeanPool().forward(hidden, _mean_cursor(torch.tensor([3, 2], dtype=torch.int64)))
    expected = torch.stack(
        [
            hidden_cpu[:3].to(torch.float32).mean(0),
            hidden_cpu[3:].to(torch.float32).mean(0),
        ]
    )
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


def test_spyre_mean_pool_ignores_trailing_pad(spyre_device):
    """Pad rows past ``sum(lens)`` must not enter the mean."""
    hidden_cpu = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [99.0, 99.0],
            [99.0, 99.0],
        ],
        dtype=torch.float16,
    )
    hidden = hidden_cpu.to(spyre_device)
    out = SpyreMeanPool().forward(hidden, _mean_cursor(torch.tensor([2, 1], dtype=torch.int64)))
    expected = torch.stack(
        [
            hidden_cpu[:2].to(torch.float32).mean(0),
            hidden_cpu[2:3].to(torch.float32).mean(0),
        ]
    )
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


def test_spyre_mean_pool_accumulates_in_float32(spyre_device):
    """Packed D2H then ``MeanPool`` must keep the fp32 accumulator (2048+1 rounds in fp16)."""
    num_small = 512
    hidden_cpu = torch.cat(
        [
            torch.full((1, 2), 2048.0, dtype=torch.float16),
            torch.ones((num_small, 2), dtype=torch.float16),
        ]
    )
    hidden = hidden_cpu.to(spyre_device)
    out = SpyreMeanPool().forward(
        hidden, _mean_cursor(torch.tensor([num_small + 1], dtype=torch.int64))
    )
    expected = hidden_cpu.to(torch.float32).mean(0, keepdim=True)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)
    assert out[0, 0].item() > 4.9


@pytest.mark.parametrize(
    "lanes,lens",
    [
        pytest.param(4, [3, 64, 10], id="pad-lane-past-the-real-count"),
        pytest.param(3, [5, 1, 64], id="grid-narrower-than-the-bucket"),
    ],
)
def test_spyre_mean_pool_grid_matches_cpu_fp32(spyre_device, lanes, lens):
    """On the rectangular path each lane is one sequence, reduced in place.

    Lanes past the real count, and rows past each length inside a lane, must not
    enter any mean. Three sequences round up to a bucket of four, which a 3-lane
    grid cannot hold -- the reduce then stops at the grid's width.
    """
    extent, hidden_size = 64, 64
    rows = torch.arange(lanes * extent, dtype=torch.float32).unsqueeze(1)
    cols = torch.arange(hidden_size, dtype=torch.float32).unsqueeze(0)
    hidden_cpu = ((rows % 13) + (cols % 5)).to(torch.float16)
    meta = _mean_cursor(torch.tensor(lens, dtype=torch.int64), grid_extent=extent)
    out = SpyreMeanPool().forward(hidden_cpu.to(spyre_device), meta)
    expected = torch.stack(
        [
            hidden_cpu[i * extent : i * extent + n].to(torch.float32).mean(0)
            for i, n in enumerate(lens)
        ]
    )
    assert out.dtype == torch.float16
    assert out.shape[0] == len(lens)
    torch.testing.assert_close(out.cpu().to(torch.float32), expected, atol=1e-2, rtol=1e-3)
