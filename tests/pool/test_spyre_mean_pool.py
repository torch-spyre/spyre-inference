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

"""Spyre MEAN pooling: packed D2H + host float32 reduction.

Host-only arithmetic lives in ``test_spyre_pooler_config.py``. The
``index_add_`` primitive probe stays in ``tests/probes/test_spyre_fallback_probes.py``.
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


def _mean_cursor(lens: torch.Tensor):
    class _Cursor:
        prompt_lens_cpu = lens

        def is_partial_prefill(self) -> bool:
            return False

    class _Meta:
        def get_pooling_cursor(self):
            return _Cursor()

    return _Meta()


def test_spyre_mean_pool_packed_d2h_matches_cpu_fp32(spyre_device):
    """SpyreMeanPool: one packed fp16 D2H + host fp32 mean.

    Hits ``convert`` of the packed ``[T, H]``, not a per-sequence copy.
    """
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


def test_spyre_mean_pool_one_gather_when_encoder_padded(spyre_device):
    """One ``index_select`` of the valid prefix, then one D2H — not per seq."""
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


def test_spyre_mean_pool_fp32_sum_after_packed_d2h(spyre_device):
    """fp16 activations must still use a float32 accumulator (2048+1 rounds in fp16).

    Sum runs on the host after one packed fp16 D2H.
    """
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
