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

"""Strict-xfail probe for torch-spyre#4517: fp16 ``exp()`` floors instead of underflowing.

``exp()`` of a large negative fp16 saturates at ``2**-24`` on the device instead of
returning zero, so softmax keeps a weight on every additively masked position and that
position's V reaches the attention output. ``tests/e2e/test_kv_cache_determinism.py``
measures the consequence and xfails for the same reason.

Minimal reproducer from torch-spyre#4517 (comment 5669447964).
"""

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

pytestmark = pytest.mark.probe

_DTYPE = torch.float16
_MASK = torch.finfo(_DTYPE).min
_REASON = (
    "torch-spyre#4517: the device's fp16 exp() saturates at 2**-24 instead of underflowing "
    "to zero, so additively masked positions leak their V into the attention output. When "
    "this passes, drop the xfails in tests/e2e/test_kv_cache_determinism.py and delete this "
    "probe."
)


@pytest.fixture(scope="module", autouse=True)
def _require_spyre() -> None:
    if not spyre_available():
        pytest.skip("Spyre device not available")


@pytest.mark.xfail(strict=True, reason=_REASON)
@pytest.mark.parametrize("exponent", [-18.0, -1000.0, _MASK])
def test_fp16_exp_underflows_to_zero(exponent: float) -> None:
    """In fp16 ``exp(x)`` is exactly zero below about -17.33."""
    got = torch.full((32, 64), exponent, dtype=_DTYPE).to("spyre").exp().to("cpu")
    # Every lane, so a partially correct lowering cannot pass on element 0.
    nonzero = int((got != 0).sum())
    assert nonzero == 0, (
        f"spyre exp({exponent:g}) = {float(got.abs().max()):g} in {nonzero}/{got.numel()} lanes"
    )


@pytest.mark.xfail(strict=True, reason=_REASON)
def test_masked_positions_do_not_reach_the_output() -> None:
    """The consequence for attention, with only the masked slots holding a value."""
    v = torch.zeros(128, 64, dtype=_DTYPE)
    v[32:] = 1000.0  # nonzero only where the mask should have zeroed the weight
    mask = torch.zeros(32, 128, dtype=_DTYPE)
    mask[:, 32:] = _MASK
    leak = float((torch.exp(mask.to("spyre")) @ v.to("spyre")).to("cpu").abs().max())
    assert leak == 0.0, f"masked V reached the output: expected 0, got {leak:g}"
