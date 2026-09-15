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

On the device, ``exp()`` of a large negative fp16 never returns zero -- it saturates at
``2**-24``, the smallest fp16 subnormal. So softmax keeps a weight on every additively
masked position and that position's V reaches the attention output, including the unused
tail slots of a partially-filled KV block that vLLM later hands to another request.

PR #873 works around it by zeroing those slots
(``SpyreAttentionImpl._clear_new_kv_block_tails``). When this probe XPASSes the backend
underflows correctly: **revert PR #873** -- the workaround, its unit test in
``tests/attention/test_spyre_attn.py`` and the e2e guard
``tests/e2e/test_kv_cache_determinism.py`` -- and delete this probe.

Follows tdoublep's minimal reproducer (torch-spyre#4517, comment 5669447964): eager ops
only, no paged attention and no model, so the probe tracks the arithmetic that is wrong
rather than one kernel's use of it.
"""

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

pytestmark = pytest.mark.probe

_DTYPE = torch.float16
_MASK = torch.finfo(_DTYPE).min
_REASON = (
    "torch-spyre#4517: the device's fp16 exp() saturates at 2**-24 instead of underflowing "
    "to zero, so additively masked positions keep a softmax weight and leak their V into the "
    "attention output. When this passes, revert PR #873 "
    "(SpyreAttentionImpl._clear_new_kv_block_tails and its tests) and delete this probe."
)


@pytest.fixture(scope="module", autouse=True)
def _require_spyre() -> None:
    if not spyre_available():
        pytest.skip("Spyre device not available")


@pytest.mark.xfail(strict=True, reason=_REASON)
@pytest.mark.parametrize("exponent", [-18.0, -1000.0, _MASK])
def test_fp16_exp_underflows_to_zero(exponent: float) -> None:
    """In fp16 ``exp(x)`` is exactly zero below about -17.33; the device returns 2**-24."""
    got = torch.full((32, 64), exponent, dtype=_DTYPE).to("spyre").exp().to("cpu")
    bits = int(got.view(torch.int16)[0, 0]) & 0xFFFF
    assert bits == 0x0000, f"spyre exp({exponent:g}) = {float(got[0, 0]):g} (bits 0x{bits:04x})"


@pytest.mark.xfail(strict=True, reason=_REASON)
def test_masked_positions_do_not_reach_the_output() -> None:
    """The consequence for attention, with only the masked slots holding a value."""
    v = torch.zeros(128, 64, dtype=_DTYPE)
    v[32:] = 1000.0  # nonzero only where the mask should have zeroed the weight
    mask = torch.zeros(32, 128, dtype=_DTYPE)
    mask[:, 32:] = _MASK
    leak = float((torch.exp(mask.to("spyre")) @ v.to("spyre")).to("cpu").abs().max())
    assert leak == 0.0, f"masked V reached the output: expected 0, got {leak:g}"
