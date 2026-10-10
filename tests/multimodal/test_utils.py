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

"""Tests for `spyre_inference/multimodal/utils.py`.

The callers cover the padding algebra on CPU; what they cannot cover is why the no-mask
path exists at all, so the on-card case below is the one that matters.
"""

from __future__ import annotations

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.multimodal.utils import padded_sdpa

_COMPILE_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "`_key_pad_mask` is `functools.lru_cache`-wrapped and Dynamo ignores the cache "
        "wrapper, tracing the body instead (pytorch/pytorch#152994). Warming it first "
        "therefore does not keep the build out of the graph, and its "
        "`[..., seq:] = finfo.min / 2` tail write lands there as an offset sub-stick "
        "write with no offset-free alternative stick dim -- `_find_alt_target_stl` on a "
        "`FixedLayout('cpu', torch.float16, size=[1, 1, 1, 64])`. A plain dict lookup, "
        "which Dynamo folds to the cached tensor, is what flips this. No tower compiles "
        "the no-mask path today (gemma 4 passes a real mask), so this is a tripwire "
        "rather than a live break; raised on #1198."
    ),
)


# CLIP ViT-B/32's 50 patches and SigLIP's 729, the stick-coprime lengths in use.
@pytest.mark.parametrize("seq", [50, 729])
@pytest.mark.parametrize("mode", ["eager", pytest.param("compile", marks=_COMPILE_XFAIL)])
def test_padded_sdpa_without_mask_matches_cpu_on_spyre(seq, mode):
    """`padded_sdpa(..., None)` on card must equal the same call on CPU.

    `_key_pad_mask` is built on the host because writing `-inf` into the tail of a
    stick is a sub-stick offset write, which has no offset-free alternative stick dim.
    Doing it on device instead fails to lower rather than returning wrong values.

    The `compile` leg asks whether a compiled block may reach that cache at all, which
    is what the next tower to opt into per-block compile needs. It is xfail today; see
    the marker.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    q, k, v = (torch.randn(1, 2, seq, 64, dtype=torch.float16) for _ in range(3))
    expected = padded_sdpa(q, k, v, None)

    device = torch.device("spyre")
    q, k, v = (x.to(device) for x in (q, k, v))

    fn = padded_sdpa
    if mode == "compile":
        padded_sdpa(q, k, v, None)  # warm the mask cache outside the graph
        torch._dynamo.reset()
        fn = torch.compile(padded_sdpa, dynamic=False, backend="inductor")

    actual = fn(q, k, v, None)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)
