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

from spyre_inference.multimodal import utils
from spyre_inference.multimodal.utils import padded_sdpa


# CLIP ViT-B/32's 50 patches and SigLIP's 729, the stick-coprime lengths in use, and
# an aligned 64, which needs no mask at all. Batch 3 is a multi-image step: the
# key-pad row is [1, 1, 1, seq_pad] and must broadcast.
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("seq", [50, 64, 729])
@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_padded_sdpa_without_mask_matches_cpu_on_spyre(batch, seq, mode):
    """`padded_sdpa(..., None)` on card must equal the same call on CPU.

    The key-pad row is built on the host because writing `-inf` into the tail of a
    stick is a sub-stick offset write, which has no offset-free alternative stick dim.
    Doing it on device instead fails to lower rather than returning wrong values.

    The `compile` leg is what the next tower to opt into per-block compile needs: the
    build must run once, at trace time, and never land in the graph.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    q, k, v = (torch.randn(batch, 2, seq, 64, dtype=torch.float16) for _ in range(3))
    expected = padded_sdpa(q, k, v, None)

    device = torch.device("spyre")
    q, k, v = (x.to(device) for x in (q, k, v))

    fn = padded_sdpa
    if mode == "compile":
        torch._dynamo.reset()
        utils._KEY_PAD_MASKS.clear()
        fn = torch.compile(padded_sdpa, dynamic=False, backend="inductor")

    actual = fn(q, k, v, None)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)

    if mode == "compile":
        if seq % 64 == 0:
            assert not utils._KEY_PAD_MASKS  # nothing to mask, nothing built
        else:
            (row,) = utils._KEY_PAD_MASKS.values()  # built once, at trace time
            fn(q, k, v, None)
            assert next(iter(utils._KEY_PAD_MASKS.values())) is row  # graph input, not rebuilt


def test_padded_sdpa_refuses_a_padded_mask_at_another_dtype():
    """A mask already padded to `seq_pad` keys is used as is, so a dtype miss has to say
    so instead of reaching `_padded_attn_mask`, which would reshape it to `[seq, seq]`."""
    q, k, v = (torch.randn(2, 2, 50, 64, dtype=torch.float16) for _ in range(3))
    mask = torch.zeros(2, 1, 1, 64, dtype=torch.float32)

    with pytest.raises(TypeError, match="float32"):
        padded_sdpa(q, k, v, mask)


def test_padded_sdpa_refuses_an_unpadded_float_mask():
    """Every caller passes a bool keep-mask or an additive mask already padded to the
    stick; an unpadded float mask is the ambiguous case and has to say so."""
    q, k, v = (torch.randn(2, 2, 50, 64, dtype=torch.float16) for _ in range(3))

    with pytest.raises(TypeError, match="bool keep-mask"):
        padded_sdpa(q, k, v, torch.zeros(50, 50, dtype=torch.float16))
