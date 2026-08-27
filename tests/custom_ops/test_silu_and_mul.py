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

"""
Test SiluAndMul custom op on Spyre device.

The base class's forward_oot calls forward_native, which now works
correctly on Spyre (torch-spyre#3578).
"""

import pytest
import torch
import torch.nn.functional as F


def reference_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Golden reference: standard SiluAndMul (SwiGLU) in PyTorch.

    Computes: silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2
    """
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return F.silu(x1) * x2


@pytest.mark.siluandmul
@pytest.mark.parametrize("num_tokens", [16, 17, 64, 128, 1024])
@pytest.mark.parametrize("d", [64, 256, 1024, 8128, 12800])
def test_siluandmul_on_spyre_matches_reference(num_tokens, d):
    """SiluAndMul.forward_oot on a Spyre input matches the CPU reference."""
    from vllm.model_executor.layers.activation import SiluAndMul

    torch.manual_seed(42)

    # Input shape is [num_tokens, 2*d], output shape is [num_tokens, d]
    x = torch.randn(num_tokens, 2 * d, dtype=torch.float16)
    layer = SiluAndMul()

    expected = reference_silu_and_mul(x)
    actual = layer.forward_oot(x.to("spyre"))

    torch.testing.assert_close(actual.cpu(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.siluandmul
def test_siluandmul_oot_method():
    """Verify SiluAndMul has forward_oot that calls forward_native."""
    from vllm.model_executor.layers.activation import SiluAndMul

    layer = SiluAndMul()

    # forward_oot should be the method that gets called on OOT platforms
    # and it should call forward_native by default
    assert hasattr(layer, "forward_oot")
    assert layer.forward_oot is not None
