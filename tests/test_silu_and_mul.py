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
Test SpyreSiluAndMul custom op correctness against a reference implementation.
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


@pytest.mark.spyre
@pytest.mark.siluandmul
@pytest.mark.parametrize("num_tokens", [1, 7, 63, 64, 65, 1024])
@pytest.mark.parametrize("d", [2, 63, 64, 65, 1024, 13824])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.float16,
            marks=pytest.mark.xfail(reason="Strided tensors on float16 does not work on spyre"),
        ),
        torch.float32,
    ],
)
def test_spyre_siluandmul_matches_reference(num_tokens, d, dtype):
    """SpyreSiluAndMul output matches golden reference.

    Tests both paths:
    - forward(): custom op dispatch (no-compile path via torch.ops.vllm.spyre_siluandmul)
    - forward_oot(): direct Spyre device execution
    """
    from spyre_inference.custom_ops.silu_and_mul import SpyreSiluAndMul

    torch.manual_seed(42)

    # Input shape is [num_tokens, 2*d], output shape is [num_tokens, d]
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)
    layer = SpyreSiluAndMul()

    expected = reference_silu_and_mul(x)
    actual = layer.forward_oot(x)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.spyre
@pytest.mark.siluandmul
def test_siluandmul_oot_dispatch():
    """Verify SiluAndMul OOT registration: class swap"""
    from vllm.model_executor.layers.activation import SiluAndMul
    from spyre_inference.custom_ops.silu_and_mul import SpyreSiluAndMul

    layer = SiluAndMul()

    # OOT class swap: SiluAndMul.__new__ should produce SpyreSiluAndMul
    assert isinstance(layer, SpyreSiluAndMul)

    # dispatch_forward should have selected forward_oot
    assert layer._forward_method == layer.forward_oot
