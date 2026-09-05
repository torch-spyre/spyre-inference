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
Test SpyreGeluAndMul custom op correctness against a reference implementation.

SpyreGeluAndMul mirrors SpyreSiluAndMul but uses F.gelu (GeGLU) for Gemma MLPs.
Two code paths:
  1. Pre-split pair (from unfuse.py) — x is an iterable (gate, up).
  2. Fused tensor [..., 2*d] — sliced on CPU to avoid Spyre memory corruption.
"""

import pytest
import torch
import torch.nn.functional as F


def reference_gelu_and_mul(x: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
    """Golden reference: standard GeluAndMul (GeGLU) in PyTorch.

    Computes: gelu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2
    """
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return F.gelu(x1, approximate=approximate) * x2


@pytest.mark.geluandmul
@pytest.mark.parametrize("num_tokens", [1, 7, 63, 64, 65, 1024])
@pytest.mark.parametrize("d", [2, 63, 64, 65, 1024, 13824])
def test_spyre_geluandmul_fused_matches_reference(num_tokens, d):
    """SpyreGeluAndMul.forward_oot on a fused Spyre tensor matches the CPU reference."""
    from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

    torch.manual_seed(42)

    # Input shape is [num_tokens, 2*d], output shape is [num_tokens, d]
    x = torch.randn(num_tokens, 2 * d, dtype=torch.float16)
    layer = SpyreGeluAndMul()

    expected = reference_gelu_and_mul(x, approximate=layer.approximate)
    actual = layer.forward_oot(x.to("spyre"))

    torch.testing.assert_close(actual.cpu(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.geluandmul
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 1024])
@pytest.mark.parametrize("d", [64, 128, 1024])
def test_spyre_geluandmul_presplit_matches_reference(num_tokens, d):
    """SpyreGeluAndMul.forward_oot on a pre-split (gate, up) pair matches reference."""
    from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

    torch.manual_seed(42)

    # Pre-split pair: gate and up are separate tensors already on device
    gate = torch.randn(num_tokens, d, dtype=torch.float16)
    up = torch.randn(num_tokens, d, dtype=torch.float16)
    layer = SpyreGeluAndMul()

    # Reference: gelu(gate) * up
    expected = F.gelu(gate, approximate=layer.approximate) * up

    # Pass as a tuple (simulating SplitSiluAndMul-style pre-split pair)
    actual = layer.forward_oot((gate.to("spyre"), up.to("spyre")))

    torch.testing.assert_close(actual.cpu(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.geluandmul
def test_spyre_geluandmul_presplit_list_input():
    """SpyreGeluAndMul.forward_oot accepts a list [gate, up] as non-Tensor input."""
    from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

    torch.manual_seed(42)
    d = 64
    gate = torch.randn(4, d, dtype=torch.float16)
    up = torch.randn(4, d, dtype=torch.float16)
    layer = SpyreGeluAndMul()

    expected = F.gelu(gate, approximate=layer.approximate) * up
    actual = layer.forward_oot([gate.to("spyre"), up.to("spyre")])

    torch.testing.assert_close(actual.cpu(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.geluandmul
def test_geluandmul_oot_dispatch():
    """Verify GeluAndMul OOT registration: class swap and forward_oot routing."""
    from vllm.model_executor.layers.activation import GeluAndMul
    from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

    layer = GeluAndMul()

    # OOT class swap: GeluAndMul.__new__ should produce SpyreGeluAndMul
    assert isinstance(layer, SpyreGeluAndMul)

    # dispatch_forward should have selected forward_oot
    assert layer._forward_method == layer.forward_oot


@pytest.mark.geluandmul
def test_geluandmul_output_shape():
    """SpyreGeluAndMul halves the last dimension (fused path)."""
    from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

    torch.manual_seed(42)
    layer = SpyreGeluAndMul()
    x = torch.randn(8, 256, dtype=torch.float16)
    out = layer.forward_oot(x.to("spyre"))
    assert out.shape == (8, 128)
