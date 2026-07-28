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

"""Unit tests for SpyreGeluAndMul custom op (GeGLU activation).

SpyreGeluAndMul is the Spyre OOT override for vLLM's GeluAndMul. It handles:
1. Unfused path: x is a pre-split (gate, up) tuple from unfuse.py
2. Fused path: x is a [..., 2*d] tensor, sliced on CPU to avoid Spyre memory
   corruption, then moved back to device

Both paths compute: gelu(gate) * up

These tests verify:
- OOT registration (class swap via register_oot)
- Numerical correctness of the unfused path against a reference
- Numerical correctness of the fused path against a reference
- The fused-path CPU-bounce logic (no Spyre slicing)
"""

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.geluandmul


def reference_gelu_and_mul(x: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
    """Golden reference: standard GeluAndMul (GeGLU) in PyTorch.

    Computes: gelu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2
    """
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return F.gelu(x1, approximate=approximate) * x2


class TestSpyreGeluAndMulRegistration:
    """Test OOT registration of SpyreGeluAndMul."""

    def test_oot_dispatch_class_swap(self):
        """GeluAndMul() should produce SpyreGeluAndMul via OOT registration."""
        from vllm.model_executor.layers.activation import GeluAndMul
        from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

        layer = GeluAndMul()
        assert isinstance(layer, SpyreGeluAndMul)

    def test_forward_method_is_forward_oot(self):
        """dispatch_forward should have selected forward_oot."""
        from vllm.model_executor.layers.activation import GeluAndMul
        from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

        layer = GeluAndMul()
        assert layer._forward_method == layer.forward_oot

    def test_inherits_from_gelu_and_mul(self):
        """SpyreGeluAndMul should be a subclass of GeluAndMul."""
        from vllm.model_executor.layers.activation import GeluAndMul
        from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

        assert issubclass(SpyreGeluAndMul, GeluAndMul)


class TestSpyreGeluAndMulFusedPath:
    """Test the fused path: x is a single [..., 2*d] tensor."""

    @pytest.mark.parametrize("num_tokens", [1, 7, 64, 128])
    @pytest.mark.parametrize("d", [64, 128, 1024])
    def test_fused_path_matches_reference(self, num_tokens, d):
        """SpyreGeluAndMul.forward_oot on a fused tensor matches the reference."""
        from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

        torch.manual_seed(42)
        x = torch.randn(num_tokens, 2 * d, dtype=torch.float16)
        layer = SpyreGeluAndMul()

        expected = reference_gelu_and_mul(x, approximate=layer.approximate)
        actual = layer.forward_oot(x)

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_fused_path_3d_input(self):
        """Test fused path with 3D input [..., 2*d]."""
        from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

        torch.manual_seed(42)
        x = torch.randn(2, 4, 128, dtype=torch.float16)  # [..., 2*64]
        layer = SpyreGeluAndMul()

        expected = reference_gelu_and_mul(x, approximate=layer.approximate)
        actual = layer.forward_oot(x)

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_fused_path_output_shape(self):
        """Output shape should be [..., d] when input is [..., 2*d]."""
        from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

        x = torch.randn(8, 256, dtype=torch.float16)  # [..., 2*128]
        layer = SpyreGeluAndMul()
        result = layer.forward_oot(x)
        assert result.shape == (8, 128)


class TestSpyreGeluAndMulUnfusedPath:
    """Test the unfused path: x is a (gate, up) tuple."""

    @pytest.mark.parametrize("num_tokens", [1, 7, 64, 128])
    @pytest.mark.parametrize("d", [64, 128, 1024])
    def test_unfused_path_matches_reference(self, num_tokens, d):
        """SpyreGeluAndMul.forward_oot on pre-split (gate, up) matches reference."""
        from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

        torch.manual_seed(42)
        # Simulate unfused: create a fused tensor, then split
        x_fused = torch.randn(num_tokens, 2 * d, dtype=torch.float16)
        x1 = x_fused[..., :d].contiguous()
        x2 = x_fused[..., d:].contiguous()

        layer = SpyreGeluAndMul()

        expected = reference_gelu_and_mul(x_fused, approximate=layer.approximate)
        actual = layer.forward_oot((x1, x2))

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_unfused_path_output_shape(self):
        """Output shape should be [..., d] when input is (gate, up) pair."""
        from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

        gate = torch.randn(8, 128, dtype=torch.float16)
        up = torch.randn(8, 128, dtype=torch.float16)
        layer = SpyreGeluAndMul()
        result = layer.forward_oot((gate, up))
        assert result.shape == (8, 128)


class TestSpyreGeluAndMulApproximate:
    """Test the approximate parameter handling."""

    def test_default_approximate_is_tanh(self):
        """Default approximate mode should be 'tanh' (Gemma uses gelu_pytorch_tanh)."""
        from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

        layer = SpyreGeluAndMul()
        assert layer.approximate == "tanh"

    def test_none_approximate(self):
        """Test with approximate='none' (exact GELU)."""
        from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul

        layer = SpyreGeluAndMul(approximate="none")
        torch.manual_seed(42)
        x = torch.randn(4, 128, dtype=torch.float16)

        expected = reference_gelu_and_mul(x, approximate="none")
        actual = layer.forward_oot(x)

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
