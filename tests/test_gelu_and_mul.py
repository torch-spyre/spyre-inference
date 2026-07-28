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

"""Unit tests for SpyreGeluAndMul (custom_ops/gelu_and_mul.py).

SpyreGeluAndMul overrides GeluAndMul.forward_oot to avoid slicing Spyre tensors
on-device (which corrupts memory). Two paths are tested:
  1. Unfused path: x is a pre-split (gate, up) tuple from unfuse.py
  2. Fused path: x is a [..., 2*d] tensor (sliced on CPU, bounced back)

All tests run on CPU — no Spyre device needed.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from spyre_inference.custom_ops.gelu_and_mul import SpyreGeluAndMul
from vllm.model_executor.layers.activation import GeluAndMul


@pytest.mark.gelu_and_mul
class TestSpyreGeluAndMulRegistration:
    """Test OOT registration of SpyreGeluAndMul."""

    def test_is_subclass_of_gelu_and_mul(self):
        """SpyreGeluAndMul inherits from GeluAndMul."""
        assert issubclass(SpyreGeluAndMul, GeluAndMul)

    def test_oot_dispatch(self):
        """GeluAndMul() instantiates SpyreGeluAndMul via OOT registration."""
        layer = GeluAndMul()
        assert isinstance(layer, SpyreGeluAndMul)

    def test_has_forward_oot_method(self):
        """SpyreGeluAndMul overrides forward_oot."""
        assert "forward_oot" in SpyreGeluAndMul.__dict__


@pytest.mark.gelu_and_mul
class TestSpyreGeluAndMulUnfusedPath:
    """Test the unfused path (x is a tuple of gate, up tensors)."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    @pytest.mark.parametrize("num_tokens", [1, 4, 16])
    def test_unfused_matches_reference(self, d, num_tokens):
        """Unfused path: gelu(gate) * up matches manual reference."""
        torch.manual_seed(42)
        gate = torch.randn(num_tokens, d, dtype=torch.float16)
        up = torch.randn(num_tokens, d, dtype=torch.float16)

        layer = GeluAndMul()
        result = layer.forward_oot((gate, up))

        expected = F.gelu(gate, approximate="tanh") * up
        assert result.shape == (num_tokens, d)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_unfused_output_dtype_preserved(self):
        """Output dtype matches input dtype."""
        gate = torch.randn(4, 64, dtype=torch.float16)
        up = torch.randn(4, 64, dtype=torch.float16)

        layer = GeluAndMul()
        result = layer.forward_oot((gate, up))
        assert result.dtype == torch.float16

    def test_unfused_output_device_preserved(self):
        """Output stays on the same device as inputs."""
        gate = torch.randn(4, 64, dtype=torch.float16)
        up = torch.randn(4, 64, dtype=torch.float16)

        layer = GeluAndMul()
        result = layer.forward_oot((gate, up))
        assert result.device.type == "cpu"


@pytest.mark.gelu_and_mul
class TestSpyreGeluAndMulFusedPath:
    """Test the fused path (x is a [..., 2*d] tensor)."""

    @pytest.mark.parametrize("d", [64, 128])
    @pytest.mark.parametrize("num_tokens", [1, 4, 8])
    def test_fused_matches_reference(self, d, num_tokens):
        """Fused path: split on CPU, gelu(gate) * up matches reference."""
        torch.manual_seed(0)
        x = torch.randn(num_tokens, 2 * d, dtype=torch.float16)

        layer = GeluAndMul()
        result = layer.forward_oot(x)

        # Reference: manual split + gelu + mul
        gate = x[..., :d]
        up = x[..., d:]
        expected = F.gelu(gate, approximate="tanh") * up

        assert result.shape == (num_tokens, d)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_fused_3d_input(self):
        """Fused path works with 3D [..., 2*d] tensors."""
        torch.manual_seed(0)
        d = 64
        x = torch.randn(2, 4, 2 * d, dtype=torch.float16)

        layer = GeluAndMul()
        result = layer.forward_oot(x)

        gate = x[..., :d]
        up = x[..., d:]
        expected = F.gelu(gate, approximate="tanh") * up

        assert result.shape == (2, 4, d)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_fused_preserves_dtype(self):
        """Fused path preserves float16 dtype."""
        x = torch.randn(4, 128, dtype=torch.float16)
        layer = GeluAndMul()
        result = layer.forward_oot(x)
        assert result.dtype == torch.float16


@pytest.mark.gelu_and_mul
class TestSpyreGeluAndMulApproximate:
    """Test that the GELU approximation matches the Gemma-style 'tanh' variant."""

    def test_uses_tanh_approximation(self):
        """SpyreGeluAndMul uses the 'tanh' gelu approximation by default."""
        layer = GeluAndMul()
        # GeluAndMul sets self.approximate = "tanh" by default
        assert layer.approximate == "tanh"

    def test_tanh_vs_none_approximation_differs(self):
        """The tanh approximation produces different values from exact gelu."""
        torch.manual_seed(0)
        gate = torch.randn(8, 64, dtype=torch.float32)
        up = torch.ones(8, 64, dtype=torch.float32)

        layer = GeluAndMul()
        result_tanh = layer.forward_oot((gate, up))

        # Exact GELU for comparison
        exact = F.gelu(gate, approximate="none") * up

        # They should be close but not identical
        assert not torch.allclose(result_tanh, exact, atol=0, rtol=0)
        # But within reasonable tolerance
        torch.testing.assert_close(result_tanh, exact, atol=2e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
