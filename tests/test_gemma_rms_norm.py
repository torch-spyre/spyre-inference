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

"""Unit tests for SpyreGemmaRMSNorm (custom_ops/gemma_rms_norm.py).

SpyreGemmaRMSNorm overrides GemmaRMSNorm.forward_oot to avoid the
float32 promotion unsupported on Spyre. Key differences from RMSNorm:
  - Weight formula: x * (1 + w) instead of x * w
  - No dtype promotion (stays in fp16)

All tests run on CPU — no Spyre device needed.
"""

import sys

import pytest
import torch

from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm
from vllm.model_executor.layers.layernorm import GemmaRMSNorm


def _reference_gemma_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Reference GemmaRMSNorm: RMS normalize then scale by (1 + weight)."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    return normed * (1.0 + weight)


@pytest.mark.gemma_rms_norm
class TestSpyreGemmaRMSNormRegistration:
    """Test OOT registration of SpyreGemmaRMSNorm."""

    def test_is_subclass(self):
        """SpyreGemmaRMSNorm inherits from GemmaRMSNorm."""
        assert issubclass(SpyreGemmaRMSNorm, GemmaRMSNorm)

    def test_oot_dispatch(self):
        """GemmaRMSNorm() instantiates SpyreGemmaRMSNorm via OOT registration."""
        layer = GemmaRMSNorm(hidden_size=64)
        assert isinstance(layer, SpyreGemmaRMSNorm)

    def test_has_forward_oot_method(self):
        """SpyreGemmaRMSNorm overrides forward_oot."""
        assert "forward_oot" in SpyreGemmaRMSNorm.__dict__


@pytest.mark.gemma_rms_norm
class TestSpyreGemmaRMSNormForward:
    """Test forward_oot behavior without residual."""

    @pytest.mark.parametrize("hidden_size", [64, 128, 256])
    @pytest.mark.parametrize("num_tokens", [1, 4, 8])
    def test_matches_reference(self, hidden_size, num_tokens):
        """forward_oot matches the reference GemmaRMSNorm implementation."""
        torch.manual_seed(42)
        layer = GemmaRMSNorm(hidden_size=hidden_size, eps=1e-6)
        layer.weight.data.normal_(std=0.02)

        x = torch.randn(num_tokens, hidden_size, dtype=torch.float16)

        result = layer.forward_oot(x)
        expected = _reference_gemma_rms_norm(x, layer.weight.data, 1e-6)

        assert result.shape == (num_tokens, hidden_size)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_output_dtype_matches_input(self):
        """Output dtype matches input dtype (no promotion)."""
        layer = GemmaRMSNorm(hidden_size=64)
        x = torch.randn(4, 64, dtype=torch.float16)
        result = layer.forward_oot(x)
        assert result.dtype == torch.float16

    def test_unit_input_normalized(self):
        """All-ones input is normalized to near-ones (RMS of ones is 1)."""
        layer = GemmaRMSNorm(hidden_size=64, eps=1e-6)
        # Set weight to zeros so (1 + w) = 1
        layer.weight.data.zero_()
        x = torch.ones(1, 64, dtype=torch.float32)

        result = layer.forward_oot(x)
        # RMS of all-ones is 1, so rsqrt(1 + eps) ≈ 1, output ≈ 1
        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    def test_weight_scaling(self):
        """Weight (1+w) scales the normalized output correctly."""
        torch.manual_seed(0)
        hidden_size = 64
        layer = GemmaRMSNorm(hidden_size=hidden_size, eps=1e-6)
        # Set weight to constant 1.0, so (1 + w) = 2.0
        layer.weight.data.fill_(1.0)

        x = torch.randn(4, hidden_size, dtype=torch.float32)

        # With weight=0: output = normalized_x * 1
        layer_zero = GemmaRMSNorm(hidden_size=hidden_size, eps=1e-6)
        layer_zero.weight.data.zero_()
        baseline = layer_zero.forward_oot(x)

        # With weight=1: output = normalized_x * 2
        result = layer.forward_oot(x)
        torch.testing.assert_close(result, baseline * 2.0, atol=1e-5, rtol=1e-5)


@pytest.mark.gemma_rms_norm
class TestSpyreGemmaRMSNormResidual:
    """Test forward_oot with residual connection."""

    def test_residual_adds_to_input(self):
        """With residual, x = x + residual before normalization."""
        torch.manual_seed(0)
        hidden_size = 64
        layer = GemmaRMSNorm(hidden_size=hidden_size, eps=1e-6)
        layer.weight.data.normal_(std=0.02)

        x = torch.randn(4, hidden_size, dtype=torch.float16)
        residual = torch.randn(4, hidden_size, dtype=torch.float16)

        result = layer.forward_oot(x, residual=residual)

        # When residual is provided, returns (output, new_residual)
        assert isinstance(result, tuple)
        output, new_residual = result

        # new_residual = x + residual
        expected_residual = x + residual
        torch.testing.assert_close(new_residual, expected_residual, atol=1e-3, rtol=1e-3)

        # output = GemmaRMSNorm(x + residual)
        expected_output = _reference_gemma_rms_norm(
            expected_residual, layer.weight.data, 1e-6
        )
        torch.testing.assert_close(output, expected_output, atol=1e-3, rtol=1e-3)

    def test_no_residual_returns_tensor(self):
        """Without residual, returns a plain tensor (not a tuple)."""
        layer = GemmaRMSNorm(hidden_size=64, eps=1e-6)
        x = torch.randn(4, 64, dtype=torch.float16)

        result = layer.forward_oot(x, residual=None)
        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, tuple)

    def test_residual_shape_preserved(self):
        """Residual output has the same shape as input."""
        hidden_size = 128
        layer = GemmaRMSNorm(hidden_size=hidden_size, eps=1e-6)
        x = torch.randn(8, hidden_size, dtype=torch.float16)
        residual = torch.randn(8, hidden_size, dtype=torch.float16)

        output, new_residual = layer.forward_oot(x, residual=residual)
        assert output.shape == (8, hidden_size)
        assert new_residual.shape == (8, hidden_size)


@pytest.mark.gemma_rms_norm
class TestSpyreGemmaRMSNormDiffersFromRMSNorm:
    """Verify GemmaRMSNorm differs from standard RMSNorm in the expected way."""

    def test_gemma_vs_standard_weight_formula(self):
        """GemmaRMSNorm uses x*(1+w) while standard RMSNorm uses x*w."""
        torch.manual_seed(0)
        hidden_size = 64
        eps = 1e-6

        gemma_layer = GemmaRMSNorm(hidden_size=hidden_size, eps=eps)
        gemma_layer.weight.data.fill_(0.5)

        x = torch.randn(4, hidden_size, dtype=torch.float32)

        gemma_result = gemma_layer.forward_oot(x)

        # Standard RMSNorm reference: x * w (where w = 0.5)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + eps)
        standard_result = normed * 0.5

        # Gemma should use (1 + 0.5) = 1.5 instead of 0.5
        gemma_expected = normed * 1.5

        torch.testing.assert_close(gemma_result, gemma_expected, atol=1e-5, rtol=1e-5)
        # And should NOT match the standard version
        assert not torch.allclose(gemma_result, standard_result)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
