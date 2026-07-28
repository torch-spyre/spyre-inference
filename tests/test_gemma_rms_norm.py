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

"""Unit tests for SpyreGemmaRMSNorm custom op.

SpyreGemmaRMSNorm is the Spyre OOT override for vLLM's GemmaRMSNorm.
Key differences from standard RMSNorm:
- Uses x * (1 + weight) instead of x * weight
- Skips float32 dtype promotion (Spyre does not support it)
- Supports optional residual add-before-norm path

These tests verify:
1. OOT registration (class swap)
2. Numerical correctness against an fp16 reference (no float32 promotion)
3. Residual path correctness
4. Weight formula: x * (1 + w) vs standard x * w
"""

import pytest
import torch


pytestmark = pytest.mark.gemma_rms_norm


def reference_gemma_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Golden reference: GemmaRMSNorm in fp16 (no float32 promotion).

    This matches the Spyre implementation behavior:
    - No dtype promotion (stays in fp16)
    - Weight formula: x * (1 + weight)
    """
    if residual is not None:
        x = x + residual
        residual = x

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    x_normed = x_normed * (1.0 + weight)

    if residual is None:
        return x_normed
    return x_normed, residual


class TestSpyreGemmaRMSNormRegistration:
    """Test OOT registration of SpyreGemmaRMSNorm."""

    def test_oot_dispatch_class_swap(self):
        """GemmaRMSNorm() should produce SpyreGemmaRMSNorm via OOT registration."""
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
        from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

        layer = GemmaRMSNorm(hidden_size=128, eps=1e-6)
        assert isinstance(layer, SpyreGemmaRMSNorm)

    def test_forward_method_is_forward_oot(self):
        """dispatch_forward should have selected forward_oot."""
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
        from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

        layer = GemmaRMSNorm(hidden_size=128, eps=1e-6)
        assert layer._forward_method == layer.forward_oot

    def test_inherits_from_gemma_rms_norm(self):
        """SpyreGemmaRMSNorm should be a subclass of GemmaRMSNorm."""
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
        from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

        assert issubclass(SpyreGemmaRMSNorm, GemmaRMSNorm)


class TestSpyreGemmaRMSNormCorrectness:
    """Test numerical correctness of SpyreGemmaRMSNorm."""

    @pytest.mark.parametrize("hidden_size", [64, 128, 256, 512])
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_forward_matches_reference(self, hidden_size, batch_size):
        """SpyreGemmaRMSNorm output matches fp16 reference without residual."""
        from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

        eps = 1e-6
        torch.manual_seed(42)

        x = torch.randn(batch_size, hidden_size, dtype=torch.float16)
        layer = SpyreGemmaRMSNorm(hidden_size=hidden_size, eps=eps)
        # Ensure weight is float16 for consistent behavior
        layer.weight.data = layer.weight.data.to(torch.float16)

        expected = reference_gemma_rms_norm(x, layer.weight.data, eps)
        actual = layer.forward_oot(x)

        torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("hidden_size", [64, 128, 256])
    def test_forward_with_residual(self, hidden_size):
        """SpyreGemmaRMSNorm with residual path matches reference."""
        from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

        eps = 1e-6
        torch.manual_seed(42)

        x = torch.randn(2, hidden_size, dtype=torch.float16)
        residual = torch.randn(2, hidden_size, dtype=torch.float16)
        layer = SpyreGemmaRMSNorm(hidden_size=hidden_size, eps=eps)
        layer.weight.data = layer.weight.data.to(torch.float16)

        expected_norm, expected_resid = reference_gemma_rms_norm(
            x, layer.weight.data, eps, residual
        )
        actual_norm, actual_resid = layer.forward_oot(x, residual)

        torch.testing.assert_close(
            actual_norm.float(), expected_norm.float(), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            actual_resid.float(), expected_resid.float(), atol=1e-2, rtol=1e-2
        )

    def test_weight_formula_uses_one_plus_weight(self):
        """Verify the (1 + weight) formula distinguishes Gemma from standard RMS."""
        from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

        hidden_size = 64
        eps = 1e-6
        torch.manual_seed(42)

        # Use a known weight to verify the formula
        layer = SpyreGemmaRMSNorm(hidden_size=hidden_size, eps=eps)
        layer.weight.data = torch.ones(hidden_size, dtype=torch.float16) * 0.5

        x = torch.randn(1, hidden_size, dtype=torch.float16)
        actual = layer.forward_oot(x)

        # Manually compute: RMSNorm(x) * (1 + 0.5) = RMSNorm(x) * 1.5
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        expected = x_normed * 1.5

        torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


class TestSpyreGemmaRMSNormResidualPath:
    """Test the residual add-before-norm path specifically."""

    def test_residual_none_returns_single_tensor(self):
        """When residual=None, forward_oot should return a single tensor."""
        from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

        layer = SpyreGemmaRMSNorm(hidden_size=64, eps=1e-6)
        layer.weight.data = layer.weight.data.to(torch.float16)
        x = torch.randn(2, 64, dtype=torch.float16)

        result = layer.forward_oot(x, residual=None)
        assert isinstance(result, torch.Tensor)

    def test_residual_present_returns_tuple(self):
        """When residual is provided, forward_oot returns (normed, residual)."""
        from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

        layer = SpyreGemmaRMSNorm(hidden_size=64, eps=1e-6)
        layer.weight.data = layer.weight.data.to(torch.float16)
        x = torch.randn(2, 64, dtype=torch.float16)
        residual = torch.randn(2, 64, dtype=torch.float16)

        result = layer.forward_oot(x, residual)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_residual_is_sum_of_x_and_input_residual(self):
        """The returned residual should be x + input_residual."""
        from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

        layer = SpyreGemmaRMSNorm(hidden_size=64, eps=1e-6)
        layer.weight.data = layer.weight.data.to(torch.float16)
        x = torch.randn(2, 64, dtype=torch.float16)
        residual = torch.randn(2, 64, dtype=torch.float16)

        _, out_residual = layer.forward_oot(x, residual)

        expected_residual = x + residual
        torch.testing.assert_close(out_residual, expected_residual)
