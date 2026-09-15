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
Test SpyreGemmaRMSNorm custom op correctness against a reference implementation.

SpyreGemmaRMSNorm mirrors SpyreRMSNorm but uses Gemma's normalization formula:
  x * (1 + weight) instead of x * weight
and skips the float32 dtype promotion (unsupported on Spyre).

Two code paths:
  1. Without residual: returns normalized x.
  2. With residual: returns (normalized x, updated residual).
"""

import pytest
import torch


def reference_gemma_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Golden reference: GemmaRMSNorm without float32 promotion.

    Gemma's formula: x_normed * (1 + weight)
    No dtype promotion to float32 (matching SpyreGemmaRMSNorm behavior).
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


@pytest.mark.gemma_rmsnorm
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512])
@pytest.mark.parametrize("use_residual", [False, True])
def test_spyre_gemma_rmsnorm_matches_reference(batch_size, hidden_size, use_residual):
    """SpyreGemmaRMSNorm output matches golden reference.

    Tests both paths:
    - forward_oot(): OOT dispatch via compiled forward
    - reference_gemma_rms_norm(): golden reference (ground truth)
    """
    from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

    eps = 1e-6
    device = "spyre"
    dtype = torch.float16
    torch.manual_seed(42)

    x = torch.randn(batch_size, hidden_size, dtype=dtype)
    layer = SpyreGemmaRMSNorm(hidden_size, eps=eps).to(dtype)
    residual = torch.randn(batch_size, hidden_size, dtype=dtype) if use_residual else None

    expected = reference_gemma_rms_norm(x, layer.weight.data, eps, residual)

    # Test forward_oot (Spyre device execution)
    layer.to(device)
    actual = layer.forward_oot(x.to(device), residual.to(device) if use_residual else None)

    if use_residual:
        expected_norm, expected_resid = expected
        actual_norm, actual_resid = actual
        torch.testing.assert_close(
            actual_norm.cpu().float(), expected_norm.float(), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            actual_resid.cpu().float(), expected_resid.float(), atol=1e-2, rtol=1e-2
        )
    else:
        torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.gemma_rmsnorm
@pytest.mark.parametrize("hidden_size", [64, 256])
def test_spyre_gemma_rmsnorm_gemma_weight_formula(hidden_size):
    """Verify SpyreGemmaRMSNorm uses x * (1 + weight) not x * weight.

    This distinguishes GemmaRMSNorm from standard RMSNorm.
    """
    from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

    eps = 1e-6
    dtype = torch.float16
    torch.manual_seed(42)

    x = torch.randn(2, hidden_size, dtype=dtype)
    layer = SpyreGemmaRMSNorm(hidden_size, eps=eps).to(dtype)

    # Set weight to zeros: (1 + 0) = 1 → output should equal plain RMS-normalized x
    layer.weight.data.zero_()
    layer.to("spyre")

    actual = layer.forward_oot(x.to("spyre"))
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    expected = x * torch.rsqrt(variance + eps)

    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.gemma_rmsnorm
def test_spyre_gemma_rmsnorm_residual_accumulation():
    """Verify residual is correctly accumulated: residual_out = x + residual_in."""
    from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

    hidden_size = 128
    eps = 1e-6
    dtype = torch.float16
    torch.manual_seed(42)

    x = torch.randn(4, hidden_size, dtype=dtype)
    residual = torch.randn(4, hidden_size, dtype=dtype)
    layer = SpyreGemmaRMSNorm(hidden_size, eps=eps).to(dtype).to("spyre")

    _, actual_resid = layer.forward_oot(x.to("spyre"), residual.to("spyre"))
    expected_resid = x + residual

    torch.testing.assert_close(
        actual_resid.cpu().float(), expected_resid.float(), atol=1e-2, rtol=1e-2
    )


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 128, dtype=torch.float32)


def mock_forward_oot(x, variance_epsilon=None, weight=None, residual=None):
    """Mock: return x + 1 (no residual path)."""
    return x + 1


def mock_forward_oot_with_residual(x, variance_epsilon=None, weight=None, residual=None):
    """Mock: return (2 * x, 2 * residual) (residual path)."""
    return 2 * x, 2 * residual


@pytest.mark.gemma_rmsnorm
@pytest.mark.parametrize("use_residual", [False, True])
def test_gemma_rmsnorm_oot_dispatch(monkeypatch, dummy_tensor, use_residual):
    """Verify GemmaRMSNorm OOT registration: class swap and forward_oot routing."""
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

    layer = GemmaRMSNorm(128, eps=1e-6)

    # OOT class swap: GemmaRMSNorm.__new__ should produce SpyreGemmaRMSNorm
    assert isinstance(layer, SpyreGemmaRMSNorm)

    # dispatch_forward should have selected forward_oot
    assert layer._forward_method == layer.forward_oot

    dummy_tensor = dummy_tensor.to(device="spyre")
    residual = torch.randn(4, 128, dtype=torch.float32, device="spyre") if use_residual else None

    # Mock _compiled_forward_spyre (called by forward_oot) with a known transform
    if residual is not None:
        monkeypatch.setattr(layer, "_compiled_forward_spyre", mock_forward_oot_with_residual)
        out_x, out_residual = layer.forward(dummy_tensor, residual)

        assert torch.allclose(out_x.cpu(), 2 * dummy_tensor.cpu())
        assert torch.allclose(out_residual.cpu(), 2 * residual.cpu())
    else:
        monkeypatch.setattr(layer, "_compiled_forward_spyre", mock_forward_oot)
        out_x = layer.forward(dummy_tensor, residual)

        assert torch.allclose(out_x.cpu(), dummy_tensor.cpu() + 1)
