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
Test SpyreRMSNorm custom op correctness against a reference implementation.
"""

import sys

import pytest
import torch


def reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """fp16 RMSNorm reference (no fp32 upcast): an oracle for the device lowering,
    not for fp16-vs-fp32 precision the op does not promise."""
    if residual is not None:
        x = x + residual
        residual = x
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    if weight is not None:
        x_normed = x_normed * weight
    if residual is not None:
        return x_normed, residual
    return x_normed


@pytest.mark.rmsnorm
@pytest.mark.parametrize("batch_size", [1])
# Hidden sizes must be a multiple of 64 (Spyre 128-byte stick / 2 bytes fp16).
@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512])
@pytest.mark.parametrize("use_residual", [False, True])
def test_spyre_rmsnorm_matches_reference(batch_size, hidden_size, use_residual):
    """SpyreRMSNorm.forward_oot on device matches the eager fp16 reference."""
    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    eps = 1e-6
    device = "spyre"
    dtype = torch.float16
    torch.manual_seed(42)

    x = torch.randn(batch_size, hidden_size, dtype=dtype)
    layer = SpyreRMSNorm(hidden_size, eps=eps).to(dtype)
    residual = torch.randn(batch_size, hidden_size, dtype=dtype) if use_residual else None

    expected = reference_rms_norm(x, layer.weight.data, eps, residual)

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


@pytest.mark.rmsnorm
def test_rmsnorm_oot_dispatch():
    """Verify RMSNorm OOT registration: class swap."""
    from vllm.model_executor.layers.layernorm import RMSNorm

    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    layer = RMSNorm(128, eps=1e-6)

    # OOT class swap: RMSNorm.__new__ should produce SpyreRMSNorm
    assert isinstance(layer, SpyreRMSNorm)

    # dispatch_forward should have selected forward_oot
    assert layer._forward_method == layer.forward_oot


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
