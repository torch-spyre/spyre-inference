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

"""Spyre GemmaRMSNorm: OOT dispatch, and on-device agreement with the fp32 reference.

Plain ``RMSNorm`` no longer has an OOT op (torch-spyre#4490 keeps the fp32 reduction in
the graph), but ``GemmaRMSNorm`` still needs one: its trailing fp32 multiply against a
STANDARD ``[hidden]`` weight is a mixed-EA form the backend can neither broadcast nor
de-stagger, so the kernel must be compiled rather than run eagerly.

Since the op forwards to vLLM's ``forward_native``, the oracle is the fp32-accumulating
upstream definition itself, so the tolerance is the device's own error rather than the
fp16-vs-fp32 gap the pre-fp32 op had to allow.
"""

import sys

import pytest
import torch

# vLLM's own float16 bound for these ops (ir/ops/layernorm.py `override_tolerance`).
_ATOL, _RTOL = 1e-2, 2e-3


def reference_gemma_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """CPU mirror of ``GemmaRMSNorm.forward_native``.

    Gemma differs from plain RMSNorm twice: the weight enters as ``1 + w``, and it is
    promoted to fp32 before the multiply. The reduction and the residual add accumulate
    in fp32 and only the result is cast back.
    """
    orig_dtype = x.dtype
    gemma_weight = weight.float() + 1.0
    x = x.float()
    residual_out = None
    if residual is not None:
        x = x + residual.float()
        residual_out = x.to(orig_dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    normed = (x * gemma_weight).to(orig_dtype)
    return normed if residual is None else (normed, residual_out)


def _install_worker_torch_wrap() -> None:
    """Mirror ``WorkerBase.__init__``, which installs the config's torch-wrap state.

    Without it the vllm_ir torch custom op wraps the norm and the Spyre backend cannot
    lower the wrapped reduction ("Multi-arg pointwise with mixed EA"), a failure no
    engine run can hit.
    """
    import vllm.ir
    from vllm.config import get_current_vllm_config

    vllm.ir.set_default_torch_wrap(
        get_current_vllm_config().compilation_config.ir_enable_torch_wrap
    )


@pytest.mark.rmsnorm
@pytest.mark.parametrize("batch_size", [1, 8])
# Hidden sizes must be a multiple of 64 (Spyre 128-byte stick / 2 bytes fp16).
@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512])
@pytest.mark.parametrize("use_residual", [False, True])
def test_spyre_gemma_rmsnorm_matches_reference(
    default_vllm_config, batch_size, hidden_size, use_residual
):
    """SpyreGemmaRMSNorm.forward_oot on device matches the upstream fp32 reference."""
    from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

    _install_worker_torch_wrap()

    eps = 1e-6
    dtype = torch.float16
    torch.manual_seed(42)

    x = torch.randn(batch_size, hidden_size, dtype=dtype)
    residual = torch.randn(batch_size, hidden_size, dtype=dtype) if use_residual else None
    layer = SpyreGemmaRMSNorm(hidden_size, eps=eps).to(dtype)
    # The default weight is all zeros (Gemma's 1 + w == 1), which hides a dropped or
    # wrongly shaped multiply.
    layer.weight.data = torch.randn(hidden_size, dtype=dtype)

    expected = reference_gemma_rms_norm(x, layer.weight.data, eps, residual)

    layer.to("spyre")
    actual = layer.forward_oot(x.to("spyre"), residual.to("spyre") if use_residual else None)

    if use_residual:
        expected_norm, expected_residual = expected
        actual_norm, actual_residual = actual
        torch.testing.assert_close(
            actual_norm.cpu().float(), expected_norm.float(), atol=_ATOL, rtol=_RTOL
        )
        torch.testing.assert_close(
            actual_residual.cpu().float(), expected_residual.float(), atol=_ATOL, rtol=_RTOL
        )
    else:
        torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=_ATOL, rtol=_RTOL)


@pytest.mark.rmsnorm
@pytest.mark.parametrize("hidden_size", [256, 512])
def test_spyre_gemma_rmsnorm_accumulates_the_variance_in_fp32(default_vllm_config, hidden_size):
    """The point of forwarding to ``forward_native``: the reduction is not fp16.

    Scaled so ``x**2`` exceeds the fp16 max (65504) and the fp16 sum-of-squares
    saturates to inf, collapsing ``rsqrt`` to 0. The device must track the fp32 oracle
    and visibly disagree with the fp16 one, which the moderate-magnitude cases above
    cannot show because there both oracles agree to within tolerance.
    """
    from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

    _install_worker_torch_wrap()

    eps = 1e-6
    torch.manual_seed(0)
    x = (torch.randn(4, hidden_size) * 64.0).half()
    layer = SpyreGemmaRMSNorm(hidden_size, eps=eps).to(torch.float16)
    layer.weight.data = torch.zeros(hidden_size, dtype=torch.float16)
    weight = layer.weight.data.clone()

    fp16_variance = x.pow(2).mean(dim=-1, keepdim=True)
    assert fp16_variance.isinf().any(), "input no longer overflows fp16; pick a larger scale"
    fp16_oracle = x * torch.rsqrt(fp16_variance + eps) * (weight + 1.0)
    fp32_oracle = reference_gemma_rms_norm(x, weight, eps)

    layer.to("spyre")
    actual = layer.forward_oot(x.to("spyre")).cpu().float()

    torch.testing.assert_close(actual, fp32_oracle.float(), atol=_ATOL, rtol=_RTOL)
    assert (actual - fp16_oracle.float()).abs().max() > 1.0


@pytest.mark.rmsnorm
def test_gemma_rmsnorm_oot_dispatch(default_vllm_config):
    """Verify GemmaRMSNorm OOT registration: class swap."""
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm

    layer = GemmaRMSNorm(128, eps=1e-6)

    # OOT class swap: GemmaRMSNorm.__new__ should produce SpyreGemmaRMSNorm
    assert isinstance(layer, SpyreGemmaRMSNorm)

    # dispatch_forward should have selected forward_oot
    assert layer._forward_method == layer.forward_oot


@pytest.mark.rmsnorm
def test_plain_rmsnorm_has_no_oot_replacement(default_vllm_config):
    """Plain RMSNorm runs upstream unchanged; only the Gemma variant keeps an op.

    Guards the removal: a re-added `RMSNorm.register_oot` would silently reintroduce the
    op that torch-spyre#4490 made unnecessary. On an OOT platform the base `CustomOp`
    still dispatches to `forward_oot`, but that is `CustomOp`'s own default, which just
    forwards to `forward_native` — so the check is that the class was not swapped and
    the method is not overridden.
    """
    from vllm.model_executor.custom_op import CustomOp
    from vllm.model_executor.layers.layernorm import RMSNorm

    layer = RMSNorm(128, eps=1e-6)

    assert type(layer) is RMSNorm, f"unexpected OOT swap to {type(layer).__name__}"
    assert type(layer).forward_oot is CustomOp.forward_oot, (
        "RMSNorm.forward_oot is overridden; the Spyre op appears to be back"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
