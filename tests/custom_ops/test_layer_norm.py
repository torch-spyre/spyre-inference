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
Test SpyreLayerNorm custom op correctness against a reference implementation.
"""

import sys

import pytest
import torch


def reference_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """The actual LayerNorm semantics every SpyreLayerNorm path must match."""
    return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)


@pytest.mark.layer_norm
@pytest.mark.parametrize("elementwise_affine", [True, False])
def test_spyre_layer_norm_cpu_matches_reference(default_vllm_config, elementwise_affine):
    """Off-device (CPU) input takes the `super().forward()` fallback branch and
    never touches the lazy Spyre kernel."""
    from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm

    torch.manual_seed(0)
    hidden_size, eps = 128, 1e-5
    layer = SpyreLayerNorm(hidden_size, eps=eps, elementwise_affine=elementwise_affine).to(
        torch.float16
    )

    x = torch.randn(4, hidden_size, dtype=torch.float16)
    actual = layer(x)
    expected = reference_layer_norm(
        x,
        layer.weight if elementwise_affine else None,
        layer.bias if elementwise_affine else None,
        eps,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)
    assert layer.spyre_compiled_kernel is None


@pytest.mark.layer_norm
@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512])
@pytest.mark.parametrize("elementwise_affine", [True, False])
def test_spyre_layer_norm_matches_reference_on_spyre(
    default_vllm_config, hidden_size, elementwise_affine
):
    """forward() on a Spyre tensor runs the decomposition-free kernel (lazily
    torch.compile'd under the default compilation mode) and matches the
    population-variance reference. Guards the crash this op works around:
    aten.layer_norm.default's native decomposition fails for boundary
    LayerNorms outside a per-block compiled graph."""
    from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm

    torch.manual_seed(1)
    eps = 1e-5
    layer = SpyreLayerNorm(hidden_size, eps=eps, elementwise_affine=elementwise_affine).to(
        torch.float16
    )
    x = torch.randn(4, hidden_size, dtype=torch.float16)

    weight = layer.weight if elementwise_affine else None
    bias = layer.bias if elementwise_affine else None
    expected = reference_layer_norm(x, weight, bias, eps)

    layer.to("spyre")
    actual = layer(x.to("spyre"))

    assert actual.device.type == "spyre"
    assert layer.spyre_compiled_kernel is not None  # lazily compiled and cached
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)

    kernel_after_first_call = layer.spyre_compiled_kernel
    layer(x.to("spyre"))
    assert layer.spyre_compiled_kernel is kernel_after_first_call  # reused, not rebuilt


@pytest.mark.layer_norm
def test_spyre_layer_norm_eager_mode_skips_torch_compile():
    """CompilationMode.NONE (enforce_eager) runs the raw _layer_norm_kernel
    directly, without ever populating a torch.compile'd kernel."""
    from vllm.config import (
        CompilationMode,
        DeviceConfig,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.config.compilation import CompilationConfig

    from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm

    # TorchSpyrePlatform.check_and_update_config keys off model_config.enforce_eager
    # (not compilation_config.mode -- see platform.py) and runs automatically inside
    # VllmConfig.__post_init__, so it overwrites a directly-set `mode=NONE` back to
    # STOCK_TORCH_COMPILE unless enforce_eager is also set here.
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        model_config=ModelConfig(dtype=torch.float16, enforce_eager=True),
    )
    with set_current_vllm_config(config):
        layer = SpyreLayerNorm(64, eps=1e-5).to(torch.float16).to("spyre")
        layer(torch.randn(2, 64, dtype=torch.float16).to("spyre"))

    assert layer.spyre_compiled_kernel is None


@pytest.mark.layer_norm
def test_spyre_layer_norm_fp32_accumulation_avoids_overflow():
    """fp16 mean/var overflows to inf once |x - mean| > 256, zeroing the row via
    rsqrt(inf). This test plants a ±400 outlier that triggers that path under a
    naive fp16 accumulation; the fp32 path must return finite, correct values.
    Runs on CPU so it needs no hardware."""
    from spyre_inference.custom_ops.layer_norm import _layer_norm_kernel

    hidden_size, eps = 128, 1e-5
    weight = torch.ones(hidden_size, dtype=torch.float16)
    bias = torch.zeros(hidden_size, dtype=torch.float16)

    # Row with an extreme outlier: |x - mean| ≈ 400, far above the fp16 255-max
    # for variance accumulation.
    x = torch.zeros(1, hidden_size, dtype=torch.float16)
    x[0, 0] = 400.0
    x[0, 1] = -400.0

    # A naive fp16 accumulation would produce var=inf → rsqrt(inf)=0 → zero row.
    var_fp16 = (x - x.mean(dim=-1, keepdim=True)).pow(2).mean(dim=-1)
    assert not torch.isfinite(var_fp16).all(), "precondition: fp16 var must overflow"

    result = _layer_norm_kernel(x, weight, bias, eps)

    assert torch.isfinite(result).all(), "fp32 accumulation must not produce inf/nan"
    # The reference (F.layer_norm, also fp32 internally) must agree.
    ref = torch.nn.functional.layer_norm(x, (hidden_size,), weight, bias, eps)
    torch.testing.assert_close(result, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.layer_norm
def test_spyre_layer_norm_inline_inside_compiled_graph(default_vllm_config):
    """Called from inside an existing torch.compile region (STOCK_TORCH_COMPILE
    compiles one transformer block at a time), forward() must inline
    _layer_norm_kernel rather than re-entering torch.compile -- and must not
    populate its own cached kernel, since it never takes that branch."""
    from vllm.platforms import current_platform

    from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm

    torch.manual_seed(2)
    hidden_size, eps = 64, 1e-5
    layer = SpyreLayerNorm(hidden_size, eps=eps).to(torch.float16)

    x = torch.randn(2, hidden_size, dtype=torch.float16)
    expected = reference_layer_norm(x, layer.weight, layer.bias, eps)

    layer.to("spyre")
    compiled = torch.compile(
        layer, backend=current_platform.simple_compile_backend, fullgraph=True, dynamic=False
    )
    actual = compiled(x.to("spyre"))

    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)
    assert layer.spyre_compiled_kernel is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
