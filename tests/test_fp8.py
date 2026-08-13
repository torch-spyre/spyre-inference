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

"""Test FP8 quantization support for the Spyre plugin.

This file provides:

1. A CPU FP8 reference (`cpu_quantize_fp8` / `cpu_dequantize_fp8`) that mirrors the
   torch-spyre op semantics, for reuse and comparison by future tests. The two
   contract tests below pin its behavior (clamping and scale-before-clamp
   ordering) so future tests can trust it as the golden reference.
2. One integration smoke test that confirms the FP8 ops run end-to-end under the
   plugin's pinned torch-spyre rev and torch.compile/inductor config.
"""

import warnings

import pytest
import torch

from torch_spyre._inductor.constants import FP8_E4M3FN_MAX

from spyre_testing_plugin.pytest_plugin import spyre_available


def cpu_quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """CPU reference for FP8 quantization: `clamp(x / scale).to(float8_e4m3fn)`."""
    return (x / scale).clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX).to(torch.float8_e4m3fn)


def cpu_dequantize_fp8(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """CPU reference for FP8 dequantization: `x_fp8.to(float16) * scale`."""
    return x_fp8.to(torch.float16) * scale


def cpu_quantize_dequantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """CPU reference for the full quantize->dequantize round-trip (float16 out)."""
    return cpu_dequantize_fp8(cpu_quantize_fp8(x, scale), scale)


@pytest.mark.fp8
def test_quantize_clamps_to_max():
    """Reference clamps values beyond +/-FP8_E4M3FN_MAX."""
    x = torch.tensor([500.0, -500.0, 1000.0], dtype=torch.float16)
    scale = torch.tensor([1.0], dtype=torch.float16)

    x_fp8_fp16 = cpu_quantize_fp8(x, scale).to(torch.float16)

    expected = torch.tensor([FP8_E4M3FN_MAX, -FP8_E4M3FN_MAX, FP8_E4M3FN_MAX], dtype=torch.float16)
    torch.testing.assert_close(x_fp8_fp16, expected, atol=0.0, rtol=0.0)


@pytest.mark.fp8
def test_quantize_applies_scale_before_clamp():
    """Reference applies the scale (x/scale) before clamping to the FP8 range."""
    x = torch.tensor([1.0, -1.0, 50.0], dtype=torch.float16)
    scale = torch.tensor([0.1], dtype=torch.float16)

    x_fp8_fp16 = cpu_quantize_fp8(x, scale).to(torch.float16)

    # x/scale = [10, -10, 500] -> clamp -> [10, -10, 448], all FP8-representable
    expected = torch.tensor([10.0, -10.0, FP8_E4M3FN_MAX], dtype=torch.float16)
    torch.testing.assert_close(x_fp8_fp16, expected, atol=0.0, rtol=0.0)


@pytest.mark.fp8
def test_spyre_fp8_roundtrip_smoke():
    """FP8 quantize->dequantize runs end-to-end on Spyre with no silent fallback."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from torch_spyre.ops.fallbacks import FallbackWarning

    torch.manual_seed(42)
    shape = (1, 128, 512)
    x_cpu = torch.randn(shape, dtype=torch.float16) * 2.0 + 1.0
    scale = torch.tensor([1.0], dtype=torch.float16)

    expected = cpu_quantize_dequantize_fp8(x_cpu, scale)

    @torch.compile(backend="inductor")
    def spyre_roundtrip(x, s):
        x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, s)
        return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, s)

    # A silent CPU fallback would change the numerical path and hide backend
    # regressions; turn it into a hard failure for this run.
    with warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        actual = spyre_roundtrip(x_cpu.to("spyre"), scale.to("spyre"))

    torch.testing.assert_close(actual.cpu(), expected, atol=0.5, rtol=0.1)
