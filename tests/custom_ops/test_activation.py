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

"""Test the gelu_new activation on Spyre (every GPT-2 model uses it)."""

import math
import sys

import pytest
import torch


def reference_gelu_new(x: torch.Tensor) -> torch.Tensor:
    """Golden reference: HF/vLLM gelu_new, evaluated in float32."""
    xf = x.float()
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * xf * (1.0 + torch.tanh(c * (xf + 0.044715 * xf.pow(3))))


@pytest.mark.parametrize("num_tokens", [1, 5, 16])
@pytest.mark.parametrize("d", [64, 256, 3072])
def test_newgelu_on_spyre_matches_reference(num_tokens, d):
    """SpyreNewGELU on a Spyre input matches the float32 reference."""
    from vllm.model_executor.layers.activation import NewGELU

    import spyre_inference.custom_ops.activation  # noqa: F401

    torch.manual_seed(42)
    # Scaled to the range a GPT-2 c_fc output actually spans.
    x = (torch.randn(num_tokens, d, dtype=torch.float32) * 4.0).to(torch.float16)
    layer = NewGELU()

    actual = layer.forward_oot(x.to("spyre"))

    torch.testing.assert_close(actual.cpu().float(), reference_gelu_new(x), atol=1e-2, rtol=1e-2)


def test_newgelu_gates_negative_inputs():
    """gelu_new must gate negative inputs towards zero, not return them unchanged."""
    from vllm.model_executor.layers.activation import NewGELU

    import spyre_inference.custom_ops.activation  # noqa: F401

    x = torch.full((1, 64), -6.0, dtype=torch.float16)
    out = NewGELU().forward_oot(x.to("spyre")).cpu().float()

    assert out.abs().max().item() < 1e-2, f"expected ~0, got {out.flatten()[0]}"


def test_newgelu_oot_dispatch():
    """Verify NewGELU OOT registration: class swap."""
    from vllm.model_executor.layers.activation import NewGELU

    from spyre_inference.custom_ops.activation import SpyreNewGELU

    layer = NewGELU()

    assert isinstance(layer, SpyreNewGELU)
    assert layer._forward_method == layer.forward_oot


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
