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

"""Tests for the Spyre GateLinear swap (custom_ops/gate_linear.py).

Models can request fp32 router logits (``out_dtype=torch.float32``). Spyre cannot
restickify fp32, so ``SpyreGateLinear`` clears ``out_dtype`` for keyword and
positional callers. MoE selects its reduction dtype separately, promoting only
when the layout permits it. The transposed-weight mixin gives the gate the
same Spyre-fast ``x @ Wᵀ`` GEMM as every other unquantized linear.
"""

import warnings

import pytest
import torch
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available

HIDDEN, EXPERTS = 128, 32


def _make_gate(out_dtype=None, bias=False, *, hidden=HIDDEN):
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    return GateLinear(
        hidden,
        EXPERTS,
        bias=bias,
        out_dtype=out_dtype,
        params_dtype=torch.float16,
        prefix="gate",
    )


@pytest.mark.parametrize("positional", [False, True])
def test_gate_linear_oot_swap_clears_out_dtype(tp_group, positional):
    """Constructing vLLM's GateLinear yields the Spyre subclass, with out_dtype
    cleared however it was passed, so router logits stay in the weight dtype."""
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    from spyre_inference.custom_ops.gate_linear import SpyreGateLinear
    from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod

    if positional:
        gate = GateLinear(HIDDEN, EXPERTS, False, torch.float32, params_dtype=torch.float16)
    else:
        gate = GateLinear(HIDDEN, EXPERTS, out_dtype=torch.float32, params_dtype=torch.float16)
    assert isinstance(gate, SpyreGateLinear)
    assert gate.out_dtype is None
    assert isinstance(gate.quant_method, SpyreUnquantizedLinearMethod)


@pytest.mark.parametrize("num_tokens", [1, 5, 16])
@pytest.mark.parametrize("use_bias", [False, True])
def test_gate_linear_matches_reference(tp_group, num_tokens, use_bias):
    """After processing weights, the gate's `x @ Wᵀ (+bias)` matches F.linear,
    and the logits come back in the fp16 weight dtype despite the fp32 out_dtype
    the layer was built with."""
    torch.manual_seed(0)
    gate = _make_gate(out_dtype=torch.float32, bias=use_bias)
    gate.weight.data.normal_(std=0.02)
    if gate.bias is not None:
        gate.bias.data.normal_(std=0.02)

    torch.manual_seed(1)
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.float16)
    bias = gate.bias.data if gate.bias is not None else None
    expected = F.linear(x, gate.weight.data, bias)

    gate.quant_method.process_weights_after_loading(gate)
    out, returned_bias = gate(x)

    assert out.dtype == torch.float16
    assert returned_bias is None
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "num_tokens,hidden,compiled",
    [
        pytest.param(1, HIDDEN, False, id="1"),
        pytest.param(8, HIDDEN, False, id="8"),
        pytest.param(7, 2816, True, id="gemma4-compiled"),
    ],
)
def test_gate_linear_matches_reference_on_spyre(tp_group, num_tokens, hidden, compiled):
    """The router-gate GEMM lowers on the card and the logits stay fp16 there:
    an out_dtype cast to fp32 would need a restickify the device cannot do."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from torch_spyre.ops.fallbacks import FallbackWarning

    torch.manual_seed(0)
    gate = _make_gate(out_dtype=torch.float32, hidden=hidden)
    gate.weight.data.normal_(std=0.02)

    torch.manual_seed(1)
    x = torch.randn(num_tokens, hidden, dtype=torch.float16)
    expected = F.linear(x.float(), gate.weight.data.float())

    gate.quant_method.process_weights_after_loading(gate)
    gate = gate.to("spyre")
    if compiled:
        gate = torch.compile(gate, fullgraph=True, dynamic=False, backend="inductor")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        out, returned_bias = gate(x.to("spyre"))
        assert out.device.type == "spyre"
        actual = out.cpu()

    assert returned_bias is None
    assert out.dtype == torch.float16
    # Same atol as the other 2816-wide fp16 reductions (test_mlp.py).
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=1e-2)
