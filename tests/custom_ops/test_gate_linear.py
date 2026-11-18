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

"""Tests for the Spyre MoE router gate (custom_ops/gate_linear.py).

The contract checks are CPU-only: the OOT substitution, the cleared
``out_dtype`` and the transposed weight are host-side decisions. The router
GEMM itself also runs on-card in the gpt-oss production shape (2880 -> 16
experts): its narrow output, one column per expert, is a shape none of the
other transposed-linear tests reach.
"""

import pytest
import torch
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.custom_ops.gate_linear import SpyreGateLinear
from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod

_HIDDEN, _EXPERTS = 64, 8


def _make_gate(input_size=_HIDDEN, output_size=_EXPERTS, out_dtype=None, **kwargs):
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    kwargs.setdefault("params_dtype", torch.float16)
    kwargs.setdefault("return_bias", False)
    return GateLinear(input_size, output_size, out_dtype=out_dtype, prefix="router", **kwargs)


@pytest.mark.mlp
def test_gate_linear_is_substituted_oot(tp_group):
    """A model's plain `GateLinear(...)` builds the Spyre layer, transposed path armed."""
    gate = _make_gate()

    assert isinstance(gate, SpyreGateLinear)
    assert isinstance(gate.quant_method, SpyreUnquantizedLinearMethod)


@pytest.mark.mlp
@pytest.mark.parametrize(
    ("out_dtype", "positional"),
    [
        (None, False),
        (torch.float32, False),
        (torch.float32, True),
    ],
    ids=["unset", "fp32_keyword", "fp32_positional"],
)
def test_fp32_router_logits_are_refused(tp_group, out_dtype, positional):
    """Models with an fp32 router must not get fp32 logits on Spyre.

    Spyre cannot restickify fp32, so the routing reduction over the logits is
    promoted in `moe.py` instead. The clear happens in `__init__` rather than
    through the argument, so a positional `out_dtype` caller is caught too.
    """
    if positional:
        from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

        gate = GateLinear(_HIDDEN, _EXPERTS, out_dtype, params_dtype=torch.float16, prefix="router")
    else:
        gate = _make_gate(out_dtype=out_dtype)

    assert gate.out_dtype is None


@pytest.mark.mlp
@pytest.mark.parametrize("bias", [False, True])
def test_router_logits_match_reference_in_the_weight_dtype(tp_group, bias):
    """After processing, forward is `x @ Wᵀ (+ bias)` matching F.linear, in fp16.

    With `out_dtype` cleared, the fp32 cast upstream's fallback tier would apply
    never happens, so the logits handed to the router stay in the weight dtype.
    """
    gate = _make_gate(out_dtype=torch.float32, bias=bias)
    torch.manual_seed(0)
    gate.weight.data.normal_(std=0.02)
    if bias:
        gate.bias.data.normal_(std=0.02)
    x = torch.randn(5, _HIDDEN, dtype=torch.float16)

    # Capture the reference BEFORE process_weights_after_loading transposes.
    expected = F.linear(x, gate.weight.data, gate.bias.data if bias else None)

    gate.quant_method.process_weights_after_loading(gate)

    assert gate.weight.shape == (_HIDDEN, _EXPERTS)  # stored transposed: [in, out]
    out = gate(x)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_router_gemm_matches_reference_on_spyre(tp_group, mode):
    """The gpt-oss router GEMM (`[T, 2880] @ [2880, 16] + bias`) on-card.

    The compile case is the production one: the router runs inside the compiled
    transformer block. 2880 -> 16 is the shape tests/models/test_gpt_oss_moe.py
    drives end to end.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    hidden, experts = 2880, 16
    gate = _make_gate(hidden, experts, bias=True)
    torch.manual_seed(0)
    gate.weight.data.normal_(std=0.02)
    gate.bias.data.normal_(std=0.02)
    x = torch.randn(7, hidden, dtype=torch.float16)
    expected = F.linear(x, gate.weight.data, gate.bias.data)

    gate.quant_method.process_weights_after_loading(gate)
    gate = gate.to("spyre")

    def fn(x):
        return gate(x)

    if mode == "compile":
        torch._dynamo.reset()
        fn = torch.compile(fn, dynamic=False, backend="inductor")

    out = fn(x.to("spyre"))
    assert out.dtype == torch.float16
    assert out.shape == expected.shape
    torch.testing.assert_close(out.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)
