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

"""Tests for the Spyre GateLinear OOT override (custom_ops/gate_linear.py).

Upstream models build their MoE router as ``GateLinear(..., out_dtype=float32)``
because fp32 logits stabilize topk. Spyre cannot restickify fp32, so the OOT
override clears ``out_dtype`` (the routing reduction is promoted to fp32 in
``moe`` instead) and mixes in the transposed-weight linear method like every
other unquantized linear. Most tests run on CPU; the numeric one runs on-card.
The end-to-end check that an MoE model routes correctly is
tests/models/test_gemma4_moe.py, which runs on Spyre hardware.
"""

import pytest
import torch
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

from spyre_inference.custom_ops.gate_linear import SpyreGateLinear
from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod

HIDDEN, NUM_EXPERTS = 128, 16


def _make_gate(out_dtype=None, *, positional=False):
    if positional:
        # (input_size, output_size, bias, out_dtype, ...): a positional caller
        # passes fp32 here, so the clear must not live in the argument handling.
        return GateLinear(HIDDEN, NUM_EXPERTS, False, out_dtype, torch.float16)
    return GateLinear(HIDDEN, NUM_EXPERTS, out_dtype=out_dtype, params_dtype=torch.float16)


@pytest.mark.mlp
@pytest.mark.parametrize("positional", [False, True])
def test_gate_linear_oot_swap_and_out_dtype_cleared(tp_group, positional):
    """Constructing vLLM's GateLinear under the platform yields the Spyre override
    with out_dtype cleared and the transposed-weight method swapped in."""
    gate = _make_gate(torch.float32, positional=positional)

    assert isinstance(gate, SpyreGateLinear)
    assert gate.out_dtype is None
    assert isinstance(gate.quant_method, SpyreUnquantizedLinearMethod)


@pytest.mark.mlp
def test_gate_weight_stored_transposed(tp_group):
    """process_weights_after_loading replaces `weight` with `weightᵀ`, contiguous."""
    torch.manual_seed(0)
    gate = _make_gate()
    gate.weight.data.normal_(std=0.02)
    original = gate.weight.data.clone()  # [num_experts, hidden]

    gate.quant_method.process_weights_after_loading(gate)

    w = gate.weight
    assert w.shape == (HIDDEN, NUM_EXPERTS)
    assert w.data.is_contiguous()
    torch.testing.assert_close(w.data, original.t(), atol=0.0, rtol=0.0)


@pytest.mark.mlp
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_router_logits_stay_in_weight_dtype(tp_group, num_tokens):
    """Router logits come back in the weight dtype and match `F.linear`.

    Upstream `out_dtype=float32` would cast them to fp32, which the card cannot
    restickify; a regression to that path shows up here as a fp32 output.
    """
    torch.manual_seed(0)
    gate = _make_gate(torch.float32)
    gate.weight.data.normal_(std=0.02)

    torch.manual_seed(1)
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.float16)
    expected = F.linear(x, gate.weight.data)

    gate.quant_method.process_weights_after_loading(gate)
    output, bias = gate(x)

    assert bias is None
    assert output.dtype == torch.float16
    torch.testing.assert_close(output.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
@pytest.mark.parametrize("mode", ["compile", "eager"])
def test_router_logits_match_reference_on_spyre(tp_group, mode):
    """The router GEMM on-card matches `F.linear`, compiled and eager.

    Compiled is the serving configuration (the router runs inside the compiled
    block), so it is the case that guards the lowering; eager proves the same
    kernels run uncompiled off the graph.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    torch.manual_seed(0)
    gate = _make_gate(torch.float32)
    gate.weight.data.normal_(std=0.02)

    torch.manual_seed(1)
    x = torch.randn(7, HIDDEN, dtype=torch.float16)
    # std-1 activations keep the output O(0.6), so atol=1e-2 is a real bound
    # rather than a tolerance wide enough to accept a wrong answer.
    expected = F.linear(x, gate.weight.data)

    gate.quant_method.process_weights_after_loading(gate)
    gate = gate.to("spyre")

    def fn(x):
        output, _ = gate(x)
        return output

    if mode == "compile":
        torch._dynamo.reset()
        fn = torch.compile(fn, dynamic=False, backend="inductor")

    actual = fn(x.to("spyre"))

    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)
