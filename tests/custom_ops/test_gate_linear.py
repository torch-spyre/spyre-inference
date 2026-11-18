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

"""Tests for the MoE router gate (custom_ops/gate_linear.py).

`SpyreGateLinear` exists for two device limits. Spyre cannot restickify fp32
(``ReStickifyOpHBM`` has no IEEE_FP32 support), so upstream's ``out_dtype``
(some models pass ``torch.float32`` for stable routing) is cleared and the
routing reduction is promoted to fp32 in ``moe`` instead. And like every
unquantized linear, the router GEMM must run on the transposed-weight fast
path (a plain matmul against a stored-transposed weight) rather than
``F.linear``.

The CPU tests guard the dtype contract and the transposed weight; the on-card
test runs the router GEMM at a stick-aligned expert count, as production MoE
models (gemma-4: 128 experts) require for their router-logits rows to be
addressable.
"""

import pytest
import torch
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


def _make_gate(hidden=128, experts=8, out_dtype=torch.float32, **kwargs):
    return GateLinear(
        input_size=hidden,
        output_size=experts,
        bias=False,
        out_dtype=out_dtype,
        params_dtype=torch.float16,
        quant_config=None,
        prefix="gate",
        **kwargs,
    )


@pytest.mark.mlp
def test_gate_linear_is_spyre_oot(tp_group):
    """Constructing vLLM's GateLinear yields the Spyre OOT subclass with the
    transposed-weight method swapped in (`quant_config=None` here)."""
    from spyre_inference.custom_ops.gate_linear import SpyreGateLinear
    from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod

    layer = _make_gate()

    assert isinstance(layer, SpyreGateLinear)
    assert isinstance(layer.quant_method, SpyreUnquantizedLinearMethod)


@pytest.mark.mlp
@pytest.mark.parametrize("pass_positionally", [False, True])
def test_out_dtype_cleared(tp_group, pass_positionally):
    """`out_dtype` is None after construction however it was passed.

    Models can request fp32 router logits; honoring that on Spyre would hand
    the routing softmax an fp32 tensor the device cannot restickify.
    """
    if pass_positionally:
        # input_size, output_size, bias, out_dtype, params_dtype
        layer = GateLinear(
            128, 8, False, torch.float32, torch.float16, quant_config=None, prefix="gate"
        )
    else:
        layer = _make_gate(out_dtype=torch.float32)
    assert layer.out_dtype is None


@pytest.mark.mlp
def test_forward_keeps_weight_dtype_and_matches_reference(tp_group):
    """The gate's output stays in the weight dtype and matches `F.linear`.

    Without the `out_dtype` clearing, upstream's fallback tier would cast the
    router logits to fp32. Runs entirely on CPU.
    """
    hidden, experts = 128, 8
    torch.manual_seed(0)
    layer = _make_gate(hidden=hidden, experts=experts, return_bias=False)
    layer.weight.data.normal_(std=0.02)

    torch.manual_seed(1)
    x = torch.randn(4, hidden, dtype=torch.float16)
    # Capture the reference BEFORE process_weights_after_loading transposes.
    expected = F.linear(x, layer.weight.data)

    layer.quant_method.process_weights_after_loading(layer)
    assert layer.weight.shape == (hidden, experts)  # stored transposed [in, out]

    out = layer(x)

    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Numeric correctness on-card
# ---------------------------------------------------------------------------


@pytest.mark.mlp
@pytest.mark.parametrize("num_tokens", [1, 8])
@pytest.mark.parametrize("mode", ["compile", "eager"])
def test_router_gemm_matches_reference_on_spyre(tp_group, num_tokens, mode):
    """The router GEMM on-card matches `F.linear` and stays fp16.

    A router-logits row spans `num_experts` fp16 values; unless that is a
    whole stick (64) the row is not addressable on device, so the expert count
    here is stick-aligned like production MoE models.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    hidden, experts = 512, 64
    torch.manual_seed(0)
    layer = _make_gate(hidden=hidden, experts=experts, return_bias=False)
    layer.weight.data.normal_(std=0.02)

    torch.manual_seed(1)
    x = torch.randn(num_tokens, hidden, dtype=torch.float16)
    expected = F.linear(x, layer.weight.data)

    layer.quant_method.process_weights_after_loading(layer)
    layer = layer.to("spyre")

    def fn(x):
        return layer(x)

    if mode == "compile":
        torch._dynamo.reset()
        fn = torch.compile(fn, dynamic=False, backend="inductor")

    actual = fn(x.to("spyre"))

    assert actual.dtype == torch.float16
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)
