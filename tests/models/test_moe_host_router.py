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

"""The TEMPORARY host fp32 router (``host_router``), on the host.

The router runs on the CPU by design, so most of this needs no card. Whether Spyre can
take the router GEMM over is probed in
``tests/probes/test_spyre_fallback_probes.py::test_spyre_fp32_accum_matmul``.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import get_current_vllm_config

EXPERTS, HIDDEN, TOP_K = 64, 128, 4


@pytest.fixture()
def model_type(monkeypatch):
    """Set the model type of the session's vLLM config, which the host router keys on."""
    text_config = get_current_vllm_config().model_config.hf_text_config

    def set_type(name: str) -> None:
        monkeypatch.setattr(text_config, "model_type", name, raising=False)

    return set_type


@pytest.fixture()
def prepared(monkeypatch):
    """Run ``host_router._prepare_layer`` on routed experts and register them, behind a runner
    stand-in, under their name in the forward context, as vLLM registers MoE runners."""
    from unittest import mock

    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
        MoERunnerInterface,
    )

    from spyre_inference import host_router

    layers = get_forward_context().no_compile_layers

    def prepare(experts):
        host_router._prepare_layer(experts)
        runner = mock.NonCallableMagicMock(spec=MoERunnerInterface)
        runner.routed_experts = experts
        monkeypatch.setitem(layers, experts.layer_name, runner)
        return experts

    return prepare


def _gate():
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    # GLM keeps its gate in fp32 (params_dtype=torch.float32, out_dtype=torch.float32).
    return GateLinear(
        HIDDEN, EXPERTS, out_dtype=torch.float32, params_dtype=torch.float32, prefix="gate"
    )


def _experts(**overrides):
    fields = {
        "layer_name": "model.layers.0.mlp.experts",
        "top_k": TOP_K,
        "global_num_experts": EXPERTS,
        "use_grouped_topk": True,
        "num_expert_group": 1,
        "topk_group": 1,
        "renormalize": True,
        "scoring_func": "sigmoid",
        "routed_scaling_factor": 1.0,
        "custom_routing_function": None,
        # Trained to balance load, which pulls every expert's biased score to one value.
        "e_score_correction_bias": torch.randn(EXPERTS) * 0.1 + 9.0,
    }
    return SimpleNamespace(**(fields | overrides))


def _vllm_fp32_table(logits, bias, scale=1.0):
    from vllm.model_executor.layers.fused_moe.router.cpu_router import select_experts

    weights, ids = select_experts(
        logits,
        logits,
        top_k=TOP_K,
        use_grouped_topk=True,
        renormalize=True,
        topk_group=1,
        num_expert_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=scale,
        e_score_correction_bias=bias,
    )
    return torch.zeros_like(logits).scatter(1, ids.long(), weights)


@pytest.mark.parametrize(
    ("name", "host"), [("glm4_moe_lite", True), ("glm_moe_dsa", True), ("gemma4_text", False)]
)
def test_only_glm_gates_compute_on_the_host(tp_group, model_type, name, host):
    """GLM's gates become the host fp32 gate; every other model keeps the Spyre gate, even
    one that asks for fp32 logits as Gemma-4's does."""
    from spyre_inference.custom_ops.gate_linear import SpyreGateLinear
    from spyre_inference.host_router import HostFp32GateLinear

    model_type(name)
    gate = _gate()
    assert isinstance(gate, SpyreGateLinear)
    assert isinstance(gate, HostFp32GateLinear) is host
    assert gate.out_dtype == (torch.float32 if host else None)


def test_host_gate_returns_fp32_host_logits(tp_group, model_type):
    """The logits are the fp32 GEMM of the checkpoint weight, which the gate's post-load
    hook captures on the host."""
    model_type("glm_moe_dsa")
    torch.manual_seed(0)
    gate = _gate()
    gate.weight.data.normal_(std=0.1)
    weight = gate.weight.data.clone()
    with pytest.raises(AssertionError, match="capture_host_weight"):
        gate(torch.zeros(1, HIDDEN, dtype=torch.float16))
    gate.quant_method.process_weights_after_loading(gate)

    x = (torch.randn(6, HIDDEN) * 0.5).half()
    logits, bias = gate(x)
    assert bias is None
    assert logits.device.type == "cpu" and logits.dtype == torch.float32
    torch.testing.assert_close(logits, x.float() @ weight.t(), atol=0, rtol=0)


@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_host_routing_table_is_vllms_router_on_the_layer_config(tp_group, prepared, scale):
    """The table is vLLM's CPU router with the layer's own routing config, scale included."""
    from spyre_inference import host_router

    torch.manual_seed(0)
    experts = prepared(_experts(routed_scaling_factor=scale))
    logits = torch.randn(6, EXPERTS)
    table = host_router.host_routing_table(experts, logits, torch.empty(0, dtype=torch.float32))
    want = _vllm_fp32_table(logits, experts.e_score_correction_bias, scale)
    torch.testing.assert_close(table, want)
    assert (table > 0).sum(-1).eq(TOP_K).all()


def test_host_router_selects_from_fp32_logits_not_fp16_ones(tp_group, model_type, prepared):
    """The point of the workaround: an expert set decided below fp16 resolution of the
    logits. Expert 1 wins by about 1e-5 in fp32 and ties after rounding to fp16."""
    from spyre_inference import host_router

    model_type("glm_moe_dsa")
    gate = _gate()
    gate.weight.data.zero_()
    gate.weight.data[0, 0], gate.weight.data[1, 0] = 1.0, 1.0 + 2e-5
    gate.quant_method.process_weights_after_loading(gate)
    bias = torch.zeros(EXPERTS)
    bias[2:] = -1.0  # Only experts 0 and 1 compete for the single slot.
    experts = prepared(
        _experts(top_k=1, e_score_correction_bias=bias, layer_name="layers.1.experts")
    )

    x = torch.zeros(1, HIDDEN, dtype=torch.float16)
    x[0, 0] = 0.75
    logits, _ = gate(x)
    assert logits.half()[0, 0] == logits.half()[0, 1], "the tie must be below fp16 resolution"
    table = host_router.host_routing_table(experts, logits, x)
    assert table.argmax(-1).item() == 1


def test_host_routing_table_round_trips_through_the_gathered_routing():
    """The gathered form reads the routing table back as its top-k weights and experts."""
    from spyre_inference.moe import HOST_ROUTING_TABLE, _routing_weights

    torch.manual_seed(0)
    ids = torch.stack([torch.randperm(EXPERTS)[:TOP_K] for _ in range(5)])
    weights = torch.rand(5, TOP_K) + 0.1
    weights = weights / weights.sum(-1, keepdim=True)
    table = torch.zeros(5, EXPERTS).scatter(1, ids, weights)
    got_w, got_ids = _routing_weights(table, TOP_K, HOST_ROUTING_TABLE, table.dtype)
    assert torch.equal(got_ids.sort(-1).values, ids.sort(-1).values)
    torch.testing.assert_close(torch.zeros_like(table).scatter(1, got_ids, got_w), table)


def test_host_routing_table_lands_on_spyre(tp_group, prepared):
    """The table goes to the activation's Spyre device, in its dtype."""
    from spyre_testing_plugin.pytest_plugin import spyre_available

    from spyre_inference import host_router

    if not spyre_available():
        pytest.skip("Spyre device not available")
    torch.manual_seed(0)
    experts = prepared(_experts(layer_name="layers.2.experts"))
    logits = torch.randn(6, EXPERTS)
    like = torch.zeros(6, HIDDEN, dtype=torch.float16).to("spyre")
    table = host_router.host_routing_table(experts, logits, like)
    assert table.device.type == "spyre" and table.dtype == torch.float16
    want = _vllm_fp32_table(logits, experts.e_score_correction_bias)
    torch.testing.assert_close(table.cpu().float(), want, atol=1e-3, rtol=1e-3)


def test_host_router_takes_the_loaded_correction_bias(tp_group, model_type, monkeypatch):
    """configure_spyre_moe_layer runs before weight loading, so the router's fp32 bias must
    come from the MoE method's post-load hook, not the construction-time placeholder."""
    from spyre_inference import moe as moe_module

    model_type("glm_moe_dsa")
    layer = _experts(layer_name="layers.3.experts", spyre_moe_recipe=object())
    layer.spyre_moe_host_router = True
    layer._ensure_moe_quant_config_init = lambda: None
    monkeypatch.setattr(moe_module, "_prepare_layer", lambda layer: None)
    loaded = torch.randn(EXPERTS) + 9.0
    layer.e_score_correction_bias = loaded
    method = object.__new__(moe_module.SpyreUnquantizedFusedMoEMethod)
    method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.spyre_host_router.e_score_correction_bias, loaded)
