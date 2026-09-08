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

"""The Gemma-4 recipe over the generic Spyre MoE backend.

Both forms compute the same function by different means, so one dense reference covers
both. They need the card; shapes are scaled down but every dim stays stick-aligned,
which is what the layout tricks in those regions depend on.
"""

import pytest
import torch
import torch.nn.functional as F

EXPERTS, HIDDEN, INTER, TOP_K = 16, 256, 128, 4


def _dense_reference(x, probs, gate, up, down, scale, top_k):
    """Top-k MoE evaluated one (token, expert) pair at a time, in float32."""
    weights, indices = torch.topk(probs.float(), top_k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    out = torch.zeros_like(x, dtype=torch.float32)
    for token in range(x.shape[0]):
        row = x[token : token + 1].float()
        for slot in range(top_k):
            expert = int(indices[token, slot])
            gated = F.gelu(row @ gate[expert].float(), approximate="tanh")
            hidden = gated * (row @ up[expert].float())
            out[token] += (
                (hidden @ down[expert].float()).squeeze(0)
                * float(weights[token, slot])
                * float(scale[expert])
            )
    return out


@pytest.fixture(scope="module")
def moe_weights():
    """Random expert stacks in the device layout, plus their host copies."""
    from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre

    torch.manual_seed(0)
    host = {
        "gate": torch.randn(EXPERTS, HIDDEN, INTER, dtype=torch.float16) * 0.05,
        "up": torch.randn(EXPERTS, HIDDEN, INTER, dtype=torch.float16) * 0.05,
        "down": torch.randn(EXPERTS, INTER, HIDDEN, dtype=torch.float16) * 0.05,
    }
    host["scale"] = torch.rand(EXPERTS, dtype=torch.float16) + 0.5
    stacks = {
        "gate": host["gate"],
        "up": host["up"],
        "down": host["down"] * host["scale"].view(EXPERTS, 1, 1),
    }
    device = {k: dma_moe_expert_weight_to_spyre(v) for k, v in stacks.items()}
    assert all(v is not None for v in device.values()), "expert stacks must take the MoE layout"
    return host, device


def _inputs(num_tokens):
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.float16) * 0.5
    logits = torch.randn(num_tokens, EXPERTS, dtype=torch.float16)
    return x, logits


def test_standard_recipe_matches_upstream_topk_softmax():
    """The generic recipe retains vLLM's standard top-k-softmax semantics."""
    from spyre_inference.moe import _routing_weights

    logits = torch.tensor([[0.5, -1.0, 2.0, 1.5]], dtype=torch.float32)
    actual, indices = _routing_weights(logits, 2, "topk_softmax")
    selected, expected_indices = torch.topk(logits, 2, dim=-1)
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(actual, torch.softmax(selected, dim=-1))


def test_gemma_recipe_uses_full_softmax_before_topk():
    """Gemma's recipe stays model-specific rather than becoming a backend default."""
    from spyre_inference.moe import _routing_weights

    logits = torch.tensor([[0.5, -1.0, 2.0, 1.5]], dtype=torch.float32)
    actual, indices = _routing_weights(logits, 2, "full_softmax")
    probs = torch.softmax(logits, dim=-1)
    expected, expected_indices = torch.topk(probs, 2, dim=-1)
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(actual, expected / expected.sum(-1, keepdim=True))


def test_recipe_rejects_unbound_full_softmax_routing():
    """Only model adapters may request a non-standard upstream routing recipe."""
    from types import SimpleNamespace

    from spyre_inference.moe import SpyreMoERecipe, _validate_recipe

    layer = SimpleNamespace(custom_routing_function=None)
    with pytest.raises(NotImplementedError, match="model-specific"):
        _validate_recipe(layer, SpyreMoERecipe("gelu_tanh", "full_softmax"))


def test_recipe_rejects_unfolded_expert_scale():
    """The current persistent kernel cannot carry per-expert scale tensors."""
    from types import SimpleNamespace

    from spyre_inference.moe import SpyreMoERecipe, _validate_recipe

    layer = SimpleNamespace(custom_routing_function=object())
    with pytest.raises(NotImplementedError, match="fold"):
        _validate_recipe(
            layer,
            SpyreMoERecipe("gelu_tanh", "full_softmax", torch.ones(EXPERTS)),
        )


def test_gathered_matches_dense_reference(moe_weights):
    """The decode form, at the single token whose combine has a legal device layout."""
    from torch_spyre._C import get_elem_in_stick
    from torch_spyre._inductor import config as spyre_config

    from spyre_inference.moe import _moe_gathered

    host, device = moe_weights
    x, logits = _inputs(1)
    stick = get_elem_in_stick(torch.float16)

    region = torch.compile(_moe_gathered, backend="inductor", fullgraph=True, dynamic=False)
    with spyre_config.patch({"frontend_pool_allocation": True}):
        actual = region(
            x.to("spyre"),
            logits.to("spyre"),
            device["gate"],
            device["up"],
            device["down"],
            TOP_K,
            stick,
            "full_softmax",
            "gelu_tanh",
        )

    expected = _dense_reference(
        x,
        torch.softmax(logits, dim=-1),
        host["gate"],
        host["up"],
        host["down"],
        host["scale"],
        TOP_K,
    )
    torch.testing.assert_close(actual.cpu().float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [24, 32])
def test_persistent_matches_dense_reference(moe_weights, num_tokens):
    """The prefill form, in the region sequence ``apply_monolithic`` uses.

    24 tokens does not divide the core count, which the work-division hint has to cope
    with.
    """
    from torch_spyre._C import get_elem_in_stick
    from torch_spyre._inductor import config as spyre_config
    from torch_spyre._inductor.wsr.propagate_named_dims import reset as reset_named_dims

    from spyre_inference.moe import (
        _moe_persistent,
        _moe_persistent_routing,
        _name_persistent_dims,
        _probs,
    )

    host, device = moe_weights
    x, logits = _inputs(num_tokens)
    x_dev = x.to("spyre")
    stick = get_elem_in_stick(torch.float16)
    identity = torch.eye(stick, dtype=torch.float16).to("spyre")

    routing = torch.compile(
        _moe_persistent_routing, backend="inductor", fullgraph=True, dynamic=False
    )
    probs = torch.compile(_probs, backend="inductor", fullgraph=True, dynamic=False)
    experts = torch.compile(_moe_persistent, backend="inductor", fullgraph=True, dynamic=False)

    with spyre_config.patch({"frontend_pool_allocation": True}):
        route = routing(probs(logits.to("spyre")), identity, TOP_K, stick)
        _name_persistent_dims(x_dev, device["gate"], device["up"], device["down"])
        try:
            with spyre_config.patch({"allow_all_ops_in_lx_planning": True}):
                actual = experts(
                    x_dev, route, device["gate"], device["up"], device["down"], "gelu_tanh"
                )
        finally:
            reset_named_dims()

    expected = _dense_reference(
        x,
        torch.softmax(logits, dim=-1),
        host["gate"],
        host["up"],
        host["down"],
        host["scale"],
        TOP_K,
    )
    torch.testing.assert_close(actual.cpu().float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("tokens", "expected"), [(1, 1), (16, 16), (24, 24), (32, 32), (40, 20), (64, 32)]
)
def test_token_cores_divides_the_token_axis(tokens, expected):
    """The token work-division split must divide the axis and not exceed the cores."""
    from torch_spyre._inductor import config as spyre_config

    from spyre_inference.moe import _token_cores

    if spyre_config.sencores != 32:
        pytest.skip(f"expectations assume SENCORES=32, got {spyre_config.sencores}")
    assert _token_cores(tokens) == expected


def test_relayout_splits_and_transposes_the_generic_expert_stacks():
    """The generic backend preserves model-specific scaling outside its weights."""

    import torch.nn as nn
    from torch_spyre._C import get_elem_in_stick

    from spyre_inference.moe import SpyreMoERecipe, _prepare_layer

    class _MoEConfig:
        tp_size = 1
        ep_size = 1
        dp_size = 1
        pcp_size = 1
        sp_size = 1

    class _RoutedExperts(nn.Module):
        """Stands in for vLLM's, which needs a whole FusedMoEConfig to build."""

        def __init__(self, w13, w2):
            super().__init__()
            self.w13_weight = nn.Parameter(w13, requires_grad=False)
            self.w2_weight = nn.Parameter(w2, requires_grad=False)
            self.moe_config = _MoEConfig()

    torch.manual_seed(0)
    w13 = torch.randn(EXPERTS, 2 * INTER, HIDDEN, dtype=torch.float16) * 0.05
    w2 = torch.randn(EXPERTS, HIDDEN, INTER, dtype=torch.float16) * 0.05
    scale = torch.rand(EXPERTS, dtype=torch.float16) + 0.5
    layer = _RoutedExperts(w13.clone(), w2.clone())
    layer.spyre_moe_recipe = SpyreMoERecipe(
        "gelu_tanh", "full_softmax", scale, fold_expert_scale_into_down=True
    )

    _prepare_layer(layer)

    assert not hasattr(layer, "w13_weight"), "the fused stacks must be freed, not kept"
    assert not hasattr(layer, "w2_weight")
    assert layer.spyre_moe_gate.shape == (EXPERTS, HIDDEN, INTER)
    assert layer.spyre_moe_up.shape == (EXPERTS, HIDDEN, INTER)
    assert layer.spyre_moe_down.shape == (EXPERTS, INTER, HIDDEN)
    assert layer.spyre_moe_route_identity.shape == (layer.spyre_moe_stick, layer.spyre_moe_stick)
    # Both follow the stacks' dtype, not a literal: a stick's element count changes with
    # it, and the identity multiplies routing weights that arrive in that same dtype.
    assert layer.spyre_moe_stick == get_elem_in_stick(w13.dtype)
    assert layer.spyre_moe_route_identity.dtype == w13.dtype

    close = {"atol": 1e-4, "rtol": 1e-2}
    torch.testing.assert_close(
        layer.spyre_moe_gate.cpu(), w13[:, :INTER, :].transpose(1, 2), **close
    )
    torch.testing.assert_close(layer.spyre_moe_up.cpu(), w13[:, INTER:, :].transpose(1, 2), **close)
    torch.testing.assert_close(
        layer.spyre_moe_down.cpu(), (w2 * scale.view(EXPERTS, 1, 1)).transpose(1, 2), **close
    )
    assert layer.spyre_moe_expert_scale is None
