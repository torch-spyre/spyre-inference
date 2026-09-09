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

The two expert forms compute the same function, so one dense reference covers both. The
tests that run them, and the relayout test, need the card: the shapes are scaled down, but
every dim stays stick-aligned because the layouts in those regions depend on it. Routing,
configuration and dispatch are host-side and need nothing.
"""

from types import SimpleNamespace

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


def test_routing_recipes_agree_under_renormalization():
    """Each recipe is pinned to its own formula, and to agreeing with the other.

    The backend only claims ``renormalize=True`` layers, and there both spellings are
    ``exp(logit)`` over the sum across the selected set. They differ only in the graph.
    """
    from spyre_inference.moe import _routing_weights

    logits = torch.tensor([[0.5, -1.0, 2.0, 1.5]], dtype=torch.float32)

    standard, standard_indices = _routing_weights(logits, 2, "topk_softmax")
    selected, expected_indices = torch.topk(logits, 2, dim=-1)
    torch.testing.assert_close(standard_indices, expected_indices)
    torch.testing.assert_close(standard, torch.softmax(selected, dim=-1))

    gemma, gemma_indices = _routing_weights(logits, 2, "full_softmax")
    top_probs, expected_indices = torch.topk(torch.softmax(logits, dim=-1), 2, dim=-1)
    torch.testing.assert_close(gemma_indices, expected_indices)
    torch.testing.assert_close(gemma, top_probs / top_probs.sum(-1, keepdim=True))

    torch.testing.assert_close(standard_indices, gemma_indices)
    torch.testing.assert_close(standard, gemma)


def test_prefill_routing_matches_the_decode_form():
    """The prefill dense form and the decode gathered form must agree on the weights."""
    from spyre_inference.moe import _routing_weights, _topk_probs

    logits = torch.tensor([[0.5, -1.0, 2.0, 1.5], [-0.25, 3.0, 0.75, -2.0]], dtype=torch.float32)
    dense = _topk_probs(logits, 2)
    weights, indices = _routing_weights(logits, 2, "topk_softmax")
    expected = torch.zeros_like(logits).scatter(-1, indices, weights)
    torch.testing.assert_close(dense, expected)
    # Exactly top_k live slots per token, already normalized, so the re-normalize inside
    # _moe_persistent_routing is the divide-by-one it is meant to be.
    assert (dense != 0).sum(-1).tolist() == [2, 2]
    torch.testing.assert_close(dense.sum(-1), torch.ones(logits.shape[0]))


def _generic_layer(*, moe_config=None, enable_eplb=False, **overrides):
    """An upstream ``RoutedExperts`` the backend would claim, with no model adapter."""
    moe = {
        "num_logical_experts": EXPERTS,
        "num_experts": EXPERTS,
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 1,
        "pcp_size": 1,
        "sp_size": 1,
    } | (moe_config or {})
    fields = {
        "custom_routing_function": None,
        "activation": SimpleNamespace(value="silu"),
        "global_num_experts": EXPERTS,
        "local_num_experts": EXPERTS,
        "renormalize": True,
        "apply_router_weight_on_input": False,
    } | overrides
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(enable_eplb=enable_eplb), **moe
        ),
        **fields,
    )


def test_post_load_claims_an_unconfigured_layer_and_warns(monkeypatch):
    """A Mixtral-shaped layer no adapter opted in is claimed by the standard recipe.

    The hook has to take it over and say so, rather than either erroring or going quiet.
    """
    from spyre_inference import moe as moe_module

    prepared, warned = [], []
    monkeypatch.setattr(moe_module, "_prepare_layer", prepared.append)
    monkeypatch.setattr(moe_module.logger, "warning_once", lambda msg, *a: warned.append(msg))
    method = object.__new__(moe_module.SpyreUnquantizedFusedMoEMethod)

    layer = _generic_layer()
    method.process_weights_after_loading(layer)
    recipe = layer.spyre_moe_recipe
    assert (recipe.routing, recipe.activation) == ("topk_softmax", "silu")
    assert recipe.expert_scale is None
    assert layer.spyre_moe_regions == {}
    assert prepared == [layer], "an unconfigured layer must still be relaid out"
    assert warned, "claiming a layer with no model adapter must warn"

    # A layer an adapter already configured keeps its recipe and stays quiet.
    warned.clear()
    adapted = _generic_layer()
    recipe = moe_module.SpyreMoERecipe("silu", "topk_softmax")
    adapted.spyre_moe_recipe = recipe
    method.process_weights_after_loading(adapted)
    assert adapted.spyre_moe_recipe is recipe
    assert not warned


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        # A DeepSeek-shaped layer needs an adapter rather than the generic recipe.
        ({"custom_routing_function": object()}, "model adapter"),
        # Only the two activations the expert regions implement may be claimed.
        ({"activation": SimpleNamespace(value="swigluoai")}, "activation="),
    ],
)
def test_default_recipe_rejects_layers_that_need_an_adapter(overrides, match):
    from spyre_inference.moe import _default_recipe

    with pytest.raises(NotImplementedError, match=match):
        _default_recipe(_generic_layer(**overrides))


_STANDARD = ("silu", "topk_softmax")


@pytest.mark.parametrize(
    ("overrides", "recipe_args", "match"),
    [
        # Every parallel axis but TP must be local: the regions hold whole expert stacks.
        ({"moe_config": {"ep_size": 2}}, _STANDARD, "local experts"),
        ({"moe_config": {"dp_size": 2}}, _STANDARD, "local experts"),
        ({"moe_config": {"pcp_size": 2}}, _STANDARD, "local experts"),
        ({"moe_config": {"sp_size": 2}}, _STANDARD, "local experts"),
        # Load balancing replicates experts the relayout would stack wrongly.
        ({"enable_eplb": True}, _STANDARD, "enable_eplb=True"),
        # The remaining upstream knobs the two Spyre forms cannot express.
        ({"local_num_experts": EXPERTS // 2}, _STANDARD, "remapped experts"),
        ({"renormalize": False}, _STANDARD, "normalized top-k"),
        ({"apply_router_weight_on_input": True}, _STANDARD, "input-weighted"),
        # The recipe's activation must be the one the layer actually asks for.
        ({}, ("gelu_tanh", "topk_softmax"), "requires activation="),
        # Only model adapters may request a non-standard upstream routing recipe.
        ({}, ("silu", "full_softmax"), "model-specific"),
        # The persistent kernel cannot carry a per-expert scale tensor beside the weights.
        ({}, ("silu", "topk_softmax", torch.ones(EXPERTS)), "fold"),
    ],
)
def test_configure_rejects_what_the_spyre_forms_cannot_express(overrides, recipe_args, match):
    """Everything unsupported has to be refused here, before the post-load hook frees w13/w2."""
    from spyre_inference.moe import SpyreMoERecipe, configure_spyre_moe_layer

    layer = _generic_layer(**overrides)
    with pytest.raises(NotImplementedError, match=match):
        configure_spyre_moe_layer(layer, SpyreMoERecipe(*recipe_args))


def test_configure_accepts_tensor_parallel_experts():
    """TP only narrows each expert's intermediate dim; MoERunner all-reduces the parts."""
    from spyre_inference.moe import SpyreMoERecipe, configure_spyre_moe_layer

    layer = _generic_layer(moe_config={"tp_size": 2})
    configure_spyre_moe_layer(layer, SpyreMoERecipe(*_STANDARD))
    assert layer.spyre_moe_regions == {}


def _dispatch_recorder(monkeypatch, fail_on=None):
    """Swap the compiled regions for recorders: dispatch order, without the device."""
    from contextlib import nullcontext

    from spyre_inference import moe as moe_module

    calls, resets = [], []

    def fake_region(layer, name, fn):
        def run(*args):
            calls.append((name, fn.__name__))
            if name == fail_on:
                raise RuntimeError("region blew up")
            return torch.zeros(1)

        return run

    monkeypatch.setattr(moe_module, "_region", fake_region)
    monkeypatch.setattr(moe_module, "_compiler_scopes", lambda: (nullcontext(), nullcontext()))
    monkeypatch.setattr(moe_module, "_name_persistent_dims", lambda *args: None)
    monkeypatch.setattr(moe_module, "_reset_named_dims", lambda: resets.append(1))
    return calls, resets


def _dispatch_layer(routing):
    from spyre_inference.moe import SpyreMoERecipe

    activation = "gelu_tanh" if routing == "full_softmax" else "silu"
    return SimpleNamespace(
        spyre_moe_recipe=SpyreMoERecipe(activation, routing),
        spyre_moe_gate=None,
        spyre_moe_up=None,
        spyre_moe_down=None,
        top_k=TOP_K,
    )


def _apply(layer, tokens):
    from spyre_inference.moe import SpyreUnquantizedFusedMoEMethod

    method = object.__new__(SpyreUnquantizedFusedMoEMethod)
    return method.apply_monolithic(layer, torch.zeros(tokens, HIDDEN), torch.zeros(tokens, EXPERTS))


def test_single_token_dispatches_to_the_gathered_form(monkeypatch):
    """One token takes the gathered region, whose combine is the only legal layout there."""
    calls, resets = _dispatch_recorder(monkeypatch)
    _apply(_dispatch_layer("full_softmax"), tokens=1)
    assert calls == [("gathered", "_gathered")]
    assert resets == [], "the gathered form declares no persistent dims to reset"


@pytest.mark.parametrize(
    ("routing", "probs_fn"), [("full_softmax", "_probs"), ("topk_softmax", "_topk_probs")]
)
def test_multi_token_dispatch_picks_the_recipe_routing(monkeypatch, routing, probs_fn):
    """The persistent form runs probs -> route -> experts, routing chosen by the recipe."""
    calls, resets = _dispatch_recorder(monkeypatch)
    _apply(_dispatch_layer(routing), tokens=8)
    assert calls == [("probs", probs_fn), ("route", "_route"), ("experts", "_experts")]
    assert resets == [1]


def test_named_dims_are_reset_when_a_region_raises(monkeypatch):
    """Unconditional reset: on an Inductor cache hit stale names leak into the next graph."""
    calls, resets = _dispatch_recorder(monkeypatch, fail_on="experts")
    with pytest.raises(RuntimeError, match="region blew up"):
        _apply(_dispatch_layer("topk_softmax"), tokens=8)
    assert resets == [1], "the reset must survive a failing region"


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

    24 tokens does not divide the core count, which the work-division hint has to cope with.
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


# One core; a split that is the whole axis; a divisor below the core count; and a
# core-count split at a longer axis.
@pytest.mark.parametrize("tokens", [1, 24, 40, 64])
def test_token_cores_is_the_largest_split_that_divides_the_token_axis(tokens):
    """The token work-division split must divide the axis and not exceed the cores."""
    from torch_spyre._inductor import config as spyre_config

    from spyre_inference.moe import _token_cores

    limit = min(tokens, spyre_config.sencores)
    cores = _token_cores(tokens)
    assert tokens % cores == 0, f"{cores} does not divide {tokens}"
    assert 1 <= cores <= limit
    assert all(tokens % larger for larger in range(cores + 1, limit + 1)), "not the largest split"


# A whole number of sticks, and a TP shard that lands mid-stick (704 // 2 = 352 for
# gemma-4-26B-A4B, scaled down here).
@pytest.mark.parametrize("inter", [INTER, INTER - 32])
def test_relayout_splits_and_transposes_the_generic_expert_stacks(inter):
    """The recipe's per-expert scale is folded into the down stack, not kept beside it."""

    import torch.nn as nn
    from torch_spyre._C import get_elem_in_stick

    from spyre_inference.moe import SpyreMoERecipe, _prepare_layer

    class _RoutedExperts(nn.Module):
        """Stands in for vLLM's, which needs a whole FusedMoEConfig to build."""

        def __init__(self, w13, w2):
            super().__init__()
            self.w13_weight = nn.Parameter(w13, requires_grad=False)
            self.w2_weight = nn.Parameter(w2, requires_grad=False)

    torch.manual_seed(0)
    w13 = torch.randn(EXPERTS, 2 * inter, HIDDEN, dtype=torch.float16) * 0.05
    w2 = torch.randn(EXPERTS, HIDDEN, inter, dtype=torch.float16) * 0.05
    scale = torch.rand(EXPERTS, dtype=torch.float16) + 0.5
    layer = _RoutedExperts(w13.clone(), w2.clone())
    layer.spyre_moe_recipe = SpyreMoERecipe(
        "gelu_tanh", "full_softmax", scale, fold_expert_scale_into_down=True
    )

    _prepare_layer(layer)

    stick = get_elem_in_stick(w13.dtype)
    width = inter + -inter % stick
    assert not hasattr(layer, "w13_weight"), "the fused stacks must be freed, not kept"
    assert not hasattr(layer, "w2_weight")
    assert layer.spyre_moe_gate.shape == (EXPERTS, HIDDEN, width)
    assert layer.spyre_moe_up.shape == (EXPERTS, HIDDEN, width)
    assert layer.spyre_moe_down.shape == (EXPERTS, width, HIDDEN)
    assert layer.spyre_moe_route_identity.shape == (layer.spyre_moe_stick, layer.spyre_moe_stick)
    # Both follow the stacks' dtype, not a literal: a stick's element count changes with
    # it, and the identity multiplies routing weights that arrive in that same dtype.
    assert layer.spyre_moe_stick == stick
    assert layer.spyre_moe_route_identity.dtype == w13.dtype

    close = {"atol": 1e-4, "rtol": 1e-2}
    gate, up = (t.cpu() for t in (layer.spyre_moe_gate, layer.spyre_moe_up))
    down = layer.spyre_moe_down.cpu()
    torch.testing.assert_close(gate[..., :inter], w13[:, :inter, :].transpose(1, 2), **close)
    torch.testing.assert_close(up[..., :inter], w13[:, inter:, :].transpose(1, 2), **close)
    torch.testing.assert_close(
        down[:, :inter], (w2 * scale.view(EXPERTS, 1, 1)).transpose(1, 2), **close
    )
    # The added lanes must be zero, which is what makes the widening inert.
    for padded in (gate[..., inter:], up[..., inter:], down[:, inter:]):
        assert not padded.count_nonzero()
