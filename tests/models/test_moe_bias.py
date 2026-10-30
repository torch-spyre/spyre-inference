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

"""Expert biases on the generic Spyre MoE backend.

A biased layer carries ``w13_bias`` [E, 2M] (gate half, then up) and ``w2_bias`` [E, H]. The
relayout places them next to the weight pools as [E, 1, F] stacks, and every expert form adds
them: gate/up after their matmuls, down after the down matmul, the down bias weighted by
routing like the rest of the expert output. The reference follows the recipe's activation and
routing, so a missing, misplaced or unpadded bias shows up as a value error. The hidden width
is 45 sticks, so the gathered form splits the down bias into chunks; the intermediate width is
stick-aligned at TP1 and lands mid-stick at TP4, so the biases are zero-padded there.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

EXPERTS, HIDDEN, TOP_K = 16, 2880, 4


def _reference(
    x, logits, w13, w13_bias, w2, w2_bias, top_k, activation="silu", routing="full_softmax"
):
    """Biased MoE in float32 on vLLM's parameter layout: ``w13`` [E, 2M, H] as
    ``[gate | up]``, ``w13_bias`` [E, 2M], ``w2`` [E, H, M], ``w2_bias`` [E, H]."""
    if routing == "full_softmax":
        weights, indices = torch.topk(torch.softmax(logits.float(), dim=-1), top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
    else:
        selected, indices = torch.topk(logits.float(), top_k, dim=-1)
        weights = torch.softmax(selected, dim=-1)
    act = partial(F.gelu, approximate="tanh") if activation == "gelu_tanh" else F.silu
    inter = w2.shape[-1]
    out = torch.zeros(x.shape, dtype=torch.float32)
    for token in range(x.shape[0]):
        row = x[token].float()
        for slot in range(top_k):
            e = int(indices[token, slot])
            gate_up = w13[e].float() @ row + w13_bias[e].float()
            activated = act(gate_up[:inter]) * gate_up[inter:]
            hidden = w2[e].float() @ activated + w2_bias[e].float()
            out[token] += hidden * float(weights[token, slot])
    return out


def _checkpoint_like(inter, seed=0, experts=EXPERTS):
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape, scale):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.float16)

    return {
        "w13": randn(experts, 2 * inter, HIDDEN, scale=0.02),
        "w13_bias": randn(experts, 2 * inter, scale=0.5),
        "w2": randn(experts, HIDDEN, inter, scale=0.02),
        "w2_bias": randn(experts, HIDDEN, scale=0.5),
    }


# Full-softmax routing, as a biased model adapter requests it (with its own routing function).
_RECIPE = ("silu", "full_softmax")
# The persistent forms also run the default top-k routing and Gemma-4's GELU with biases.
_RECIPES = [
    pytest.param(_RECIPE, id="silu-full"),
    pytest.param(("silu", "topk_softmax"), id="silu-topk"),
    pytest.param(("gelu_tanh", "full_softmax"), id="gelu-full"),
]


def _routing(hidden_states, gating_output, topk, renormalize):
    weights, ids = torch.topk(torch.softmax(gating_output.float(), dim=-1), topk, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    return weights.to(gating_output.dtype), ids.to(torch.int32)


def _biased_layer():
    """An upstream ``RoutedExperts`` with expert biases, as the backend would claim it."""
    from spyre_inference.moe import SpyreUnquantizedFusedMoEMethod

    moe = {
        "num_logical_experts": EXPERTS,
        "num_experts": EXPERTS,
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 1,
        "pcp_size": 1,
        "sp_size": 1,
        "is_lora_enabled": False,
        "has_bias": True,
    }
    return SimpleNamespace(
        moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(enable_eplb=False), **moe),
        custom_routing_function=_routing,
        activation=SimpleNamespace(value="silu"),
        global_num_experts=EXPERTS,
        local_num_experts=EXPERTS,
        renormalize=True,
        apply_router_weight_on_input=False,
        quant_config=None,
        quant_method=object.__new__(SpyreUnquantizedFusedMoEMethod),
    )


def test_configure_accepts_a_biased_layer():
    from spyre_inference.moe import SpyreMoERecipe, configure_spyre_moe_layer

    layer = _biased_layer()
    configure_spyre_moe_layer(layer, SpyreMoERecipe(*_RECIPE))
    assert layer.spyre_moe_regions == {}


class _RoutedExperts(torch.nn.Module):
    """Stands in for vLLM's, which needs a whole FusedMoEConfig to build."""

    def __init__(self, params, has_bias=True):
        super().__init__()
        for key, value in params.items():
            if has_bias or not key.endswith("_bias"):
                name = key if key.endswith("_bias") else f"{key}_weight"
                param = torch.nn.Parameter(value.clone(), requires_grad=False)
                self.register_parameter(name, param)
        self.top_k = TOP_K


def _prepared(inter, monkeypatch, has_bias=True, experts=EXPERTS, recipe=_RECIPE):
    from spyre_inference.moe import SpyreMoERecipe, _prepare_layer

    monkeypatch.delenv("SPYRE_MOE_CHUNKS", raising=False)
    params = _checkpoint_like(inter, experts=experts)
    layer = _RoutedExperts(params, has_bias=has_bias)
    layer.spyre_moe_recipe = SpyreMoERecipe(*recipe)
    _prepare_layer(layer)
    return layer, params


# Stick-aligned at TP1; mid-stick at TP4 (720 -> 768 padded) and TP8 (360 -> 384 padded).
_INTERS = [pytest.param(2880, id="tp1"), pytest.param(720, id="tp4"), pytest.param(360, id="tp8")]


@pytest.mark.parametrize("inter", _INTERS)
def test_relayout_places_the_biases_next_to_the_pools(inter, monkeypatch):
    """Gate/up biases split like ``w13``, zero in the padded lanes; the down bias whole and
    in the gathered form's chunks. The host biases stay, so upstream's quant config can
    still read them."""
    from torch_spyre._C import get_elem_in_stick

    layer, p = _prepared(inter, monkeypatch)
    stick = get_elem_in_stick(torch.float16)
    width = inter + -inter % stick
    torch.testing.assert_close(layer.w13_bias.detach(), p["w13_bias"])
    torch.testing.assert_close(layer.w2_bias.detach(), p["w2_bias"])

    close = {"atol": 1e-4, "rtol": 1e-2}
    b = layer.spyre_moe_biases
    gate_b, up_b = b.gate.cpu(), b.up.cpu()
    assert gate_b.shape == up_b.shape == (EXPERTS, 1, width)
    torch.testing.assert_close(gate_b[:, 0, :inter], p["w13_bias"][:, :inter], **close)
    torch.testing.assert_close(up_b[:, 0, :inter], p["w13_bias"][:, inter:], **close)
    torch.testing.assert_close(b.down.cpu()[:, 0], p["w2_bias"], **close)
    chunks = layer.spyre_moe_chunks
    torch.testing.assert_close(
        b.down_chunks.cpu(), p["w2_bias"].reshape(EXPERTS * chunks, 1, HIDDEN // chunks), **close
    )
    for padded in (gate_b[..., inter:], up_b[..., inter:]):
        assert not padded.count_nonzero()


def test_relayout_without_biases_keeps_none(monkeypatch):
    layer, _ = _prepared(720, monkeypatch, has_bias=False)
    assert layer.spyre_moe_biases is None


@pytest.mark.parametrize("kind", ["hidden", "experts"])
def test_bad_bias_shape_is_rejected_before_relayout(kind, monkeypatch):
    """The check uses the weight shapes, and it runs before any pool is copied to the device.

    A wrong expert count on both biases used to match them against each other and pass.
    """
    from spyre_inference import moe as moe_module
    from spyre_inference.moe import SpyreMoERecipe

    copied = []
    monkeypatch.setattr(
        moe_module,
        "_to_spyre_expert_weight",
        lambda *args, **kwargs: copied.append(1),
    )
    params = _checkpoint_like(720)
    if kind == "hidden":
        params["w2_bias"] = params["w2_bias"][:, :-1]
    else:
        params["w13_bias"] = params["w13_bias"][: EXPERTS // 2]
        params["w2_bias"] = params["w2_bias"][: EXPERTS // 2]
    layer = _RoutedExperts(params)
    layer.spyre_moe_recipe = SpyreMoERecipe(*_RECIPE)
    with pytest.raises(ValueError, match="w2_bias"):
        moe_module._prepare_layer(layer)
    assert copied == []


def _inputs(tokens, seed, experts=EXPERTS):
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(tokens, HIDDEN, generator=gen) * 0.5).to(torch.float16)
    logits = torch.randn(tokens, experts, generator=gen)
    # Keep the top-k boundary clear of fp16 near-ties, which the card may break either way.
    order = torch.sort(logits, dim=-1, descending=True).indices
    lift = torch.zeros_like(logits).scatter_(-1, order[:, :TOP_K], 0.25)
    return x, (logits + lift).to(torch.float16)


@pytest.mark.parametrize("inter", _INTERS)
def test_gathered_matches_the_biased_reference(inter, monkeypatch):
    """One decode token through the gathered region the dispatcher picks for it."""
    from contextlib import nullcontext

    from spyre_inference import moe as moe_module

    layer, p = _prepared(inter, monkeypatch)
    layer.spyre_moe_regions = {}
    monkeypatch.setattr(moe_module, "_compiler_scopes", lambda: (nullcontext(), nullcontext()))
    x, logits = _inputs(1, seed=inter)
    method = object.__new__(moe_module.SpyreUnquantizedFusedMoEMethod)
    actual = method.apply_monolithic(layer, x.to("spyre"), logits.to("spyre"))
    assert set(layer.spyre_moe_regions) == {"gathered"}
    expected = _reference(x, logits, p["w13"], p["w13_bias"], p["w2"], p["w2_bias"], TOP_K)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("inter", _INTERS)
@pytest.mark.parametrize("tokens", [2, 8])
def test_gathered_batch_matches_the_biased_reference(inter, tokens, monkeypatch):
    """A stick-addressable batch of 2-8 tokens through the gathered_batch region.

    Row ``t`` of the router logits starts at ``t * experts``. That stride has to be a whole
    number of sticks or the dispatcher refuses the gathered batch and takes the all-expert
    path. Sixteen experts do not fill an fp16 stick, so this case uses one stick of experts.
    """
    from contextlib import nullcontext

    from torch_spyre._C import get_elem_in_stick

    from spyre_inference import moe as moe_module

    experts = get_elem_in_stick(torch.float16)
    layer, p = _prepared(inter, monkeypatch, experts=experts)
    layer.spyre_moe_regions = {}
    monkeypatch.setattr(moe_module, "_compiler_scopes", lambda: (nullcontext(), nullcontext()))
    x, logits = _inputs(tokens, seed=tokens + inter, experts=experts)
    x_dev, logits_dev = x.to("spyre"), logits.to("spyre")
    stick = get_elem_in_stick(torch.float16)
    assert all(
        t.storage_offset() % stick == 0 and t.stride(0) % stick == 0 for t in (x_dev, logits_dev)
    )
    method = object.__new__(moe_module.SpyreUnquantizedFusedMoEMethod)
    actual = method.apply_monolithic(layer, x_dev, logits_dev)
    assert set(layer.spyre_moe_regions) == {"gathered_batch"}
    expected = _reference(x, logits, p["w13"], p["w13_bias"], p["w2"], p["w2_bias"], TOP_K)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("inter", _INTERS)
@pytest.mark.parametrize("tokens", [16, 64])
@pytest.mark.parametrize("recipe", _RECIPES)
def test_persistent_matches_the_biased_reference(inter, tokens, recipe, monkeypatch):
    """A prefill-sized batch through the all-expert regions, as ``apply_monolithic`` runs them.

    Top-k routing takes the ``topk_probs`` / ``route_selected`` regions instead of
    ``probs`` / ``route``.
    """
    from spyre_inference import moe as moe_module

    layer, p = _prepared(inter, monkeypatch, recipe=recipe)
    layer.spyre_moe_regions = {}
    x, logits = _inputs(tokens, seed=tokens)
    method = object.__new__(moe_module.SpyreUnquantizedFusedMoEMethod)
    actual = method.apply_monolithic(layer, x.to("spyre"), logits.to("spyre"))
    assert "experts" in layer.spyre_moe_regions
    if recipe[1] == "topk_softmax":
        assert "route_selected" in layer.spyre_moe_regions
    expected = _reference(x, logits, p["w13"], p["w13_bias"], p["w2"], p["w2_bias"], TOP_K, *recipe)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("inter", [pytest.param(2880, id="tp1"), pytest.param(720, id="tp4")])
@pytest.mark.parametrize("recipe", _RECIPES)
def test_traced_dispatch_matches_the_biased_reference(inter, recipe, monkeypatch):
    """Traced into one graph with in-graph producers, as a compiled block runs one stick of
    tokens: the in-graph persistent form, biases included.

    ``tp1`` is stick-aligned (no intermediate pad); ``tp4`` pads mid-stick.
    """
    import warnings

    from torch_spyre._C import get_elem_in_stick
    from torch_spyre.ops.fallbacks import FallbackWarning

    from spyre_inference import moe as moe_module

    layer, p = _prepared(inter, monkeypatch, recipe=recipe)
    tokens = get_elem_in_stick(torch.float16)
    x, logits = _inputs(tokens, seed=1)

    def block(x, logits):
        return moe_module.SpyreUnquantizedFusedMoEMethod.apply_monolithic(
            None, layer, x * 1.0, logits * 1.0
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FallbackWarning)
        compiled = torch.compile(block, backend="inductor", fullgraph=True, dynamic=False)
        actual = compiled(x.to("spyre"), logits.to("spyre"))
    fallbacks = [str(w.message) for w in caught if issubclass(w.category, FallbackWarning)]
    assert not fallbacks, f"the traced dispatch fell back to CPU: {fallbacks}"
    expected = _reference(x, logits, p["w13"], p["w13_bias"], p["w2"], p["w2_bias"], TOP_K, *recipe)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=3e-2, rtol=3e-2)
