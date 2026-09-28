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
every dim the device sees stays stick-aligned, because the layouts in those regions depend
on it. The intermediate dim reaches that alignment the way a TP shard does — zero-widened
when it lands mid-stick — so the forms are exercised at both a native and a widened width.
Routing semantics, configuration and dispatch are host-side; only the promoted softmax's
lowering needs the card.
"""

import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

EXPERTS, HIDDEN, INTER, TOP_K = 16, 256, 128, 4
# gemma-4-26B-A4B's expert count: a whole number of fp16 sticks, so its routing promotes.
GEMMA4_EXPERTS = 128


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


# A whole number of sticks, and a TP shard that lands mid-stick with the same remainder
# gemma-4-26B-A4B leaves at TP=2 (704 // 2 = 352).
@pytest.fixture(scope="module", params=[INTER, INTER - 32], ids=["native", "widened"])
def moe_weights(request):
    """Random expert stacks in the device layout, plus their host copies.

    ``_prepare_layer`` zero-widens an intermediate dim that does not span whole sticks, so
    the device stacks can be wider than the host copies the reference is computed from. The
    added lanes must not reach the result: that inertness is what lets TP narrow ``M``.
    """
    from torch_spyre._C import get_elem_in_stick
    from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre

    inter = request.param
    pad = -inter % get_elem_in_stick(torch.float16)
    torch.manual_seed(0)
    host = {
        "gate": torch.randn(EXPERTS, HIDDEN, inter, dtype=torch.float16) * 0.05,
        "up": torch.randn(EXPERTS, HIDDEN, inter, dtype=torch.float16) * 0.05,
        "down": torch.randn(EXPERTS, inter, HIDDEN, dtype=torch.float16) * 0.05,
    }
    host["scale"] = torch.rand(EXPERTS, dtype=torch.float16) + 0.5
    stacks = {
        "gate": host["gate"],
        "up": host["up"],
        "down": host["down"] * host["scale"].view(EXPERTS, 1, 1),
    }
    if pad:
        # The same widening, on the same axes, that ``_prepare_layer`` applies.
        stacks["gate"] = F.pad(stacks["gate"], (0, pad))
        stacks["up"] = F.pad(stacks["up"], (0, pad))
        stacks["down"] = F.pad(stacks["down"], (0, 0, 0, pad))
    device = {k: dma_moe_expert_weight_to_spyre(v) for k, v in stacks.items()}
    assert all(v is not None for v in device.values()), "expert stacks must take the MoE layout"
    return host, device


def _inputs(num_tokens):
    gen = torch.Generator().manual_seed(num_tokens)
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.float16, generator=gen) * 0.5
    logits = torch.randn(num_tokens, EXPERTS, dtype=torch.float16, generator=gen)
    return x, logits


def test_routing_recipes_agree_under_renormalization():
    """Each recipe is pinned to its own formula, and to agreeing with the other.

    The backend only claims ``renormalize=True`` layers, and there both spellings are
    ``exp(logit)`` over the sum across the selected set. They differ only in the graph.
    """
    from spyre_inference.moe import _routing_weights

    logits = torch.tensor([[0.5, -1.0, 2.0, 1.5]], dtype=torch.float32)

    standard, standard_indices = _routing_weights(logits, 2, "topk_softmax", logits.dtype)
    selected, expected_indices = torch.topk(logits, 2, dim=-1)
    torch.testing.assert_close(standard_indices, expected_indices)
    torch.testing.assert_close(standard, torch.softmax(selected, dim=-1))

    gemma, gemma_indices = _routing_weights(logits, 2, "full_softmax", logits.dtype)
    top_probs, expected_indices = torch.topk(torch.softmax(logits, dim=-1), 2, dim=-1)
    torch.testing.assert_close(gemma_indices, expected_indices)
    torch.testing.assert_close(gemma, top_probs / top_probs.sum(-1, keepdim=True))

    torch.testing.assert_close(standard_indices, gemma_indices)
    torch.testing.assert_close(standard, gemma)


def test_routing_promotes_only_across_whole_transport_dtype_sticks(monkeypatch):
    """The gate is the TRANSPORT dtype's stick, not fp32's: it is the cast's source that
    has to be unpadded. The two rules differ, and getting it wrong is not caught by the
    aligned/unaligned pair alone -- a count that spans whole fp32 sticks but a partial
    fp16 one would promote and then fail to lower at one token, so it is asserted here.
    """
    from torch_spyre._C import get_elem_in_stick

    from spyre_inference import moe as moe_module

    warned = []
    monkeypatch.setattr(moe_module.logger, "warning_once", lambda msg, *args: warned.append(args))

    stick = get_elem_in_stick(torch.float16)
    assert moe_module._route_reduce_dtype(2 * stick, torch.float16) is torch.float32
    assert warned == [], "the promoted reduction is not a degraded path"
    assert moe_module._route_reduce_dtype(stick + 1, torch.float16) is torch.float16
    assert warned == [(stick + 1, torch.float16, stick)]

    # Whole fp32 sticks, partial fp16 stick: 96 experts on today's geometry. Measured on
    # a card, every such count (32/96/160/224) fails to lower at 1, 2 and 8 tokens, while
    # 64/128/192 lower at all three -- so the fp16 stick is the rule, at every shape.
    fp32_stick = get_elem_in_stick(torch.float32)
    misaligned = 3 * fp32_stick
    assert misaligned % fp32_stick == 0 and misaligned % stick != 0
    warned.clear()
    assert moe_module._route_reduce_dtype(misaligned, torch.float16) is torch.float16
    assert warned == [(misaligned, torch.float16, stick)]


def test_probs_reduce_in_the_given_dtype_and_return_the_transport_dtype():
    from spyre_inference.moe import _probs

    logits = torch.randn(4, GEMMA4_EXPERTS, dtype=torch.float16)
    promoted = _probs(logits, torch.float32)
    assert promoted.dtype == torch.float16
    torch.testing.assert_close(promoted, torch.softmax(logits.float(), dim=-1).half())
    torch.testing.assert_close(_probs(logits, torch.float16), torch.softmax(logits, dim=-1))


@pytest.mark.parametrize("tokens", [1, 8])
def test_promoted_routing_softmax_lowers_on_spyre(tokens):
    """``frontend_pool_allocation`` is the config ``apply_monolithic`` runs the region under;
    a fallback that only manifests there would otherwise escape. One token is the decode
    shape and 8 a prefill one; the promoted softmax has to lower at both.
    """
    from spyre_testing_plugin.pytest_plugin import spyre_available
    from torch_spyre._inductor import config as spyre_config
    from torch_spyre.ops.fallbacks import FallbackWarning

    from spyre_inference.moe import _probs

    if not spyre_available():
        pytest.skip("Spyre device not available")

    logits = torch.randn(tokens, GEMMA4_EXPERTS, dtype=torch.float16)
    region = torch.compile(_probs, backend="inductor", fullgraph=True, dynamic=False)
    with (
        spyre_config.patch({"frontend_pool_allocation": True}),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always", FallbackWarning)
        actual = region(logits.to("spyre"), torch.float32)

    fallbacks = [str(w.message) for w in caught if issubclass(w.category, FallbackWarning)]
    assert not fallbacks, f"the promoted routing softmax fell back to CPU: {fallbacks}"
    torch.testing.assert_close(
        actual.cpu(), torch.softmax(logits.float(), dim=-1).half(), atol=2e-3, rtol=2e-3
    )


def test_dense_topk_weights_are_softmax_over_the_selected_logits():
    """The dense prefill weights, against the formula rather than against the decode form.

    Both forms now share ``_routing_weights``, so they agree by construction; what still
    needs pinning is the formula that shared helper implements.
    """
    from spyre_inference.moe import _topk_probs

    logits = torch.tensor([[0.5, -1.0, 2.0, 1.5], [-0.25, 3.0, 0.75, -2.0]], dtype=torch.float32)
    dense = _topk_probs(logits, 2)
    expected_weights, indices = torch.topk(logits, 2, dim=-1)
    expected = torch.zeros_like(logits).scatter(
        -1, indices, torch.softmax(expected_weights, dim=-1)
    )
    torch.testing.assert_close(dense, expected)


def test_selected_routing_matches_the_general_routing_form():
    """The topk path skips the re-normalize the full-softmax path needs.

    That is only legal because these weights already have exactly ``top_k`` live slots
    summing to one, so the two forms must come out identical.
    """
    from torch_spyre._C import get_elem_in_stick

    from spyre_inference.moe import (
        _moe_persistent_routing,
        _moe_persistent_selected_routing,
        _topk_probs,
    )

    torch.manual_seed(0)
    logits = torch.randn(8, EXPERTS, dtype=torch.float16)
    dense = _topk_probs(logits, TOP_K)
    assert (dense != 0).sum(-1).tolist() == [TOP_K] * logits.shape[0]
    torch.testing.assert_close(dense.sum(-1), torch.ones(logits.shape[0], dtype=torch.float16))

    stick = get_elem_in_stick(torch.float16)
    identity = torch.eye(stick, dtype=torch.float16)
    torch.testing.assert_close(
        _moe_persistent_selected_routing(dense, identity, stick),
        _moe_persistent_routing(dense, identity, TOP_K, stick),
    )


def _unquantized_method():
    """What vLLM installs for an unquantized layer — the Spyre OOT subclass, here."""
    from spyre_inference.moe import SpyreUnquantizedFusedMoEMethod

    return object.__new__(SpyreUnquantizedFusedMoEMethod)


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
        "is_lora_enabled": False,
        "has_bias": False,
    } | (moe_config or {})
    fields = {
        "custom_routing_function": None,
        "activation": SimpleNamespace(value="silu"),
        "global_num_experts": EXPERTS,
        "local_num_experts": EXPERTS,
        "renormalize": True,
        "apply_router_weight_on_input": False,
        "quant_config": None,
        "quant_method": _unquantized_method(),
    } | overrides
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(enable_eplb=enable_eplb), **moe
        ),
        **fields,
    )


def test_post_load_rejects_an_unconfigured_layer(monkeypatch):
    """An architecture must opt in before the backend destroys source weights."""
    from spyre_inference import moe as moe_module

    prepared = []
    monkeypatch.setattr(moe_module, "_prepare_layer", prepared.append)
    method = object.__new__(moe_module.SpyreUnquantizedFusedMoEMethod)

    with pytest.raises(NotImplementedError, match="explicit model-specific recipe"):
        method.process_weights_after_loading(_generic_layer())
    assert prepared == []


def test_post_load_advertises_no_preprocessed_weight_support():
    """The destructive relayout cannot consume a vLLM preprocessed weight cache."""
    from spyre_inference.moe import SpyreUnquantizedFusedMoEMethod

    assert not SpyreUnquantizedFusedMoEMethod.supports_pre_processed_weights


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
        ({"quant_method": object()}, _STANDARD, "quantized experts"),
        ({"moe_config": {"is_lora_enabled": True}}, _STANDARD, "LoRA experts"),
        ({"moe_config": {"has_bias": True}}, _STANDARD, "expert biases"),
        # The recipe's activation must be the one the layer actually asks for.
        ({}, ("gelu_tanh", "topk_softmax"), "requires activation="),
        # Only model adapters may request a non-standard upstream routing recipe.
        ({}, ("silu", "full_softmax"), "model-specific"),
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


def test_configure_claims_an_unquantized_layer_of_a_quantized_model():
    """A quantized checkpoint hands its ignored MoE layers back unquantized; claim those."""
    from spyre_inference.moe import SpyreMoERecipe, configure_spyre_moe_layer

    layer = _generic_layer(quant_config=object())
    configure_spyre_moe_layer(layer, SpyreMoERecipe(*_STANDARD))
    assert layer.spyre_moe_recipe.routing == "topk_softmax"


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
            return torch.zeros(args[1].shape[0] if name == "gathered_batch" else 1)

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
        # Divides both widths ``_apply`` builds, so these tests hit the token bound, not the guard.
        spyre_moe_stick=16,
        spyre_moe_route_dtype=torch.float16,
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


def test_a_small_batch_uses_the_compiled_gathered_loop(monkeypatch):
    """Below the bound the packed batch is handled by one compiled gathered loop."""
    monkeypatch.setenv("SPYRE_MOE_GATHERED_MAX_TOKENS", "4")
    calls, resets = _dispatch_recorder(monkeypatch)
    out = _apply(_dispatch_layer("full_softmax"), tokens=3)
    assert calls == [("gathered_batch", "_gathered_tokens")]
    assert out.shape[0] == 3, "the per-token results must be reassembled into one batch"
    assert resets == [], "the gathered form declares no persistent dims to reset"


def test_a_batch_whose_rows_are_not_stick_addressable_takes_the_all_expert_form(monkeypatch):
    """An expert count narrower than a stick leaves rows unaddressable, so gathered is skipped."""
    monkeypatch.setenv("SPYRE_MOE_GATHERED_MAX_TOKENS", "4")
    calls, resets = _dispatch_recorder(monkeypatch)
    layer = _dispatch_layer("full_softmax")
    layer.spyre_moe_stick = 64
    _apply(layer, tokens=2)
    assert calls == [("probs", "_probs"), ("route", "_route"), ("experts", "_experts")]
    assert resets == [1]


def test_a_batch_at_an_unaddressable_storage_offset_takes_the_all_expert_form(monkeypatch):
    """Both widths span whole sticks here, so a width-only check would wrongly admit the batch."""
    from spyre_inference.moe import SpyreUnquantizedFusedMoEMethod

    monkeypatch.setenv("SPYRE_MOE_GATHERED_MAX_TOKENS", "4")
    calls, resets = _dispatch_recorder(monkeypatch)
    layer = _dispatch_layer("full_softmax")
    method = object.__new__(SpyreUnquantizedFusedMoEMethod)
    offset = layer.spyre_moe_stick // 2
    x = torch.zeros(2 * HIDDEN + offset)[offset:].view(2, HIDDEN)
    logits = torch.zeros(2 * EXPERTS + offset)[offset:].view(2, EXPERTS)
    assert x.shape[-1] % layer.spyre_moe_stick == 0, "the widths must be stick multiples"
    assert logits.shape[-1] % layer.spyre_moe_stick == 0, "the widths must be stick multiples"

    method.apply_monolithic(layer, x, logits)
    assert calls == [("probs", "_probs"), ("route", "_route"), ("experts", "_experts")]
    assert resets == [1]


def test_above_the_gathered_bound_the_all_expert_form_takes_the_batch(monkeypatch):
    """The bound is the seam: one token past it the whole batch goes all-expert."""
    monkeypatch.setenv("SPYRE_MOE_GATHERED_MAX_TOKENS", "2")
    calls, resets = _dispatch_recorder(monkeypatch)
    _apply(_dispatch_layer("full_softmax"), tokens=3)
    assert calls == [("probs", "_probs"), ("route", "_route"), ("experts", "_experts")]
    assert resets == [1]


@pytest.mark.parametrize(
    ("routing", "routing_fn", "route_fn"),
    [("full_softmax", "_probs", "_route"), ("topk_softmax", "_topk_probs", "_route_selected")],
)
def test_multi_token_dispatch_picks_the_recipe_routing(monkeypatch, routing, routing_fn, route_fn):
    """The persistent form runs routing -> route -> experts for the selected recipe."""
    calls, resets = _dispatch_recorder(monkeypatch)
    _apply(_dispatch_layer(routing), tokens=8)
    assert calls == [
        ("probs" if routing == "full_softmax" else "topk_probs", routing_fn),
        ("route" if routing == "full_softmax" else "route_selected", route_fn),
        ("experts", "_experts"),
    ]
    assert resets == [1]


def test_named_dims_are_reset_when_a_region_raises(monkeypatch):
    """Unconditional reset: on an Inductor cache hit stale names leak into the next graph."""
    calls, resets = _dispatch_recorder(monkeypatch, fail_on="experts")
    with pytest.raises(RuntimeError, match="region blew up"):
        _apply(_dispatch_layer("topk_softmax"), tokens=8)
    assert resets == [1], "the reset must survive a failing region"


def test_gathered_matches_dense_reference(moe_weights):
    """The decode form, at the single token whose combine has a legal device layout.

    ``EXPERTS`` does not span whole sticks, so routing reduces in the transport dtype.
    """
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
            logits.dtype,
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


# Row ``t`` of the router logits starts at ``t * num_experts``, which must span whole sticks to
# be addressable. ``EXPERTS`` above deliberately does not, so the fallback is covered too.
STICK_EXPERTS = 64


@pytest.fixture(scope="module")
def stick_aligned_moe_weights():
    """Expert stacks whose count spans whole sticks, so a row slice is addressable."""
    from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre

    torch.manual_seed(0)
    host = {
        "gate": torch.randn(STICK_EXPERTS, HIDDEN, INTER, dtype=torch.float16) * 0.05,
        "up": torch.randn(STICK_EXPERTS, HIDDEN, INTER, dtype=torch.float16) * 0.05,
        "down": torch.randn(STICK_EXPERTS, INTER, HIDDEN, dtype=torch.float16) * 0.05,
    }
    host["scale"] = torch.ones(STICK_EXPERTS, dtype=torch.float16)
    device = {
        "gate": dma_moe_expert_weight_to_spyre(host["gate"]),
        "up": dma_moe_expert_weight_to_spyre(host["up"]),
        "down": dma_moe_expert_weight_to_spyre(host["down"]),
    }
    assert all(v is not None for v in device.values()), "expert stacks must take the MoE layout"
    return host, device


# 3 is the bucket the e2e quality gate decodes at; ``T`` sets the row count each row is copied
# out of, so 4 does not subsume it.
@pytest.mark.parametrize("num_tokens", [2, 3, 4])
def test_gathered_loop_matches_dense_reference(stick_aligned_moe_weights, num_tokens):
    """The per-token driver over a packed batch, against the same dense reference.

    A reused region output would give a row another token's experts, which only values catch.
    A CPU fallback would still pass the tolerance while inverting the point, hence that assert.
    """
    from torch_spyre._C import get_elem_in_stick
    from torch_spyre.ops.fallbacks import FallbackWarning

    from spyre_inference.moe import SpyreMoERecipe, _gathered_tokens

    host, device = stick_aligned_moe_weights
    gen = torch.Generator().manual_seed(num_tokens)
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.float16, generator=gen) * 0.5
    logits = torch.randn(num_tokens, STICK_EXPERTS, dtype=torch.float16, generator=gen)
    layer = SimpleNamespace(
        spyre_moe_recipe=SpyreMoERecipe("gelu_tanh", "full_softmax"),
        spyre_moe_gate=device["gate"],
        spyre_moe_up=device["up"],
        spyre_moe_down=device["down"],
        spyre_moe_stick=get_elem_in_stick(torch.float16),
        # The transport dtype, so the routing softmax matches the reference exactly.
        spyre_moe_route_dtype=torch.float16,
        spyre_moe_regions={},
        top_k=TOP_K,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FallbackWarning)
        region = torch.compile(_gathered_tokens, backend="inductor", fullgraph=True, dynamic=False)
        actual = region(
            layer,
            x.to("spyre"),
            logits.to("spyre"),
        )

    fallbacks = [str(w.message) for w in caught if issubclass(w.category, FallbackWarning)]
    assert not fallbacks, f"the gathered loop fell back to CPU: {fallbacks}"

    expected = _dense_reference(
        x,
        torch.softmax(logits, dim=-1),
        host["gate"],
        host["up"],
        host["down"],
        host["scale"],
        TOP_K,
    )
    assert actual.shape == x.shape
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
        route = routing(probs(logits.to("spyre"), logits.dtype), identity, TOP_K, stick)
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


class _RoutedExperts(torch.nn.Module):
    """Stands in for vLLM's, which needs a whole FusedMoEConfig to build."""

    def __init__(self, w13, w2):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2, requires_grad=False)


def test_prepare_layer_rejects_an_unaligned_hidden_size_before_relayout():
    from torch_spyre._C import get_elem_in_stick

    from spyre_inference.moe import SpyreMoERecipe, _prepare_layer

    stick = get_elem_in_stick(torch.float16)
    hidden = HIDDEN - 1
    w13 = torch.empty(EXPERTS, 2 * INTER, hidden, dtype=torch.float16)
    w2 = torch.empty(EXPERTS, hidden, INTER, dtype=torch.float16)
    layer = _RoutedExperts(w13, w2)
    layer.spyre_moe_recipe = SpyreMoERecipe("gelu_tanh", "full_softmax")

    match = rf"down expert-stack free dim {hidden}.*{stick}-element stick.*hidden_size"
    with pytest.raises(ValueError, match=match):
        _prepare_layer(layer)

    assert hasattr(layer, "w13_weight")
    assert hasattr(layer, "w2_weight")
    assert not hasattr(layer, "spyre_moe_gate")


# A whole number of sticks, and a TP shard that lands mid-stick (704 // 2 = 352 for
# gemma-4-26B-A4B, scaled down here).
@pytest.mark.parametrize("inter", [INTER, INTER - 32])
def test_relayout_splits_and_transposes_the_generic_expert_stacks(inter):
    """A model recipe may prepare down weights before generic relayout."""

    from torch_spyre._C import get_elem_in_stick

    from spyre_inference.moe import SpyreMoERecipe, _prepare_layer

    torch.manual_seed(0)
    w13 = torch.randn(EXPERTS, 2 * inter, HIDDEN, dtype=torch.float16) * 0.05
    w2 = torch.randn(EXPERTS, HIDDEN, inter, dtype=torch.float16) * 0.05
    scale = torch.rand(EXPERTS, dtype=torch.float16) + 0.5
    layer = _RoutedExperts(w13.clone(), w2.clone())
    layer.spyre_moe_recipe = SpyreMoERecipe(
        "gelu_tanh",
        "full_softmax",
        prepare_down_weight=lambda weight: weight * scale.view(EXPERTS, 1, 1),
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
