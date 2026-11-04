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

"""The gpt-oss recipe over the generic Spyre MoE backend.

On top of expert biases (``test_moe_bias.py``), gpt-oss needs the ``swigluoai`` activation
and gate/up rows interleaved in ``w13``. The reference below is the transformers
``GptOssExperts`` math, evaluated on the checkpoint layout (interleaved, unpadded), so a
mistake in de-interleaving the weights or the gate/up bias shows up as a value error. Shapes
keep gpt-oss's hidden width, 2880 (45 sticks, so the gathered form picks 15 chunks), and its
per-rank expert width at TP1, TP4 and TP8; the expert count is cut.
"""

from types import SimpleNamespace

import pytest
import torch

EXPERTS, HIDDEN, TOP_K = 16, 2880, 4
LIMIT, ALPHA = 7.0, 1.702


def _swiglu_oai(gate_up: torch.Tensor) -> torch.Tensor:
    # transformers GptOssExperts._apply_gate, on interleaved gate/up columns.
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(-LIMIT, LIMIT)
    return (up + 1) * gate * torch.sigmoid(gate * ALPHA)


def _reference(x, logits, w13, w13_bias, w2, w2_bias, top_k):
    """gpt-oss MoE in float32 on vLLM's parameter layout: ``w13`` [E, 2M, H] interleaved,
    ``w13_bias`` [E, 2M] interleaved, ``w2`` [E, H, M], ``w2_bias`` [E, H]."""
    weights, indices = torch.topk(logits.float(), top_k, dim=-1)
    weights = torch.softmax(weights, dim=-1)
    out = torch.zeros(x.shape, dtype=torch.float32)
    for token in range(x.shape[0]):
        row = x[token].float()
        for slot in range(top_k):
            e = int(indices[token, slot])
            gate_up = w13[e].float() @ row + w13_bias[e].float()
            hidden = w2[e].float() @ _swiglu_oai(gate_up) + w2_bias[e].float()
            out[token] += hidden * float(weights[token, slot])
    return out


def _checkpoint_like(inter, seed=0):
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape, scale):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.float16)

    return {
        "w13": randn(EXPERTS, 2 * inter, HIDDEN, scale=0.02),
        "w13_bias": randn(EXPERTS, 2 * inter, scale=0.5),
        "w2": randn(EXPERTS, HIDDEN, inter, scale=0.02),
        "w2_bias": randn(EXPERTS, HIDDEN, scale=0.5),
    }


def test_deinterleave_gate_up_splits_even_and_odd_rows_into_halves():
    from spyre_inference.moe import deinterleave_gate_up

    w13 = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    halves = deinterleave_gate_up(w13)
    torch.testing.assert_close(halves[:, :3], w13[:, 0::2])
    torch.testing.assert_close(halves[:, 3:], w13[:, 1::2])
    # A [E, 2M] bias is de-interleaved by the same rule.
    bias = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    expected = torch.cat([bias[:, 0::2], bias[:, 1::2]], 1)
    torch.testing.assert_close(deinterleave_gate_up(bias), expected)


def test_swigluoai_activation_matches_transformers():
    """Clamp limits and the ``up + 1`` offset, on separate gate/up tensors."""
    from spyre_inference.moe import _activation

    gen = torch.Generator().manual_seed(0)
    gate_up = torch.randn(4, 2 * 64, generator=gen) * 6  # past the clamp on both sides
    actual = _activation(gate_up[..., ::2], gate_up[..., 1::2], "swigluoai")
    torch.testing.assert_close(actual, _swiglu_oai(gate_up))


def test_routing_function_is_softmax_over_the_selected_logits():
    """``full_softmax`` + renormalize equals gpt-oss's softmax over its top-k logits."""
    from spyre_inference.models.gpt_oss import gpt_oss_routing

    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(8, EXPERTS, generator=gen)
    weights, ids = gpt_oss_routing(None, logits, TOP_K, renormalize=True)
    top, top_ids = torch.topk(logits, TOP_K, dim=-1)
    torch.testing.assert_close(ids.long(), top_ids)
    torch.testing.assert_close(weights, torch.softmax(top, dim=-1))


def _gpt_oss_layer(**moe_overrides):
    """A ``RoutedExperts`` as vLLM builds it for gpt-oss."""
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
        "swiglu_limit": None,
        "swiglu_alpha": None,
    } | moe_overrides
    return SimpleNamespace(
        moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(enable_eplb=False), **moe),
        custom_routing_function=None,
        activation=SimpleNamespace(value="swigluoai"),
        global_num_experts=EXPERTS,
        local_num_experts=EXPERTS,
        renormalize=True,
        apply_router_weight_on_input=False,
        quant_config=None,
        quant_method=object.__new__(SpyreUnquantizedFusedMoEMethod),
    )


def test_configure_accepts_a_biased_swigluoai_layer():
    from spyre_inference.models.gpt_oss import GPT_OSS_MOE_RECIPE, gpt_oss_routing
    from spyre_inference.moe import configure_spyre_moe_layer

    layer = _gpt_oss_layer()
    layer.custom_routing_function = gpt_oss_routing
    configure_spyre_moe_layer(layer, GPT_OSS_MOE_RECIPE)
    assert layer.spyre_moe_recipe is GPT_OSS_MOE_RECIPE


@pytest.mark.parametrize(
    "overrides", [{"swiglu_limit": 10.0}, {"swiglu_alpha": 1.0}], ids=["limit", "alpha"]
)
def test_configure_rejects_swigluoai_constants_the_kernel_does_not_use(overrides):
    """The activation hard-codes vLLM's swigluoai_and_mul constants; refuse others."""
    from spyre_inference.models.gpt_oss import GPT_OSS_MOE_RECIPE, gpt_oss_routing
    from spyre_inference.moe import configure_spyre_moe_layer

    layer = _gpt_oss_layer(**overrides)
    layer.custom_routing_function = gpt_oss_routing
    with pytest.raises(NotImplementedError, match="swigluoai uses limit="):
        configure_spyre_moe_layer(layer, GPT_OSS_MOE_RECIPE)


class _RoutedExperts(torch.nn.Module):
    """Stands in for vLLM's, which needs a whole FusedMoEConfig to build."""

    def __init__(self, params):
        super().__init__()
        for key, value in params.items():
            name = key if key.endswith("_bias") else f"{key}_weight"
            param = torch.nn.Parameter(value.clone(), requires_grad=False)
            self.register_parameter(name, param)
        self.top_k = TOP_K


def _prepared(inter, monkeypatch):
    from spyre_inference.models.gpt_oss import GPT_OSS_MOE_RECIPE
    from spyre_inference.moe import _prepare_layer

    monkeypatch.delenv("SPYRE_MOE_CHUNKS", raising=False)
    params = _checkpoint_like(inter)
    layer = _RoutedExperts(params)
    layer.spyre_moe_recipe = GPT_OSS_MOE_RECIPE
    _prepare_layer(layer)
    return layer, params


# gpt-oss-20b at TP1; gpt-oss-120b at TP4 (720 -> 768 padded) and TP8 (360 -> 384 padded).
_INTERS = [pytest.param(2880, id="tp1"), pytest.param(720, id="tp4"), pytest.param(360, id="tp8")]


@pytest.mark.parametrize("inter", _INTERS)
def test_relayout_deinterleaves_the_weights_and_the_gate_up_bias(inter, monkeypatch):
    """Gate and up pools, and the gate and up biases, get the even and odd rows."""
    layer, p = _prepared(inter, monkeypatch)

    close = {"atol": 1e-4, "rtol": 1e-2}
    gate, up = layer.spyre_moe_gate.cpu(), layer.spyre_moe_up.cpu()
    torch.testing.assert_close(gate[..., :inter], p["w13"][:, 0::2].transpose(1, 2), **close)
    torch.testing.assert_close(up[..., :inter], p["w13"][:, 1::2].transpose(1, 2), **close)
    b = layer.spyre_moe_biases
    torch.testing.assert_close(b.gate.cpu()[:, 0, :inter], p["w13_bias"][:, 0::2], **close)
    torch.testing.assert_close(b.up.cpu()[:, 0, :inter], p["w13_bias"][:, 1::2], **close)


def _inputs(tokens, seed):
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(tokens, HIDDEN, generator=gen) * 0.5).to(torch.float16)
    logits = torch.randn(tokens, EXPERTS, generator=gen)
    # Keep the top-k boundary clear of fp16 near-ties, which the card may break either way.
    order = torch.sort(logits, dim=-1, descending=True).indices
    lift = torch.zeros_like(logits).scatter_(-1, order[:, :TOP_K], 0.25)
    return x, (logits + lift).to(torch.float16)


@pytest.mark.parametrize("inter", _INTERS)
def test_gathered_matches_the_gpt_oss_reference(inter, monkeypatch):
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
@pytest.mark.parametrize("tokens", [16, 64])
def test_persistent_matches_the_gpt_oss_reference(inter, tokens, monkeypatch):
    """A prefill-sized batch through the all-expert regions, as ``apply_monolithic`` runs them."""
    from spyre_inference import moe as moe_module

    layer, p = _prepared(inter, monkeypatch)
    layer.spyre_moe_regions = {}
    x, logits = _inputs(tokens, seed=tokens)
    method = object.__new__(moe_module.SpyreUnquantizedFusedMoEMethod)
    actual = method.apply_monolithic(layer, x.to("spyre"), logits.to("spyre"))
    assert "experts" in layer.spyre_moe_regions
    expected = _reference(x, logits, p["w13"], p["w13_bias"], p["w2"], p["w2_bias"], TOP_K)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("inter", [pytest.param(720, id="tp4")])
def test_traced_dispatch_matches_the_gpt_oss_reference(inter, monkeypatch):
    """Traced into one graph with in-graph producers, as a compiled block runs one stick of
    tokens: the in-graph persistent form, biases included."""
    import warnings

    from torch_spyre._C import get_elem_in_stick
    from torch_spyre.ops.fallbacks import FallbackWarning

    from spyre_inference import moe as moe_module

    layer, p = _prepared(inter, monkeypatch)
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
    expected = _reference(x, logits, p["w13"], p["w13_bias"], p["w2"], p["w2_bias"], TOP_K)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=3e-2, rtol=3e-2)
