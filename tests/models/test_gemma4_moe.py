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

"""The Spyre Gemma-4 MoE layer: its expert dispatch, and its decoder-layer plumbing.

Both dispatch forms compute the same function by different means (see
``spyre_inference.models._gemma4_moe``), so one dense reference covers both. Those
tests need the card; shapes are scaled down but keep every dim stick-aligned, which
is what the layout tricks in those regions depend on.

The decoder-layer tests at the bottom are the other half: the Spyre path re-splits
upstream's ``Gemma4DecoderLayer.forward`` around its expert dispatch, and they hold
that copy to the original. They are plain host torch and need no device.
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
    # The per-expert output scale is folded into `down` at load time (see
    # spyre_relayout_weights), so the device stacks carry it and the reference
    # applies it separately.
    host["scale"] = torch.rand(EXPERTS, dtype=torch.float16) + 0.5
    scaled_down = host["down"] * host["scale"].view(EXPERTS, 1, 1)
    stacks = {"gate": host["gate"], "up": host["up"], "down": scaled_down}
    device = {k: dma_moe_expert_weight_to_spyre(v) for k, v in stacks.items()}
    assert all(v is not None for v in device.values()), "expert stacks must take the MoE layout"
    return host, device


def _inputs(num_tokens):
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.float16) * 0.5
    probs = torch.softmax(torch.randn(num_tokens, EXPERTS, dtype=torch.float16), dim=-1)
    return x, probs


def test_gathered_matches_dense_reference(moe_weights):
    """The decode form: gather the selected experts' weights, BMM, combine over K.

    Single token only — the only shape whose combine step has a legal device layout.
    """
    from torch_spyre._C import get_elem_in_stick
    from torch_spyre._inductor import config as spyre_config

    from spyre_inference.models._gemma4_moe import _moe_gathered

    host, device = moe_weights
    x, probs = _inputs(1)
    stick = get_elem_in_stick(torch.float16)

    region = torch.compile(_moe_gathered, backend="inductor", fullgraph=True, dynamic=False)
    with spyre_config.patch({"frontend_pool_allocation": True}):
        actual = region(
            x.to("spyre"),
            probs.to("spyre"),
            device["gate"],
            device["up"],
            device["down"],
            TOP_K,
            stick,
        )

    expected = _dense_reference(
        x, probs, host["gate"], host["up"], host["down"], host["scale"], TOP_K
    )
    torch.testing.assert_close(actual.cpu().float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [24, 32])
def test_persistent_matches_dense_reference(moe_weights, num_tokens):
    """The prefill form: dense routing weights times every expert's output.

    Mirrors the region sequence in ``SpyreGemma4MoEDecoderLayer.forward``: the
    routing runs in a graph of its own, then the expert matmul under an eagerly
    declared named-dims context. 24 tokens does not divide the core count, which the
    work-division hint has to cope with.
    """
    from torch_spyre._C import get_elem_in_stick
    from torch_spyre._inductor import config as spyre_config
    from torch_spyre._inductor.wsr.propagate_named_dims import reset as reset_named_dims

    from spyre_inference.models._gemma4_moe import (
        _moe_persistent,
        _moe_persistent_routing,
        _name_persistent_dims,
    )

    host, device = moe_weights
    x, probs = _inputs(num_tokens)
    x_dev = x.to("spyre")
    stick = get_elem_in_stick(torch.float16)
    identity = torch.eye(stick, dtype=torch.float16).to("spyre")

    routing = torch.compile(
        _moe_persistent_routing, backend="inductor", fullgraph=True, dynamic=False
    )
    experts = torch.compile(_moe_persistent, backend="inductor", fullgraph=True, dynamic=False)

    with spyre_config.patch({"frontend_pool_allocation": True}):
        route = routing(probs.to("spyre"), identity, TOP_K, stick)
        _name_persistent_dims(x_dev, device["gate"], device["up"], device["down"])
        try:
            with spyre_config.patch({"allow_all_ops_in_lx_planning": True}):
                actual = experts(x_dev, route, device["gate"], device["up"], device["down"])
        finally:
            reset_named_dims()

    expected = _dense_reference(
        x, probs, host["gate"], host["up"], host["down"], host["scale"], TOP_K
    )
    torch.testing.assert_close(actual.cpu().float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("tokens", "expected"), [(1, 1), (16, 16), (24, 24), (32, 32), (40, 20), (64, 32)]
)
def test_token_cores_divides_the_token_axis(tokens, expected):
    """The token work-division split must divide the axis and not exceed the cores."""
    from torch_spyre._inductor import config as spyre_config

    from spyre_inference.models._gemma4_moe import _token_cores

    if spyre_config.sencores != 32:
        pytest.skip(f"expectations assume SENCORES=32, got {spyre_config.sencores}")
    assert _token_cores(tokens) == expected


def test_relayout_splits_transposes_and_folds_the_scale():
    """The load-time weight hook: what `spyre_relayout_weights` leaves for the regions.

    `w13 [E,2M,H]` splits into `gate`/`up` `[E,H,M]`, `w2 [E,H,M]` becomes
    `down [E,M,H]` carrying `per_expert_scale`, and both sources are freed.
    """
    import torch.nn as nn
    from torch_spyre._C import get_elem_in_stick

    from spyre_inference.models._gemma4_moe import SpyreGemma4MoEDecoderLayer

    class _RoutedExperts(nn.Module):
        def __init__(self, w13, w2):
            super().__init__()
            self.w13_weight = nn.Parameter(w13, requires_grad=False)
            self.w2_weight = nn.Parameter(w2, requires_grad=False)

    class _Layer(nn.Module):
        def __init__(self, experts, scale):
            super().__init__()
            self.routed_experts = experts
            self.moe = nn.Module()
            self.moe.per_expert_scale = nn.Parameter(scale, requires_grad=False)

        def spyre_experts(self):
            return self.routed_experts

        # The real hook, under the name the model runner's post-load walk looks for.
        spyre_relayout_weights = SpyreGemma4MoEDecoderLayer.spyre_relayout_weights

    torch.manual_seed(0)
    w13 = torch.randn(EXPERTS, 2 * INTER, HIDDEN, dtype=torch.float16) * 0.05
    w2 = torch.randn(EXPERTS, HIDDEN, INTER, dtype=torch.float16) * 0.05
    scale = torch.rand(EXPERTS, dtype=torch.float16) + 0.5
    experts = _RoutedExperts(w13.clone(), w2.clone())
    layer = _Layer(experts, scale.clone())

    layer.spyre_relayout_weights()

    assert not hasattr(experts, "w13_weight"), "the fused stacks must be freed, not kept"
    assert not hasattr(experts, "w2_weight")
    assert layer.spyre_gate.shape == (EXPERTS, HIDDEN, INTER)
    assert layer.spyre_up.shape == (EXPERTS, HIDDEN, INTER)
    assert layer.spyre_down.shape == (EXPERTS, INTER, HIDDEN)
    assert layer.spyre_route_identity.shape == (layer.spyre_stick, layer.spyre_stick)
    # Both come from the stacks' dtype, not a literal: the identity multiplies the
    # routing weights, which arrive in that same dtype, and a stick's element count
    # changes with it.
    assert layer.spyre_stick == get_elem_in_stick(w13.dtype)
    assert layer.spyre_route_identity.dtype == w13.dtype

    # Not bit-exact: the device round-trip rounds a few fp16 elements by one ulp. The
    # tolerance is still far tighter than the 0.5x-1.5x per-expert scale, so a dropped
    # or misapplied fold would still fail here.
    close = {"atol": 1e-4, "rtol": 1e-2}
    torch.testing.assert_close(layer.spyre_gate.cpu(), w13[:, :INTER, :].transpose(1, 2), **close)
    torch.testing.assert_close(layer.spyre_up.cpu(), w13[:, INTER:, :].transpose(1, 2), **close)
    torch.testing.assert_close(
        layer.spyre_down.cpu(), (w2 * scale.view(EXPERTS, 1, 1)).transpose(1, 2), **close
    )


# ---------------------------------------------------------------------------
# Decoder-layer plumbing
#
# The Spyre path does not call upstream's ``Gemma4DecoderLayer.forward``: it
# splits the same body into ``_attn_block`` / ``_persistent_prologue`` /
# ``_combine_block`` so the expert dispatch can sit between two compiled
# regions. That split is a hand copy of upstream's residual-and-sandwich-norm
# ordering, and nothing but these tests keeps the copy honest.
# ---------------------------------------------------------------------------

DTYPE = torch.float16


class _StubAttention(torch.nn.Module):
    """Stands in for ``Gemma4Attention``, which needs a KV cache and a backend.

    ``positions`` and the forwarded ``**kwargs`` both enter the result, and the
    probe is a required argument, so a path that drops either fails.
    """

    def __init__(self, *, hidden_size: int, **kwargs) -> None:
        super().__init__()
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False, dtype=DTYPE)

    def forward(self, positions, hidden_states, attn_probe, **kwargs):
        return self.o_proj(hidden_states) + attn_probe * positions[:, None].to(hidden_states.dtype)


class _StubMoE(torch.nn.Module):
    """Stands in for ``Gemma4MoE``, whose ``FusedMoE`` kernels are CUDA-only.

    Keeps the real module's own ``per_expert_scale``; the fixture then replaces
    ``forward`` with the one expert dispatch both paths call.
    """

    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.per_expert_scale = torch.nn.Parameter(torch.ones(config.num_experts))


def _tiny_moe_text_config(**overrides):
    """Upstream's own Gemma-4 text config, scaled down to one small MoE layer."""
    from transformers import Gemma4TextConfig

    return Gemma4TextConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=128,
        # spyre_init rejects per-layer embeddings, so the supported shape has none.
        hidden_size_per_layer_input=0,
        enable_moe_block=True,
        num_experts=EXPERTS,
        top_k_experts=TOP_K,
        moe_intermediate_size=INTER,
        **overrides,
    )


# The one config switch that changes a layer's shape: on a KV-shared layer it
# doubles the *dense* MLP's intermediate size, which both paths run as
# ``layer.mlp``. LAYER_IDX is the last layer, so it is the shared one.
DOUBLE_WIDE_MLP = {"use_double_wide_mlp": True, "num_kv_shared_layers": 1}
LAYER_IDX = 1


def _spyre_moe(layer, expert_input, probs):
    """The one expert dispatch both paths run: the gathered form is plain torch."""
    from spyre_inference.models._gemma4_moe import _moe_gathered

    return _moe_gathered(
        expert_input,
        probs,
        layer.spyre_gate,
        layer.spyre_up,
        layer.spyre_down,
        layer.spyre_top_k,
        layer.spyre_stick,
    )


@pytest.fixture
def build_moe_layer(tp_group):
    """Builds a real ``Gemma4DecoderLayer`` with an MoE block, eager and on the host.

    Upstream's own: the five RMSNorms, the dense MLP, the router, ``layer_scalar``
    and — the point of the exercise — ``Gemma4DecoderLayer.forward``. Only the two
    submodules that cannot run here are replaced, and both paths call the identical
    stand-in, so any difference between them is plumbing and nothing else.

    ``CompilationMode.NONE`` is what keeps this off the card: the OOT layers'
    ``compile_when_outermost`` kernels would otherwise compile for the device.
    """
    from torch_spyre._C import get_elem_in_stick
    from vllm.config import (
        CompilationMode,
        DeviceConfig,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.config.compilation import CompilationConfig
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.models import gemma4
    from vllm.utils.torch_utils import set_default_torch_dtype

    vllm_config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        model_config=ModelConfig(dtype=DTYPE),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE, custom_ops=["all"]),
    )

    def _build(**config_overrides):
        torch.manual_seed(0)
        # The loader's dtype context, and not for convenience: the norm weights and
        # ``layer_scalar`` are created without an explicit dtype, so outside it every
        # norm would silently upcast the layer to float32.
        with set_default_torch_dtype(DTYPE):
            layer = gemma4.Gemma4DecoderLayer(
                _tiny_moe_text_config(**config_overrides), prefix=f"model.layers.{LAYER_IDX}"
            )

        # vLLM's linear layers come out of torch.empty, so every weight has to be
        # written. Norm weights multiply an already normalized tensor: centred on
        # 1.0, or five norms in sequence would shrink the activations to fp16 zero.
        for module in layer.modules():
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor):
                weight.data.normal_(mean=1.0 if isinstance(module, RMSNorm) else 0.0, std=0.05)
        layer.router.scale.data.normal_(mean=1.0, std=0.05)
        # Not 1.0: the combine step's final multiply has to be observable.
        layer.layer_scalar.fill_(0.75)

        # The load-time pass the loader runs over every linear layer; the Spyre
        # linear method stores its transposed weight there and reads it in apply.
        for module in layer.modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(module)

        # The expert stacks in the layout spyre_relayout_weights leaves behind
        # (per-expert scale already folded into ``down``); its own test covers
        # the load-time hook that builds them on the device.
        torch.manual_seed(1)
        layer.spyre_gate = torch.randn(EXPERTS, HIDDEN, INTER, dtype=DTYPE) * 0.05
        layer.spyre_up = torch.randn(EXPERTS, HIDDEN, INTER, dtype=DTYPE) * 0.05
        layer.spyre_down = torch.randn(EXPERTS, INTER, HIDDEN, dtype=DTYPE) * 0.05
        layer.spyre_top_k = TOP_K
        layer.spyre_stick = get_elem_in_stick(DTYPE)

        # Upstream's MoE takes router logits, the Spyre regions take probabilities:
        # the softmax lands on the same side of the boundary either way, so both
        # paths reach _moe_gathered with bit-identical inputs.
        def _moe_forward(expert_input, router_logits):
            return _spyre_moe(layer, expert_input, torch.softmax(router_logits, dim=-1))

        layer.moe.forward = _moe_forward
        return layer

    with set_current_vllm_config(vllm_config), pytest.MonkeyPatch.context() as patch:
        patch.setattr(gemma4, "Gemma4Attention", _StubAttention)
        patch.setattr(gemma4, "Gemma4MoE", _StubMoE)
        yield _build


def _layer_inputs(num_tokens):
    torch.manual_seed(2)
    return (
        # Non-zero, so the stub's positions term cannot vanish.
        torch.arange(num_tokens) + 5,
        torch.randn(num_tokens, HIDDEN, dtype=DTYPE) * 0.5,
        # The model runner threads attention metadata through the layer as
        # **kwargs; this stands in for it.
        {"attn_probe": torch.randn(num_tokens, HIDDEN, dtype=DTYPE) * 0.1},
    )


def _upstream_layer_output(layer, num_tokens):
    """What upstream's own ``forward`` makes of these inputs, plus the inputs."""
    positions, hidden_states, kwargs = _layer_inputs(num_tokens)
    expected, residual = layer(
        positions=positions, hidden_states=hidden_states, residual=None, **kwargs
    )
    # Upstream carries no fused residual here, which is why the Spyre forward can
    # return None as its second element.
    assert residual is None
    # Guards against a vacuous comparison: fp16 underflow to zero on both sides
    # would otherwise pass whatever the plumbing did.
    assert torch.isfinite(expected).all()
    assert expected.abs().max() > 1e-3
    return expected, positions, hidden_states, kwargs


@pytest.mark.parametrize("config", [{}, DOUBLE_WIDE_MLP], ids=["plain", "double_wide_mlp"])
@pytest.mark.parametrize("num_tokens", [1, 8])
def test_persistent_plumbing_matches_upstream_forward(build_moe_layer, num_tokens, config):
    """``_persistent_prologue`` + experts + ``_combine_block`` == upstream's forward.

    The prefill form's regions, minus the expert math itself: the sandwich norms,
    the second residual capture, what the router and the expert block are each fed,
    the dense/MoE combine and the layer scalar. Exact equality — both sides run the
    same ops on the same tensors in the same order, so anything but a bit-for-bit
    match is a real reordering.
    """
    from spyre_inference.models._gemma4_moe import _combine_block, _persistent_prologue

    layer = build_moe_layer(**config)
    expected, positions, hidden_states, kwargs = _upstream_layer_output(layer, num_tokens)

    residual, probs, expert_input = _persistent_prologue(layer, positions, hidden_states, **kwargs)
    moe_out = _spyre_moe(layer, expert_input, probs)
    actual = _combine_block(layer, residual, moe_out)

    assert actual.shape == (num_tokens, HIDDEN)
    assert actual.dtype == DTYPE
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("config", [{}, DOUBLE_WIDE_MLP], ids=["plain", "double_wide_mlp"])
def test_gathered_layer_matches_upstream_forward(build_moe_layer, config):
    """``_gathered_layer`` == upstream's forward, at the one token it is used for.

    The decode form is a single region, so this runs its whole body — including the
    real ``_moe_gathered``, which is plain torch and needs no device.
    """
    from spyre_inference.models._gemma4_moe import _gathered_layer

    layer = build_moe_layer(**config)
    expected, positions, hidden_states, kwargs = _upstream_layer_output(layer, 1)

    actual = _gathered_layer(layer, positions, hidden_states, **kwargs)

    assert actual.shape == (1, HIDDEN)
    assert actual.dtype == DTYPE
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
