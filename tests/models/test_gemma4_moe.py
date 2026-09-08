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

"""The Spyre Gemma-4 MoE expert dispatch: its two forms and its weight relayout.

Both dispatch forms compute the same function by different means, so one dense
reference covers both. They need the card; shapes are scaled down but keep every dim
stick-aligned, which is what the layout tricks in those regions depend on.

There is nothing to test around them: the dispatch hangs off vLLM's own
``UnquantizedFusedMoEMethod`` seam, so ``Gemma4DecoderLayer.forward`` runs upstream's
body unmodified.
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
    # _relayout_experts), so the device stacks carry it and the reference
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
    """The decode form, at the single token whose combine has a legal device layout."""
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
    """The prefill form, in the region sequence ``apply_monolithic`` uses.

    24 tokens does not divide the core count, which the work-division hint has to cope
    with.
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
    """`w13 [E,2M,H]` -> `gate`/`up` `[E,H,M]`, `w2` -> scaled `down [E,M,H]`, both freed."""
    import weakref

    import torch.nn as nn
    from torch_spyre._C import get_elem_in_stick

    from spyre_inference.models._gemma4_moe import _relayout_experts

    class _MoEConfig:
        tp_size = 1
        ep_size = 1

    class _RoutedExperts(nn.Module):
        """Stands in for vLLM's, which needs a whole FusedMoEConfig to build."""

        def __init__(self, w13, w2, owner):
            super().__init__()
            self.w13_weight = nn.Parameter(w13, requires_grad=False)
            self.w2_weight = nn.Parameter(w2, requires_grad=False)
            self.moe_config = _MoEConfig()
            self.spyre_moe_owner = weakref.ref(owner)

    torch.manual_seed(0)
    w13 = torch.randn(EXPERTS, 2 * INTER, HIDDEN, dtype=torch.float16) * 0.05
    w2 = torch.randn(EXPERTS, HIDDEN, INTER, dtype=torch.float16) * 0.05
    scale = torch.rand(EXPERTS, dtype=torch.float16) + 0.5
    owner = nn.Module()
    owner.per_expert_scale = nn.Parameter(scale.clone(), requires_grad=False)
    layer = _RoutedExperts(w13.clone(), w2.clone(), owner)

    _relayout_experts(layer)

    assert not hasattr(layer, "w13_weight"), "the fused stacks must be freed, not kept"
    assert not hasattr(layer, "w2_weight")
    assert layer.spyre_gate.shape == (EXPERTS, HIDDEN, INTER)
    assert layer.spyre_up.shape == (EXPERTS, HIDDEN, INTER)
    assert layer.spyre_down.shape == (EXPERTS, INTER, HIDDEN)
    assert layer.spyre_route_identity.shape == (layer.spyre_stick, layer.spyre_stick)
    # Both come from the stacks' dtype, not a literal: the identity multiplies the
    # routing weights, which arrive in that same dtype, and a stick's element count
    # changes with it.
    assert layer.spyre_stick == get_elem_in_stick(w13.dtype)
    assert layer.spyre_route_identity.dtype == w13.dtype

    # Not bit-exact: the device round-trip rounds a few fp16 elements by one ulp. Still
    # far tighter than the 0.5x-1.5x scale, so a dropped fold would fail here.
    close = {"atol": 1e-4, "rtol": 1e-2}
    torch.testing.assert_close(layer.spyre_gate.cpu(), w13[:, :INTER, :].transpose(1, 2), **close)
    torch.testing.assert_close(layer.spyre_up.cpu(), w13[:, INTER:, :].transpose(1, 2), **close)
    torch.testing.assert_close(
        layer.spyre_down.cpu(), (w2 * scale.view(EXPERTS, 1, 1)).transpose(1, 2), **close
    )
