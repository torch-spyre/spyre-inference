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

"""Spyre sparse-MoE path for Gemma-4 26B-A4B (``enable_moe_block``).

vLLM dispatches Gemma-4's expert block through ``FusedMoE``, whose kernels are
CUDA / Triton only. This module supplies a Spyre dispatch instead, ported from the
``hf_adapters`` ``hf_gemma4_moe`` adapter. Two forms, both reading the expert
stacks :func:`relayout_moe_experts` lays out:

*Gathered*, for a single-token decode step: gather only the selected experts'
weights, one row per top-k slot, and contract with per-row BMMs. One graph for the
whole layer. Its per-row combine has no legal device layout above one token.

*Persistent*, for everything else: evaluate every expert over every token as one
batched ``[E,T,H] x [E,H,M]`` matmul, with dense routing weights zeroing the
unselected pairs. Reads each expert weight once regardless of token count. Needs
four graphs: its expert matmul is tiled by ``spyre_hint`` scopes resolving against
named dims the driver declares *eagerly*, between compilations, and the routing
weights must sit in a graph of their own between the softmax and that context — so
the layer opts out of the model runner's whole-block compile.

Both forms require ``torch.compile``; see
:meth:`SpyreGemma4MoEDecoderLayer.spyre_init`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.models.gemma4 import Gemma4MoE, Gemma4Router

logger = init_logger(__name__)

# ``frontend_pool_allocation`` has the front end allocate the scratch pool as a
# real tensor and pass its address in, instead of the backend self-allocating.
_MOE_COMPILER_CONFIG = {"frontend_pool_allocation": True}
# The persistent form's all-expert matmul is not in the default LX planning set.
_PERSISTENT_COMPILER_CONFIG = {"allow_all_ops_in_lx_planning": True}


def _topk(probs: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k over the expert axis, padded to two rows at T=1.

    A length-1 reduction aborts the Spyre compiler ("stick expression 1"), so a
    single-token batch duplicates its row and slices the result back.
    """
    tokens = probs.shape[0]
    padded = probs.expand(2, -1).contiguous() if tokens == 1 else probs
    weights, indices = torch.topk(padded, top_k, dim=-1)
    return weights[:tokens], indices[:tokens]


def _gather_indices(indices: torch.Tensor, top_k: int, stick: int) -> torch.Tensor:
    """Turn topk's fp16 expert ids into device int32 gather indices.

    The ids have to travel through a full stick before the int32 cast, and the
    ``.contiguous()`` is what restickifies them. Materializing the widening with a
    pointwise op instead (``relu``, as the routing path does) yields an index layout
    the backend's gather-index conversion rejects (``fmod(dsDim, size) == 0``).
    """
    tokens = indices.shape[0]
    widened = indices[..., None].expand(tokens, top_k, stick).contiguous()
    address = widened.to(torch.float32)[..., : stick // 2].to(torch.int32)
    return address[..., 0]


def _moe_gathered(
    x: torch.Tensor,
    probs: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    top_k: int,
    stick: int,
) -> torch.Tensor:
    """Expert FFN over the selected experts only: gather weights, BMM, combine.

    ``x`` is ``[T,H]``; ``gate``/``up`` are ``[E,H,M]`` and ``down`` is
    ``[E,M,H]``. Returns ``[T,H]``.
    """
    tokens, hidden = x.shape
    weights, indices = _topk(probs, top_k)
    weights = weights / weights.sum(-1, keepdim=True)
    indices = _gather_indices(indices, top_k, stick)

    rows = tokens * top_k
    inter = gate.shape[-1]
    # Materialize the K-batch stride: a stride-0 expand drops the batch dim from
    # the BMM's layout order and the scheduler rejects the result.
    inputs = x[:, None, :].expand(tokens, top_k, hidden).contiguous().reshape(rows, 1, hidden)
    gate_out = torch.bmm(inputs, gate[indices].reshape(rows, hidden, inter))
    up_out = torch.bmm(inputs, up[indices].reshape(rows, hidden, inter))
    activated = F.gelu(gate_out, approximate="tanh") * up_out
    expert_out = torch.bmm(activated, down[indices].reshape(rows, inter, hidden))
    expert_out = expert_out.reshape(tokens, top_k, hidden)

    # The routing weight is folded into the H-carrying tensor: a bare [T,K] product
    # has no legal layout. The per-expert output scale is already in ``down`` (see
    # relayout_moe_experts), so nothing else joins it here.
    return (expert_out * weights[..., None]).sum(dim=1)


def _moe_persistent_routing(
    probs: torch.Tensor,
    route_identity: torch.Tensor,
    top_k: int,
    stick: int,
) -> torch.Tensor:
    """Dense ``[T,E,1]`` routing weights: renormalized top-k probs, zero elsewhere."""
    _, selected = _topk(probs, top_k)
    weights = torch.ops.spyre.keep_by_index(
        probs,  # ty: ignore[invalid-argument-type]
        selected,  # ty: ignore[invalid-argument-type]
        -1,  # ty: ignore[invalid-argument-type]
        0.0,  # ty: ignore[invalid-argument-type]
    )
    weights = weights / weights.sum(-1, keepdim=True)

    # ReLU materializes the expansion; the identity matmul puts it on a stick.
    packed = torch.relu(weights.unsqueeze(-1).expand(-1, -1, stick))
    return (packed @ route_identity)[..., :1]


def _token_cores(tokens: int) -> int:
    """How many ways to spread the token axis over cores.

    ``work_div`` splits a dim across that many cores, so it has to divide it
    exactly, and vLLM's compile buckets (16, 24, 40, 56, …) are not all multiples of
    the core count — hence the largest legal divisor rather than the count itself.
    """
    from torch_spyre._inductor import config as spyre_config

    limit = min(tokens, spyre_config.sencores)
    return max(split for split in range(1, limit + 1) if tokens % split == 0)


def _reset_named_dims() -> None:
    """Drop the driver-declared named dims so they cannot leak into the next graph."""
    from torch_spyre._inductor.wsr.propagate_named_dims import reset

    reset()


def _moe_persistent(
    x: torch.Tensor,
    route: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """Expert FFN over every expert, summed against ``route``.

    ``x`` is ``[T,H]``, ``route`` is ``[T,E,1]``, weights as in
    ``_moe_gathered``. Returns ``[T,H]``.
    """
    from torch_spyre._inductor.propagate_hints import spyre_hint

    experts = gate.shape[0]
    h = x.unsqueeze(0)
    with spyre_hint(named_dims=["E", "T", "ONE"]):
        route = route.permute(1, 0, 2).contiguous().clone()

    with spyre_hint(num_tiles_per_dim={"E": experts}, work_div={"T": _token_cores(x.shape[0])}):
        activated = F.gelu(torch.matmul(h, gate), approximate="tanh") * torch.matmul(h, up)
        # A genuine .sum reduction collapses the expert axis to a rank-3 buffer;
        # folding it into the matmul leaves a rank-4 view the backend rejects.
        return (torch.matmul(activated, down) * route).sum(dim=0)


def _name_persistent_dims(
    x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor
) -> None:
    """Declare the named dims the persistent form's tiling hints resolve against.

    The propagation pass reads these names off the *real* input tensors during
    lowering, so this must run eagerly, before the region is traced — which is why
    the FFN gets a graph of its own.
    """
    from torch_spyre._inductor.wsr.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
    )

    experts, hidden, inter = gate.shape
    for name, extent in (
        ("E", experts),
        ("T", x.shape[0]),
        ("H", hidden),
        ("M", inter),
        ("ONE", 1),
    ):
        declare_tensor_dim(name, extent)
    name_tensor_dims(x, ["T", "H"])
    name_tensor_dims(gate, ["E", "H", "M"])
    name_tensor_dims(up, ["E", "H", "M"])
    name_tensor_dims(down, ["E", "M", "H"])


def _attn_block(
    layer: SpyreGemma4MoEDecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **kwargs,
):
    """``input_layernorm`` -> attention -> ``post_attention_layernorm`` -> residual add.

    Gemma-4 norms the attention *output* before adding the residual (a "sandwich"
    norm), so the add cannot be fused into the next norm.
    """
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)
    attn_out = layer.self_attn(positions=positions, hidden_states=normed, **kwargs)
    return residual + layer.post_attention_layernorm(attn_out)


def _combine_block(
    layer: SpyreGemma4MoEDecoderLayer, residual: torch.Tensor, moe_out: torch.Tensor
):
    dense = layer.mlp(layer.pre_feedforward_layernorm(residual))
    dense = layer.post_feedforward_layernorm_1(dense)
    moe = layer.post_feedforward_layernorm_2(moe_out.to(residual.dtype))
    ffn_out = layer.post_feedforward_layernorm(dense + moe)
    return (residual + ffn_out) * layer.layer_scalar


def _persistent_prologue(
    layer: SpyreGemma4MoEDecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **kwargs,
):
    """Attention, the router probabilities, and the expert block's normed input.

    Softmax puts its stick on the token axis and ``keep_by_index`` needs it on the
    expert axis, so the probabilities have to cross a graph boundary — which restores
    the default layout — before the routing weights are built.
    """
    residual = _attn_block(layer, positions, hidden_states, **kwargs)
    probs = torch.softmax(layer.router(residual), dim=-1)
    return residual, probs, layer.pre_feedforward_layernorm_2(residual)


def _persistent_route(layer: SpyreGemma4MoEDecoderLayer, probs: torch.Tensor) -> torch.Tensor:
    """Dense routing weights.

    Its own graph for two independent reasons: it cannot share one with the softmax
    (layout, see ``_persistent_prologue``) nor with ``_moe_persistent``, because
    ``keep_by_index`` reduces over the top-k axis, which the named-dims propagation
    pass cannot map onto its probability input.
    """
    return _moe_persistent_routing(
        probs, layer.spyre_route_identity, layer.spyre_top_k, layer.spyre_stick
    )


def _persistent_experts(
    layer: SpyreGemma4MoEDecoderLayer, expert_input: torch.Tensor, route: torch.Tensor
) -> torch.Tensor:
    """All-expert FFN. The only region that needs the eager named-dims context."""
    return _moe_persistent(expert_input, route, layer.spyre_gate, layer.spyre_up, layer.spyre_down)


def _gathered_layer(
    layer: SpyreGemma4MoEDecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **kwargs,
):
    """Attention + router + gathered-expert FFN + combine, all in one graph."""
    residual = _attn_block(layer, positions, hidden_states, **kwargs)
    probs = torch.softmax(layer.router(residual), dim=-1)
    moe_out = _moe_gathered(
        layer.pre_feedforward_layernorm_2(residual),
        probs,
        layer.spyre_gate,
        layer.spyre_up,
        layer.spyre_down,
        layer.spyre_top_k,
        layer.spyre_stick,
    )
    return _combine_block(layer, residual, moe_out)


class SpyreGemma4MoEDecoderLayer(Gemma4DecoderLayer):
    """A Gemma-4 MoE decoder layer that dispatches its experts on Spyre.

    Instances are retyped into this class by :func:`adapt_moe_layers` rather than
    constructed: ``Gemma4Model`` names ``Gemma4DecoderLayer`` directly in its
    ``make_layers`` call, so there is no class hook to inject through
    (``models._token_type`` retypes the BERT embedding the same way).
    """

    # The model runner must not wrap this layer in one whole-block graph; see ``forward``.
    spyre_compiles_own_regions = True

    # Upstream types these as ``| None`` because a dense layer leaves them unset;
    # this class only ever wraps a layer that has them.
    router: Gemma4Router
    moe: Gemma4MoE
    pre_feedforward_layernorm_2: RMSNorm
    post_feedforward_layernorm_1: RMSNorm
    post_feedforward_layernorm_2: RMSNorm

    # Set by spyre_init / relayout_moe_experts.
    spyre_top_k: int
    spyre_stick: int
    spyre_gate: torch.Tensor
    spyre_up: torch.Tensor
    spyre_down: torch.Tensor
    spyre_route_identity: torch.Tensor

    def spyre_init(self) -> None:
        """Per-instance setup, in place of the ``__init__`` a retype skips."""
        from vllm.config import CompilationMode, get_current_vllm_config

        assert not self.hidden_size_per_layer_input, (
            "Spyre Gemma-4 MoE does not support per-layer embeddings (PLE); "
            f"hidden_size_per_layer_input={self.hidden_size_per_layer_input}."
        )
        experts = self.spyre_experts()
        moe_config = experts.moe_config
        if moe_config.tp_size > 1 or moe_config.ep_size > 1:
            raise NotImplementedError(
                "Spyre Gemma-4 MoE does not support tensor or expert parallelism "
                f"(tp_size={moe_config.tp_size}, ep_size={moe_config.ep_size}). "
                "Run with --tensor-parallel-size 1."
            )
        if get_current_vllm_config().compilation_config.mode is CompilationMode.NONE:
            raise NotImplementedError(
                "Spyre Gemma-4 MoE requires torch.compile — torch.ops.spyre.keep_by_index, "
                "which builds the prefill routing weights, has no eager implementation. "
                "Run without --enforce-eager."
            )
        self._spyre_regions: dict[str, Any] = {}
        self.spyre_top_k = int(experts.top_k)

    def spyre_experts(self) -> RoutedExperts:
        return self.moe.experts.routed_experts

    def _spyre_region(self, name: str, fn: Any) -> Any:
        """``dynamic=False`` is mandatory: the Spyre backend rejects SymInt shapes.

        All layers share each region's code object, so layers 2..N hit the Inductor
        cache.
        """
        region = self._spyre_regions.get(name)
        if region is None:
            region = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)
            self._spyre_regions[name] = region
        return region

    # ty: the override narrows the second element to None, which is what upstream's
    # own body returns despite annotating it as a Tensor.
    def forward(  # ty: ignore[invalid-method-override]
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        per_layer_input: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Gathered experts for a single-token decode step, persistent otherwise."""
        from torch_spyre._inductor import config as spyre_config

        with spyre_config.patch(_MOE_COMPILER_CONFIG):
            if hidden_states.shape[0] == 1:
                out = self._spyre_region("gathered_layer", _gathered_layer)(
                    self, positions, hidden_states, **kwargs
                )
            else:
                # Four graphs: the expert matmul's named-dims context has to be
                # declared eagerly, and the routing must sit outside it.
                residual, probs, expert_input = self._spyre_region(
                    "prologue", _persistent_prologue
                )(self, positions, hidden_states, **kwargs)
                route = self._spyre_region("route", _persistent_route)(self, probs)
                _name_persistent_dims(expert_input, self.spyre_gate, self.spyre_up, self.spyre_down)
                try:
                    with spyre_config.patch(_PERSISTENT_COMPILER_CONFIG):
                        moe_out = self._spyre_region("experts", _persistent_experts)(
                            self, expert_input, route
                        )
                finally:
                    # Unconditional: a stale _enabled flag would leak the persistent form's
                    # named dims into the next region's compilation whenever this one is an
                    # Inductor cache hit and never reaches the propagation pass.
                    _reset_named_dims()
                out = self._spyre_region("combine", _combine_block)(self, residual, moe_out)
        return out, None


def adapt_moe_layers(layers: Iterable[nn.Module]) -> None:
    """Retype Gemma-4's MoE decoder layers onto the Spyre expert dispatch.

    Runs before the checkpoint is loaded; :func:`relayout_moe_experts` finishes the
    job once the expert weights are in.
    """
    adapted = 0
    for layer in layers:
        if not getattr(layer, "enable_moe_block", False):
            continue
        if type(layer) is not Gemma4DecoderLayer:
            raise RuntimeError(
                f"expected Gemma4DecoderLayer, got {type(layer).__name__}; the Spyre "
                "Gemma-4 MoE dispatch needs updating for this vLLM version."
            )
        layer.__class__ = SpyreGemma4MoEDecoderLayer
        cast("SpyreGemma4MoEDecoderLayer", layer).spyre_init()
        adapted += 1
    if adapted:
        logger.info(
            "Spyre: %d Gemma-4 MoE layers dispatch experts through the Spyre persistent / "
            "gathered paths instead of FusedMoE (whose kernels are CUDA-only).",
            adapted,
        )


def _to_spyre_expert_weight(weight: torch.Tensor) -> torch.Tensor:
    from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre

    moved = dma_moe_expert_weight_to_spyre(weight)
    return moved if moved is not None else weight.contiguous().to("spyre")


def relayout_moe_experts(layers: Iterable[nn.Module]) -> None:
    """Move every adapted layer's expert stacks into the layout its regions read."""
    for layer in layers:
        if isinstance(layer, SpyreGemma4MoEDecoderLayer):
            _relayout_experts(layer)


def _relayout_experts(layer: SpyreGemma4MoEDecoderLayer) -> None:
    """Split, transpose and move one layer's expert stacks, freeing the originals.

    ``FusedMoE`` stores ``w13`` as ``[E, 2M, H]`` (gate rows then up rows) and
    ``w2`` as ``[E, H, M]``. The Spyre regions contract on the *second* axis, so
    gate/up become ``[E, H, M]`` and down becomes ``[E, M, H]``. Gate and up stay two
    stacks rather than one fused ``[E, H, 2M]``: the gathered and persistent forms
    favour opposite layouts here, and only one of them can be stored.

    The device cannot hold the stacks and their relaid-out copies at once, so each is
    converted and freed before the next one starts.
    """
    from torch_spyre._C import get_elem_in_stick

    # ``get_parameter`` rather than attribute access: create_weights registers both
    # stacks dynamically, so only the lookup is typed.
    experts = layer.spyre_experts()
    w13 = experts.get_parameter("w13_weight").data
    w2_shape = tuple(experts.get_parameter("w2_weight").shape)
    num_experts, twice_inter, hidden = w13.shape
    inter = twice_inter // 2
    dtype = w13.dtype
    assert w2_shape == (num_experts, hidden, inter), (
        f"unexpected Gemma-4 expert weight shapes: w13={tuple(w13.shape)} w2={w2_shape}"
    )

    layer.spyre_gate = _to_spyre_expert_weight(w13[:, :inter, :].transpose(1, 2))
    layer.spyre_up = _to_spyre_expert_weight(w13[:, inter:, :].transpose(1, 2))
    del experts.w13_weight, w13

    w2 = experts.get_parameter("w2_weight").data
    # Fold the per-expert output scale into ``down`` instead of gathering it every
    # step. It multiplies the already renormalized routing weight, so the fold is
    # exact, and the checkpoint's values sit within 2% of 1.0, well inside fp16.
    w2.mul_(layer.moe.per_expert_scale.data.detach().to(w2.dtype).view(num_experts, 1, 1))
    layer.spyre_down = _to_spyre_expert_weight(w2.transpose(1, 2))
    del experts.w2_weight, w2

    # Elements per stick, from the stacks' own dtype: the top-k indices and the
    # routing weights are both widened onto a full stick before the compiler will
    # gather or restickify with them.
    layer.spyre_stick = get_elem_in_stick(dtype)
    # Identity for the routing-weight restickify. It has to originate on the
    # host: Spyre has no on-device eye/diag kernel.
    layer.spyre_route_identity = torch.eye(layer.spyre_stick, dtype=dtype).to("spyre")
    logger.info_once(
        "Spyre: relaid out the Gemma-4 MoE expert stacks (%d experts, hidden=%d, "
        "intermediate=%d) for on-device gather and matmul.",
        num_experts,
        hidden,
        inter,
    )
