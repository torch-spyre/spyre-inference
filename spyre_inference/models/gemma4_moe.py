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

Gemma-4's MoE layers run a dense MLP and a 128-expert top-8 block in parallel.
vLLM dispatches the expert block through ``FusedMoE``, whose kernels are CUDA /
Triton only, so Spyre needs its own dispatch. This module supplies it, ported
from the ``hf_adapters`` ``hf_gemma4_moe`` adapter, which is the reference
implementation for what the torch-spyre compiler accepts here.

Two dispatch forms, one per phase:

*Gathered*, for a single-token decode step: only the selected experts' weights
are gathered, one row per top-k slot, and contracted with per-row BMMs. Reads
``K`` expert weights instead of all ``E`` — the whole point of a sparse MoE — and
it is the latency-critical path, so it is one graph for the entire layer.

*Persistent*, for everything else: every expert is evaluated over every token as
one batched ``[E,T,H] x [E,H,M]`` matmul, and dense routing weights zero out the
unselected ``(token, expert)`` pairs. Reads each expert weight once regardless of
token count, which is what makes it the right form for a prefill chunk (the
gathered form would read ``T*K`` of them). It also has no alternative: the
gathered form's per-row combine has no legal device layout above one token.

The persistent form needs four graphs, because its expert matmul is tiled by
``spyre_hint`` scopes that resolve against named dims the driver has to declare
*eagerly*, between compilations, and its routing weights have to sit in a graph of
their own between the softmax and that context. So a Gemma-4 MoE layer compiles
its own regions and opts out of the model runner's whole-block compile; each
region documents its own boundary.

Both forms require ``torch.compile``. ``--enforce-eager`` is rejected at layer
construction: the persistent routing region's ``torch.ops.spyre.keep_by_index``
exists only as an Inductor lowering and silently returns ``None`` when called
eagerly.

Both forms read one shared device-resident expert stack, laid out by
:class:`SpyreFusedMoEMethod` below.
"""

from __future__ import annotations

import weakref
from typing import Any

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

logger = init_logger(__name__)

# Compiler config for the MoE regions, matching the reference adapter.
# ``frontend_pool_allocation`` has the front end allocate the scratch pool as a
# real tensor and pass its address in, rather than letting the backend
# self-allocate. Both flags are read only during codegen, so scoping them to the
# region calls costs nothing once the region is traced.
_MOE_COMPILER_CONFIG = {"frontend_pool_allocation": True}
# The persistent form's all-expert matmul is not in the default LX planning set.
_PERSISTENT_COMPILER_CONFIG = {"allow_all_ops_in_lx_planning": True}


# --------------------------------------------------------------------------- #
# Compiled device regions
# --------------------------------------------------------------------------- #


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

    The value has to travel through a full stick before the int32 cast: widen
    [T,K] to [T,K,stick], make it contiguous so the layout pass moves the stick
    dim onto that axis (this is the restickify), widen to fp32 for address
    arithmetic, take the one fp32 stick, then read lane 0.

    The ``.contiguous()`` is load-bearing. Materializing the widening with a
    pointwise op instead (``relu``, as the routing path does) yields an index
    layout the backend's gather-index conversion rejects outright
    (``fmod(dsDim, size) == 0``).
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

    Unlike the reference adapter this needs no ``spyre_hint`` row tiling: measured
    at the real shapes, every tile from 2 to 64 lands on the same 5.90 ms and the
    region lowers without the scope at all.
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
    # SpyreFusedMoEMethod), so nothing else joins it here.
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

    ``work_div`` splits a dim across that many cores and so has to divide it
    exactly. The reference adapter hardcodes the full core count, which only
    holds when the token count is a multiple of it — vLLM's compile buckets
    (16, 24, 40, 56, …) are not, so take the largest legal divisor instead.
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

    ``spyre_hint(num_tiles_per_dim={"E": ...}, work_div={"T": ...})`` names dims
    that propagate from the graph's inputs, and the propagation pass reads those
    names off the *real* input tensors during lowering. So this has to run
    eagerly, before the region is traced — which is why the layer keeps its FFN
    in a graph of its own.
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


# --------------------------------------------------------------------------- #
# Decoder-layer regions
# --------------------------------------------------------------------------- #


def _attn_block(layer: Any, positions: torch.Tensor, hidden_states: torch.Tensor, **kwargs):
    """``input_layernorm`` -> attention -> ``post_attention_layernorm`` -> residual add.

    Gemma-4 norms the attention *output* before adding the residual (a "sandwich"
    norm), so the add cannot be fused into the next norm.
    """
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)
    attn_out = layer.self_attn(positions=positions, hidden_states=normed, **kwargs)
    return residual + layer.post_attention_layernorm(attn_out)


def _combine_block(layer: Any, residual: torch.Tensor, moe_out: torch.Tensor):
    """Dense MLP in parallel with the MoE output, then the sandwich norms + scalar."""
    dense = layer.mlp(layer.pre_feedforward_layernorm(residual))
    dense = layer.post_feedforward_layernorm_1(dense)
    moe = layer.post_feedforward_layernorm_2(moe_out.to(residual.dtype))
    ffn_out = layer.post_feedforward_layernorm(dense + moe)
    return (residual + ffn_out) * layer.layer_scalar


def _persistent_prologue(
    layer: Any, positions: torch.Tensor, hidden_states: torch.Tensor, **kwargs
):
    """Attention, the router probabilities, and the expert block's normed input.

    The router finishes here rather than alongside the routing weights: softmax
    puts its stick on the token axis and ``keep_by_index`` needs it on the expert
    axis, so the probabilities have to cross a graph boundary — which restores the
    default layout — before the routing weights are built.
    """
    residual = _attn_block(layer, positions, hidden_states, **kwargs)
    probs = torch.softmax(layer.router(residual), dim=-1)
    return residual, probs, layer.pre_feedforward_layernorm_2(residual)


def _persistent_route(layer: Any, probs: torch.Tensor) -> torch.Tensor:
    """Dense routing weights.

    Its own graph for two independent reasons: it must not share one with the
    softmax (layout, see ``_persistent_prologue``) nor with ``_moe_persistent``,
    because ``keep_by_index`` is a reduction over the top-k axis that the
    named-dims propagation pass cannot map (its first input dep, the
    probabilities, does not carry that axis).
    """
    moe = layer.moe
    return _moe_persistent_routing(
        probs, moe.spyre_route_identity, moe.spyre_top_k, moe.spyre_stick
    )


def _persistent_experts(
    layer: Any, expert_input: torch.Tensor, route: torch.Tensor
) -> torch.Tensor:
    """All-expert FFN. The only region that needs the eager named-dims context."""
    moe = layer.moe
    return _moe_persistent(expert_input, route, moe.spyre_gate, moe.spyre_up, moe.spyre_down)


def _gathered_layer(layer: Any, positions: torch.Tensor, hidden_states: torch.Tensor, **kwargs):
    """Attention + router + gathered-expert FFN + combine, all in one graph."""
    moe = layer.moe
    residual = _attn_block(layer, positions, hidden_states, **kwargs)
    probs = torch.softmax(layer.router(residual), dim=-1)
    moe_out = _moe_gathered(
        layer.pre_feedforward_layernorm_2(residual),
        probs,
        moe.spyre_gate,
        moe.spyre_up,
        moe.spyre_down,
        moe.spyre_top_k,
        moe.spyre_stick,
    )
    return _combine_block(layer, residual, moe_out)


# --------------------------------------------------------------------------- #
# Layer patch
# --------------------------------------------------------------------------- #


def _spyre_region(layer: Any, name: str, fn: Any) -> Any:
    """Return ``fn`` compiled as its own graph, built on first use and memoized.

    ``dynamic=False`` is mandatory: the Spyre backend rejects SymInt shapes. All
    layers share each region's code object, so layers 2..N hit the Inductor cache.
    """
    region = layer._spyre_regions.get(name)
    if region is None:
        region = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)
        layer._spyre_regions[name] = region
    return region


def _spyre_moe_layer_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    per_layer_input: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Gemma-4 decoder layer on Spyre: gathered experts for a single-token decode
    step, all-expert persistent dispatch for anything wider."""
    from torch_spyre._inductor import config as spyre_config

    if not self.enable_moe_block:
        return self._spyre_dense_forward(
            positions, hidden_states, residual, per_layer_input=per_layer_input, **kwargs
        )

    with spyre_config.patch(_MOE_COMPILER_CONFIG):
        if hidden_states.shape[0] == 1:
            out = _spyre_region(self, "gathered_layer", _gathered_layer)(
                self, positions, hidden_states, **kwargs
            )
        else:
            # Persistent form, four graphs: the expert matmul needs a named-dims
            # context declared eagerly (so it cannot share a graph with what runs
            # before it), and the routing must sit between the two — out of that
            # context, and downstream of the softmax's own graph.
            residual, probs, expert_input = _spyre_region(self, "prologue", _persistent_prologue)(
                self, positions, hidden_states, **kwargs
            )
            route = _spyre_region(self, "route", _persistent_route)(self, probs)
            moe = self.moe
            _name_persistent_dims(expert_input, moe.spyre_gate, moe.spyre_up, moe.spyre_down)
            try:
                with spyre_config.patch(_PERSISTENT_COMPILER_CONFIG):
                    moe_out = _spyre_region(self, "experts", _persistent_experts)(
                        self, expert_input, route
                    )
            finally:
                # Unconditional: a stale _enabled flag would leak the persistent form's
                # named dims into the next region's compilation whenever this one is an
                # Inductor cache hit and never reaches the propagation pass.
                _reset_named_dims()
            out = _spyre_region(self, "combine", _combine_block)(self, residual, moe_out)
    return out, None


def install_spyre_patches() -> None:
    """Route Gemma-4 MoE decoder layers through the Spyre expert dispatch."""
    from vllm.config import CompilationMode, get_cached_compilation_config
    from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer

    if getattr(Gemma4DecoderLayer, "_spyre_moe_patched", False):
        return

    orig_init = Gemma4DecoderLayer.__init__
    orig_forward = Gemma4DecoderLayer.forward

    def __init__(self, config, *args, **kwargs) -> None:
        orig_init(self, config, *args, **kwargs)
        if not self.enable_moe_block:
            return
        assert not self.hidden_size_per_layer_input, (
            "Spyre Gemma-4 MoE does not support per-layer embeddings (PLE); "
            f"hidden_size_per_layer_input={self.hidden_size_per_layer_input}."
        )
        # The model runner must not wrap this layer in one whole-block graph: the
        # persistent path needs named-dims context set eagerly between two of its
        # compilations. See _spyre_moe_layer_forward.
        self.spyre_compiles_own_regions = True
        if get_cached_compilation_config().mode is CompilationMode.NONE:
            # Fail here rather than with an AttributeError 30 layers deep: the
            # persistent path's `torch.ops.spyre.keep_by_index` exists only as an
            # Inductor lowering and returns None when called eagerly.
            raise NotImplementedError(
                "Spyre Gemma-4 MoE requires torch.compile — torch.ops.spyre.keep_by_index, "
                "which builds the prefill routing weights, has no eager implementation. "
                "Run without --enforce-eager."
            )
        self._spyre_regions: dict[str, Any] = {}
        self.moe.spyre_top_k = int(config.top_k_experts)
        # The expert relayout runs in the RoutedExperts weight hook, which is handed
        # that module alone; point it back at the Gemma4MoE that owns
        # per_expert_scale and holds the relaid-out stacks the regions read. A
        # weakref, because assigning a Module to one of its own descendants
        # registers it as a submodule and every module walk then recurses forever.
        _routed_experts(self.moe).spyre_moe_owner = weakref.ref(self.moe)

    Gemma4DecoderLayer.__init__ = __init__  # ty: ignore[invalid-assignment]
    Gemma4DecoderLayer._spyre_dense_forward = orig_forward
    Gemma4DecoderLayer.forward = _spyre_moe_layer_forward  # ty: ignore[invalid-assignment]
    Gemma4DecoderLayer._spyre_moe_patched = True
    logger.info(
        "Spyre: Gemma-4 MoE layers dispatch experts through the Spyre persistent / "
        "gathered paths instead of FusedMoE (whose kernels are CUDA-only)."
    )


# --------------------------------------------------------------------------- #
# Expert-weight layout
# --------------------------------------------------------------------------- #


def _routed_experts(moe: Any) -> Any:
    """The ``RoutedExperts`` under a ``Gemma4MoE`` — the module owning w13/w2."""
    for module in moe.modules():
        if hasattr(module, "w13_weight") and hasattr(module, "w2_weight"):
            return module
    raise AssertionError("Gemma4MoE has no w13_weight/w2_weight to relay out")


def _to_spyre_expert_weight(weight: torch.Tensor) -> torch.Tensor:
    """DMA an ``[E, C, F]`` expert stack in the gather- and matmul-friendly layout."""
    from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre

    moved = dma_moe_expert_weight_to_spyre(weight)
    return moved if moved is not None else weight.contiguous().to("spyre")


@UnquantizedFusedMoEMethod.register_oot(name="UnquantizedFusedMoEMethod")
class SpyreFusedMoEMethod(UnquantizedFusedMoEMethod):
    """Rebuild Gemma-4's expert stacks on device, in the layout the regions read.

    This class computes nothing: vLLM's out-of-tree MoE backend builds no expert
    kernel, and Gemma-4 never asks it to — the decoder-layer patch above calls the
    regions directly and ``FusedMoE.forward`` stays dead code. It exists for
    ``process_weights_after_loading``, the one upstream hook that runs once the
    checkpoint is loaded and before the model reaches the device, which is where a
    device-specific expert layout has to happen.
    """

    def process_weights_after_loading(self, layer: Any) -> None:
        super().process_weights_after_loading(layer)
        owner = getattr(layer, "spyre_moe_owner", None)
        if owner is None:
            # Not a Gemma-4 MoE: no Spyre expert layout is defined for it, and its
            # forward would already have failed on the empty OOT backend.
            return
        _relayout_experts(layer, owner())


def _relayout_experts(experts: Any, moe: Any) -> None:
    """Split, transpose and move one layer's expert stacks, freeing the originals.

    ``FusedMoE`` stores ``w13`` as ``[E, 2M, H]`` (gate rows then up rows) and
    ``w2`` as ``[E, H, M]``. The Spyre regions contract on the *second* axis, so
    gate/up become ``[E, H, M]`` and down becomes ``[E, M, H]``. Gate and up stay
    two stacks: fusing them into ``[E, H, 2M]`` is 7% better for the gathered form
    and 6.5x worse for the persistent one, and there is only room for one layout.

    The stacks are ~45 GB for this checkpoint and the device cannot hold both them
    and the relaid-out copies, so each is converted and freed before the next one
    starts; host peak stays at one layer's worth.
    """
    from torch_spyre._C import get_elem_in_stick

    w13: torch.Tensor = experts.w13_weight.data
    num_experts, twice_inter, hidden = w13.shape
    inter = twice_inter // 2
    assert experts.w2_weight.shape == (num_experts, hidden, inter), (
        f"unexpected Gemma-4 expert weight shapes: w13={tuple(w13.shape)} "
        f"w2={tuple(experts.w2_weight.shape)}"
    )

    moe.spyre_gate = _to_spyre_expert_weight(w13[:, :inter, :].transpose(1, 2))
    moe.spyre_up = _to_spyre_expert_weight(w13[:, inter:, :].transpose(1, 2))
    del experts.w13_weight, w13

    w2: torch.Tensor = experts.w2_weight.data
    # Fold the per-expert output scale into ``down`` instead of gathering it
    # alongside the expert weights every step. It multiplies the already
    # renormalized routing weight, so pushing it onto that expert's rows is
    # exact; the checkpoint's values sit within 2% of 1.0, well inside fp16.
    w2.mul_(moe.per_expert_scale.data.detach().to(w2.dtype).view(num_experts, 1, 1))
    moe.spyre_down = _to_spyre_expert_weight(w2.transpose(1, 2))
    del experts.w2_weight, w2

    # Elements per Spyre stick at the platform's forced compute dtype. The top-k
    # indices and the routing weights are both widened onto a full stick before
    # the compiler will gather or restickify with them.
    moe.spyre_stick = get_elem_in_stick(torch.float16)
    # Identity for the routing-weight restickify. It has to originate on the
    # host: Spyre has no on-device eye/diag kernel.
    moe.spyre_route_identity = torch.eye(moe.spyre_stick, dtype=torch.float16).to("spyre")
    logger.info_once(
        "Spyre: relaid out the Gemma-4 MoE expert stacks (%d experts, hidden=%d, "
        "intermediate=%d) for on-device gather and matmul.",
        num_experts,
        hidden,
        inter,
    )
