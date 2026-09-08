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

"""Spyre routed-expert dispatch, plugged into vLLM's own ``FusedMoE`` seams.

vLLM's unquantized MoE oracle has a kernel for CUDA, ROCm, XPU and CPU, and for an
out-of-tree platform it selects ``UnquantizedMoeBackend.OOT`` — no kernel, and
``process_weights_after_loading`` returns early, "OOT handles internally". This
module is that OOT handling: a ``CustomOp.register_oot`` replacement for
``UnquantizedFusedMoEMethod`` that lays the expert stacks out for Spyre at load
time and computes the experts on device.

Everything above the expert compute stays upstream's. ``Gemma4DecoderLayer.forward``
runs unmodified, so the model runner compiles the block as usual, and ``MoERunner``
still drives the padding, the routed-input transform and the layer plumbing. vLLM
reaches the experts through ``torch.ops.vllm.moe_forward``, an opaque custom op, so
this code runs *eagerly inside* the block's compiled graph — the same seam the
attention backend uses — which is what lets it drive compiled regions of its own
without the runner having to leave the layer uncompiled.

Two forms, both reading the stacks :func:`_relayout_experts` builds:

*Gathered*, for a single-token decode step: gather only the selected experts'
weights and contract with per-row BMMs, in one graph. Its combine step has no legal
device layout above one token.

*Persistent*, for everything else: every expert over every token as one batched
``[E,T,H] x [E,H,M]`` matmul, with dense routing weights zeroing the unselected
pairs. It needs three graphs, because its ``spyre_hint`` tiling resolves against
named dims the driver has to declare *eagerly*, between compilations.
"""

from __future__ import annotations

import weakref
from functools import cache
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts as _RoutedExperts,
    )
    from vllm.model_executor.models.gemma4 import Gemma4MoE

    class RoutedExperts(_RoutedExperts):
        """Type-only view of the Spyre attributes hung on vLLM's ``RoutedExperts``.

        ``nn.Module.__getattr__`` types every dynamically added attribute as
        ``Tensor | Module``, so without this the regions cannot read them.
        """

        spyre_moe_owner: weakref.ref[Gemma4MoE]
        spyre_regions: dict[str, Any]
        spyre_stick: int
        spyre_gate: torch.Tensor
        spyre_up: torch.Tensor
        spyre_down: torch.Tensor
        spyre_route_identity: torch.Tensor


logger = init_logger(__name__)

# ``frontend_pool_allocation`` has the front end allocate the scratch pool as a
# real tensor and pass its address in, instead of the backend self-allocating.
_MOE_COMPILER_CONFIG = {"frontend_pool_allocation": True}
# The persistent form's all-expert matmul is not in the default LX planning set.
_PERSISTENT_COMPILER_CONFIG = {"allow_all_ops_in_lx_planning": True}


@cache
def _compiler_scopes() -> tuple[Any, Any]:
    """The two torch-spyre config scopes the forms compile under.

    Built once and reused: ``patch`` defines a fresh class per call, and the objects
    it returns are reusable as long as neither is entered inside itself.
    """
    from torch_spyre._inductor import config as spyre_config

    return (
        spyre_config.patch(_MOE_COMPILER_CONFIG),
        spyre_config.patch(_PERSISTENT_COMPILER_CONFIG),
    )


def _topk(probs: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k over the expert axis, padded to two rows at T=1.

    A length-1 reduction aborts the Spyre compiler ("stick expression 1").
    """
    tokens = probs.shape[0]
    padded = probs.expand(2, -1).contiguous() if tokens == 1 else probs
    weights, indices = torch.topk(padded, top_k, dim=-1)
    return weights[:tokens], indices[:tokens]


def _gather_indices(indices: torch.Tensor, top_k: int, stick: int) -> torch.Tensor:
    """Turn topk's fp16 expert ids into device int32 gather indices.

    The ids have to travel through a full stick before the int32 cast, and the
    ``.contiguous()`` is what restickifies them; widening with a pointwise op instead
    yields a layout the backend's gather-index conversion rejects.
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

    ``x`` ``[T,H]``, ``gate``/``up`` ``[E,H,M]``, ``down`` ``[E,M,H]``, out ``[T,H]``.
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
    # has no legal layout. The per-expert output scale is already in ``down``.
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

    ``work_div`` has to divide the dim exactly, and vLLM's compile buckets (16, 24,
    40, …) are not all multiples of the core count — hence the largest legal divisor.
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
    """Expert FFN over every expert, summed against ``route`` ``[T,E,1]``.

    ``x`` ``[T,H]``, weights as in ``_moe_gathered``, out ``[T,H]``.
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

    The propagation pass reads these names off the *real* input tensors, so this has
    to run eagerly — which is why the FFN gets a graph of its own.
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


def _probs(router_logits: torch.Tensor) -> torch.Tensor:
    """Gemma-4 routing: softmax over *all* experts, before the top-k.

    Its own region on the persistent path: softmax puts its stick on the token axis
    and ``keep_by_index`` needs it on the expert axis.
    """
    return torch.softmax(router_logits, dim=-1)


def _gathered(layer: RoutedExperts, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
    return _moe_gathered(
        x,
        _probs(router_logits),
        layer.spyre_gate,
        layer.spyre_up,
        layer.spyre_down,
        layer.top_k,
        layer.spyre_stick,
    )


def _route(layer: RoutedExperts, probs: torch.Tensor) -> torch.Tensor:
    """Dense routing weights, in a graph of their own.

    They can share one with neither the softmax (layout, see ``_probs``) nor
    ``_moe_persistent``: ``keep_by_index`` reduces over the top-k axis, which the
    named-dims propagation cannot map onto its probability input.
    """
    return _moe_persistent_routing(
        probs, layer.spyre_route_identity, layer.top_k, layer.spyre_stick
    )


def _experts(layer: RoutedExperts, x: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
    """All-expert FFN. The only region that needs the eager named-dims context."""
    return _moe_persistent(x, route, layer.spyre_gate, layer.spyre_up, layer.spyre_down)


def _region(layer: RoutedExperts, name: str, fn: Any) -> Any:
    """One of the layer's memoized compiled regions; ``dynamic=False`` — no SymInts.

    Memoized on the layer rather than on the quant method: vLLM may share one method
    instance across layers, and each region closes over its layer's own weights.
    """
    region = layer.spyre_regions.get(name)
    if region is None:
        region = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)
        layer.spyre_regions[name] = region
    return region


@CustomOp.register_oot(name="UnquantizedFusedMoEMethod")
class SpyreUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """Unquantized routed experts on Spyre, in place of the backend the oracle lacks.

    Inert for any MoE model that has not opted in: only a layer carrying the
    ``spyre_moe_owner`` backref :func:`adapt_moe_layers` sets is taken over, and any
    other layer falls through to upstream, which fails the same way it does without
    this class.
    """

    @property
    def is_monolithic(self) -> bool:
        """Routing runs in :meth:`apply_monolithic`, not in the caller.

        vLLM's modular path calls ``select_experts`` *before* ``apply``, in the eager
        context of the ``moe_forward`` op where Spyre has no ``topk``; monolithic
        lands it inside a compiled region instead.
        """
        return True

    def process_weights_after_loading(self, layer: _RoutedExperts) -> None:
        super().process_weights_after_loading(layer)
        if getattr(layer, "spyre_moe_owner", None) is not None:
            _relayout_experts(cast("RoutedExperts", layer))

    def apply_monolithic(
        self,
        layer: _RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gathered experts for a single-token decode step, persistent otherwise."""
        if getattr(layer, "spyre_moe_owner", None) is None:
            return super().apply_monolithic(layer, x, router_logits, input_ids)

        layer = cast("RoutedExperts", layer)
        moe_scope, persistent_scope = _compiler_scopes()
        with moe_scope:
            if x.shape[0] == 1:
                return _region(layer, "gathered", _gathered)(layer, x, router_logits)

            probs = _region(layer, "probs", _probs)(router_logits)
            route = _region(layer, "route", _route)(layer, probs)
            _name_persistent_dims(x, layer.spyre_gate, layer.spyre_up, layer.spyre_down)
            try:
                with persistent_scope:
                    return _region(layer, "experts", _experts)(layer, x, route)
            finally:
                # Unconditional: on an Inductor cache hit the propagation pass never
                # runs, and the stale names would leak into the next compilation.
                _reset_named_dims()


def adapt_moe_layers(layers: Iterable[nn.Module]) -> None:
    """Opt each Gemma-4 MoE layer's ``RoutedExperts`` into the Spyre dispatch.

    vLLM's post-load hook is handed the ``RoutedExperts``, which does not know the
    ``Gemma4MoE`` holding ``per_expert_scale``; the backref supplies it. A weakref,
    so it is not a submodule cycle (a plain child->parent attribute makes module
    walks recurse forever).

    Raises:
        NotImplementedError: under ``--enforce-eager``, or for a PLE checkpoint.
    """
    from vllm.config import CompilationMode, get_current_vllm_config

    adapted = 0
    for layer in layers:
        moe = getattr(layer, "moe", None)
        if moe is None:
            continue
        if get_current_vllm_config().compilation_config.mode is CompilationMode.NONE:
            raise NotImplementedError(
                "Spyre Gemma-4 MoE requires torch.compile — torch.ops.spyre.keep_by_index, "
                "which builds the prefill routing weights, has no eager implementation. "
                "Run without --enforce-eager."
            )
        if layer.hidden_size_per_layer_input:
            raise NotImplementedError(
                "Spyre Gemma-4 MoE does not support per-layer embeddings (PLE); "
                f"hidden_size_per_layer_input={layer.hidden_size_per_layer_input}."
            )
        routed = moe.experts.routed_experts
        routed.spyre_moe_owner = weakref.ref(moe)
        routed.spyre_regions = {}
        adapted += 1
    if adapted:
        logger.info(
            "Spyre: %d Gemma-4 MoE layers dispatch their experts through the Spyre "
            "gathered / persistent forms; vLLM's unquantized MoE oracle has no "
            "out-of-tree backend.",
            adapted,
        )


def _to_spyre_expert_weight(weight: torch.Tensor) -> torch.Tensor:
    from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre

    moved = dma_moe_expert_weight_to_spyre(weight)
    return moved if moved is not None else weight.contiguous().to("spyre")


def _relayout_experts(layer: RoutedExperts) -> None:
    """Split, transpose and move one layer's expert stacks, freeing the originals.

    ``create_weights`` stores ``w13`` as ``[E, 2M, H]`` (gate rows then up rows) and
    ``w2`` as ``[E, H, M]``; the Spyre regions contract on the *second* axis, so
    gate/up become ``[E, H, M]`` and down ``[E, M, H]``. Gate and up stay separate
    stacks because the two forms favour opposite layouts for a fused one.

    The device cannot hold both layouts at once, so each stack is converted and freed
    before the next one starts.
    """
    from torch_spyre._C import get_elem_in_stick

    moe_config = layer.moe_config
    if moe_config.tp_size > 1 or moe_config.ep_size > 1:
        raise NotImplementedError(
            "Spyre Gemma-4 MoE does not support tensor or expert parallelism "
            f"(tp_size={moe_config.tp_size}, ep_size={moe_config.ep_size}). "
            "Run with --tensor-parallel-size 1."
        )

    # ``get_parameter``, not attribute access: create_weights registers both stacks
    # dynamically, so only the lookup is typed.
    w13 = layer.get_parameter("w13_weight").data
    w2_shape = tuple(layer.get_parameter("w2_weight").shape)
    num_experts, twice_inter, hidden = w13.shape
    inter = twice_inter // 2
    dtype = w13.dtype
    assert w2_shape == (num_experts, hidden, inter), (
        f"unexpected Gemma-4 expert weight shapes: w13={tuple(w13.shape)} w2={w2_shape}"
    )

    layer.spyre_gate = _to_spyre_expert_weight(w13[:, :inter, :].transpose(1, 2))
    layer.spyre_up = _to_spyre_expert_weight(w13[:, inter:, :].transpose(1, 2))
    del layer.w13_weight, w13

    owner = layer.spyre_moe_owner()
    assert owner is not None, "the Gemma4MoE owning these experts was collected"
    w2 = layer.get_parameter("w2_weight").data
    # Fold the per-expert output scale into ``down`` instead of gathering it every
    # step: it multiplies the already renormalized routing weight, so this is exact.
    w2.mul_(owner.per_expert_scale.data.detach().to(w2.dtype).view(num_experts, 1, 1))
    layer.spyre_down = _to_spyre_expert_weight(w2.transpose(1, 2))
    del layer.w2_weight, w2

    # Elements per stick: both the top-k indices and the routing weights are widened
    # onto a full stick before the compiler will gather or restickify with them.
    layer.spyre_stick = get_elem_in_stick(dtype)
    # Identity for the routing-weight restickify; host-built, as Spyre has no eye kernel.
    layer.spyre_route_identity = torch.eye(layer.spyre_stick, dtype=dtype).to("spyre")

    logger.info_once(
        "Spyre: relaid out the Gemma-4 MoE expert stacks (%d experts, hidden=%d, "
        "intermediate=%d) for on-device gather and matmul.",
        num_experts,
        hidden,
        inter,
    )
