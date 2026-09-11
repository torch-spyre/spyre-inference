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

"""Spyre backend for vLLM's unquantized routed experts.

vLLM's unquantized MoE oracle selects ``UnquantizedMoeBackend.OOT`` without building a
kernel, leaving ``UnquantizedFusedMoEMethod`` for the platform to implement. Model
adapters opt a layer in with a ``SpyreMoERecipe``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts as _RoutedExperts,
    )

    class RoutedExperts(_RoutedExperts):
        spyre_moe_recipe: SpyreMoERecipe
        spyre_moe_regions: dict[str, Any]
        spyre_moe_stick: int
        spyre_moe_gate: torch.Tensor
        spyre_moe_up: torch.Tensor
        spyre_moe_down: torch.Tensor
        spyre_moe_route_identity: torch.Tensor


logger = init_logger(__name__)

_MOE_COMPILER_CONFIG = {"frontend_pool_allocation": True}
_PERSISTENT_COMPILER_CONFIG = {"allow_all_ops_in_lx_planning": True}


@dataclass(frozen=True)
class SpyreMoERecipe:
    """The upstream MoE semantics implemented by Spyre's current kernels.

    The routing function follows vLLM's canonical ``(topk_weights, topk_ids)``
    contract. The persistent form expands that result to Spyre's dense route
    layout without selecting top-k a second time.
    """

    activation: Literal["gelu_tanh", "silu"]
    routing: Literal["full_softmax", "topk_softmax"]
    prepare_down_weight: Callable[[torch.Tensor], torch.Tensor] | None = None


def _validate_recipe(layer: _RoutedExperts, recipe: SpyreMoERecipe) -> None:
    if recipe.routing == "full_softmax" and layer.custom_routing_function is None:
        raise NotImplementedError(
            "Full-softmax routing requires a model-specific Spyre MoE recipe."
        )


@cache
def _compiler_scopes() -> tuple[Any, Any]:
    from torch_spyre._inductor import config as spyre_config

    return (
        spyre_config.patch(_MOE_COMPILER_CONFIG),
        spyre_config.patch(_PERSISTENT_COMPILER_CONFIG),
    )


def configure_spyre_moe_layer(layer: _RoutedExperts, recipe: SpyreMoERecipe) -> None:
    """Opt one upstream ``RoutedExperts`` layer into Spyre, before weight loading.

    The post-load hook replaces vLLM's source layout with a second, transposed
    device layout. It frees source parameters because holding both layouts would
    double MoE weight memory, so unsupported configurations must be rejected here.
    """
    moe = layer.moe_config
    _validate_recipe(layer, recipe)
    if not isinstance(layer.quant_method, UnquantizedFusedMoEMethod):
        raise NotImplementedError("Spyre MoE backend does not support quantized experts.")
    if moe.is_lora_enabled:
        raise NotImplementedError("Spyre MoE backend does not support LoRA experts.")
    if moe.has_bias:
        raise NotImplementedError("Spyre MoE backend does not support expert biases.")
    parallel = moe.moe_parallel_config
    # TP only narrows each expert's ``M``, and MoERunner all-reduces the partial sums.
    # The other axes would split the expert stacks the regions hold whole.
    unsupported = {
        "ep_size": moe.ep_size,
        "dp_size": moe.dp_size,
        "pcp_size": moe.pcp_size,
        "sp_size": moe.sp_size,
    }
    active = {name: value for name, value in unsupported.items() if value != 1}
    if active or parallel.enable_eplb:
        detail = ", ".join(f"{name}={value}" for name, value in active.items())
        if parallel.enable_eplb:
            detail = f"{detail}, enable_eplb=True" if detail else "enable_eplb=True"
        raise NotImplementedError(f"Spyre MoE backend requires local experts ({detail}).")
    if (
        layer.global_num_experts != layer.local_num_experts
        or moe.num_logical_experts != moe.num_experts
    ):
        raise NotImplementedError("Spyre MoE backend does not support remapped experts.")
    if not layer.renormalize:
        raise NotImplementedError("Spyre MoE backend requires normalized top-k routing weights.")
    if layer.apply_router_weight_on_input:
        raise NotImplementedError("Spyre MoE backend does not support input-weighted routing.")
    if layer.activation.value != recipe.activation:
        raise NotImplementedError(
            f"Spyre MoE recipe requires activation={recipe.activation!r}, "
            f"got {layer.activation.value!r}."
        )
    layer = cast("RoutedExperts", layer)
    layer.spyre_moe_recipe = recipe
    layer.spyre_moe_regions = {}


def _topk(values: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = values.shape[0]
    # A single-row top-k does not lower; pad to two rows and drop the copy.
    padded = values.expand(2, -1).contiguous() if tokens == 1 else values
    weights, indices = torch.topk(padded, top_k, dim=-1)
    return weights[:tokens], indices[:tokens]


def _routing_weights(
    router_logits: torch.Tensor,
    top_k: int,
    routing: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if routing == "full_softmax":
        selected_values, indices = _topk(torch.softmax(router_logits, dim=-1), top_k)
        weights = selected_values / selected_values.sum(-1, keepdim=True)
    else:
        selected_logits, indices = _topk(router_logits, top_k)
        weights = torch.softmax(selected_logits, dim=-1)
    return weights, indices


def _gather_indices(indices: torch.Tensor, top_k: int, stick: int) -> torch.Tensor:
    tokens = indices.shape[0]
    widened = indices[..., None].expand(tokens, top_k, stick).contiguous()
    address = widened.to(torch.float32)[..., : stick // 2].to(torch.int32)
    return address[..., 0]


def _activation(x: torch.Tensor, up: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "gelu_tanh":
        return F.gelu(x, approximate="tanh") * up
    return F.silu(x) * up


def _moe_gathered(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    top_k: int,
    stick: int,
    routing: str,
    activation: str,
) -> torch.Tensor:
    tokens, hidden = x.shape
    weights, indices = _routing_weights(router_logits, top_k, routing)
    indices = _gather_indices(indices, top_k, stick)
    rows, inter = tokens * top_k, gate.shape[-1]
    inputs = x[:, None, :].expand(tokens, top_k, hidden).contiguous().reshape(rows, 1, hidden)
    gate_out = torch.bmm(inputs, gate[indices].reshape(rows, hidden, inter))
    up_out = torch.bmm(inputs, up[indices].reshape(rows, hidden, inter))
    expert_out = torch.bmm(
        _activation(gate_out, up_out, activation), down[indices].reshape(rows, inter, hidden)
    ).reshape(tokens, top_k, hidden)
    return (expert_out * weights[..., None]).sum(dim=1)


def _moe_persistent_routing(
    values: torch.Tensor,
    route_identity: torch.Tensor,
    top_k: int,
    stick: int,
) -> torch.Tensor:
    _, selected = _topk(values, top_k)
    weights = torch.ops.spyre.keep_by_index(
        values,  # ty: ignore[invalid-argument-type]
        selected,  # ty: ignore[invalid-argument-type]
        -1,  # ty: ignore[invalid-argument-type]
        0.0,  # ty: ignore[invalid-argument-type]
    )
    weights = weights / weights.sum(-1, keepdim=True)
    packed = torch.relu(weights.unsqueeze(-1).expand(-1, -1, stick))
    return (packed @ route_identity)[..., :1]


def _moe_persistent_selected_routing(
    dense_topk_probs: torch.Tensor, route_identity: torch.Tensor, stick: int
) -> torch.Tensor:
    """Convert dense weights already selected by vLLM-style top-k routing."""
    packed = torch.relu(dense_topk_probs.unsqueeze(-1).expand(-1, -1, stick))
    return (packed @ route_identity)[..., :1]


def _token_cores(tokens: int) -> int:
    from torch_spyre._inductor import config as spyre_config

    limit = min(tokens, spyre_config.sencores)
    return max(split for split in range(1, limit + 1) if tokens % split == 0)


def _name_persistent_dims(
    x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor
) -> None:
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


def _moe_persistent(
    x: torch.Tensor,
    route: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    from torch_spyre._inductor.propagate_hints import spyre_hint

    experts = gate.shape[0]
    with spyre_hint(named_dims=["E", "T", "ONE"]):
        route = route.permute(1, 0, 2).contiguous().clone()
    with spyre_hint(num_tiles_per_dim={"E": experts}, work_div={"T": _token_cores(x.shape[0])}):
        h = x.unsqueeze(0)
        activated = _activation(torch.matmul(h, gate), torch.matmul(h, up), activation)
        return (torch.matmul(activated, down) * route).sum(dim=0)


def _gathered(layer: RoutedExperts, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
    recipe = layer.spyre_moe_recipe
    return _moe_gathered(
        x,
        router_logits,
        layer.spyre_moe_gate,
        layer.spyre_moe_up,
        layer.spyre_moe_down,
        layer.top_k,
        layer.spyre_moe_stick,
        recipe.routing,
        recipe.activation,
    )


def _probs(router_logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(router_logits, dim=-1)


def _topk_probs(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Materialize canonical vLLM top-k weights in dense expert order."""
    topk_weights, topk_ids = _routing_weights(router_logits, top_k, "topk_softmax")
    return torch.zeros_like(router_logits).scatter(-1, topk_ids, topk_weights)


def _route(layer: RoutedExperts, probs: torch.Tensor) -> torch.Tensor:
    return _moe_persistent_routing(
        probs,
        layer.spyre_moe_route_identity,
        layer.top_k,
        layer.spyre_moe_stick,
    )


def _route_selected(layer: RoutedExperts, topk_probs: torch.Tensor) -> torch.Tensor:
    return _moe_persistent_selected_routing(
        topk_probs, layer.spyre_moe_route_identity, layer.spyre_moe_stick
    )


def _experts(layer: RoutedExperts, x: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
    return _moe_persistent(
        x,
        route,
        layer.spyre_moe_gate,
        layer.spyre_moe_up,
        layer.spyre_moe_down,
        layer.spyre_moe_recipe.activation,
    )


def _region(layer: RoutedExperts, name: str, fn: Any) -> Any:
    region = layer.spyre_moe_regions.get(name)
    if region is None:
        region = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)
        layer.spyre_moe_regions[name] = region
    return region


def _reset_named_dims() -> None:
    from torch_spyre._inductor.wsr.propagate_named_dims import reset

    reset()


def _to_spyre_expert_weight(weight: torch.Tensor, pad: tuple[int, ...]) -> torch.Tensor:
    """Move one expert stack to the device in the gather-friendly MoE layout.

    ``dma_moe_expert_weight_to_spyre`` only takes an ``[E, C, F]`` stack whose free dim
    spans whole sticks; ``pad`` is the ``F.pad`` spec that widens it to one.
    """
    from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre

    if any(pad):
        weight = F.pad(weight, pad)
    moved = dma_moe_expert_weight_to_spyre(weight)
    return moved if moved is not None else weight.contiguous().to("spyre")


def _prepare_layer(layer: RoutedExperts) -> None:
    from torch_spyre._C import get_elem_in_stick

    if hasattr(layer, "spyre_moe_gate"):
        raise RuntimeError(
            "Spyre MoE weights are immutable after loading; weight reload is unsupported."
        )
    w13 = layer.get_parameter("w13_weight").data
    w2_shape = tuple(layer.get_parameter("w2_weight").shape)
    experts, twice_inter, hidden = w13.shape
    if twice_inter % 2:
        raise ValueError(f"Spyre MoE requires a gated w13 stack, got {tuple(w13.shape)}.")
    inter = twice_inter // 2
    if w2_shape != (experts, hidden, inter):
        raise ValueError(
            f"unexpected MoE expert weight shapes: w13={tuple(w13.shape)} w2={w2_shape}"
        )

    # TP divides ``inter`` by the rank count, so it need not span whole sticks. Widening
    # is inert: the added lanes activate to zero, against zero rows of ``down``.
    stick = get_elem_in_stick(w13.dtype)
    pad = -inter % stick
    layer.spyre_moe_gate = _to_spyre_expert_weight(w13[:, :inter, :].transpose(1, 2), (0, pad))
    layer.spyre_moe_up = _to_spyre_expert_weight(w13[:, inter:, :].transpose(1, 2), (0, pad))
    del layer.w13_weight, w13
    w2 = layer.get_parameter("w2_weight").data
    transform_down = layer.spyre_moe_recipe.prepare_down_weight
    if transform_down is not None:
        w2 = transform_down(w2)
    layer.spyre_moe_down = _to_spyre_expert_weight(w2.transpose(1, 2), (0, 0, 0, pad))
    del layer.w2_weight, w2

    dtype = layer.spyre_moe_gate.dtype
    layer.spyre_moe_stick = stick
    layer.spyre_moe_route_identity = torch.eye(stick, dtype=dtype).to("spyre")
    logger.info_once(
        "Spyre: relaid out routed-expert stacks (%d experts, hidden=%d, intermediate=%d%s).",
        experts,
        hidden,
        inter,
        f" padded to {inter + pad}" if pad else "",
    )


@CustomOp.register_oot(name="UnquantizedFusedMoEMethod")
class SpyreUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """OOT bridge from vLLM's unquantized MoE method to ``SpyreMoERecipe``."""

    # The source expert parameters are replaced with device-specific stacks.
    supports_pre_processed_weights = False

    @property
    def is_monolithic(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: _RoutedExperts) -> None:
        if getattr(layer, "spyre_moe_recipe", None) is None:
            raise NotImplementedError(
                "Spyre MoE backend requires an explicit model-specific recipe; "
                "this architecture has not opted in."
            )
        _prepare_layer(cast("RoutedExperts", layer))

    def apply_monolithic(
        self,
        layer: _RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer = cast("RoutedExperts", layer)
        moe_scope, persistent_scope = _compiler_scopes()
        with moe_scope:
            if x.shape[0] == 1:
                return _region(layer, "gathered", _gathered)(layer, x, router_logits)
            recipe = layer.spyre_moe_recipe
            if recipe.routing == "full_softmax":
                probs = _region(layer, "probs", _probs)(router_logits)
                route = _region(layer, "route", _route)(layer, probs)
            else:
                topk_probs = _region(layer, "topk_probs", _topk_probs)(router_logits, layer.top_k)
                route = _region(layer, "route_selected", _route_selected)(layer, topk_probs)
            _name_persistent_dims(x, layer.spyre_moe_gate, layer.spyre_moe_up, layer.spyre_moe_down)
            try:
                with persistent_scope:
                    return _region(layer, "experts", _experts)(layer, x, route)
            finally:
                _reset_named_dims()
