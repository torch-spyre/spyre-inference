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
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _moe_forward,
    _moe_forward_shared,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from spyre_inference import envs

if TYPE_CHECKING:
    from torch_spyre._C import SpyreTensorLayout
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts as _RoutedExperts,
    )

    class RoutedExperts(_RoutedExperts):
        spyre_moe_recipe: SpyreMoERecipe
        spyre_moe_regions: dict[str, Any]
        spyre_moe_stick: int
        spyre_moe_chunks: int
        spyre_moe_route_dtype: torch.dtype
        spyre_moe_gate: torch.Tensor
        spyre_moe_up: torch.Tensor
        spyre_moe_down: torch.Tensor
        spyre_moe_gate_alias: torch.Tensor
        spyre_moe_up_alias: torch.Tensor
        spyre_moe_down_alias: torch.Tensor
        spyre_moe_route_identity: torch.Tensor


logger = init_logger(__name__)

_MOE_COMPILER_CONFIG = {"frontend_pool_allocation": True}
_PERSISTENT_COMPILER_CONFIG = {"allow_all_ops_in_lx_planning": True}

# Budget for one gathered gate/up/down weight operand per core, not the device's total LX.
_MOE_GATHER_WEIGHT_BUDGET_PER_CORE_BYTES = 1_500_000


def _derive_moe_chunks(
    hidden: int,
    inter: int,
    top_k: int,
    stick: int,
    element_size: int,
    max_cores: int,
    override: int | None = None,
) -> int | None:
    """Choose a core-filling, stick-legal chunk count within the gathered-weight budget.

    The gathered kernel splits work only across its ``top_k * chunks`` entries. The per-core
    footprint includes every selected weight entry assigned to that core; down has the same
    footprint. ``None`` means no automatic split fits the residency estimate; layer preparation
    falls back to a C=1 gathered layout.
    """
    if hidden <= 0 or inter <= 0 or top_k <= 0 or stick <= 0 or element_size <= 0:
        raise ValueError("MoE chunk selection requires positive dimensions, top_k, and stick size")
    if max_cores <= 0:
        raise ValueError(f"MoE chunk selection requires at least one core, got {max_cores}")
    if hidden % stick:
        raise ValueError(
            f"Spyre MoE hidden size {hidden} must be a multiple of {stick}-element sticks."
        )
    if override is not None:
        if override <= 0:
            raise ValueError(f"SPYRE_MOE_CHUNKS must be positive, got {override}")
        if hidden % (override * stick):
            raise ValueError(
                f"SPYRE_MOE_CHUNKS={override} requires hidden size {hidden} to split into "
                f"{override} whole {stick}-element-stick chunks."
            )
        return override

    padded_inter = inter + (-inter % stick)
    hidden_sticks = hidden // stick
    best: tuple[int, int, int] | None = None
    for chunks in range(1, hidden_sticks + 1):
        if hidden_sticks % chunks:
            continue
        entries = top_k * chunks
        cores_used = max(
            split for split in range(1, min(entries, max_cores) + 1) if entries % split == 0
        )
        entries_per_core = entries // cores_used
        bytes_per_core = entries_per_core * (hidden // chunks) * padded_inter * element_size
        if bytes_per_core > _MOE_GATHER_WEIGHT_BUDGET_PER_CORE_BYTES:
            continue
        candidate = (cores_used, -chunks, chunks)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best is not None else None


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


def _route_reduce_dtype(experts: int, dtype: torch.dtype) -> torch.dtype:
    # An fp32 rescale needs whole sticks in the source dtype ("cannot rescale device layout").
    from torch_spyre._C import get_elem_in_stick

    stick = get_elem_in_stick(dtype)
    if experts % stick == 0:
        return torch.float32
    logger.warning_once(
        "Spyre: reducing the routing softmax over %d experts in %s; an fp32 reduction "
        "requires the expert count to be a multiple of %d.",
        experts,
        dtype,
        stick,
    )
    return dtype


def _probs(router_logits: torch.Tensor, reduce_dtype: torch.dtype) -> torch.Tensor:
    return torch.softmax(router_logits.to(reduce_dtype), dim=-1).to(router_logits.dtype)


def _routing_weights(
    router_logits: torch.Tensor,
    top_k: int,
    routing: str,
    reduce_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if routing == "full_softmax":
        selected_values, indices = _topk(_probs(router_logits, reduce_dtype), top_k)
        weights = selected_values / selected_values.sum(-1, keepdim=True)
    else:
        # ``reduce_dtype`` gates the expert dim; this reduces over ``top_k`` instead.
        selected_logits, indices = _topk(router_logits, top_k)
        weights = torch.softmax(selected_logits, dim=-1)
    return weights, indices


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
    chunks: int,
    stick: int,
    reduce_dtype: torch.dtype,
    routing: str,
    activation: str,
) -> torch.Tensor:
    tokens, hidden = x.shape
    if hidden % chunks:
        raise ValueError(
            f"gathered MoE hidden size {hidden} must divide evenly into {chunks} chunks"
        )
    if gate.ndim != 3 or gate.shape[0] % chunks:
        raise ValueError(
            "gathered gate pool must have chunked [E*chunks, hidden/chunks, inter] shape"
        )
    slice_rows = hidden // chunks
    experts = gate.shape[0] // chunks
    inter = gate.shape[-1]
    expected_gate_shape = (experts * chunks, slice_rows, inter)
    if gate.shape != expected_gate_shape or up.shape != expected_gate_shape:
        raise ValueError(
            f"gathered gate/up pools must have shape {expected_gate_shape}, "
            f"got gate={tuple(gate.shape)}, up={tuple(up.shape)}"
        )
    expected_down_shape = (experts * chunks, inter, slice_rows)
    if down.shape != expected_down_shape:
        raise ValueError(
            f"gathered down pool must have chunked shape {expected_down_shape}, "
            f"got {tuple(down.shape)}"
        )
    weights, raw_indices = _routing_weights(router_logits, top_k, routing, reduce_dtype)
    rows = tokens * top_k
    # Entry e*chunks+c selects chunk c of expert e; entries are stored chunk-major.
    expert_fp32 = (
        raw_indices.reshape(-1)[:, None].expand(rows, stick).contiguous().to(torch.float32)
    )
    entries = torch.cat([expert_fp32 * float(chunks) + float(c) for c in range(chunks)], dim=0)
    slice_index = entries[..., : stick // 2].to(torch.int32)[:, 0]
    # (chunk, row) batch order: chunk c's rows cover every expert of the step,
    # so each chunk block of the x-operand repeats the token's chunk slice.
    inputs = torch.cat(
        [
            x.reshape(tokens, 1, chunks, slice_rows)[:, :, c]
            .reshape(tokens, 1, slice_rows)
            .expand(tokens, top_k, slice_rows)
            .reshape(rows, 1, slice_rows)
            for c in range(chunks)
        ],
        dim=0,
    )
    # Finish each stack's projection before processing the next to limit live intermediates.
    gate_out = (
        torch.bmm(inputs, gate[slice_index].reshape(rows * chunks, slice_rows, inter))
        .reshape(chunks, rows, 1, inter)
        .sum(dim=0)
    )
    up_out = (
        torch.bmm(inputs, up[slice_index].reshape(rows * chunks, slice_rows, inter))
        .reshape(chunks, rows, 1, inter)
        .sum(dim=0)
    )
    # Reuse the activated rows across output chunks and select each down-weight slice by expert.
    activated = _activation(gate_out, up_out, activation)
    act_rep = activated.expand(chunks, rows, 1, inter).reshape(rows * chunks, 1, inter)
    # ``slice_index`` encodes e*chunks+c in chunk,row order, matching the down alias.
    parts = torch.bmm(act_rep, down[slice_index].reshape(rows * chunks, inter, slice_rows)).reshape(
        chunks, tokens, top_k, slice_rows
    )
    expert_out = torch.cat([parts[c] for c in range(chunks)], dim=-1)
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

    # Whole-expert pools [E, hidden, inter] / [E, inter, hidden].
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
    # Process routed experts over token tiles using the persistent weight pools.
    from torch_spyre._inductor.propagate_hints import spyre_hint
    from torch_spyre._inductor.wsr import for_each_tile

    with spyre_hint(named_dims=["E", "T", "ONE"]):
        route = route.permute(1, 0, 2).contiguous().clone()

    def expert_body(acc, tiles):
        x, route_tile, gate_tile, up_tile, down_tile = tiles
        activated = _activation(torch.matmul(x, gate_tile), torch.matmul(x, up_tile), activation)
        return acc + (torch.matmul(activated, down_tile) * route_tile).squeeze(0), None

    with spyre_hint(work_div={"T": _token_cores(x.shape[0])}):
        result, _ = for_each_tile(
            expert_body,
            (x, route, gate, up, down),
            dims=(None, 0, 0, 0, 0),
            tile_size=1,
            init=torch.zeros_like(x),
        )
    return result


def _moe_persistent_in_graph(
    x: torch.Tensor,
    route: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """``_moe_persistent`` inside a block graph, where nothing names x's token axis."""
    from torch_spyre._inductor.propagate_hints import spyre_hint
    from torch_spyre._inductor.wsr import for_each_tile

    with spyre_hint(named_dims=["E", "T", "ONE"]):
        route = route.permute(1, 0, 2).contiguous().clone()

    def expert_body(acc, tiles):
        x, route_tile, gate_tile, up_tile, down_tile = tiles
        activated = _activation(torch.matmul(x, gate_tile), torch.matmul(x, up_tile), activation)
        return acc + torch.matmul(activated, down_tile) * route_tile, None

    # Loop-body ops only receive this scope's hints, so T is named here; every body op stays
    # at [1, T, C] (C is M or H) so one name list fits them all.
    init = torch.zeros_like(x).unsqueeze(0)
    with spyre_hint(work_div={"T": _token_cores(x.shape[0])}, named_dims=["ONE", "T", "C"]):
        result, _ = for_each_tile(
            expert_body,
            (x, route, gate, up, down),
            dims=(None, 0, 0, 0, 0),
            tile_size=1,
            init=init,
        )
    return result.squeeze(0)


def _gathered(layer: RoutedExperts, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
    recipe = layer.spyre_moe_recipe
    chunks = layer.spyre_moe_chunks
    gate, up, down = (
        layer.spyre_moe_gate_alias,
        layer.spyre_moe_up_alias,
        layer.spyre_moe_down_alias,
    )
    return _moe_gathered(
        x,
        router_logits,
        gate,
        up,
        down,
        layer.top_k,
        chunks,
        layer.spyre_moe_stick,
        layer.spyre_moe_route_dtype,
        recipe.routing,
        recipe.activation,
    )


def _gathered_tokens(
    layer: RoutedExperts, x: torch.Tensor, router_logits: torch.Tensor
) -> torch.Tensor:
    # The gathered kernel only lowers at one token. ``dynamic=False`` specializes this loop to
    # the packed bucket, so slicing, expert calls, and assembly stay in one compiled region.
    rows = [
        _gathered(layer, x[token : token + 1], router_logits[token : token + 1])
        for token in range(x.shape[0])
    ]
    return torch.cat(rows)


def _rows_are_stick_addressable(x: torch.Tensor, router_logits: torch.Tensor, stick: int) -> bool:
    # Cloning row ``t`` bakes its flat storage offset, ``storage_offset() + t * stride(0)``, into
    # the kernel coordinate, and the backend can only bake whole sticks: hence both terms.
    return all(
        t.storage_offset() % stick == 0 and t.stride(0) % stick == 0 for t in (x, router_logits)
    )


def _topk_probs(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Materialize canonical vLLM top-k weights in dense expert order."""
    topk_weights, topk_ids = _routing_weights(
        router_logits, top_k, "topk_softmax", router_logits.dtype
    )
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
        # Deferred import: spyre_inference.v1.worker imports this module's package.
        from spyre_inference.v1.worker import compile_guard

        compile_guard.watch(fn, f"MoE region {name!r}")
    return region


def _reset_named_dims() -> None:
    from torch_spyre._inductor.wsr.propagate_named_dims import reset

    reset()


def _expert_kernel_layout(weight: torch.Tensor) -> SpyreTensorLayout:
    """``[E, C, F]`` as ``[E, F // stick, C, stick]``, the nn.Linear order (torch-spyre #1339)."""
    from torch_spyre._C import SpyreTensorLayout

    return SpyreTensorLayout(list(weight.shape), list(weight.stride()), weight.dtype, [1, 0, 2])


def _to_spyre_expert_weight(
    weight: torch.Tensor, pad: tuple[int, ...], *, kernel_order: bool = False
) -> torch.Tensor:
    """Move an ``[E, C, F]`` expert stack, widened by the ``F.pad`` spec ``pad``, to the device."""
    from torch_spyre.model_utils import dma_moe_expert_weight_to_spyre

    if any(pad):
        weight = F.pad(weight, pad)
    if not kernel_order:
        moved = dma_moe_expert_weight_to_spyre(weight)
        assert moved is not None
        return moved
    if not torch.spyre.is_initialized():
        torch.empty(0, dtype=weight.dtype, device="spyre")
    weight = weight.contiguous()
    layout = _expert_kernel_layout(weight)
    assert weight.shape[-1] % layout.elems_per_stick() == 0, "the free dim must span whole sticks"
    # Lands on the current Spyre device, as the ``dma_*`` helpers do.
    return weight.to(device_layout=layout)  # ty: ignore[no-matching-overload]


def _check_pool_layout(pool: torch.Tensor, expected: SpyreTensorLayout) -> None:
    from torch_spyre._C import get_spyre_tensor_layout

    actual = get_spyre_tensor_layout(pool)
    if (actual.device_size, actual.stride_map) != (expected.device_size, expected.stride_map):
        raise RuntimeError(
            f"MoE pool device layout {actual.device_size} / {actual.stride_map} differs from "
            f"the {expected.device_size} / {expected.stride_map} its chunk alias reinterprets."
        )


def _chunk_pool_alias(pool: torch.Tensor, chunks: int) -> torch.Tensor:
    """Expose a whole-expert [E, contract, free] pool as [E*chunks, contract/chunks, free].

    The view shares storage; entry e*chunks+c selects chunk c of expert e. The contract axis
    is outer to the free-dim sticks, so chunking it preserves flat physical order.
    """
    from torch_spyre import _C
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    experts, contract, free = pool.shape
    if contract % chunks:
        raise ValueError(f"gate/up contract dim {contract} must divide into {chunks} chunks")
    eps = SpyreTensorLayout([experts, contract, free], pool.dtype).elems_per_stick()
    device_dtype = get_device_dtype(pool.dtype)
    _check_pool_layout(
        pool,
        SpyreTensorLayout(
            [experts, contract, free // eps, eps], [contract * free, free, eps, 1], device_dtype
        ),
    )
    c4 = contract // chunks
    layout = SpyreTensorLayout(
        [experts * chunks, c4, free // eps, eps],
        [c4 * free, free, eps, 1],
        device_dtype,
    )
    return _C.reinterpret_tensor_with_layout(
        pool,
        [experts * chunks, c4, free],
        [c4 * free, free, 1],
        0,
        layout,
    )


def _down_chunk_pool_alias(pool: torch.Tensor, chunks: int) -> torch.Tensor:
    """Expose down weights as expert-major [E*chunks, M, H/chunks] entries.

    The view shares storage. Each e*chunks+c entry selects output chunk c for expert e.
    The H-stick axis is outer to M, so chunking it preserves flat physical order.
    """
    from torch_spyre import _C
    from torch_spyre._C import SpyreTensorLayout, get_elem_in_stick

    experts, inter, hidden = pool.shape
    stick = get_elem_in_stick(pool.dtype)
    if hidden % chunks or (hidden // chunks) % stick:
        raise ValueError(
            f"down hidden size {hidden} must divide into {chunks} stick-aligned chunks"
        )
    _check_pool_layout(pool, _expert_kernel_layout(pool))
    chunk_hidden = hidden // chunks
    size = [experts * chunks, inter, chunk_hidden]
    stride = [inter * chunk_hidden, chunk_hidden, 1]
    layout = SpyreTensorLayout(size, stride, pool.dtype, [1, 0, 2])
    return _C.reinterpret_tensor_with_layout(pool, size, stride, 0, layout)


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

    from torch_spyre._inductor import config as spyre_config

    # All shape, alignment, and scratch-budget checks run before the first device relayout.
    stick = get_elem_in_stick(w13.dtype)
    chunks = _derive_moe_chunks(
        hidden,
        inter,
        layer.top_k,
        stick,
        w13.element_size(),
        spyre_config.sencores,
        envs.SPYRE_MOE_CHUNKS,
    )
    fallback_to_c1 = chunks is None
    if chunks is None:
        # The budget estimates residency, not correctness. C=1 still gathers selected
        # experts, while persistent execution visits every expert.
        logger.info_once(
            "Spyre MoE: no gathered split fits the residency estimate for top_k=%d, "
            "hidden=%d, intermediate=%d; using C=1 gathered fallback.",
            layer.top_k,
            hidden,
            inter,
        )
        chunks = 1
    # TP can leave intermediate shards mid-stick; zero-padding is inert because those
    # activation lanes multiply zero rows in the down stack.
    pad = -inter % stick
    # Share gate/up allocations between persistent and gathered chunk views.
    layer.spyre_moe_gate = _to_spyre_expert_weight(w13[:, :inter, :].transpose(1, 2), (0, pad))
    layer.spyre_moe_up = _to_spyre_expert_weight(w13[:, inter:, :].transpose(1, 2), (0, pad))
    layer.spyre_moe_chunks = chunks
    layer.spyre_moe_gate_alias = _chunk_pool_alias(layer.spyre_moe_gate, chunks)
    layer.spyre_moe_up_alias = _chunk_pool_alias(layer.spyre_moe_up, chunks)
    del layer.w13_weight, w13
    w2 = layer.get_parameter("w2_weight").data
    transform_down = layer.spyre_moe_recipe.prepare_down_weight
    if transform_down is not None:
        w2 = transform_down(w2)
    # Keep the complete stack for persistent compute and expose a view for gathered decode.
    layer.spyre_moe_down = _to_spyre_expert_weight(
        w2.transpose(1, 2), (0, 0, 0, pad), kernel_order=True
    )
    layer.spyre_moe_down_alias = _down_chunk_pool_alias(layer.spyre_moe_down, chunks)
    if not fallback_to_c1:
        logger.info_once(
            "Spyre MoE: selected %d gathered weight chunks (top_k=%d, cores=%d).",
            chunks,
            layer.top_k,
            spyre_config.sencores,
        )
    del layer.w2_weight, w2

    dtype = layer.spyre_moe_gate.dtype
    layer.spyre_moe_stick = stick
    layer.spyre_moe_route_dtype = (
        _route_reduce_dtype(experts, dtype)
        if layer.spyre_moe_recipe.routing == "full_softmax"
        else dtype
    )
    layer.spyre_moe_route_identity = torch.eye(stick, dtype=dtype).to("spyre")
    logger.info_once(
        "Spyre: relaid out routed-expert stacks (%d experts, hidden=%d, intermediate=%d%s).",
        experts,
        hidden,
        inter,
        f" padded to {inter + pad}" if pad else "",
    )


@MoERunner.register_oot(name="MoERunner")
class SpyreMoERunner(MoERunner):
    """Takes vLLM's direct MoE entry, as upstream does on CPU and TPU, for stick-aligned batches.

    A compiled block then compiles the MoE dispatch along with the rest of its graph. In-graph
    routing only lowers when the token axis spans whole sticks, so other batch sizes keep the
    opaque ``torch.ops.vllm.moe_forward`` op and its compiled regions.
    """

    def _select_forward(self) -> Callable:
        from torch_spyre._C import get_elem_in_stick

        self._spyre_stick = get_elem_in_stick(self.moe_config.in_dtype)
        return self._spyre_forward_entry

    def _spyre_forward_entry(self, hidden_states: torch.Tensor, *args: Any) -> Any:
        return self._entry_for(hidden_states.shape[0])(hidden_states, *args)

    def _entry_for(self, tokens: int) -> Callable:
        shared = self.shared_experts is not None
        if tokens % self._spyre_stick == 0:
            return _moe_forward_shared if shared else _moe_forward
        return torch.ops.vllm.moe_forward_shared if shared else torch.ops.vllm.moe_forward


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
        # MoERunner._forward_impl initializes this lazily. Traced, that check becomes a guard
        # which warmup's first call flips, so the first real request would re-trace the block.
        layer._ensure_moe_quant_config_init()

    def apply_monolithic(
        self,
        layer: _RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer = cast("RoutedExperts", layer)
        tokens = x.shape[0]
        if torch.compiler.is_compiling():
            # Traced into the block graph, which SpyreMoERunner only does for stick-aligned
            # batches: the persistent form runs inline, without compiled regions or their
            # scopes. An in-graph fp32 softmax cannot be lowered, so routing stays in the
            # logits' dtype.
            # Note: This branching here is solely done to support eager and compiled
            # execution of the experts.
            recipe = layer.spyre_moe_recipe
            if recipe.routing == "full_softmax":
                route = _route(layer, _probs(router_logits, router_logits.dtype))
            else:
                route = _route_selected(layer, _topk_probs(router_logits, layer.top_k))
            return _moe_persistent_in_graph(
                x,
                route,
                layer.spyre_moe_gate,
                layer.spyre_moe_up,
                layer.spyre_moe_down,
                recipe.activation,
            )
        moe_scope, persistent_scope = _compiler_scopes()
        # A single row is handed to the region whole, so no row slice needs an addressable offset.
        if tokens == 1 or (
            tokens <= envs.SPYRE_MOE_GATHERED_MAX_TOKENS
            and _rows_are_stick_addressable(x, router_logits, layer.spyre_moe_stick)
        ):
            with moe_scope:
                if tokens == 1:
                    return _region(layer, "gathered", _gathered)(layer, x, router_logits)
                return _region(layer, "gathered_batch", _gathered_tokens)(layer, x, router_logits)
        with moe_scope:
            recipe = layer.spyre_moe_recipe
            if recipe.routing == "full_softmax":
                probs = _region(layer, "probs", _probs)(router_logits, layer.spyre_moe_route_dtype)
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
