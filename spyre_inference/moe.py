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

import contextvars
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
        spyre_moe_biases: _ExpertBiases | None
        spyre_moe_chunk_scale: torch.Tensor
        spyre_moe_chunk_offsets: torch.Tensor
        spyre_moe_route_identity: torch.Tensor


logger = init_logger(__name__)

_MOE_COMPILER_CONFIG = {"frontend_pool_allocation": True}
_PERSISTENT_COMPILER_CONFIG = {"allow_all_ops_in_lx_planning": True}


def _derive_moe_chunks(
    hidden: int,
    top_k: int,
    stick: int,
    max_cores: int,
    override: int | None = None,
) -> int:
    """Choose the fewest stick-legal chunks that spread the gathered entries over the most cores.

    The gathered kernel splits work only across its ``top_k * chunks`` entries. A core holds
    ``top_k * hidden * inter / cores_used`` weight elements whatever the chunk count, so the
    split that fills the most cores also has the smallest per-core working set.
    """
    if hidden <= 0 or top_k <= 0 or stick <= 0:
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

    hidden_sticks = hidden // stick
    best_cores, best_chunks = 0, 1
    for chunks in range(1, hidden_sticks + 1):
        if hidden_sticks % chunks:
            continue
        entries = top_k * chunks
        cores_used = max(
            split for split in range(1, min(entries, max_cores) + 1) if entries % split == 0
        )
        if cores_used > best_cores:
            best_cores, best_chunks = cores_used, chunks
    return best_chunks


@dataclass(frozen=True)
class SpyreMoERecipe:
    """The upstream MoE semantics implemented by Spyre's current kernels.

    The routing function follows vLLM's canonical ``(topk_weights, topk_ids)``
    contract. The persistent form expands that result to Spyre's dense route
    layout without selecting top-k a second time.
    """

    activation: Literal["gelu_tanh", "silu", "swigluoai"]
    routing: Literal["full_softmax", "topk_softmax"]
    # Called per chunk of experts while the down stack streams to the card: ``stack`` is
    # ``w2[experts]``, so the whole transformed stack is never built on the host.
    prepare_down_weight: Callable[[torch.Tensor, slice], torch.Tensor] | None = None
    # G experts per trip of the traced all-expert loop for stick-aligned batches of
    # ``expert_group_tokens`` = (fewest, most) tokens; other batches run one. Where grouping
    # pays depends on the model's expert shapes, so a recipe that sets G > 1 sets the range.
    expert_group: int = 1
    expert_group_tokens: tuple[int, int] | None = None

    def experts_per_trip(self, tokens: int) -> int:
        """Experts per trip of the traced all-expert loop for a batch of ``tokens``."""
        if self.expert_group == 1 or self.expert_group_tokens is None:
            return 1
        fewest, most = self.expert_group_tokens
        return self.expert_group if fewest <= tokens <= most else 1


# vLLM's swigluoai_and_mul constants. Its MoEActivation.SWIGLUOAI also defines the w13 layout:
# gate and up rows interleaved, [gate0, up0, gate1, ...], as in gpt-oss checkpoints.
_SWIGLUOAI_LIMIT, _SWIGLUOAI_ALPHA = 7.0, 1.702


def deinterleave_gate_up(weight: torch.Tensor) -> torch.Tensor:
    """``[E, 2M, ...]`` with gate on even rows, up on odd, as ``[gate | up]`` halves."""
    return torch.cat([weight[:, 0::2], weight[:, 1::2]], dim=1)


@dataclass(frozen=True)
class _ExpertBiases:
    """Per-expert bias stacks, ``[E, 1, F]`` so each tiles and gathers like its weight pool.

    ``down_chunks`` is ``down`` split along the hidden axis into the gathered form's
    ``[E*chunks, 1, hidden/chunks]`` entries; the persistent forms read ``down`` whole.
    """

    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    down_chunks: torch.Tensor


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


def _expert_group_fits(layer: _RoutedExperts, group: int) -> bool:
    # A traced batch is whole sticks of tokens, which the cores must split evenly.
    from torch_spyre._C import get_elem_in_stick
    from torch_spyre._inductor import config as spyre_config

    cores = spyre_config.sencores
    return (
        1 <= group <= cores
        and layer.local_num_experts % group == 0
        and cores % group == 0
        and get_elem_in_stick(layer.moe_config.in_dtype) % cores == 0
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
    # prepare_down_weight scales the down stack (Gemma folds per_expert_scale there) and
    # does not see w2_bias, so a biased expert would add an unscaled d_e.
    if moe.has_bias and recipe.prepare_down_weight is not None:
        raise NotImplementedError(
            "Spyre MoE backend does not support expert biases together with "
            "prepare_down_weight; that transform scales the down stack and would "
            "leave w2_bias unscaled."
        )
    if layer.activation.value != recipe.activation:
        raise NotImplementedError(
            f"Spyre MoE recipe requires activation={recipe.activation!r}, "
            f"got {layer.activation.value!r}."
        )
    if recipe.activation == "swigluoai":
        # The constants are vLLM's swigluoai_and_mul, which the layer's own limit/alpha do not
        # reach on any backend; a model that sets other values would be silently wrong.
        limit, alpha = moe.swiglu_limit, moe.swiglu_alpha
        if (limit is not None and limit != _SWIGLUOAI_LIMIT) or (
            alpha is not None and alpha != _SWIGLUOAI_ALPHA
        ):
            raise NotImplementedError(
                f"Spyre swigluoai uses limit={_SWIGLUOAI_LIMIT} and alpha={_SWIGLUOAI_ALPHA}; "
                f"the layer asks for limit={limit}, alpha={alpha}."
            )
    if recipe.expert_group != 1:
        fewest, most = recipe.expert_group_tokens or (0, 0)
        if recipe.expert_group < 1 or not 1 <= fewest <= most:
            raise ValueError(
                f"SpyreMoERecipe.expert_group={recipe.expert_group} needs a group of at least 1 "
                "and expert_group_tokens=(fewest, most) with 1 <= fewest <= most, got "
                f"{recipe.expert_group_tokens}."
            )
        if _expert_group_fits(layer, recipe.expert_group):
            logger.info_once(
                "Spyre MoE: %d experts per trip in the in-graph expert loop for stick-aligned "
                "batches of %d to %d tokens; other batches keep one expert per trip.",
                recipe.expert_group,
                fewest,
                most,
            )
        else:
            # The core count belongs to the deployment, so the recipe's group yields to it.
            logger.warning_once(
                "Spyre MoE: the recipe's %d experts per trip do not fit this layer's experts "
                "and the cores; running one expert per trip.",
                recipe.expert_group,
            )
            recipe = replace(recipe, expert_group=1)
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
    if activation == "swigluoai":
        # gpt-oss: (clamp(up, +-limit) + 1) * gate * sigmoid(alpha * gate), gate clamped above.
        gate = torch.clamp(x, max=_SWIGLUOAI_LIMIT)
        up = torch.clamp(up, -_SWIGLUOAI_LIMIT, _SWIGLUOAI_LIMIT)
        return (up + 1) * gate * torch.sigmoid(gate * _SWIGLUOAI_ALPHA)
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
    *,
    chunk_scale: torch.Tensor,
    chunk_offsets: torch.Tensor,
    biases: _ExpertBiases | None = None,
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
    # Entry e*chunks+c selects chunk c of expert e, in (row, chunk) order. Broadcasting against
    # the [1, chunks, 1] device offsets, not a cat of per-chunk index tensors, keeps the region
    # one kernel without per-call constants.
    expert_fp32 = (
        raw_indices.reshape(-1)[:, None].expand(rows, stick).contiguous().to(torch.float32)
    )
    entries = (expert_fp32[:, None, :] * chunk_scale + chunk_offsets).reshape(rows * chunks, stick)
    # Every lane of the one-stick-wide entries holds the entry, so take it as the max over the
    # stick: the float32 holds each pair of sticks in an interleaved order, and a reduction
    # reads it whole where a slice would split a pair.
    slice_index = entries.amax(-1).to(torch.int32)
    inputs = (
        x.reshape(tokens, 1, chunks, slice_rows)
        .expand(tokens, top_k, chunks, slice_rows)
        .contiguous()
        .reshape(rows * chunks, 1, slice_rows)
    )
    gate_out = torch.bmm(inputs, gate[slice_index].reshape(rows * chunks, slice_rows, inter))
    up_out = torch.bmm(inputs, up[slice_index].reshape(rows * chunks, slice_rows, inter))
    if chunks > 1:
        gate_out = gate_out.reshape(rows, chunks, 1, inter).sum(dim=1)
        up_out = up_out.reshape(rows, chunks, 1, inter).sum(dim=1)
    if biases is not None:
        # Gathered per expert, read off the same fp32 lanes as the entry index; a gate/up bias
        # is the expert's whole row, the down bias its chunk of the hidden axis.
        expert_index = expert_fp32.amax(-1).to(torch.int32)
        gate_out = gate_out + biases.gate[expert_index]
        up_out = up_out + biases.up[expert_index]
    activated = _activation(gate_out, up_out, activation)
    act_rep = activated[:, None].expand(rows, chunks, 1, inter).reshape(rows * chunks, 1, inter)
    # Output chunk c of every row is the c-th hidden slice, so the parts reshape in place.
    parts = torch.bmm(act_rep, down[slice_index].reshape(rows * chunks, inter, slice_rows))
    if biases is not None:
        parts = parts + biases.down_chunks[slice_index]
    expert_out = parts.reshape(tokens, top_k, hidden)
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
    biases: _ExpertBiases | None = None,
) -> torch.Tensor:
    # Process routed experts over token tiles using the persistent weight pools.
    from torch_spyre._inductor.propagate_hints import spyre_hint
    from torch_spyre._inductor.wsr import for_each_tile

    with spyre_hint(named_dims=["E", "T", "ONE"]):
        route = route.permute(1, 0, 2).contiguous().clone()

    operands, dims = _expert_loop_operands(x, route, gate, up, down, biases)

    def expert_body(acc, tiles):
        x, route_tile, gate_tile, up_tile, down_tile, *bias = tiles
        expert_out = _expert_tile(x, gate_tile, up_tile, down_tile, bias, activation)
        return acc + (expert_out * route_tile).squeeze(0), None

    result, _ = for_each_tile(
        expert_body,
        operands,
        dims=dims,
        tile_size=1,
        init=torch.zeros_like(x),
    )
    return result


def _expert_loop_operands(x, route, gate, up, down, biases):
    """The persistent loops' operands: the bias stacks tile on the expert axis with the pools."""
    operands: tuple[torch.Tensor, ...] = (x, route, gate, up, down)
    dims: tuple[int | None, ...] = (None, 0, 0, 0, 0)
    if biases is not None:
        operands += (biases.gate, biases.up, biases.down)
        dims += (0, 0, 0)
    return operands, dims


def _expert_tile(x, gate_tile, up_tile, down_tile, bias, activation):
    """One expert's output for every token; ``bias`` is the tiled gate/up/down biases or empty."""
    gate_out = torch.matmul(x, gate_tile)
    up_out = torch.matmul(x, up_tile)
    if bias:
        gate_out = gate_out + bias[0]
        up_out = up_out + bias[1]
    out = torch.matmul(_activation(gate_out, up_out, activation), down_tile)
    return out + bias[2] if bias else out


def _moe_persistent_in_graph(
    x: torch.Tensor,
    route: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    activation: str,
    biases: _ExpertBiases | None = None,
    group: int = 1,
) -> torch.Tensor:
    """``_moe_persistent`` inside a block graph, where nothing names x's token axis."""
    from torch_spyre._inductor.propagate_hints import spyre_hint
    from torch_spyre._inductor.wsr import for_each_tile

    with spyre_hint(named_dims=["E", "T", "ONE"]):
        route = route.permute(1, 0, 2).contiguous().clone()

    operands, dims = _expert_loop_operands(x, route, gate, up, down, biases)

    def expert_body(acc, tiles):
        x, route_tile, gate_tile, up_tile, down_tile, *bias = tiles
        expert_out = _expert_tile(x, gate_tile, up_tile, down_tile, bias, activation)
        return acc + expert_out * route_tile, None

    if group > 1:
        from torch_spyre._inductor import config as spyre_config

        cores = spyre_config.sencores
        # A trip's tiles carry G experts on a leading axis, which the body's ops broadcast
        # over. One core map {G, T} serves x's per-group copy, the trip and the accumulators;
        # a broadcast x gives the same values but is slower at 256 rows.
        with spyre_hint(named_dims=["G", "T", "C"], work_div={"G": group, "T": cores // group}):
            x_groups = x.expand(group, *x.shape).contiguous()
            result, _ = for_each_tile(
                expert_body,
                (x_groups, *operands[1:]),
                dims=dims,
                tile_size=group,
                init=x.new_zeros(group, *x.shape),
            )
        with spyre_hint(named_dims=["T", "C"], work_div={"T": cores}):
            return result.sum(0)

    # Loop-body ops only receive this scope's hints, so T is named here; every body op stays
    # at [1, T, C] (C is M or H) so one name list fits them all.
    init = torch.zeros_like(x).unsqueeze(0)
    with spyre_hint(named_dims=["ONE", "T", "C"]):
        result, _ = for_each_tile(
            expert_body,
            operands,
            dims=dims,
            tile_size=1,
            init=init,
        )
    # Sum, not squeeze: an in-graph all_reduce of a view of a for_each_tile result can leave
    # each rank with its own partial (torch-spyre#6062).
    return result.sum(dim=0)


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
        chunk_scale=layer.spyre_moe_chunk_scale,
        chunk_offsets=layer.spyre_moe_chunk_offsets,
        biases=layer.spyre_moe_biases,
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
    """Materialize canonical vLLM top-k weights in dense expert order.

    A softmax over the row with the unselected experts at -inf is the softmax over the
    selected logits. Unlike scattering those weights back, it lowers on Spyre.
    """
    _, selected = _topk(router_logits, top_k)
    masked = torch.ops.spyre.keep_by_index(
        router_logits,  # ty: ignore[invalid-argument-type]
        selected,  # ty: ignore[invalid-argument-type]
        -1,  # ty: ignore[invalid-argument-type]
        float("-inf"),  # ty: ignore[invalid-argument-type]
    )
    return torch.softmax(masked, dim=-1)


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
        layer.spyre_moe_biases,
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


# Host bytes a chunk of an expert stack stages: the copy is host-bound, and a chunk's
# transpose stays in cache where a whole stack's does not.
_EXPERT_CHUNK_BYTES = 32 << 20


def _gather_layout(weight: torch.Tensor) -> SpyreTensorLayout | None:
    """The ``[E, C, F]`` gather layout of the ``dma_*`` helpers; None off whole sticks."""
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    experts, contract, free = weight.shape
    eps = SpyreTensorLayout(list(weight.shape), weight.dtype).elems_per_stick()
    if free % eps:
        return None
    return SpyreTensorLayout(
        [experts, contract, free // eps, eps],
        [contract * free, free, eps, 1],
        get_device_dtype(weight.dtype),
        {(0,): experts, (1,): contract, (2, 3): free},
    )


def _to_spyre_expert_weight(
    weight: torch.Tensor,
    pad: tuple[int, ...],
    *,
    kernel_order: bool = False,
    prepare: Callable[[torch.Tensor, slice], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Move an ``[E, C, F]`` expert stack, widened by the ``F.pad`` spec ``pad``, to the device
    in chunks of experts, each transformed by ``prepare(chunk, experts)`` when given."""
    from torch_spyre._C import copy_tensor
    from torch_spyre.model_utils import spyre_empty_with_layout

    if not torch.spyre.is_initialized():
        torch.empty(0, dtype=weight.dtype, device="spyre")
    assert len(pad) <= 4, "pad never touches the expert axis, so chunks pad like the stack"
    padded = torch.empty(weight.shape, dtype=weight.dtype, device="meta")
    if any(pad):
        padded = F.pad(padded, pad)
    layout = _expert_kernel_layout(padded) if kernel_order else _gather_layout(padded)
    assert layout is not None, "the free dim must span whole sticks"
    # Lands on the current Spyre device, as the ``dma_*`` helpers do.
    dst = spyre_empty_with_layout(padded.size(), padded.stride(), weight.dtype, layout, device=None)
    experts = weight.shape[0]
    per_expert = padded[0].numel() * weight.element_size()
    chunk = max(1, min(experts, _EXPERT_CHUNK_BYTES // max(1, per_expert)))
    for start in range(0, experts, chunk):
        part = weight[start : start + chunk]
        if prepare is not None:
            part = prepare(part, slice(start, min(start + chunk, experts)))
        part = F.pad(part, pad) if any(pad) else part.contiguous()
        copy_tensor(part, dst[start : start + chunk], non_blocking=False)
    return dst


def _chunked_down_transform(
    transform: Callable[[torch.Tensor, slice], torch.Tensor],
) -> Callable[[torch.Tensor, slice], torch.Tensor]:
    def prepare(part: torch.Tensor, experts: slice) -> torch.Tensor:
        return transform(part.transpose(1, 2), experts).transpose(1, 2)

    return prepare


def _chunk_entry_constants(chunks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Device scale and offsets of the gathered entry index ``e*chunks + c``.

    Inputs rather than in-graph scalars, so the region needs no per-call constants; the
    size-1 stick dim lets them broadcast against the widened fp32 expert index.
    """
    scale = torch.full((1, 1, 1), float(chunks), dtype=torch.float32).to("spyre")
    offsets = torch.arange(chunks, dtype=torch.float32).reshape(1, chunks, 1).to("spyre")
    return scale, offsets


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
    expected = _gather_layout(pool)
    assert expected is not None, "a gather pool spans whole sticks"
    _check_pool_layout(pool, expected)
    eps = expected.elems_per_stick()
    device_dtype = get_device_dtype(pool.dtype)
    c4 = contract // chunks
    layout = SpyreTensorLayout(
        [experts * chunks, c4, free // eps, eps],
        [c4 * free, free, eps, 1],
        device_dtype,
        {(0,): experts * chunks, (1,): c4, (2, 3): free},
    )
    return _C.reinterpret_tensor_with_layout(
        pool,
        (experts * chunks, c4, free),
        (c4 * free, free, 1),
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
    size = (experts * chunks, inter, chunk_hidden)
    stride = (inter * chunk_hidden, chunk_hidden, 1)
    layout = SpyreTensorLayout(size, stride, pool.dtype, [1, 0, 2])
    return _C.reinterpret_tensor_with_layout(pool, size, stride, 0, layout)


def _prepare_biases(
    layer: RoutedExperts, inter: int, pad: int, chunks: int
) -> _ExpertBiases | None:
    """Relayout ``w13_bias`` / ``w2_bias`` next to the pools.

    ``_prepare_layer`` has already checked both shapes against the weights. The host
    parameters stay. They are tiny next to the weight stacks, and upstream's
    ``get_fused_moe_quant_config`` still reads them. The kernels read ``spyre_moe_biases``.
    """
    # vLLM registers both exactly when ``moe_config.has_bias``.
    if not hasattr(layer, "w13_bias"):
        return None
    w13_bias = layer.get_parameter("w13_bias").data
    if layer.spyre_moe_recipe.activation == "swigluoai":
        w13_bias = deinterleave_gate_up(w13_bias)
    w2_bias = layer.get_parameter("w2_bias").data
    experts, hidden = w2_bias.shape
    # Every rank adds this full down bias before MoERunner's all-reduce, as upstream's
    # kernels do. The model loader must zero ``w2_bias`` off rank 0. gpt-oss's loader
    # does; vLLM's generic RoutedExperts weight loader copies it onto every rank.
    biases = _ExpertBiases(
        gate=_to_spyre_expert_weight(w13_bias[:, :inter].unsqueeze(1), (0, pad)),
        up=_to_spyre_expert_weight(w13_bias[:, inter:].unsqueeze(1), (0, pad)),
        down=_to_spyre_expert_weight(w2_bias.unsqueeze(1), ()),
        down_chunks=_to_spyre_expert_weight(
            w2_bias.reshape(experts * chunks, 1, hidden // chunks), ()
        ),
    )
    return biases


# Routed-expert layers prepared on threads while a model loads. The pool belongs to one
# ``prepare_moe_layers_on_threads`` context: leaving it waits, re-raises the first failure
# and joins the threads, also when the load raises.

_DEFAULT_PREP_THREADS = 8
_MAX_PREP_THREADS = 16

_prep_thread = threading.local()


def _prep_thread_count() -> int:
    """``SPYRE_MOE_PREP_THREADS``, else min(8, torch's threads, this rank's CPU share); 1 to 16."""
    threads = envs.SPYRE_MOE_PREP_THREADS
    if threads < 0:
        raise ValueError(f"SPYRE_MOE_PREP_THREADS must be 0 or more, got {threads}")
    if threads == 0:
        threads = min(_DEFAULT_PREP_THREADS, torch.get_num_threads())
    return max(1, min(threads, _MAX_PREP_THREADS))


def _start_prep_thread(device_index: int) -> None:
    torch.spyre.set_device(device_index)
    _prep_thread.stream = torch.spyre.Stream()


class _MoePrep:
    def __init__(self) -> None:
        self._threads: int | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._futures: list[Future[None]] = []
        self._started = False

    def prepare(self, layer: RoutedExperts) -> None:
        if self._threads is None:
            self._threads = _prep_thread_count()
        if self._threads <= 1 or not self._started:
            # The first layer runs here, which also starts the runtime the threads share.
            _prepare_layer(layer)
            self._started = True
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self._threads,
                thread_name_prefix="spyre-moe-prep",
                initializer=_start_prep_thread,
                initargs=(torch.spyre.current_device(),),
            )
            logger.info(
                "Spyre MoE: preparing routed-expert layers on %d threads, each staging "
                "%d MiB of host copies at a time.",
                self._threads,
                _EXPERT_CHUNK_BYTES >> 20,
            )
        self._futures.append(self._pool.submit(self._prepare_on_thread, layer))

    @staticmethod
    def _prepare_on_thread(layer: RoutedExperts) -> None:
        with torch.spyre.stream(_prep_thread.stream):
            _prepare_layer(layer)

    def wait(self) -> None:
        futures, self._futures = self._futures, []
        for future in futures:
            future.result()

    def shutdown(self) -> None:
        self._futures.clear()
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None


_active_prep: contextvars.ContextVar[_MoePrep | None] = contextvars.ContextVar(
    "spyre_moe_prep", default=None
)


@contextmanager
def prepare_moe_layers_on_threads() -> Iterator[None]:
    """Prepare the MoE layers of the model loaded inside on ``SPYRE_MOE_PREP_THREADS`` threads.

    The first layer runs on the caller, which starts the runtime; leaving the context waits
    for the rest and re-raises the first failure. Outside it, layers run on the caller.
    """
    prep = _MoePrep()
    token = _active_prep.set(prep)
    try:
        yield
        prep.wait()
    finally:
        _active_prep.reset(token)
        prep.shutdown()


def _prepare_layer_soon(layer: RoutedExperts) -> None:
    prep = _active_prep.get()
    if prep is None:
        _prepare_layer(layer)
    else:
        prep.prepare(layer)


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
    # A bias with the wrong expert count matches its sibling and
    # would otherwise gather out of range.
    if hasattr(layer, "w13_bias"):
        w13_bias_shape = tuple(layer.get_parameter("w13_bias").shape)
        w2_bias_shape = tuple(layer.get_parameter("w2_bias").shape)
        if w13_bias_shape != (experts, 2 * inter) or w2_bias_shape != (experts, hidden):
            raise ValueError(
                f"unexpected MoE expert bias shapes: w13_bias={w13_bias_shape} "
                f"w2_bias={w2_bias_shape}, expected {(experts, 2 * inter)} and "
                f"{(experts, hidden)}"
            )

    from torch_spyre._inductor import config as spyre_config

    # All shape and alignment checks run before the first device relayout.
    stick = get_elem_in_stick(w13.dtype)
    chunks = _derive_moe_chunks(
        hidden, layer.top_k, stick, spyre_config.sencores, envs.SPYRE_MOE_CHUNKS
    )
    # TP can leave intermediate shards mid-stick; zero-padding is inert because those
    # activation lanes multiply zero rows in the down stack. A gate/up bias pads with zeros
    # too, and its padded activation lanes still meet those zero rows.
    pad = -inter % stick
    if layer.spyre_moe_recipe.activation == "swigluoai":
        # gate on the even rows, up on the odd ones (deinterleave_gate_up), as views.
        gate, up = w13[:, 0::2, :], w13[:, 1::2, :]
    else:
        gate, up = w13[:, :inter, :], w13[:, inter:, :]
    # Share gate/up allocations between persistent and gathered chunk views.
    layer.spyre_moe_gate = _to_spyre_expert_weight(gate.transpose(1, 2), (0, pad))
    layer.spyre_moe_up = _to_spyre_expert_weight(up.transpose(1, 2), (0, pad))
    layer.spyre_moe_chunks = chunks
    layer.spyre_moe_gate_alias = _chunk_pool_alias(layer.spyre_moe_gate, chunks)
    layer.spyre_moe_up_alias = _chunk_pool_alias(layer.spyre_moe_up, chunks)
    del layer.w13_weight, w13, gate, up  # gate and up are views of w13
    w2 = layer.get_parameter("w2_weight").data
    transform_down = layer.spyre_moe_recipe.prepare_down_weight
    prepare = _chunked_down_transform(transform_down) if transform_down is not None else None
    # Keep the complete stack for persistent compute and expose a view for gathered decode.
    layer.spyre_moe_down = _to_spyre_expert_weight(
        w2.transpose(1, 2), (0, 0, 0, pad), kernel_order=True, prepare=prepare
    )
    layer.spyre_moe_down_alias = _down_chunk_pool_alias(layer.spyre_moe_down, chunks)
    layer.spyre_moe_chunk_scale, layer.spyre_moe_chunk_offsets = _chunk_entry_constants(chunks)
    logger.info_once(
        "Spyre MoE: selected %d gathered weight chunks (top_k=%d, cores=%d).",
        chunks,
        layer.top_k,
        spyre_config.sencores,
    )
    del layer.w2_weight, w2
    layer.spyre_moe_biases = _prepare_biases(layer, inter, pad, chunks)

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

    def __init__(self, moe) -> None:
        super().__init__(moe)
        # The source expert parameters are replaced with device-specific stacks. Set
        # after super().__init__, which since 0.31 assigns this per instance.
        self.supports_pre_processed_weights = False

    @property
    def is_monolithic(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: _RoutedExperts) -> None:
        if getattr(layer, "spyre_moe_recipe", None) is None:
            raise NotImplementedError(
                "Spyre MoE backend requires an explicit model-specific recipe; "
                "this architecture has not opted in."
            )
        _prepare_layer_soon(cast("RoutedExperts", layer))
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
                layer.spyre_moe_biases,
                group=recipe.experts_per_trip(tokens),
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
