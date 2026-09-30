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

"""Spyre ``Attention.forward``: the KV write is traced, and so is small pure-decode attention.

``install()``, called from the attention metadata builder, binds the forward below onto
each eligible layer instance; every other ``Attention`` keeps upstream's forward and its
``unified_kv_cache_update`` op. Any other step keeps the opaque core: a mixed step's
per-sequence loop cannot be captured with ``fullgraph=True``.
"""

import collections
import types
import weakref
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backend import AttentionType

from spyre_inference import envs

if TYPE_CHECKING:
    from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadata

logger = init_logger(__name__)

# vLLM reserves block 0 as `BlockPool.null_block`, so no sequence is ever given its
# slots. `index_copy_` has no skip index, so they absorb writes with nowhere to go.
_NULL_SLOT = 0

# Set by the model runner: a block compiled without fullgraph could graph-break inside
# for_each_tile silently rather than fail, so decode attention stays opaque there.
outer_graph_fullgraph: bool = True

_DECODE_PATH_LOG_EVERY = 256


class SlotMapping:
    """This step's slot mapping on device, shared by every split layer."""

    def __init__(self, layers: list[Attention]) -> None:
        self._layers = layers
        self._device: torch.device | None = None
        # One tensor, or one per KV head: whatever the impl's KV store indexes by.
        self.slots: torch.Tensor | list[torch.Tensor] | None = None

    def _resolve_device(self) -> torch.device | None:
        if self._device is None:
            # `install` runs before bind_kv_cache, so a layer whose cache never arrives
            # still has the empty default and indexing it would raise.
            self._layers = [layer for layer in self._layers if len(layer.kv_cache) > 0]
            if not self._layers:
                return None
            self._device = self._layers[0].kv_cache[0].device
            # Must exist before tracing; see SpyreAttentionImpl.kv_slot_views.
            for layer in self._layers:
                layer.impl.kv_slot_views(layer.kv_cache)  # ty: ignore[possibly-missing-attribute]
        return self._device

    def _write_index(self, slot_mapping: torch.Tensor, device: torch.device):
        """The device-side KV store index in this group's cache layout."""
        return self._layers[0].impl.kv_write_index(slot_mapping, device)  # ty: ignore[possibly-missing-attribute]

    def publish(self, slot_mapping: torch.Tensor) -> None:
        """Mirror a step's host slot mapping to device for the traced write to read."""
        device = self._resolve_device()
        if device is None:
            return
        self.slots = self._write_index(slot_mapping.clamp(min=_NULL_SLOT), device)

    def publish_null(self, num_tokens: int) -> None:
        device = self._resolve_device()
        if device is None:
            return
        self.slots = self._write_index(
            torch.full((num_tokens,), _NULL_SLOT, dtype=torch.int64), device
        )


class DecodeGrid:
    """This step's batched-decode index tensors on device, or None off the inline path."""

    def __init__(self, slots: SlotMapping) -> None:
        self._slots = slots
        # Whether any layer traces from this grid; otherwise only the mirror runs.
        self.inline = False
        self.steps: collections.Counter[str] = collections.Counter()
        self.opaque_keys: collections.Counter[tuple] = collections.Counter()
        self.rep_row_ids: torch.Tensor | None = None
        self.page_ids: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None

    def publish(self, attn_metadata: "SpyreAttentionMetadata") -> None:
        self.rep_row_ids = self.page_ids = self.mask = None
        path = self._publish(attn_metadata)
        if self.inline:
            self._count(path, attn_metadata)

    def _publish(self, attn_metadata: "SpyreAttentionMetadata") -> str:
        pure_decode = attn_metadata.num_decode_seqs == attn_metadata.num_seqs
        per_seq = "per_seq" if pure_decode else "mixed"
        if attn_metadata.padded_num_seqs is None:
            return per_seq
        device = self._slots._resolve_device()
        if device is None:
            return per_seq
        impl = next(
            (
                layer.impl
                for layer in self._slots._layers
                if layer.impl._batched_decode_preconditions_met(attn_metadata)  # ty: ignore[possibly-missing-attribute]
            ),
            None,
        )
        if impl is None:
            return per_seq
        impl._mirror_batched_decode_indices(attn_metadata, device)  # ty: ignore[possibly-missing-attribute]
        assert attn_metadata.blocks_per_chunk is not None
        if not pure_decode:
            return "mixed"
        b_seqs, bpc = attn_metadata.padded_num_seqs, attn_metadata.blocks_per_chunk
        if (
            not self.inline
            or not impl.inline_batched_decode(b_seqs, bpc)  # ty: ignore[possibly-missing-attribute]
            # A gather selecting every page faults the device (torch-spyre#4033); the
            # recorder skips these variants too, so they stay on the opaque path.
            or b_seqs * bpc >= self._slots._layers[0].kv_cache[0].shape[0]
        ):
            return "batched_opaque"
        self.rep_row_ids = attn_metadata.rep_row_ids_dev
        self.page_ids = attn_metadata.chunk_page_ids_dev
        self.mask = attn_metadata.mask_by_chunk_dev
        return "inline"

    def _count(self, path: str, attn_metadata: "SpyreAttentionMetadata") -> None:
        from spyre_inference.v1.attention.backends.spyre_attn import is_warmup_complete

        if not is_warmup_complete():
            return
        self.steps[path] += 1
        if path == "batched_opaque":
            self.opaque_keys[(attn_metadata.padded_num_seqs, attn_metadata.blocks_per_chunk)] += 1
        total = sum(self.steps.values())
        if total % _DECODE_PATH_LOG_EVERY == 0:
            logger.info(
                "Decode attention over %d steps: %s; opaque batched (num_seqs, "
                "blocks_per_chunk): %s.",
                total,
                dict(self.steps),
                dict(self.opaque_keys),
            )


_holders: weakref.WeakSet[SlotMapping] = weakref.WeakSet()
_grids: weakref.WeakSet[DecodeGrid] = weakref.WeakSet()


def publish_null_slots(num_tokens: int, keep_decode_grids: bool = False) -> None:
    """Point every token at the null block ahead of a run that builds no metadata.

    Warmup would otherwise trace a second graph without the KV write, and a dummy run
    after real inference would scatter into whichever step's slots ran last. The decode
    grids clear too, so such a run traces the opaque attention path, unless warmup has
    just published them to trace the inline one.
    """
    for holder in _holders:
        holder.publish_null(num_tokens)
    if not keep_decode_grids:
        clear_decode_grids()


def clear_decode_grids() -> None:
    for grid in _grids:
        grid.rep_row_ids = grid.page_ids = grid.mask = None


def _spyre_attention_forward(
    self: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    grid = cast(DecodeGrid | None, getattr(self, "spyre_decode_grid", None))
    # The inline path returns [tokens, num_heads * head_size] at the query's dtype.
    if grid is not None and (output_shape is not None or output_dtype not in (None, query.dtype)):
        grid = None
    if output_dtype is None:
        output_dtype = query.dtype
    if output_shape is None:
        output_shape = torch.Size((query.shape[0], self.num_heads * self.head_size_v))
    output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
    hidden_size = output_shape[-1]

    query = query.view(-1, self.num_heads, self.head_size)
    output = output.view(-1, self.num_heads, self.head_size_v)
    if key is not None:
        key = key.view(-1, self.num_kv_heads, self.head_size)
    if value is not None:
        value = value.view(-1, self.num_kv_heads, self.head_size_v)

    dep = None
    slots = cast(SlotMapping, self.spyre_slots).slots
    if slots is not None and key is not None and value is not None:
        # `dep` makes "scatter before read" a real data dependency, which is otherwise
        # invisible because the op reaches its cache through the forward context.
        dep = self.impl.do_kv_cache_update(self, key, value, self.kv_cache, slots)

    if grid is not None and grid.mask is not None and dep is not None:
        return _inline_batched_decode(self, grid, query).view(-1, hidden_size)

    # Staged here, not in the impl: the copies are then traced into the block
    # graph, which warmup already compiles at every token bucket.
    staging = getattr(self.impl, "staging_buffers", None)
    buffers = staging(query.device) if staging is not None else None
    rows = query.shape[0]
    if buffers is None:
        q_in, out_buf = query, output
    else:
        q_in, out_buf = buffers
        q_in[:rows] = query

    torch.ops.vllm.unified_attention_with_output(
        q_in,  # ty: ignore[invalid-argument-type]
        key,  # ty: ignore[invalid-argument-type]
        value,  # ty: ignore[invalid-argument-type]
        out_buf,  # ty: ignore[invalid-argument-type]
        _encode_layer_name(self.layer_name),  # ty: ignore[invalid-argument-type]
        kv_cache_dummy_dep=dep,  # ty: ignore[invalid-argument-type]
    )
    if buffers is not None:
        output.copy_(out_buf[:rows])
    return output.view(-1, hidden_size)


def _inline_batched_decode(self: Attention, grid: DecodeGrid, query: torch.Tensor) -> torch.Tensor:
    # Static arguments come from the grid's shapes, which Dynamo guards on; pages are read
    # through the views the KV write scattered into, so the graph mutates one input.
    impl = self.impl
    mask, rep_row_ids = cast(torch.Tensor, grid.mask), cast(torch.Tensor, grid.rep_row_ids)
    b_seqs = mask.shape[-4]
    k_rows, v_rows = impl.kv_slot_views(self.kv_cache)  # ty: ignore[possibly-missing-attribute]
    page_shape = self.kv_cache[0].shape
    # Staged even here: the kernel's row gather has no legal layout from the in-graph
    # query's tiled one, only from the staging buffer's row-outermost one.
    q_in, _ = impl.staging_buffers(query.device)  # ty: ignore[possibly-missing-attribute]
    q_in[: query.shape[0]] = query
    attn = impl.inline_decode_kernel(  # ty: ignore[possibly-missing-attribute]
        q_in,
        rep_row_ids,
        k_rows.view(page_shape),
        v_rows.view(page_shape),
        grid.page_ids,
        mask,
        impl.scale,  # ty: ignore[possibly-missing-attribute]
        b_seqs,
        rep_row_ids.shape[0] // b_seqs,
        impl.num_kv_heads,  # ty: ignore[possibly-missing-attribute]
        impl.num_queries_per_kv,  # ty: ignore[possibly-missing-attribute]
        impl.block_size,  # ty: ignore[possibly-missing-attribute]
        impl.head_size,  # ty: ignore[possibly-missing-attribute]
        impl.logits_soft_cap,  # ty: ignore[possibly-missing-attribute]
    )
    # The body bucket and the num_seqs bucket coincide under the default buckets; a
    # custom compile_sizes can split them. Padded rows stay finite for MoE routing.
    rows = query.shape[0]
    if b_seqs > rows:
        return attn[:rows]
    if b_seqs < rows:
        return torch.cat([attn, attn.new_zeros(rows - b_seqs, *attn.shape[1:])])
    return attn


def _inline_decode_refusal(layer: Attention) -> str | None:
    impl = layer.impl
    if getattr(impl, "inline_decode_kernel", None) is None:
        return f"{type(impl).__name__} has no traceable batched decode kernel"
    if not impl._batched_decode_supported():  # ty: ignore[possibly-missing-attribute]
        return "batched decode is off (SPYRE_BATCHED_DECODE=0, eager attention, or ALiBi)"
    if envs.SPYRE_ATTN_MAX_CORES:
        return "SPYRE_ATTN_MAX_CORES caps every attention compile, which a block graph cannot scope"
    if not outer_graph_fullgraph:
        return "the block compiles without fullgraph (Spyre FP8 linears)"
    return None


def _can_split(layer: Attention) -> bool:
    """Only Spyre paged attention, and only where upstream's own prologue is a no-op."""
    return (
        # Encoder-only impls inherit `do_kv_cache_update` from the paged one and would
        # otherwise scatter into an unbound cache.
        layer.attn_type == AttentionType.DECODER
        and hasattr(layer.impl, "do_kv_cache_update")
        and layer.kv_sharing_target_layer_name is None
        and layer.query_quant is None
    )


def install(layers: Iterable[Attention]) -> tuple[SlotMapping, DecodeGrid]:
    """Opt eligible layers into the traced KV write; returns their shared holders."""
    split = [layer for layer in layers if _can_split(layer)]
    slot_mapping = SlotMapping(split)
    _holders.add(slot_mapping)
    decode_grid = DecodeGrid(slot_mapping)
    _grids.add(decode_grid)

    refusal = next((r for r in map(_inline_decode_refusal, split) if r is not None), None)
    decode_grid.inline = bool(split) and refusal is None
    for layer in split:
        layer.spyre_slots = slot_mapping  # ty: ignore[invalid-assignment]
        layer.spyre_decode_grid = decode_grid if decode_grid.inline else None  # ty: ignore[invalid-assignment]
        layer.forward = types.MethodType(  # ty: ignore[invalid-assignment]
            _spyre_attention_forward, layer
        )

    if split:
        logger.info(
            "Scattering the KV cache inside the outer graph for %d attention layers.",
            len(split),
        )
        if refusal is None:
            logger.info("Tracing batched decode attention into the outer graph.")
        else:
            logger.info("Batched decode attention stays behind the opaque op: %s.", refusal)
    return slot_mapping, decode_grid
