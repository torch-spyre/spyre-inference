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

"""Spyre ``Attention.forward``: the KV write is traced, the attention core stays opaque.

``install()``, called from the attention metadata builder, binds the forward below onto
each eligible layer instance; every other ``Attention`` keeps upstream's forward and its
``unified_kv_cache_update`` op. The core must stay opaque: its per-sequence Python loop
cannot be captured with ``fullgraph=True``.
"""

import types
import weakref
from collections.abc import Iterable
from typing import cast

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backend import AttentionType

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert

logger = init_logger(__name__)

# vLLM reserves block 0 as `BlockPool.null_block`, so no sequence is ever given its
# slots. `index_copy_` has no skip index, so they absorb writes with nowhere to go.
_NULL_SLOT = 0


class SlotMapping:
    """This step's slot mapping on device, shared by every split layer."""

    def __init__(self, layers: list[Attention]) -> None:
        self._layers = layers
        self._device: torch.device | None = None
        self.slots: torch.Tensor | None = None
        # Per batch row, the narrow-query slot it belongs to; see narrow_query_buffer.
        self.narrow_rows: torch.Tensor | None = None
        self._narrow_spare: int | None = None
        self._published = False
        # Keyed by (num_kv_heads, block_size): a hybrid model's layers need not share
        # either.
        self._folded: dict[tuple[int, int], list[torch.Tensor]] = {}

    def _resolve_device(self) -> torch.device | None:
        if self._device is None:
            # `install` runs before bind_kv_cache, so a layer whose cache never arrives
            # still has the empty default and indexing it would raise.
            self._layers = [layer for layer in self._layers if len(layer.kv_cache) > 0]
            if not self._layers:
                return None
            self._device = self._layers[0].kv_cache[0].device
            self._narrow_spare = getattr(self._layers[0].impl, "narrow_spare_slot", None)
            # Must exist before tracing; see SpyreAttentionImpl.kv_slot_views.
            for layer in self._layers:
                layer.impl.kv_slot_views(layer.kv_cache)
        return self._device

    def publish(
        self, slot_mapping: torch.Tensor, narrow_pairs: list[tuple[int, int]] | None = None
    ) -> None:
        """Mirror a step's host slot mapping to device for the traced write to read."""
        self._publish_host(slot_mapping.clamp(min=_NULL_SLOT), narrow_pairs)

    def publish_null(self, num_tokens: int) -> None:
        self._publish_host(torch.full((num_tokens,), _NULL_SLOT, dtype=torch.int64), None)

    def _publish_host(
        self, host: torch.Tensor, narrow_pairs: list[tuple[int, int]] | None = None
    ) -> None:
        device = self._resolve_device()
        if device is None:
            return
        self._published = True
        self._folded.clear()
        # int64: index_copy_ takes only a long index. Every row a decode does not own
        # points at the spare slot, so the scatter covers the whole bucket at one shape.
        if self._narrow_spare is not None:
            rows = torch.full((host.shape[0],), self._narrow_spare, dtype=torch.int64)
            for row, slot in narrow_pairs or ():
                if row < rows.shape[0]:
                    rows[row] = slot
            self.narrow_rows = convert(rows, device=device)
        if not envs.SPYRE_LX_KV_LAYOUT:
            self.slots = convert(host, device=device)
            return
        # Built here rather than on demand in `slots_for`, which runs inside the model's
        # compiled graph: tensor arithmetic there gets traced into it.
        for num_kv_heads, block_size in self._layer_shapes():
            pages = torch.div(host, block_size, rounding_mode="floor")
            offsets = host - pages * block_size
            self._folded[(num_kv_heads, block_size)] = [
                convert((pages * num_kv_heads + h) * block_size + offsets, device=device)
                for h in range(num_kv_heads)
            ]

    def _layer_shapes(self) -> set[tuple[int, int]]:
        # shape[1] is block_size in both the slot-major and the folded frame.
        return {(layer.num_kv_heads, layer.kv_cache[0].shape[1]) for layer in self._layers}

    def slots_for(
        self, num_kv_heads: int, block_size: int
    ) -> torch.Tensor | list[torch.Tensor] | None:
        """This step's store index in whichever frame the KV cache is in.

        Runs inside the compiled graph, so it must stay a lookup and touch no tensor.
        """
        if not self._published:
            return None
        if not envs.SPYRE_LX_KV_LAYOUT:
            return self.slots
        return self._folded.get((num_kv_heads, block_size))


_holders: weakref.WeakSet[SlotMapping] = weakref.WeakSet()


def publish_null_slots(num_tokens: int) -> None:
    """Point every token at the null block ahead of a run that builds no metadata.

    Warmup would otherwise trace a second graph without the KV write, and a dummy run
    after real inference would scatter into whichever step's slots ran last.
    """
    for holder in _holders:
        holder.publish_null(num_tokens)


def _spyre_attention_forward(
    self: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
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
    kv_cache = self.kv_cache
    # shape[1] is block_size in both the slot-major and the folded frame.
    slots = (
        cast(SlotMapping, self.spyre_slots).slots_for(self.num_kv_heads, kv_cache[0].shape[1])
        if len(kv_cache) > 0
        else None
    )
    if slots is not None and key is not None and value is not None:
        # `dep` makes "scatter before read" a real data dependency, which is otherwise
        # invisible because the op reaches its cache through the forward context.
        dep = self.impl.do_kv_cache_update(self, key, value, kv_cache, slots)

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
        # Scatter every row to its decode slot in one op, so a one-row sequence reads a
        # buffer of max_num_seqs + 1 rows instead of the batch-wide one -- index_select
        # costs O(source), and this source is small whatever the batch. A gather would
        # need an index shorter than the source (torch-spyre#4033), impossible when the
        # token bucket equals max_num_seqs; a scatter has no such rule. Rows no decode
        # owns land on the spare slot, which nothing reads. Traced, never eager: eager
        # index_copy_ falls back to CPU and segfaults.
        narrow_buf = getattr(self.impl, "narrow_query_buffer", None)
        narrow = narrow_buf(query.device) if narrow_buf is not None else None
        if narrow is not None:
            slots = cast(SlotMapping, self.spyre_slots).narrow_rows
            assert slots is not None and slots.shape[0] == rows, (
                "narrow scatter rows must be published at the token bucket's width"
            )
            narrow.index_copy_(0, slots, query)

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


def install(layers: Iterable[Attention]) -> SlotMapping:
    """Opt eligible layers into the traced KV write; returns their shared slot holder."""
    split = [layer for layer in layers if _can_split(layer)]
    slot_mapping = SlotMapping(split)
    _holders.add(slot_mapping)

    for layer in split:
        layer.spyre_slots = slot_mapping  # ty: ignore[invalid-assignment]
        layer.forward = types.MethodType(  # ty: ignore[invalid-assignment]
            _spyre_attention_forward, layer
        )

    if split:
        logger.info(
            "Scattering the KV cache inside the outer graph for %d attention layers.",
            len(split),
        )
    return slot_mapping
