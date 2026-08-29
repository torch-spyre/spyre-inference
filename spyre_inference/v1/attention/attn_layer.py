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

    def publish(self, slot_mapping: torch.Tensor) -> None:
        """Mirror a step's host slot mapping to device for the traced write to read."""
        device = self._resolve_device()
        if device is None:
            return
        self.slots = convert(slot_mapping.clamp(min=_NULL_SLOT), device=device)

    def publish_null(self, num_tokens: int) -> None:
        device = self._resolve_device()
        if device is None:
            return
        self.slots = convert(
            torch.full((num_tokens,), _NULL_SLOT, dtype=torch.int64), device=device
        )


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
    slots = cast(SlotMapping, self.spyre_slots).slots
    if slots is not None and key is not None and value is not None:
        # `dep` makes "scatter before read" a real data dependency, which is otherwise
        # invisible because the op reaches its cache through the forward context.
        dep = self.impl.do_kv_cache_update(self, key, value, self.kv_cache, slots)

    torch.ops.vllm.unified_attention_with_output(
        query,  # ty: ignore[invalid-argument-type]
        key,  # ty: ignore[invalid-argument-type]
        value,  # ty: ignore[invalid-argument-type]
        output,  # ty: ignore[invalid-argument-type]
        _encode_layer_name(self.layer_name),  # ty: ignore[invalid-argument-type]
        kv_cache_dummy_dep=dep,  # ty: ignore[invalid-argument-type]
    )
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
        and not layer.calculate_kv_scales
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
