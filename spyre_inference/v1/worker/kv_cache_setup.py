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

"""Attention-group window split and per-layer KV page allocation, shared by both
model runners."""

from dataclasses import replace
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.worker.utils import AttentionGroup

from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyrePagedKVCache,
)

logger = init_logger(__name__)


def split_attn_groups_by_window(
    attn_groups: list[list[AttentionGroup]], layers: dict[str, Any]
) -> None:
    """Give each distinct per-layer sliding window its own attention group, in place."""
    # `FullAttentionSpec.merge` collapses a hybrid decoder onto the one window it finds,
    # clamping its full-attention layers too (gemma-3-1b-it: 512 on all 26). Spyre masks
    # per group, not per layer as upstream does, so the group is what has to split.
    for kv_cache_group_id, groups in enumerate(attn_groups):
        split_groups: list[AttentionGroup] = []
        for group in groups:
            spec = group.kv_cache_spec
            windows: dict[int | None, list[str]] = {}
            for layer_name in group.layer_names:
                window = getattr(layers.get(layer_name), "sliding_window", None)
                windows.setdefault(window, []).append(layer_name)
            # SlidingWindowSpec is single-window by construction; UniformTypeKVCacheSpecs
            # is already unwrapped per layer upstream.
            if not isinstance(spec, FullAttentionSpec) or len(windows) <= 1:
                split_groups.append(group)
                continue
            logger.info(
                "Split %d attention layers into %d groups by sliding window %s.",
                len(group.layer_names),
                len(windows),
                {w: len(n) for w, n in windows.items()},
            )
            for window, layer_names in windows.items():
                split_groups.append(
                    AttentionGroup(
                        group.backend,
                        layer_names,
                        replace(spec, sliding_window=window),
                        kv_cache_group_id,
                    )
                )
        attn_groups[kv_cache_group_id] = split_groups


def allocate_spyre_kv_caches(
    kv_cache_config: KVCacheConfig,
    static_ctx: dict[str, Any],
    shared_layers: dict[str, str],
    device: torch.device,
) -> dict[str, SpyrePagedKVCache]:
    """One SpyrePagedKVCache per layer, in the layout its attention impl allocates."""
    # One spec per layer. disable_hybrid_kv_cache_manager (set in the platform)
    # collapses hybrid models into a single group; when the layers' head shapes differ
    # its spec is a UniformTypeKVCacheSpecs, which unwraps to the real per-layer specs
    # so each layer keeps its own num_kv_heads/head_size. A merged spec is direct.
    spec_by_layer = {}
    for group in kv_cache_config.kv_cache_groups:
        per_layer = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        if per_layer is not None:
            spec_by_layer.update(per_layer)
        else:
            spec_by_layer.update({ln: group.kv_cache_spec for ln in group.layer_names})

    kv_caches: dict[str, SpyrePagedKVCache] = {}
    # A KVCacheTensor lists several layers; sharing one page buffer among them
    # would let each layer's KV write clobber the others'.
    num_blocks = kv_cache_config.num_blocks
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        for layer_name in kv_cache_tensor.layers:
            spec = spec_by_layer[layer_name]
            # The layout belongs to the backend; a layer without an impl (fixture
            # stubs) gets the token-major default.
            impl = getattr(static_ctx.get(layer_name), "impl", None)
            impl_cls = type(impl) if isinstance(impl, SpyreAttentionImpl) else SpyreAttentionImpl
            kv_caches[layer_name] = impl_cls.allocate_pages(num_blocks, spec, device)

    for layer_name, target in shared_layers.items():
        kv_caches[layer_name] = kv_caches[target]
    return kv_caches
