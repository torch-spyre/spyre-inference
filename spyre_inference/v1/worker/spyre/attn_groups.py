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

"""MRV2: attention-group construction (vllm/v1/worker/gpu/attn_utils.py:init_attn_backend).

Upstream's init_attn_backend, then the sliding-window split V1 also uses
(split_attn_groups_by_window).
"""

from collections.abc import Callable
from typing import Any

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import get_attn_cg_support
from vllm.v1.worker.ubatch_utils import get_num_ubatches
from vllm.v1.worker.utils import AttentionGroup

from spyre_inference.v1.worker.kv_cache_setup import split_attn_groups_by_window


def init(
    upstream_init_attn_backend: Callable[..., Any],
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    device: torch.device,
    *args: Any,
    **kwargs: Any,
) -> tuple[list[list[AttentionGroup]], Any, list[int]]:
    attn_groups, attn_cg_support, kernel_block_sizes = upstream_init_attn_backend(
        kv_cache_config, vllm_config, device, *args, **kwargs
    )
    split_attn_groups_by_window(attn_groups, get_layers_from_vllm_config(vllm_config, Attention))
    # Upstream builds builders before the split, so the groups it creates have none.
    split = False
    for kv_cache_group_id, groups in enumerate(attn_groups):
        for group in groups:
            if group.metadata_builders:
                continue
            split = True
            group.create_metadata_builders(
                vllm_config=vllm_config,
                device=device,
                kernel_block_size=(
                    kernel_block_sizes[kv_cache_group_id]
                    if kv_cache_group_id < len(kernel_block_sizes)
                    else None
                ),
                num_metadata_builders=get_num_ubatches(vllm_config.parallel_config),
            )
    if split:
        # Upstream computed it over the pre-split groups.
        attn_cg_support = get_attn_cg_support(attn_groups, vllm_config)
    return attn_groups, attn_cg_support, kernel_block_sizes
