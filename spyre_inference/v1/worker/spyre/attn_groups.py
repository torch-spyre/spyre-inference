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

Splits each group by per-layer sliding window after init_attn_backend
runs: FullAttentionSpec.merge collapses a hybrid decoder onto one window,
clamping its full-attention layers too. MRV2 counterpart of V1's
_split_attn_groups_by_layer_window (spyre_model_runner.py).
"""

from typing import TYPE_CHECKING, Any

from vllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2


def init(runner: "SpyreModelRunnerV2", kv_cache_config: KVCacheConfig, **kwargs: Any) -> None:
    """Build runner.attn_groups and runner.kernel_block_sizes, split by window."""
    raise NotImplementedError
