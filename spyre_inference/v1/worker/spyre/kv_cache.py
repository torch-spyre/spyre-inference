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

"""MRV2: SpyreModelRunnerV2.initialize_kv_cache body
(vllm/v1/worker/gpu/model_runner.py:GPUModelRunner.initialize_kv_cache).

Upstream has no override hook for its init_attn_backend / BlockTables
calls, so how init() reaches attn_groups.init() / SpyreBlockTables (full
copy, monkeypatch, or otherwise) is open; see the plan's kv_cache seam
discussion. Also waits on #945 (KVCacheTensor's schema changes at 0.29).
"""

from typing import TYPE_CHECKING, Any

from vllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2


def init(runner: "SpyreModelRunnerV2", kv_cache_config: KVCacheConfig, **kwargs: Any) -> None:
    """Allocate the KV cache, routing attn_groups.init() / SpyreBlockTables
    in place of upstream's init_attn_backend / BlockTables."""
    raise NotImplementedError
