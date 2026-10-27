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

"""MRV2: Spyre model runner (vllm/v1/worker/gpu/model_runner.py:GPUModelRunner).

A thin coordinator: every method here delegates to the component module
that owns the concern.
"""

from contextlib import AbstractContextManager
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from spyre_inference.v1.worker.spyre import bucketing, kv_cache, load, warmup
from spyre_inference.v1.worker.spyre.pool import output as pool_output


class SpyreModelRunnerV2(GPUModelRunner):
    """MRV2: vllm/v1/worker/gpu/model_runner.py:GPUModelRunner, on device=cpu."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, torch.device("cpu"))
        self.spyre_device = device

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        load.load_and_wrap(self, *args, **kwargs)

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        kv_cache.init(
            self,
            kv_cache_config,
            is_profiling=is_profiling,
            kv_cache_allocation_context=kv_cache_allocation_context,
        )
        self.cudagraph_manager = bucketing.SpyreBucketManager.from_runner(self)

    def pool(self, *args: Any, **kwargs: Any) -> Any:
        return pool_output.pool_sync(self, *args, **kwargs)

    def warming_up_model(self) -> None:
        warmup.run(self)
