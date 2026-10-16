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

"""MRV2: Spyre bucket padding via ModelCudaGraphManager.dispatch
(vllm/v1/worker/gpu/cudagraph_utils.py).

Spyre has no cudagraph capture, so dispatch() only picks the padded bucket
size and always reports CUDAGraphMode.NONE.
"""

from typing import TYPE_CHECKING

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
)

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2


class SpyreBucketManager(ModelCudaGraphManager):
    """MRV2: ModelCudaGraphManager, replacing capture/dispatch with bucket padding."""

    @classmethod
    def from_runner(cls, runner: "SpyreModelRunnerV2") -> "SpyreBucketManager":
        """Build the bucket manager from an already-initialized runner."""
        raise NotImplementedError

    def bucket(self, num_tokens: int) -> int:
        """Pad num_tokens up to the next bucket (SPYRE_ATTN_QUERY_BUCKETS)."""
        raise NotImplementedError

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        max_query_len: int | None = None,
        num_ubatches: int = 1,
    ) -> BatchExecutionDescriptor:
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=self.bucket(num_tokens),
            num_reqs=num_reqs,
            num_active_loras=self._resolve_effective_loras(num_active_loras),
        )
