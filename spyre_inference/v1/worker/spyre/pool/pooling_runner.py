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

"""MRV2: pooling (vllm/v1/worker/gpu/pool/pooling_runner.py:PoolingRunner).

Host cursor instead of a GPU pooling cursor, no dim-0 slice (Spyre pads
differently than CUDA), grid-direct CLS/MEAN pooling, and unpadding before
the pooler sees hidden_states.
"""

import torch
from vllm.v1.outputs import PoolerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner
from vllm.v1.worker.gpu.states import RequestState


class SpyrePoolingRunner(PoolingRunner):
    """MRV2: PoolingRunner, pooling synchronously with a host cursor."""

    def pool(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[PoolerOutput, list[bool]]:
        raise NotImplementedError
