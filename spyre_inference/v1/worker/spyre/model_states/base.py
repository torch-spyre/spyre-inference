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

"""MRV2: Spyre mixin over vllm/v1/worker/gpu/model_states/interface.py:ModelState.

Mixed into whichever upstream ModelState a model resolves to. Owns the one
hook every Spyre model state needs regardless of shape: multimodal
embeddings staged on Spyre rather than gathered on the GPU-native path.
"""

from typing import TYPE_CHECKING

import torch
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.states import RequestState


class SpyreModelStateMixin:
    """MRV2: ModelState, staging inputs_embeds on Spyre instead of the GPU."""

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: "RequestState",
    ) -> torch.Tensor | None:
        raise NotImplementedError

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        raise NotImplementedError
