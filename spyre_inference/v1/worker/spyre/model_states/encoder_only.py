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

"""MRV2: encoder-only model state (vllm/v1/worker/gpu/model_states/encoder_only.py:
EncoderOnlyModelState).

Dispatches through an EncoderPlan (rectangular grid vs. ragged rows)
instead of upstream's non-causal metadata path directly.
"""

from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from spyre_inference.v1.worker.spyre.model_states.base import SpyreModelStateMixin
from spyre_inference.v1.worker.spyre.model_states.encoder_plan import EncoderPlan


class SpyreEncoderOnlyModelState(SpyreModelStateMixin, EncoderOnlyModelState):
    """MRV2: EncoderOnlyModelState, dispatching through an EncoderPlan."""

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        raise NotImplementedError

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _build_encoder_plan(self, input_batch: InputBatch) -> EncoderPlan:
        raise NotImplementedError
