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

"""MRV2: resolves the Spyre ModelState
(vllm/v1/worker/gpu/model_states/__init__.py:init_model_state).

Mirrors upstream's dispatch, but to the Spyre mixin over whichever upstream
class it resolves to, rather than to the upstream class itself.
"""

import functools

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states import resolve_model_state_cls
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState
from vllm.v1.worker.gpu.model_states.interface import ModelState

from spyre_inference.v1.worker.spyre.model_states.base import SpyreModelStateMixin
from spyre_inference.v1.worker.spyre.model_states.default import SpyreDefaultModelState
from spyre_inference.v1.worker.spyre.model_states.encoder_only import (
    SpyreEncoderOnlyModelState,
)

_SPYRE_STATES: dict[type[ModelState], type[ModelState]] = {
    DefaultModelState: SpyreDefaultModelState,
    EncoderOnlyModelState: SpyreEncoderOnlyModelState,
}


@functools.cache
def spyre_model_state_cls(upstream_cls: type[ModelState]) -> type[ModelState]:
    spyre_cls = _SPYRE_STATES.get(upstream_cls)
    if spyre_cls is None:
        spyre_cls = type(f"Spyre{upstream_cls.__name__}", (SpyreModelStateMixin, upstream_cls), {})
    return spyre_cls


def resolve_spyre_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
    spyre_device: torch.device,
) -> ModelState:
    """Pick and construct the Spyre-mixed ModelState; ``device`` is the host."""
    cls = resolve_model_state_cls(vllm_config, model)
    # As init_model_state does: reject a state that would silently ignore this.
    if vllm_config.model_config.enable_prompt_embeds and not cls.supports_prompt_embeds:
        raise ValueError(f"--enable-prompt-embeds not supported with {cls.__name__}.")
    return spyre_model_state_cls(cls)(
        vllm_config,
        model,
        encoder_cache,
        device,
        spyre_device,  # ty: ignore[too-many-positional-arguments]
    )
