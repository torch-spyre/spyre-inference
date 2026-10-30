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

Mixed into whichever upstream ModelState a model resolves to. Upstream's states
allocate on ``device``, which for this runner is the host; ``spyre_device`` is
where tensors built for the accelerator itself go (encoder attention masks).

A multimodal state's EncoderRunner is the Spyre one (mm/encoder_runner.py), whose
inputs_embeds staging buffer lives on Spyre. Upstream's prepare_inputs_embeds and
dummy_inputs_embeds then run as-is. The mixin sits first in the MRO, so it must not
define a method it does not implement.
"""

import functools
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states import interface as upstream_interface

from spyre_inference.v1.worker.spyre.mm.encoder_runner import SpyreEncoderRunner


class SpyreModelStateMixin:
    """MRV2: ModelState, aware of the Spyre device beside the host one."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
        spyre_device: torch.device,
    ) -> None:
        # Upstream's ModelState.__init__ builds its EncoderRunner by module-level name.
        encoder_runner_cls = upstream_interface.EncoderRunner
        upstream_interface.EncoderRunner = functools.partial(  # ty: ignore[invalid-assignment]
            SpyreEncoderRunner, spyre_device=spyre_device
        )
        try:
            super().__init__(vllm_config, model, encoder_cache, device)  # ty: ignore[too-many-positional-arguments]
        finally:
            upstream_interface.EncoderRunner = encoder_runner_cls
        self.spyre_device = spyre_device

    def get_supported_generation_tasks(self) -> Any:
        # Upstream's interface checks (is_text_generation_model, supports_transcription)
        # fail on load.py's _SpyreModelWrapper, whose __init__ takes no vllm_config.
        model = self.model
        bare = getattr(model, "_model", model)
        self.model = getattr(bare, "_orig_mod", bare)
        try:
            return super().get_supported_generation_tasks()
        finally:
            self.model = model
