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

The multimodal embeds hook (``prepare_inputs_embeds``) lands here with #1242. Until
then upstream's runs: the mixin sits first in the MRO, so it must not define a method
it does not implement.
"""

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


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
        super().__init__(vllm_config, model, encoder_cache, device)  # ty: ignore[too-many-positional-arguments]
        self.spyre_device = spyre_device
