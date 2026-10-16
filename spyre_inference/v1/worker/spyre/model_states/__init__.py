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

from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


def resolve_spyre_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
) -> Any:
    """Pick and construct the Spyre-mixed ModelState for this model."""
    raise NotImplementedError
