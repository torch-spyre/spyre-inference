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

"""Muse Glimmer (text backbone) on Spyre: upstream's model, its final norm in the model
dtype (upstream builds it with an fp32 weight, which the logits-row gather after it
cannot lower on Spyre).

The vision tower is not adapted, so a checkpoint carrying one serves text only; see
``disable_vision``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models import muse_glimmer as mg

from spyre_inference.models._prelaunch import disable_modalities

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)


def disable_vision(engine_args: EngineArgs, hf_config: PretrainedConfig | None) -> None:
    """Serve a Muse Glimmer checkpoint with a vision tower text only, refusing image or
    video input asked for in ``limit_mm_per_prompt``.

    ``language_model_only`` makes upstream build the tower as a placeholder, so its weights
    are neither loaded nor moved to the card, and the frontend refuses image and video input
    per request. Images are not supported on Spyre yet: the tower would run op by op, and
    its host-built position tables have no device layout.
    """
    if engine_args.language_model_only:
        return
    if getattr(hf_config, "model_type", None) != "muse_glimmer":
        return
    if not mg._muse_glimmer_has_vision(hf_config):
        return
    disable_modalities(engine_args, ("image", "video"), "Muse Glimmer")
    engine_args.language_model_only = True
    logger.info("Muse Glimmer: serving text only (language_model_only=True).")


class SpyreMuseGlimmerForCausalLM(mg.MuseGlimmerForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        norm = self.model.norm
        self.model.norm = RMSNorm(norm.hidden_size, eps=norm.variance_epsilon)
