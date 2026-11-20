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

"""Spreading an image's vision encoder pass over engine steps (SPYRE_VISION_BLOCKS_PER_STEP).

Experimental and off by default. Upstream encodes every scheduled image inside the step that
prefills it, so the decodes batched with it wait for the whole tower. Instead the scheduler
plans, each step, which image starts and how many tower blocks it runs, and the runner
carries the plan out before the step's forward. The image's prefill is held back until the
step that finishes its pass.

The plan rides on the ``SchedulerOutput`` as the ``VISION_SLICES_ATTR`` attribute, which
upstream's dataclass pickles to every TP worker with the rest of it. The scheduler counts
steps rather than waiting for a reply from the runner: without async scheduling every
``schedule()`` is followed by exactly one ``execute_model()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

from spyre_inference import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.multimodal.inputs import MultiModalKwargsItem

logger = init_logger(__name__)

VISION_SLICES_ATTR = "spyre_vision_slices"


@dataclass
class VisionSliceStart:
    """An image whose pass begins this step; the runner does not know its request yet."""

    mm_hash: str
    modality: str
    data: MultiModalKwargsItem


@dataclass
class VisionSlices:
    """One step's plan, applied in field order."""

    # Passes to drop: their request finished before the pass did.
    cancel: list[str] = field(default_factory=list)
    start: list[VisionSliceStart] = field(default_factory=list)
    # Blocks each pass runs this step; ``None`` runs the rest and caches the output.
    run: dict[str, int | None] = field(default_factory=dict)


@dataclass(frozen=True)
class VisionSlicing:
    """The engine-wide settings, fixed at scheduler construction."""

    blocks_per_step: int
    # Steps one pass takes: the last runs whatever blocks remain, so a depth read wrong
    # from the config changes only how evenly the pass is spread.
    steps: int
    modalities: tuple[str, ...]


def vision_slicing(vllm_config: VllmConfig) -> VisionSlicing | None:
    """How this engine spreads its encoder passes, or ``None`` when it does not."""
    from vllm.config import CompilationMode

    from spyre_inference.models import spyre_model_cls
    from spyre_inference.models._vision_encoder import SpyreVisionEncoderMixin

    blocks = envs.SPYRE_VISION_BLOCKS_PER_STEP
    model_config = vllm_config.model_config
    if (
        blocks <= 0
        or model_config.runner_type != "generate"
        or not model_config.is_multimodal_model
    ):
        return None
    model_cls = spyre_model_cls(model_config.architecture)
    if model_cls is None or not issubclass(model_cls, SpyreVisionEncoderMixin):
        return None
    if vllm_config.compilation_config.mode is CompilationMode.NONE:
        logger.warning("SPYRE_VISION_BLOCKS_PER_STEP is ignored: the vision tower is not compiled.")
        return None

    mm_config = model_config.multimodal_config
    conflicts = {
        # Step counting needs schedule() and execute_model() in lockstep.
        "async scheduling": bool(vllm_config.scheduler_config.async_scheduling),
        # The worker resolves shm items only for newly scheduled requests.
        "mm_processor_cache_type='shm'": (
            mm_config is not None and mm_config.mm_processor_cache_type == "shm"
        ),
        "LoRA": vllm_config.lora_config is not None,
        "an encoder cache transfer connector": vllm_config.ec_transfer_config is not None,
        "a custom encoder cache manager": (
            vllm_config.ec_manager_config.get_encoder_cache_manager_obj()
            not in (None, EncoderCacheManager)
        ),
    }
    for feature, present in conflicts.items():
        if present:
            raise ValueError(f"SPYRE_VISION_BLOCKS_PER_STEP does not support {feature}.")

    vision_config = getattr(model_config.hf_config, "vision_config", None)
    depth = getattr(vision_config, "num_hidden_layers", None)
    if not isinstance(depth, int):
        raise ValueError(
            "SPYRE_VISION_BLOCKS_PER_STEP needs the vision tower's depth, but "
            f"{model_config.architecture}'s config has no vision_config.num_hidden_layers."
        )
    steps = -(-depth // blocks)
    if steps < 2:
        logger.info(
            "SPYRE_VISION_BLOCKS_PER_STEP=%d covers the whole %d-block tower; images encode "
            "in one step.",
            blocks,
            depth,
        )
        return None
    logger.info(
        "Spreading each image's %d-block vision pass over %d steps while requests run.",
        depth,
        steps,
    )
    return VisionSlicing(blocks, steps, tuple(model_cls.vision_modalities))
