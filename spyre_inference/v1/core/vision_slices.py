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
prefills it, so the decodes batched with it wait for the whole tower. Instead an in-process
encoder cache connector, which the platform installs when the env var is on, runs the tower a
few blocks per step while other requests run, and hands upstream the output as an external
load.

On the scheduler side, ``ensure_cache_available`` starts a pass and defers the request while
one of its images is mid-pass; upstream defers it in place, so FCFS order holds. Once a pass
has run all but its last slice, ``has_cache_item`` turns true and upstream allocates the entry,
and the last slice runs in the step that allocates it, so a finished output never sits outside
the cache's accounting. ``build_connector_meta`` sends each step's plan to every worker inside
the ``SchedulerOutput``, and MRV2 calls ``start_load_caches`` before it gathers the step's
encoder outputs, steps with no tokens included. The connector counts steps rather than waiting
for a reply from the workers: without async scheduling every ``schedule()`` is followed by
exactly one ``execute_model()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.request import RequestStatus

from spyre_inference import envs

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.multimodal.inputs import MultiModalKwargsItem
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

    from spyre_inference.v1.worker.spyre_encoder_graph import SpyreEncoderGraphManager

logger = init_logger(__name__)


@dataclass
class VisionSliceStart:
    """An image whose pass begins this step; the worker does not know its request yet."""

    mm_hash: str
    modality: str
    data: MultiModalKwargsItem


@dataclass
class VisionSlices(ECConnectorMetadata):
    """One step's plan, applied in field order."""

    # Passes to drop: every request waiting on them finished first.
    cancel: list[str] = field(default_factory=list)
    start: list[VisionSliceStart] = field(default_factory=list)
    # Blocks each pass runs this step; ``None`` runs the rest and caches the output.
    run: dict[str, int | None] = field(default_factory=dict)


@dataclass(frozen=True)
class VisionSlicing:
    """The engine-wide settings, fixed when the platform installs the connector."""

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
    ec_config = vllm_config.ec_transfer_config
    conflicts = {
        # Step counting needs schedule() and execute_model() in lockstep.
        "async scheduling": bool(vllm_config.scheduler_config.async_scheduling),
        # The worker resolves shm items only for newly scheduled requests.
        "mm_processor_cache_type='shm'": (
            mm_config is not None and mm_config.mm_processor_cache_type == "shm"
        ),
        "LoRA": vllm_config.lora_config is not None,
        # vLLM builds one encoder cache connector per engine.
        "another encoder cache connector": (
            ec_config is not None and ec_config.ec_connector != SpyreVisionSliceConnector.__name__
        ),
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


@dataclass
class _VisionPass:
    """One image's encoder pass, from its first slice until upstream allocates its entry."""

    # Slices before the last, which waits for the entry.
    slices_left: int
    # Sent with the pass's first slice; ``None`` once sent.
    start: VisionSliceStart | None
    # Requests deferred on the pass; it is cancelled once they have all finished.
    waiters: set[str]
    # Upstream allocated the entry this step, so the last slice goes with this step's plan.
    allocated: bool = False


class SpyreVisionSliceConnector(ECConnectorBase):
    """Produces the encoder output of an image over several steps, in place of upstream's
    one-step encode, while other requests are running (see the module docstring)."""

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole) -> None:
        super().__init__(vllm_config, role)
        assert vllm_config.ec_transfer_config is not None
        extra = vllm_config.ec_transfer_config.ec_connector_extra_config
        self._slicing = VisionSlicing(
            extra["blocks_per_step"], extra["steps"], tuple(extra["modalities"])
        )
        # Scheduler side. By mm_hash, in the order they run.
        self._scheduler: Scheduler | None = None
        self._passes: dict[str, _VisionPass] = {}
        self._cancelled: list[str] = []
        # Whether the last step ran other requests, whose decodes a one-step encode would stall.
        self._busy = False
        # Worker side.
        self._encoder: SpyreEncoderGraphManager | None = None
        self._device: torch.device | None = None

    # ---- Scheduler side ----

    def bind_scheduler(self, scheduler: Scheduler) -> None:
        """The scheduler whose encoder cache it reads, to never pass an image already cached
        or one that can never fit."""
        self._scheduler = scheduler

    def ensure_cache_available(self, request: Request, num_computed_tokens: int) -> bool:
        needed = self._uncached_images(request, num_computed_tokens)
        passes = [self._passes.get(mm_hash) for mm_hash in needed]
        if any(passes):
            for vision_pass in passes:
                if vision_pass is not None:
                    vision_pass.waiters.add(request.request_id)
            # Upstream encodes any image without a pass in one step.
            return all(p.slices_left == 0 for p in passes if p is not None)
        if not needed or not self._wants_pass(request, needed):
            return True
        # One pass at a time, in the order requests are scanned.
        if not self._passes:
            for mm_hash, input_id in needed.items():
                feature = request.mm_features[input_id]
                assert feature.data is not None
                self._passes[mm_hash] = _VisionPass(
                    self._slicing.steps - 1,
                    VisionSliceStart(mm_hash, feature.modality, feature.data),
                    {request.request_id},
                )
        return False

    def has_cache_item(self, identifier: str) -> bool:
        vision_pass = self._passes.get(identifier)
        return vision_pass is not None and vision_pass.slices_left == 0

    def update_state_after_alloc(self, request: Request, index: int) -> None:
        mm_hash = request.mm_features[index].identifier
        vision_pass = self._passes.get(mm_hash)
        if vision_pass is None:
            return
        if vision_pass.slices_left:
            # Upstream is encoding it itself after all.
            self._cancel(mm_hash)
        else:
            vision_pass.allocated = True

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> VisionSlices:
        slices = VisionSlices(cancel=self._cancelled)
        self._cancelled = []
        for mm_hash, vision_pass in list(self._passes.items()):
            if vision_pass.allocated:
                slices.run[mm_hash] = None
                del self._passes[mm_hash]
        current = next(((h, p) for h, p in self._passes.items() if p.slices_left), None)
        if current is not None:
            mm_hash, vision_pass = current
            if vision_pass.start is not None:
                slices.start.append(vision_pass.start)
                vision_pass.start = None
            # With nothing else scheduled there is no decode to protect.
            num_slices = vision_pass.slices_left if not scheduler_output.num_scheduled_tokens else 1
            slices.run[mm_hash] = num_slices * self._slicing.blocks_per_step
            vision_pass.slices_left -= num_slices
        self._busy = bool(scheduler_output.num_scheduled_tokens)
        return slices

    def request_finished(self, request: Request) -> tuple[bool, dict[str, Any] | None]:
        for mm_hash, vision_pass in list(self._passes.items()):
            vision_pass.waiters.discard(request.request_id)
            if not vision_pass.waiters:
                self._cancel(mm_hash)
        return False, None

    def reset(self) -> None:
        """Cancel every pass, as the encoder cache was reset."""
        for mm_hash in list(self._passes):
            self._cancel(mm_hash)

    def _uncached_images(self, request: Request, num_computed_tokens: int) -> dict[str, int]:
        """mm_hash -> first input id, for the request's images still to be read and not cached."""
        assert self._scheduler is not None
        cached = self._scheduler.encoder_cache_manager.cached
        needed: dict[str, int] = {}
        for i, f in enumerate(request.mm_features):
            if (
                f.modality in self._slicing.modalities
                and f.mm_position.offset + f.mm_position.length > num_computed_tokens
                and f.identifier not in cached
            ):
                needed.setdefault(f.identifier, i)
        return needed

    def _wants_pass(self, request: Request, needed: dict[str, int]) -> bool:
        """Whether the request's images get a pass rather than upstream's one-step encode."""
        assert self._scheduler is not None
        if not self._busy or request.num_computed_tokens or request.status != RequestStatus.WAITING:
            return False
        if any(request.mm_features[i].data is None for i in needed.values()):
            return False
        # Images that can never be cached together would wait forever, and every image
        # request behind them; upstream encodes them a chunk at a time instead.
        sizes = [request.get_num_encoder_embeds(i) for i in needed.values()]
        return (
            sum(sizes) <= self._scheduler.encoder_cache_manager.cache_size
            and max(sizes) <= self._scheduler.max_num_encoder_input_tokens
        )

    def _cancel(self, mm_hash: str) -> None:
        del self._passes[mm_hash]
        self._cancelled.append(mm_hash)

    # ---- Worker side ----

    def bind_encoder(self, encoder: SpyreEncoderGraphManager, device: torch.device) -> None:
        """The runner's encoder manager, and the device it batches multimodal kwargs on."""
        self._encoder, self._device = encoder, device

    def start_load_caches(self, encoder_cache: dict[str, torch.Tensor], **kwargs: Any) -> None:
        slices = self._get_connector_metadata()
        assert isinstance(slices, VisionSlices)
        assert self._encoder is not None and self._device is not None
        encoder_cache.update(self._encoder.run_slices(slices, self._device))

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs: Any
    ) -> None:
        # A consumer: nothing leaves this instance.
        return None
