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

# SPDX-License-Identifier: Apache-2.0

import functools

from vllm.exceptions import VLLMValidationError
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest

from spyre_inference import envs


def _validate_before_admission() -> None:
    """Run ``TorchSpyreScheduler.validate_request`` on each request EngineCore receives.

    It hooks ``preprocess_add_request``, the step EngineCoreProc runs on its input thread:
    a ``VLLMValidationError`` raised there finishes only that request, with
    ``FinishReason.ERROR`` and the message, whereas one from ``add_request`` or later
    kills the engine with every in-flight request.
    """
    from vllm.v1.engine.core import EngineCore

    original = EngineCore.preprocess_add_request
    if getattr(original, "_spyre_patched", False):
        return

    @functools.wraps(original)
    def preprocess_add_request(self, request: EngineCoreRequest):
        if isinstance(self.scheduler, TorchSpyreScheduler):
            self.scheduler.validate_request(request)
        return original(self, request)

    preprocess_add_request._spyre_patched = True
    EngineCore.preprocess_add_request = preprocess_add_request  # ty: ignore[invalid-assignment]


class TorchSpyreScheduler(Scheduler):
    """V1 scheduler that caps how many sequences may prefill in one batch.

    Prefill attention runs one kernel per sequence, padding each to its own query bucket, so
    the short leftover chunk upstream uses to top up a batch costs a full-width
    prefill however few tokens it carries.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Pooling runners never decode, so there is no leftover chunk to avoid and
        # serialising their prefills only gives up batching.
        self.max_num_partial_prefills = (
            0
            if self.vllm_config.model_config.runner_type == "pooling"
            else envs.SPYRE_MAX_NUM_PARTIAL_PREFILLS
        )
        _validate_before_admission()

    def validate_request(self, request: EngineCoreRequest) -> None:
        """Reject multimodal items the worker cannot run.

        ``TorchSpyrePlatform.validate_request`` covers the Python frontend only; the Rust
        one skips it and never sees the limits ``apply_prelaunch_overrides`` sets, but the
        engine's config carries them.
        """
        from spyre_inference.platform import TorchSpyrePlatform

        mm_config = self.vllm_config.model_config.multimodal_config
        for feature in request.mm_features or ():
            # Upstream lets precomputed embeddings past a zero limit under enable_mm_embeds.
            if mm_config is None or (
                not mm_config.enable_mm_embeds
                and mm_config.get_limit_per_prompt(feature.modality) == 0
            ):
                raise VLLMValidationError(
                    f"This server does not accept {feature.modality} input.",
                    parameter="messages",
                    value=feature.modality,
                )
            TorchSpyrePlatform.check_vision_item(
                feature.modality, feature.mm_position.get_num_embeds()
            )

    def schedule(self, *args, **kwargs) -> SchedulerOutput:
        if self.max_num_partial_prefills <= 0:
            return super().schedule(*args, **kwargs)

        prefilling = sum(
            request.num_computed_tokens < request.num_prompt_tokens for request in self.running
        )
        free_slots = max(self.max_num_partial_prefills - prefilling, 0)
        # The waiting loop re-reads this every iteration and appends one request per
        # admission, so a lowered cap stops it after `free_slots` of them while still
        # letting it skip candidates. Steps that preempt skip the loop entirely.
        # Lower the admission cap, not max_num_running_reqs: since 0.31 the loop gates
        # on max_num_active_reqs, and max_num_running_reqs is the runner slot count
        # that `assert len(self.running) <= ...` checks after the loop.
        max_num_active_reqs = self.max_num_active_reqs
        self.max_num_active_reqs = min(
            max_num_active_reqs,
            len(self.running) + self.num_waiting_for_streaming_input + free_slots,
        )
        try:
            return super().schedule(*args, **kwargs)
        finally:
            self.max_num_active_reqs = max_num_active_reqs
