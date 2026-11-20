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

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler

from spyre_inference import envs
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    choose_rectangle_admissions,
    encoder_shape_tables,
)

# How far past the oldest waiting request rectangle admission looks for companions.
# Past this the oldest request is still admitted; the rest wait for a later step.
_RECT_ADMISSION_WINDOW = 256


class TorchSpyreScheduler(Scheduler):
    """V1 scheduler that caps how many sequences may prefill in one batch.

    Prefill attention runs one kernel per sequence, padding each to its own query bucket, so
    the short leftover chunk upstream uses to top up a batch costs a full-width
    prefill however few tokens it carries.

    Pooling never decodes, so that cap is off. Instead, when chunked prefill is off,
    a step admits only the requests that fill one encoder rectangle: the oldest
    waiting request, plus the companions that carry the most real tokens without
    exceeding that rectangle's width. Upstream admits by real token count and will
    otherwise take one sequence more than the rectangle holds, which drops the step
    onto the ragged path.
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

    def schedule(self, *args, **kwargs) -> SchedulerOutput:
        if self._rectangle_admission_applies():
            return self._schedule_pooling_rectangle(*args, **kwargs)
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

    def _rectangle_admission_applies(self) -> bool:
        """Pooling, FCFS, whole prompts. Chunked prefill and priority keep upstream."""
        if not envs.SPYRE_ENCODER_RECT_ADMISSION:
            return False
        if self.vllm_config.model_config.runner_type != "pooling":
            return False
        if self.scheduler_config.enable_chunked_prefill:
            return False
        return self.policy == SchedulingPolicy.FCFS

    def _schedule_pooling_rectangle(self, *args, **kwargs) -> SchedulerOutput:
        chosen = self._rectangle_waiting_requests()
        if not chosen:
            return super().schedule(*args, **kwargs)
        # ``prepend_request`` is appendleft, so walk the chosen list backwards to
        # leave the oldest request at the front. Upstream then admits that prefix
        # and stops when the next one does not fit, which this cap makes be the
        # first request outside the rectangle.
        self.waiting.remove_requests(chosen)
        for request in reversed(chosen):
            self.waiting.prepend_request(request)
        max_num_active_reqs = self.max_num_active_reqs
        self.max_num_active_reqs = min(
            max_num_active_reqs,
            len(self.running) + self.num_waiting_for_streaming_input + len(chosen),
        )
        try:
            return super().schedule(*args, **kwargs)
        finally:
            self.max_num_active_reqs = max_num_active_reqs

    def _rectangle_waiting_requests(self) -> list:
        if not self.waiting:
            return []
        window = []
        for index, request in enumerate(self.waiting):
            if index >= _RECT_ADMISSION_WINDOW:
                break
            window.append(request)
        running = [left for request in self.running if (left := _tokens_left(request)) > 0]
        free_slots = (
            self.max_num_active_reqs - len(self.running) - self.num_waiting_for_streaming_input
        )
        indices = choose_rectangle_admissions(
            [_tokens_left(request) for request in window],
            encoder_shape_tables(self.vllm_config).rectangles,
            token_budget=self.max_num_scheduled_tokens,
            max_new=max(0, free_slots),
            running_lengths=running,
        )
        if not indices:
            return []
        return [window[index] for index in indices]


def _tokens_left(request) -> int:
    return max(0, int(request.num_tokens) - int(request.num_computed_tokens))
