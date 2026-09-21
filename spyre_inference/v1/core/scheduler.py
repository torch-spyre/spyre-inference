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
from vllm.v1.core.sched.scheduler import Scheduler

from spyre_inference import envs


class TorchSpyreScheduler(Scheduler):
    """V1 scheduler that caps how many sequences may prefill in one batch.

    Attention runs one kernel per sequence, padding each to its own query bucket, so
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
        max_num_running_reqs = self.max_num_running_reqs
        self.max_num_running_reqs = min(
            max_num_running_reqs,
            len(self.running) + self.num_waiting_for_streaming_input + free_slots,
        )
        try:
            return super().schedule(*args, **kwargs)
        finally:
            self.max_num_running_reqs = max_num_running_reqs
