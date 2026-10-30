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
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from spyre_inference import envs
from spyre_inference.v1.attention.spyre_attn_bucketer import batched_decode_query_len

# Upstream's padded-draft sentinel: the rejection sampler rejects it, the runner clamps
# it to a valid id before the embedding, and structured output strips it.
_PAD_DRAFT_TOKEN_ID = -1


class TorchSpyreScheduler(Scheduler):
    """V1 scheduler that caps how many sequences may prefill in one batch.

    Prefill attention runs one kernel per sequence, padding each to its own query bucket, so
    the short leftover chunk upstream uses to top up a batch costs a full-width
    prefill however few tokens it carries.

    Under speculative decoding it also pads every draft list out to k, so each
    verification has the single query width the batched decode kernel serves.
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
        self._pad_drafts = batched_decode_query_len(self.vllm_config) > 1

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        # After super(), so grammar validation has already cut the real drafts.
        super().update_draft_token_ids(draft_token_ids)
        if not self._pad_drafts:
            return
        for req_id in draft_token_ids.req_ids:
            request = self.requests.get(req_id)
            # The requests upstream leaves without drafts.
            if request is None or request.is_finished() or request.is_prefill_chunk:
                continue
            missing = self.num_spec_tokens - len(request.spec_token_ids)
            if missing > 0:
                # A new list: spec_token_ids may alias one the caller still holds.
                request.spec_token_ids = request.spec_token_ids + [_PAD_DRAFT_TOKEN_ID] * missing

    def schedule(self, *args, **kwargs) -> SchedulerOutput:
        output = self._schedule(*args, **kwargs)
        if self._pad_drafts:
            # Padding is not a proposal: keep it out of the acceptance stats, which
            # subtract these counts from the drafted ones.
            padded = {
                req_id: tokens.count(_PAD_DRAFT_TOKEN_ID)
                for req_id, tokens in output.scheduled_spec_decode_tokens.items()
                if _PAD_DRAFT_TOKEN_ID in tokens
            }
            if padded:
                invalid = output.num_invalid_spec_tokens or {}
                output.num_invalid_spec_tokens = {**invalid, **padded}
        return output

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        # Upstream counts a draft before subtracting the invalid tokens, so a step whose
        # drafts are all padding would still count as one and deflate the mean
        # acceptance length.
        if num_invalid_spec_tokens and num_draft_tokens == num_invalid_spec_tokens.get(
            request_id, 0
        ):
            return spec_decoding_stats
        return super().make_spec_decoding_stats(
            spec_decoding_stats,
            num_draft_tokens,
            num_accepted_tokens,
            num_invalid_spec_tokens,
            request_id,
        )

    def _schedule(self, *args, **kwargs) -> SchedulerOutput:
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
