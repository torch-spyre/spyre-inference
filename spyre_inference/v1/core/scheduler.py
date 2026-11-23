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

from collections.abc import Iterable
from dataclasses import dataclass

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from spyre_inference import envs
from spyre_inference.v1.core.vision_slices import (
    VISION_SLICES_ATTR,
    VisionSlices,
    VisionSliceStart,
    vision_slicing,
)


@dataclass
class _VisionPass:
    """One image's encoder pass, from allocation until the step that finishes it."""

    steps_left: int
    # Sent with the pass's first slice; ``None`` once sent.
    start: VisionSliceStart | None


class TorchSpyreScheduler(Scheduler):
    """V1 scheduler that caps how many sequences may prefill in one batch.

    Prefill attention runs one kernel per sequence, padding each to its own query bucket, so
    the short leftover chunk upstream uses to top up a batch costs a full-width
    prefill however few tokens it carries.

    With SPYRE_VISION_BLOCKS_PER_STEP (experimental) it also spreads each image's encoder pass
    over several steps while other requests run (see ``vision_slices``). One pass runs at a
    time, in the order requests wait. A waiting request is held out of upstream's scan while
    it has an image a pass would encode, so upstream never encodes one inside a single step,
    and so is any request, those holding KV blocks included, with an image a pass is encoding.
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
        self._vision_slicing = vision_slicing(self.vllm_config)
        # By mm_hash, in the order they run; the first is the one running.
        self._vision_passes: dict[str, _VisionPass] = {}
        # Request id -> the mm_hashes it allocated passes for.
        self._vision_owners: dict[str, list[str]] = {}
        self._vision_cancelled: list[str] = []

    def schedule(self, *args, **kwargs) -> SchedulerOutput:
        if self._vision_slicing is None:
            return self._schedule_capped(*args, **kwargs)
        slices = self._plan_vision_slices()
        # Upstream drains kv_holding_waiting first (e.g. a streaming session's next chunk),
        # and its requests never start a pass, so only an in-flight one holds them.
        held = [
            (self.waiting, [r for r in self.waiting if self._held_for_vision(r)]),
            (
                self.kv_holding_waiting,
                [r for r in self.kv_holding_waiting if self._reads_vision_pass(r)],
            ),
        ]
        for queue, requests in held:
            if requests:
                queue.remove_requests(requests)
        try:
            output = self._schedule_capped(*args, **kwargs)
        finally:
            for queue, requests in held:
                if requests:
                    # FCFS prepends in reverse, so reverse first to keep their order.
                    queue.prepend_requests(reversed(requests))
        if slices.cancel or slices.start or slices.run:
            setattr(output, VISION_SLICES_ATTR, slices)
        return output

    def _schedule_capped(self, *args, **kwargs) -> SchedulerOutput:
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

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        finished = super().finish_requests(request_ids, finished_status)
        cached = self.encoder_cache_manager.cached
        for request in finished:
            for mm_hash in self._vision_owners.pop(request.request_id, ()):
                # The entry was allocated but never filled, yet upstream's free leaves it
                # reading as cached. Only the owner could reach it (same-image requests
                # are held), so evict it before a later request takes it as encoded.
                if mm_hash in self._vision_passes and not cached.get(mm_hash):
                    self._cancel_vision_pass(mm_hash)
        return finished

    def reset_encoder_cache(self) -> None:
        super().reset_encoder_cache()
        self._vision_cancelled.extend(self._vision_passes)
        self._vision_passes.clear()
        self._vision_owners.clear()

    def _plan_vision_slices(self) -> VisionSlices:
        """This step's slices: start a pass if none runs, then advance or finish the current."""
        assert self._vision_slicing is not None
        slices = VisionSlices(cancel=self._vision_cancelled)
        self._vision_cancelled = []
        if not self._vision_passes and self.running:
            for request in self.waiting:
                if self._wants_vision_pass(request):
                    # The head waits for cache space rather than letting later ones pass.
                    self._allocate_vision_passes(request)
                    break
        if not self._vision_passes:
            return slices

        mm_hash, current = next(iter(self._vision_passes.items()))
        if current.start is not None:
            slices.start.append(current.start)
            current.start = None
        # With nothing running there is no decode left to protect.
        if current.steps_left <= 1 or not self.running:
            slices.run[mm_hash] = None
            del self._vision_passes[mm_hash]
            self._vision_owners = {
                req_id: hashes
                for req_id, hashes in self._vision_owners.items()
                if any(h in self._vision_passes for h in hashes)
            }
        else:
            slices.run[mm_hash] = self._vision_slicing.blocks_per_step
            current.steps_left -= 1
        return slices

    def _held_for_vision(self, request: Request) -> bool:
        return self._reads_vision_pass(request) or (
            bool(self.running) and self._wants_vision_pass(request)
        )

    def _reads_vision_pass(self, request: Request) -> bool:
        """Whether an image of the request is mid-pass: its entry already reads as cached."""
        return any(f.identifier in self._vision_passes for f in request.mm_features)

    def _wants_vision_pass(self, request: Request) -> bool:
        """A new request with an image to encode, that a pass can take.

        Not one whose image a running request may still reach: that request's
        prefill would find the image cached before its pass finished.
        """
        if request.num_computed_tokens or request.status != RequestStatus.WAITING:
            return False
        new = self._new_vision_items(request)
        if not new or any(request.mm_features[i].data is None for i in new.values()):
            return False
        # Images that can never be cached together would hold the request (and every one
        # queued behind it) forever; upstream encodes them a chunk at a time instead.
        sizes = [request.get_num_encoder_embeds(i) for i in new.values()]
        if (
            sum(sizes) > self.encoder_cache_manager.cache_size
            or max(sizes) > self.max_num_encoder_input_tokens
        ):
            return False
        elsewhere = {
            f.identifier
            for other in (*self.running, *self.kv_holding_waiting)
            for f in other.mm_features
        }
        return elsewhere.isdisjoint(new)

    def _new_vision_items(self, request: Request) -> dict[str, int]:
        """mm_hash -> first input id, for the request's images not cached yet."""
        assert self._vision_slicing is not None
        cached = self.encoder_cache_manager.cached
        new: dict[str, int] = {}
        for i, f in enumerate(request.mm_features):
            if f.modality in self._vision_slicing.modalities and f.identifier not in cached:
                new.setdefault(f.identifier, i)
        return new

    def _allocate_vision_passes(self, request: Request) -> None:
        assert self._vision_slicing is not None
        manager = self.encoder_cache_manager
        new = self._new_vision_items(request)
        reserved = 0
        for input_id in new.values():
            if not manager.can_allocate(
                request, input_id, self.max_num_encoder_input_tokens, reserved
            ):
                return
            reserved += request.get_num_encoder_embeds(input_id)
        for mm_hash, input_id in new.items():
            manager.allocate(request, input_id)
            feature = request.mm_features[input_id]
            assert feature.data is not None
            self._vision_passes[mm_hash] = _VisionPass(
                self._vision_slicing.steps,
                VisionSliceStart(mm_hash, feature.modality, feature.data),
            )
        self._vision_owners[request.request_id] = list(new)

    def _cancel_vision_pass(self, mm_hash: str) -> None:
        """Drop the pass and evict its entry, as ``can_allocate`` evicts a freeable one."""
        del self._vision_passes[mm_hash]
        self._vision_cancelled.append(mm_hash)
        manager = self.encoder_cache_manager
        manager.num_free_slots += manager.freeable.pop(mm_hash)
        del manager.cached[mm_hash]
        del manager.num_embeds_by_hash[mm_hash]
        manager.freed.append(mm_hash)
