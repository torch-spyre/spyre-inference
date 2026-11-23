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

"""Upstream's vision encoder budget manager, with compiled graphs in place of CUDA graphs."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import torch
from vllm.logger import init_logger
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from spyre_inference.custom_ops.utils import convert

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph

    from spyre_inference.models._vision_encoder import SpyreVisionEncoderMixin

logger = init_logger(__name__)

# Arguments of one compiled replay, in the order ``_execute_segments`` runs them.
_QueuedCall = tuple[dict[str, Any], int, str, tuple]
# budget, path, axis keys, device slot, device inputs already copied.
_Armed = tuple[int, str, tuple, int, dict[str, torch.Tensor]]


class _PlannedBatch(NamedTuple):
    indices: list[int]
    mm_kwargs: dict[str, Any]
    axis_keys: tuple
    path_budgets: dict[str, int | None]
    path_tokens: dict[str, int]


class _Segment(NamedTuple):
    """One packed group. ``index_map`` sends a local item index back to the caller's."""

    mm_kwargs: dict[str, Any]
    planned: list[_PlannedBatch]
    index_map: list[int]


class SpyreEncoderGraphManager(EncoderCudaGraphManager):
    """Compiles the encoder at each token budget during warmup, then serves through it.

    Inherited from upstream: budget selection, user overrides, replay padding, and the
    per-item output split. Capture compiles the model's graphs at that budget's static
    shape and keeps the upload as device slot 0. Packing is a copy of upstream's greedy
    pack (``_plan_batches``), so the next batch is known before the current forward
    returns. The manager counts as captured while the host staging buffers are live.

    Packed batches do not depend on each other. After the device forward, the next batch
    at the same ``(path, budget)`` is padded into slot 1 and its upload is enqueued. The
    download of the current result is what waits for that upload. A different path or
    budget uses its own slot 0, which is not in flight.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        dtype: torch.dtype,
        model: SpyreVisionEncoderMixin,
    ) -> None:
        cap = model.vision_max_items_per_call
        by_tokens = model.vision_items_limited_by_tokens
        if by_tokens and cap is None:
            raise ValueError(
                f"{type(model).__name__} sets vision_items_limited_by_tokens, which needs "
                "vision_max_items_per_call (e.g. SEGMENT_ITEMS for segment_buffer())."
            )
        # Upstream bounds items per call by the smallest budget for its per-item capture split,
        # which Spyre never does: an opted-in model hides the user's limit and applies it below.
        comp_config = vllm_config.compilation_config
        user_max_items = comp_config.encoder_cudagraph_max_vision_items_per_batch
        if by_tokens:
            comp_config.encoder_cudagraph_max_vision_items_per_batch = 0
        try:
            super().__init__(vllm_config, device, dtype, cast("SupportsEncoderCudaGraph", model))
        finally:
            comp_config.encoder_cudagraph_max_vision_items_per_batch = user_max_items
        # One graph set per budget: upstream's extra capture axes are not supported.
        if self._capture_axes:
            raise NotImplementedError("encoder capture_axes are not supported on Spyre.")
        # Tensor parallelism only: the encoder runs on the decoder's ranks.
        if self.use_dp:
            raise NotImplementedError(
                "mm_encoder_tp_mode='data' is not supported on Spyre; the vision "
                "encoder runs tensor-parallel with the decoder."
            )
        # Round every budget up so its patch count is a whole vision_patch_multiple.
        step = model.vision_patch_multiple // model.patches_per_token

        def aligned(budgets: list[int]) -> list[int]:
            return sorted({-(-b // step) * step if b > 0 else b for b in budgets})

        self.token_budgets = aligned(self.token_budgets)
        self.path_token_budgets = {p: aligned(b) for p, b in self.path_token_budgets.items()}
        logger.info("Vision encoder budgets: %s", self.path_token_budgets)
        # Fail at startup rather than compile mid-request when user budgets miss an item.
        largest_item = model.get_encoder_cudagraph_budget_range(vllm_config)[1]
        if self.token_budgets[-1] < largest_item:
            raise ValueError(
                f"largest encoder budget {self.token_budgets[-1]} is below the largest "
                f"item ({largest_item} tokens), which would then compile "
                "mid-request. Raise compilation_config.encoder_cudagraph_token_budgets."
            )
        if by_tokens:
            assert cap is not None
            self.max_batch_size = min(cap, user_max_items) if user_max_items > 0 else cap
            if (
                self.max_frames_per_batch
                and comp_config.encoder_cudagraph_max_frames_per_batch is None
            ):
                self.max_frames_per_batch = self.max_batch_size * self.config.max_frames_per_video
        # Otherwise respect a layout limit on items per encoder call when packing. Upstream
        # derived the frame limit from the wider item count, so it shrinks with it.
        elif cap is not None and cap < self.max_batch_size:
            self.max_batch_size = cap
            self.max_frames_per_batch = min(
                self.max_frames_per_batch, cap * self.config.max_frames_per_video
            )
        self._pack_tokens = model.vision_pack_tokens
        self._pack_budgets: dict[str, list[int]] = {}
        if self._pack_tokens is not None:
            for path, budgets in self.path_token_budgets.items():
                fit = self._path_fitting_budget(path, self._pack_tokens)
                self._pack_budgets[path] = (
                    budgets if fit is None else [b for b in budgets if b <= fit]
                )
        self._staging: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        # Slot 0 is the capture upload. Slot 1 is allocated on the first overlap for a
        # budget, so a step that never splits does not keep a second device copy.
        self._host_slot1: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        self._device_slots: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}
        self._queue: list[_QueuedCall] | None = None
        self._armed: _Armed | None = None
        # Slot in flight for one (path, budget). Another key's slot 0 is idle.
        self._current_key: tuple[str, int] | None = None
        self._current_slot = 0

    def _execute_local(self, mm_kwargs: dict[str, Any]) -> list[torch.Tensor]:
        """Upstream's greedy packing, with each item above ``vision_pack_tokens`` run alone.

        The small group and each lone large item are separate packs, but they share one
        queue. A Gemma 4 step with two large images, or a small group plus a large image,
        still stages the next forward under the current one.
        """
        pack = self._pack_tokens
        if pack is None:
            count = len(self._get_item_specs(mm_kwargs))
            segment = _Segment(mm_kwargs, self._plan_batches(mm_kwargs), list(range(count)))
            return self._execute_segments([segment])
        tokens = self._get_per_item_out_tokens(mm_kwargs)
        small = [i for i, t in enumerate(tokens) if t <= pack]
        budgets = self.path_token_budgets
        segments: list[_Segment] = []
        try:
            # Upstream reads path_token_budgets per call: the packed group sees only the
            # capped budgets, and a lone large item every budget.
            self.path_token_budgets = self._pack_budgets
            if small:
                if len(small) == len(tokens):
                    group, index_map = mm_kwargs, list(range(len(tokens)))
                else:
                    group = self._select_items(mm_kwargs, small)[0]
                    index_map = small
                segments.append(_Segment(group, self._plan_batches(group), index_map))
        finally:
            self.path_token_budgets = budgets
        for i, t in enumerate(tokens):
            if t > pack:
                item = self._select_items(mm_kwargs, [i])[0]
                segments.append(_Segment(item, self._plan_batches(item), [i]))
        return self._execute_segments(segments)

    def _plan_batches(self, mm_kwargs: dict[str, Any]) -> list[_PlannedBatch]:
        """Greedy pack copied from ``EncoderCudaGraphManager._execute_local``.

        The copy exists so the next batch is known before the current forward returns.
        ``upgrade-vllm`` has to re-sync ``should_defer_item`` and the fit check; the
        drift test compares this plan with that upstream method.
        """
        item_specs = self._get_item_specs(mm_kwargs)
        paths = tuple(self.path_token_budgets)
        per_item_path_tokens = {
            path: [spec.get_path_output_tokens(path) for spec in item_specs] for path in paths
        }
        per_item_out_tokens = [spec.output_tokens for spec in item_specs]
        max_path_budgets = {path: max(self.path_token_budgets[path]) for path in paths}
        sorted_indices = sorted(range(len(item_specs)), key=lambda i: per_item_out_tokens[i])

        batches: list[tuple[list[int], dict[str, int | None]]] = []
        current_batch: list[int] = []
        current_tokens = dict.fromkeys(paths, 0)

        def append_current_batch() -> None:
            if not current_batch:
                return
            path_budgets = {
                path: self._path_fitting_budget(path, current_tokens[path]) for path in paths
            }
            batches.append((list(current_batch), path_budgets))

        def should_defer_item(item_tokens: dict[str, int]) -> bool:
            """Whether closing the batch strictly reduces summed budgets."""
            for path in paths:
                item = item_tokens[path]
                if item == 0:
                    continue
                cur = current_tokens[path]
                b_cur = self._path_fitting_budget(path, cur)
                b_alone = self._path_fitting_budget(path, item)
                b_merged = self._path_fitting_budget(path, cur + item)
                # Only called when the item fits; lookups land on a budget.
                assert b_cur is not None and b_alone is not None and b_merged is not None
                # Only defer items that alone outweigh the whole current batch.
                if b_cur + b_alone < b_merged and b_alone >= b_cur:
                    return True
            return False

        for orig_idx in sorted_indices:
            item_tokens = {path: per_item_path_tokens[path][orig_idx] for path in paths}
            fits = len(current_batch) < self.max_batch_size and all(
                current_tokens[path] + item_tokens[path] <= max_path_budgets[path] for path in paths
            )
            if current_batch and (not fits or should_defer_item(item_tokens)):
                append_current_batch()
                current_batch = []
                current_tokens = dict.fromkeys(paths, 0)
            current_batch.append(orig_idx)
            for path in paths:
                current_tokens[path] += item_tokens[path]
        append_current_batch()

        planned: list[_PlannedBatch] = []
        for indices, path_budgets in batches:
            selected, axis_keys = self._select_items(mm_kwargs, indices)
            path_tokens = {
                path: sum(per_item_path_tokens[path][i] for i in indices) for path in paths
            }
            planned.append(_PlannedBatch(indices, selected, axis_keys, path_budgets, path_tokens))
        return planned

    def _execute_segments(self, segments: list[_Segment]) -> list[torch.Tensor]:
        """Run every planned batch on one queue, staging the next under the current forward."""
        queue: list[_QueuedCall] = []
        for segment in segments:
            for batch in segment.planned:
                for path, path_tokens in batch.path_tokens.items():
                    if path_tokens == 0:
                        continue
                    budget = batch.path_budgets[path]
                    if budget is not None:
                        queue.append((batch.mm_kwargs, budget, path, batch.axis_keys))
        outputs: dict[int, torch.Tensor] = {}
        self._queue = queue
        self._armed = None
        try:
            for segment in segments:
                per_item_out_tokens = [
                    spec.output_tokens for spec in self._get_item_specs(segment.mm_kwargs)
                ]
                local: dict[int, torch.Tensor] = {}
                for batch in segment.planned:
                    graph_outputs: dict[str, torch.Tensor] = {}
                    all_eager = True
                    for path, path_tokens in batch.path_tokens.items():
                        if path_tokens == 0:
                            continue
                        budget = batch.path_budgets[path]
                        if budget is None:
                            with torch.inference_mode():
                                output = self.model.encoder_eager_forward(
                                    batch.mm_kwargs, path=path
                                )
                        else:
                            all_eager = False
                            output = self._run_budget_graph(
                                batch.mm_kwargs, budget, path=path, axis_keys=batch.axis_keys
                            )
                            assert output is not None
                        graph_outputs[path] = output
                    if all_eager:
                        self.graph_misses += len(batch.indices)
                    self.model.postprocess_encoder_output(
                        graph_outputs,
                        batch.indices,
                        per_item_out_tokens,
                        local,
                        clone=True,
                        batch_mm_kwargs=batch.mm_kwargs,
                    )
                for local_index, orig_index in enumerate(segment.index_map):
                    outputs[orig_index] = local[local_index]
        finally:
            self._queue = None
            self._armed = None
        count = max((index for segment in segments for index in segment.index_map), default=-1)
        return [outputs[i] for i in range(count + 1)]

    def _capture_budget_graph(
        self, token_budget: int, path: str = "default", axis_keys: tuple = ()
    ) -> None:
        """Compile the encoder at this budget, keeping its host buffers for replay."""
        values = self.model.prepare_encoder_cudagraph_capture_inputs(
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
            torch.device("cpu"),
            self.dtype,
            path,
        ).values
        t0 = time.time()
        device_inputs = self._upload(values)
        # Same entry as replay. ``encoder_cudagraph_forward`` only adds the host download.
        model = cast("SpyreVisionEncoderMixin", self.model)
        with torch.inference_mode():
            model._encoder_device_forward(device_inputs, path=path)
        logger.info("Vision encoder budget %d compiled in %.1fs.", token_budget, time.time() - t0)
        self._staging.setdefault(path, {})[token_budget] = values
        self._device_slots[(path, token_budget, 0)] = device_inputs

    def _run_budget_graph(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
        path: str = "default",
        axis_keys: tuple = (),
    ) -> torch.Tensor | None:
        """Run one packed batch through the budget's compiled graphs; ``None`` if uncompiled.

        When a following call is queued, its host padding and H2D run after this forward
        is launched and before the output is copied back. An unstaged successor is left
        unarmed, so the next staged batch starts from slot 0.
        """
        if self._queue is not None:
            if not self._queue:
                raise RuntimeError("encoder batch queue underflow")
            queued_kwargs, queued_budget, queued_path, queued_axes = self._queue.pop(0)
            # Identity, not equality: the kwargs hold tensors, and dict equality would
            # raise "Boolean value of Tensor with more than one value is ambiguous".
            if (
                queued_kwargs is not mm_kwargs
                or queued_budget != token_budget
                or queued_path != path
                or queued_axes != axis_keys
            ):
                raise RuntimeError("encoder batch queue does not match the call")
        staging = self._staging.get(path, {}).get(token_budget)
        num_items = len(self._get_item_specs(mm_kwargs))
        if staging is None:
            self.graph_misses += num_items
            self._armed = None
            return None
        armed = self._armed
        if armed is not None and armed[:3] == (token_budget, path, axis_keys):
            _budget, _path, _axis, slot, device_inputs = armed
            self._armed = None
        else:
            self._armed = None
            slot = 0
            device_inputs = self._fill_and_copy(mm_kwargs, token_budget, path, slot)
        self._current_key = (path, token_budget)
        self._current_slot = slot
        model = cast("SpyreVisionEncoderMixin", self.model)
        with torch.inference_mode():
            output = model._encoder_device_forward(device_inputs, path=path)
        # This batch's output is not an input of the next one. Padding and the H2D
        # enqueue happen before the download waits for the device.
        self._arm_next()
        output = convert(output, device="cpu")
        self.graph_hits += num_items
        return output

    def _arm_next(self) -> None:
        """Pad the queued batch into the idle slot and enqueue its H2D."""
        queue = self._queue
        if not queue:
            return
        mm_kwargs, token_budget, path, axis_keys = queue[0]
        if self._staging.get(path, {}).get(token_budget) is None:
            return
        # Only the key this forward is reading needs the other slot. A different
        # path or budget still has slot 0 free.
        if self._current_key == (path, token_budget):
            slot = 1 - self._current_slot
        else:
            slot = 0
        device_inputs = self._fill_and_copy(mm_kwargs, token_budget, path, slot)
        self._armed = (token_budget, path, axis_keys, slot, device_inputs)

    def _host_for(self, path: str, token_budget: int, slot: int) -> dict[str, torch.Tensor]:
        if slot == 0:
            return self._staging[path][token_budget]
        return self._host_slot1[path][token_budget]

    def _ensure_slot1(self, path: str, token_budget: int) -> None:
        """Allocate the idle host and device buffers the first time a budget overlaps."""
        if token_budget in self._host_slot1.get(path, {}):
            return
        values = self._staging[path][token_budget]
        alternate = {key: value.clone() for key, value in values.items()}
        self._host_slot1.setdefault(path, {})[token_budget] = alternate
        self._device_slots[(path, token_budget, 1)] = self._upload(alternate)

    def _fill_and_copy(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
        path: str,
        slot: int,
    ) -> dict[str, torch.Tensor]:
        """Pad this batch into the host slot and copy it to the matching device slot."""
        if slot == 1:
            self._ensure_slot1(path, token_budget)
        host = self._host_for(path, token_budget, slot)
        replay = self.model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs, self.max_batch_size, self.max_frames_per_batch, path
        )
        for key, buf in host.items():
            # The mixin validates replay buffers: every key is present and none is None.
            self.config.padding_logics[key](buf, cast(torch.Tensor, replay.values[key]))
        device = self._device_slots[(path, token_budget, slot)]
        for key, buf in host.items():
            dst = device[key]
            if dst.data_ptr() == buf.data_ptr() and dst.storage_offset() == buf.storage_offset():
                # CPU replay keeps the host buffer as the device buffer.
                continue
            if dst.device.type == "spyre" and slot == 1:
                # The current download syncs the stream before the next forward reads
                # this slot. Slot 0 is read on the next line, so that copy stays blocking:
                # a non_blocking H2D elsewhere in this repo must be followed by
                # torch.spyre.synchronize before the tensors are consumed.
                dst.copy_(buf, non_blocking=True)
            else:
                dst.copy_(buf)
        return device

    def is_captured(self) -> bool:
        # Captured once a budget is staged; upstream's check reads the CUDA graph pool,
        # which Spyre leaves None.
        return bool(self._staging)

    def clear(self) -> None:
        # The compiled graphs live on the model; only the staging buffers are dropped.
        super().clear()
        self._staging.clear()
        self._host_slot1.clear()
        self._device_slots.clear()
        self._queue = None
        self._armed = None
        self._current_key = None
        self._current_slot = 0

    def _upload(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Copy host buffers to the encoder's device."""
        return {key: convert(value, device=self.device) for key, value in values.items()}
