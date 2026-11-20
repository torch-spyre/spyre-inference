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
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.logger import init_logger
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from spyre_inference.custom_ops.utils import convert

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph

    from spyre_inference.models._vision_encoder import SpyreVisionEncoderMixin

logger = init_logger(__name__)


class SpyreEncoderGraphManager(EncoderCudaGraphManager):
    """Compiles the encoder at each token budget during warmup, then serves through it.

    Inherited from upstream: budget selection and user overrides, replay padding, and the
    per-item output split. Overridden: capture compiles the model's graphs at that budget's
    static shape, replay copies the inputs into host staging buffers, uploads them once and
    runs those graphs, and the manager counts as captured while those buffers are staged.
    Kept from vLLM 0.33, because each budget is one tower compile at warmup and any other
    shape compiles mid-request: the power-of-two budget grid, and greedy packing of a batch
    into budgets, where 0.34 runs one batch per path and goes eager past the largest budget.
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

    @staticmethod
    def _generate_budgets(min_budget: int, max_budget: int, cap: int | None = None) -> list[int]:
        """vLLM 0.33's power-of-two budgets from min_budget, plus max_budget. ``cap``
        (encoder_cudagraph_max_token_budget) is ignored: past it a batch would compile."""
        budgets: list[int] = []
        b = min_budget
        while b <= max_budget:
            budgets.append(b)
            b *= 2
        if not budgets or budgets[-1] < max_budget:
            budgets.append(max_budget)
        return budgets

    @classmethod
    def _item_multiple_budgets(cls, item_tokens: int, max_budget: int) -> list[int]:
        return cls._generate_budgets(item_tokens, max_budget)

    def _execute_local(self, mm_kwargs: dict[str, Any]) -> list[torch.Tensor]:
        """Greedy packing, with each item above ``vision_pack_tokens`` run alone."""
        pack = self._pack_tokens
        if pack is None:
            return self._execute_packed(mm_kwargs)
        tokens = self._get_per_item_out_tokens(mm_kwargs)
        small = [i for i, t in enumerate(tokens) if t <= pack]
        budgets = self.path_token_budgets
        outputs: dict[int, torch.Tensor] = {}
        try:
            # Upstream reads path_token_budgets per call: the packed group sees only the
            # capped budgets, and a lone large item every budget.
            self.path_token_budgets = self._pack_budgets
            if len(small) == len(tokens):
                return self._execute_packed(mm_kwargs)
            if small:
                group = self._select_items(mm_kwargs, small)[0]
                outputs.update(zip(small, self._execute_packed(group), strict=True))
            self.path_token_budgets = budgets
            for i, t in enumerate(tokens):
                if t > pack:
                    (outputs[i],) = self._execute_packed(self._select_items(mm_kwargs, [i])[0])
        finally:
            self.path_token_budgets = budgets
        return [outputs[i] for i in range(len(tokens))]

    def _execute_packed(self, mm_kwargs: dict[str, Any]) -> list[torch.Tensor]:
        """vLLM 0.33's ``_execute_local``: smallest items first, into batches that each fit a budget."""
        item_specs = self._get_item_specs(mm_kwargs)
        num_items = len(item_specs)
        paths = tuple(self.path_token_budgets)
        per_item_path_tokens = {
            path: [spec.get_path_output_tokens(path) for spec in item_specs] for path in paths
        }
        per_item_out_tokens = [spec.output_tokens for spec in item_specs]
        max_path_budgets = {path: max(self.path_token_budgets[path]) for path in paths}

        batches: list[tuple[list[int], dict[str, int | None]]] = []
        current_batch: list[int] = []
        current_tokens = dict.fromkeys(paths, 0)

        def append_current_batch() -> None:
            if current_batch:
                path_budgets = {
                    path: self._path_fitting_budget(path, current_tokens[path]) for path in paths
                }
                batches.append((list(current_batch), path_budgets))

        def should_defer_item(item_tokens: dict[str, int]) -> bool:
            for path in paths:
                item = item_tokens[path]
                if item == 0:
                    continue
                cur = current_tokens[path]
                b_cur = self._path_fitting_budget(path, cur)
                b_alone = self._path_fitting_budget(path, item)
                b_merged = self._path_fitting_budget(path, cur + item)
                assert b_cur is not None and b_alone is not None and b_merged is not None
                # Only defer items that alone outweigh the whole current batch.
                if b_cur + b_alone < b_merged and b_alone >= b_cur:
                    return True
            return False

        for orig_idx in sorted(range(num_items), key=lambda i: per_item_out_tokens[i]):
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

        outputs_by_orig_idx: dict[int, torch.Tensor] = {}
        for batch_indices, path_budgets in batches:
            batch_mm_kwargs, axis_keys = self._select_items(mm_kwargs, batch_indices)
            graph_outputs: dict[str, torch.Tensor] = {}
            all_eager = True
            for path in paths:
                if sum(per_item_path_tokens[path][i] for i in batch_indices) == 0:
                    continue
                token_budget = path_budgets[path]
                if token_budget is None:
                    with torch.inference_mode():
                        output = self.model.encoder_eager_forward(batch_mm_kwargs, path=path)
                else:
                    all_eager = False
                    output = self._run_budget_graph(
                        batch_mm_kwargs, token_budget, path=path, axis_keys=axis_keys
                    )
                    assert output is not None
                graph_outputs[path] = output
            if all_eager:
                self.graph_misses += len(batch_indices)
            self.model.postprocess_encoder_output(
                graph_outputs,
                batch_indices,
                per_item_out_tokens,
                outputs_by_orig_idx,
                clone=True,
                batch_mm_kwargs=batch_mm_kwargs,
            )
        return [outputs_by_orig_idx[i] for i in range(num_items)]

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
        with torch.inference_mode():
            self.model.encoder_cudagraph_forward(self._upload(values), path=path)
        logger.info("Vision encoder budget %d compiled in %.1fs.", token_budget, time.time() - t0)
        self._staging.setdefault(path, {})[token_budget] = values

    def _run_budget_graph(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
        path: str = "default",
        axis_keys: tuple = (),
    ) -> torch.Tensor | None:
        """Run one packed batch through the budget's compiled graphs; ``None`` if uncompiled."""
        staging = self._staging.get(path, {}).get(token_budget)
        num_items = len(self._get_item_specs(mm_kwargs))
        if staging is None:
            self.graph_misses += num_items
            return None
        replay = self.model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs, self.max_batch_size, self.max_frames_per_batch, path
        )
        for key, buf in staging.items():
            # The mixin validates replay buffers: every key is present and none is None.
            self.config.padding_logics[key](buf, cast(torch.Tensor, replay.values[key]))
        with torch.inference_mode():
            output = self.model.encoder_cudagraph_forward(self._upload(staging), path=path)
        self.graph_hits += num_items
        return output

    def is_captured(self) -> bool:
        # Captured once a budget is staged; upstream's check reads the CUDA graph pool,
        # which Spyre leaves None.
        return bool(self._staging)

    def clear(self) -> None:
        # The compiled graphs live on the model; only the staging buffers are dropped.
        super().clear()
        self._staging.clear()

    def _upload(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Copy host buffers to the encoder's device."""
        return {key: convert(value, device=self.device) for key, value in values.items()}
