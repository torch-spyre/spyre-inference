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

    Inherited from upstream: budget selection and user overrides, packing items into budgets,
    replay padding, and the per-item output split. Overridden: capture compiles the model's
    graphs at that budget's static shape, and replay copies the inputs into host staging
    buffers, uploads them once and runs those graphs.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        dtype: torch.dtype,
        model: SpyreVisionEncoderMixin,
    ) -> None:
        super().__init__(vllm_config, device, dtype, cast("SupportsEncoderCudaGraph", model))
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
        # Fail at startup rather than compile mid-request when user budgets miss an item.
        largest_item = model.get_encoder_cudagraph_budget_range(vllm_config)[1]
        if self.token_budgets[-1] < largest_item:
            raise ValueError(
                f"largest encoder budget {self.token_budgets[-1]} is below the largest "
                f"item ({largest_item} tokens), which would then compile "
                "mid-request. Raise compilation_config.encoder_cudagraph_token_budgets."
            )
        # Respect a layout limit on items per encoder call when packing. Upstream derived the
        # frame limit from the wider item count, so it shrinks with it.
        cap = model.vision_max_items_per_call
        if cap is not None and cap < self.max_batch_size:
            self.max_batch_size = cap
            self.max_frames_per_batch = min(
                self.max_frames_per_batch, cap * self.config.max_frames_per_video
            )
        self._staging: dict[str, dict[int, dict[str, torch.Tensor]]] = {}

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

    def _upload(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Copy host buffers to the encoder's device."""
        return {key: convert(value, device=self.device) for key, value in values.items()}
