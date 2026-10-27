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

"""MRV2: encoder-only model state (vllm/v1/worker/gpu/model_states/encoder_only.py:
EncoderOnlyModelState).

Dispatches through an EncoderPlan (rectangular grid vs. ragged rows)
instead of upstream's non-causal metadata path directly. Plan selection and
grid expansion are V1's own functions, shared rather than copied.

MRV2 calls prepare_attn before prepare_inputs, so the plan is built first.
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, cast

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from spyre_inference.models.roberta import offset_host_positions, roberta_position_delta
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    plan_encoder_step,
    publish_encoder_grid,
)
from spyre_inference.v1.worker.encoder_grid import (
    EncoderDispatchStats,
    encoder_pad_token_id,
    expand_body_to_grid,
)
from spyre_inference.v1.worker.spyre.model_states.base import SpyreModelStateMixin
from spyre_inference.v1.worker.spyre.model_states.encoder_plan import EncoderPlan
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    encoder_shape_tables,
    expand_packed_token_types,
    group_width_caps,
)


class SpyreEncoderOnlyModelState(SpyreModelStateMixin, EncoderOnlyModelState):
    """MRV2: EncoderOnlyModelState, dispatching through an EncoderPlan."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
        spyre_device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device, spyre_device)
        tables = encoder_shape_tables(vllm_config)
        self._rectangles = list(tables.rectangles)
        # Per body, ascending. A step padded to 2048 must not use the 8192 caps.
        self._width_caps_by_rows = {
            body: group_width_caps(groups) for body, groups in tables.tier_groups
        }
        self._bodies = tuple(body for body, _groups in tables.tier_groups)
        # Grouping only pays off through the compiled kernels.
        self._batched = vllm_config.compilation_config.mode is CompilationMode.STOCK_TORCH_COMPILE
        # The platform sets float16 for every engine run; the field is typed wider.
        self._mask_dtype = cast(torch.dtype, self.dtype)
        self._position_offset = roberta_position_delta(
            getattr(self.model_config, "hf_config", None)
        )
        self._pad_token_id = encoder_pad_token_id(self.model_config)

        # Warmup only, set through forced_plan(); a forced rectangle wins over ragged.
        self._forced_rect: tuple[int, int] | None = None
        self._force_ragged = False

        self.encoder_plan: EncoderPlan | None = None
        # profile_run skips prepare_attn, so its batch must not inherit the last grid.
        self._planned_batch: InputBatch | None = None
        self.dispatch_stats = EncoderDispatchStats()

    @contextmanager
    def forced_plan(
        self, rect: tuple[int, int] | None = None, ragged: bool = False
    ) -> Generator[None]:
        """Warmup only: plan every step inside as ``rect``, a declared ``(extent, width)``
        rectangle, or as ragged, whatever the dummy batch's own shape."""
        if rect is not None:
            # Warmup must not trace a rectangle serving never takes.
            if self.supports_mm_inputs:
                raise ValueError("multimodal encoders run packed until #1242")
            if rect not in self._rectangles:
                raise ValueError(f"{rect} is not a declared (extent, width): {self._rectangles}")
        previous = self._forced_rect, self._force_ragged
        self._forced_rect, self._force_ragged = rect, ragged
        try:
            yield
        finally:
            self._forced_rect, self._force_ragged = previous

    def _width_caps_for(self, rows: int) -> dict[int, int]:
        """Caps of the body this step belongs to.

        A rectangle's row count is its body, so that key is exact. A ragged step's
        row count is the packed token count, which is not itself a body: 192 tokens
        against a 512-row body still use that body's caps, or same-extent requests
        stop grouping.
        """
        if rows in self._width_caps_by_rows:
            return self._width_caps_by_rows[rows]
        for body in self._bodies:
            if body >= rows:
                return self._width_caps_by_rows[body]
        if self._bodies:
            return self._width_caps_by_rows[self._bodies[-1]]
        return {}

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        # Not in the forward: the plan's D2H/H2D would become graph nodes there.
        attn_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            ubatch_idx,
            model_specific_attn_metadata,
        )
        rows = input_batch.num_tokens_after_padding
        # The expansion is an identity only over a body exactly the forced size.
        # That size is the rectangle's own grid, not the largest tier.
        if self._forced_rect is not None:
            extent, width = self._forced_rect
            body = extent * width
            if rows != body:
                raise RuntimeError(f"forced {self._forced_rect} needs {body} rows, got {rows}")
        # A rectangle is its body. A 2048 or 4096 step on an 8K model still has one;
        # offering the whole ladder would lay a short grid over a longer buffer.
        # Multimodal stays packed until #1242: its body arrives as inputs_embeds,
        # which prepare_inputs never sees.
        rectangles = (
            []
            if self._force_ragged or self.supports_mm_inputs
            else [(length, batch) for length, batch in self._rectangles if length * batch == rows]
        )
        attn_plan = plan_encoder_step(
            attn_metadata.values(),
            rectangles=rectangles,
            width_cap_for=self._width_caps_for(rows),
            device=self.spyre_device,
            dtype=self._mask_dtype,
            batched=self._batched,
            forced_rect=self._forced_rect,
        )
        plan = None
        if attn_plan is not None:
            plan = EncoderPlan(attn_plan=attn_plan)
            self.dispatch_stats.record(attn_plan, rows)
        self._set_plan(plan, input_batch)
        return attn_metadata

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        """Lay the packed body out as the rectangle's grid; offset RoBERTa positions."""
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if self._planned_batch is not input_batch:
            self._set_plan(None, None)
        positions = model_inputs.get("positions")
        if positions is None:
            positions = input_batch.positions

        grid = self.encoder_plan.grid if self.encoder_plan is not None else None
        # A model that masks multimodal rows after embedding (CLIP, models/clip.py)
        # needs its mask in the same row layout as the embeddings, as in V1.
        align_token_mask = getattr(self.model, "spyre_align_token_mask", None)
        if align_token_mask is not None:
            align_token_mask(input_batch.num_tokens if grid is None else sum(grid[2]), grid)
        if grid is None:
            if self._position_offset:
                model_inputs["positions"] = offset_host_positions(positions, self._position_offset)
            return model_inputs

        ids, _, pos = expand_body_to_grid(
            input_batch.input_ids,
            None,
            positions,
            grid,
            pad_token_id=self._pad_token_id,
            position_offset=self._position_offset,
        )
        model_inputs["input_ids"] = ids
        model_inputs["positions"] = pos
        token_types = model_inputs.get("token_type_ids")
        if token_types is not None:
            extent, width, query_lens = grid
            model_inputs["token_type_ids"] = expand_packed_token_types(
                token_types[: sum(query_lens)].cpu(), query_lens, width, extent
            ).to(token_types.device)
        return model_inputs

    def _set_plan(self, plan: EncoderPlan | None, input_batch: InputBatch | None) -> None:
        self.encoder_plan = plan
        self._planned_batch = input_batch
        publish_encoder_grid(plan.attn_plan if plan is not None else None)
