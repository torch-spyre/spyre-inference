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

"""Host-side encoder step helpers: the rectangular path's dense grid, and dispatch stats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.pool.spyre_pooler import select_rows
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    encoder_grid_source_rows,
    expand_packed_to_encoder_grid,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig

    # Not imported at runtime: the V1 runner imports spyre_encoder_attn lazily.
    from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
        EncoderGroupPlan,
        EncoderRectPlan,
    )

logger = init_logger(__name__)


def encoder_pad_token_id(model_config: ModelConfig) -> int:
    """Pad id for batch-pad filler tokens; their outputs are masked and dropped."""
    pad = getattr(getattr(model_config, "hf_config", None), "pad_token_id", None)
    return int(pad) if isinstance(pad, int) else 0


def expand_body_to_grid(
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    positions: torch.Tensor,
    grid: tuple[int, int, list[int]],
    *,
    pad_token_id: int,
    position_offset: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """``(input_ids, inputs_embeds, positions)`` for the ``(extent, width, query_lens)`` grid.

    The `input_ids` path packs only integer tensors (tens of KB), avoiding a per-layer
    activation gather. Multimodal pooling with only `inputs_embeds` instead gathers the
    embeddings into the grid on the device, with an index built on the host, so they
    never round-trip through the host. Each result is on its input's device.
    """
    extent, width, query_lens = grid
    num_tokens = sum(query_lens)

    if input_ids is not None:
        ids, pos = expand_packed_to_encoder_grid(
            input_ids[:num_tokens].cpu(),
            positions[:num_tokens].cpu(),
            query_lens,
            width,
            extent,
            pad_token_id=pad_token_id,
            position_offset=position_offset,
        )
        assert ids.shape[0] == width * extent, (ids.shape[0], width * extent)
        return convert(ids, input_ids.device), inputs_embeds, convert(pos, positions.device)

    # A multimodal pooling model (e.g. CLIP) preprocesses straight to embeddings; expand
    # those instead of ids. Positions still need the same padding, so reuse that helper
    # with dummy ids (discarded).
    assert inputs_embeds is not None, "upstream must supply ids or embeds"
    _, pos = expand_packed_to_encoder_grid(
        torch.zeros(num_tokens, dtype=torch.int64),
        positions[:num_tokens].cpu(),
        query_lens,
        width,
        extent,
        position_offset=position_offset,
    )
    # Gathered from the whole body buffer, not a `[:num_tokens]` slice: the gather
    # specialises on its source's row count, which then stays fixed.
    grid_embeds = select_rows(
        inputs_embeds,
        torch.tensor(encoder_grid_source_rows(query_lens, width, extent), dtype=torch.int64),
    )
    assert grid_embeds.shape[0] == width * extent, (grid_embeds.shape[0], width * extent)
    return None, grid_embeds, convert(pos, positions.device)


# Steps between encoder dispatch-ratio log lines.
_ENCODER_DISPATCH_LOG_EVERY = 200


@dataclass
class EncoderDispatchStats:
    """Path counts and body occupancy of the encoder steps, logged periodically.

    The counters live in the worker process, so a log line is the only way they reach
    whoever is running a benchmark. Occupancy is the number worth watching: the body is
    one fixed shape, so a step that is narrow or short pays for rows it does not use,
    and that is the cost the rectangular path trades away per-layer data movement for.
    """

    rect_steps: int = 0
    ragged_steps: int = 0
    real_tokens: int = 0
    body_rows: int = 0
    seqs: int = 0

    def record(self, plan: EncoderRectPlan | list[EncoderGroupPlan] | None, rows: int) -> None:
        if plan is None:
            return
        if isinstance(plan, list):
            self.ragged_steps += 1
            query_lens = [length for group in plan for length in group.query_lens]
            num_seqs = sum(group.group for group in plan)
        else:
            self.rect_steps += 1
            query_lens = plan.query_lens
            num_seqs = len(plan.query_lens)
        self.real_tokens += sum(query_lens)
        self.body_rows += rows
        self.seqs += num_seqs
        steps = self.rect_steps + self.ragged_steps
        if steps % _ENCODER_DISPATCH_LOG_EVERY:
            return
        logger.info(
            "Encoder dispatch over %d steps: %d fast, %d slow; mean %.2f seqs/step; "
            "body occupancy %.1f%% (%d real tokens in %d rows).",
            steps,
            self.rect_steps,
            self.ragged_steps,
            self.seqs / steps,
            100.0 * self.real_tokens / max(1, self.body_rows),
            self.real_tokens,
            self.body_rows,
        )
