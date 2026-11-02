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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.models import idefics3

from spyre_inference.custom_ops.utils import convert
from spyre_inference.models._vision_encoder import (
    SEGMENT_ITEMS,
    GatherTable,
    SpyreVisionEncoderMixin,
    VisionBuffer,
    offset_major,
    one_hot_segments,
    segment_buffer,
    segment_mask,
    unbind_items,
)
from spyre_inference.multimodal.utils import (
    align_up,
    linear_weight,
    padded_sdpa,
    rope_perm_matrix,
    rope_rotate_matmul,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def tile_position_ids(patch_mask: torch.Tensor, side: int) -> torch.Tensor:
    """``[T, hp, wp]`` patch mask -> ``[T, hp * wp]`` indices into the position embedding.

    Upstream's ``Idefics2VisionEmbeddings.get_position_ids`` for a 3-D mask, kept on the host:
    upstream moves its result to the position embedding's device, which is the card once the
    table is a ``GatherTable``.
    """
    _, hp, wp = patch_mask.shape
    boundaries = torch.arange(1 / side, 1.0, 1 / side)
    nb_h = patch_mask[:, :, 0].sum(dim=1).clamp(min=1)[:, None]
    nb_w = patch_mask[:, 0, :].sum(dim=1).clamp(min=1)[:, None]
    i = torch.arange(hp * wp)[None]
    bucket_h = torch.bucketize((i // wp) / nb_h, boundaries, right=True)
    bucket_w = torch.bucketize((i % wp) / nb_w, boundaries, right=True)
    return torch.where(patch_mask.flatten(1), bucket_h * side + bucket_w, 0)


class SpyreIdefics3ForConditionalGeneration(SpyreVisionEncoderMixin, idefics3.Idefics3ForConditionalGeneration):

    vision_max_items_per_call = SEGMENT_ITEMS

    # --- construction ---

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._init_spyre_vision(prefix)

    def _init_spyre_vision(self, prefix: str) -> None:
        """Set the mixin's attributes. Separate from ``__init__`` so tests can run it on upstream
        modules without a full ``VllmConfig``."""
        vision = self.config.vision_config
        # One image token is a pixel-shuffle window of scale_factor**2 patches.
        self.patches_per_token = self.config.scale_factor**2
        self.out_hidden_size = self.config.text_config.hidden_size
        self.vision_buffers = {
            "patches": VisionBuffer(align_up(vision.num_channels * vision.patch_size**2)),
            "segments": segment_buffer(),
            # Index into the learned position embedding.
            "positions": VisionBuffer(None, pad=0, dtype=torch.int64),
        }

    # --- SpyreVisionEncoderMixin hooks ---

    def vision_item_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]:
        # An item is an image: ``num_patches`` tiles of ``image_seq_len`` tokens each.
        return [n * self.model.image_seq_len for n in self._get_num_patches_list(mm_kwargs)]

    def vision_inputs(self, mm_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Offset-major patches of every tile, each flattened as ``(c, py, px)`` like the conv
        weight, with its one-hot tile and its position-embedding index.

        Upstream attends within a tile, so the tile, not the image, is the segment.
        """
        vision = self.config.vision_config
        p, m = vision.patch_size, self.config.scale_factor
        pixel_values = convert(mm_kwargs["pixel_values"], device="cpu")
        pixel_mask = convert(mm_kwargs["pixel_attention_mask"], device="cpu")
        tiles, c, h, w = pixel_values.shape
        if tiles > SEGMENT_ITEMS:
            raise ValueError(f"{tiles} tiles in one call; at most {SEGMENT_ITEMS} fit the one-hot")
        assert h % (p * m) == 0 and w % (p * m) == 0, f"{h}x{w} is not a multiple of {p * m}"
        hp, wp = h // p, w // p

        grids = pixel_values.reshape(tiles, c, hp, p, wp, p).permute(0, 2, 4, 1, 3, 5)
        grids = grids.reshape(tiles, hp, wp, -1)
        pos = tile_position_ids(
            self.model.get_patch_attention_mask(pixel_mask), vision.image_size // p
        ).reshape(tiles, hp, wp)
        flat_patches = [offset_major(grids[t], m) for t in range(tiles)]
        positions = [offset_major(pos[t], m) for t in range(tiles)]
        segments = [torch.full((m * m, (hp // m) * (wp // m)), t) for t in range(tiles)]

        flat = torch.cat(flat_patches, dim=1).flatten(0, 1).to(torch.float16)
        width = cast(int, self.vision_buffers["patches"].width)
        return {
            "patches": F.pad(flat, (0, width - flat.shape[1])),
            "segments": one_hot_segments(torch.cat(segments, dim=1).flatten()),
            "positions": torch.cat(positions, dim=1).flatten().to(torch.int64),
        }

    def vision_embed(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        embeddings = self.model.vision_model.embeddings
        # [patch width, D] since process_weights_after_loading.
        patch_weight_t = cast(torch.Tensor, embeddings.patch_embedding.weight)
        bias = cast(torch.Tensor, embeddings.patch_embedding.bias)
        position = self.spyre_position_table.gather(values["positions"])
        x = (torch.matmul(values["patches"], patch_weight_t) + bias + position).unsqueeze(0)
        # No pre-norm and no RoPE: the blocks take only the same-tile mask.
        return x, {"attention_mask": segment_mask(values["segments"], x.dtype)}
    
    def vision_blocks(self) -> nn.ModuleList:
        return self.model.vision_model.encoder.layers

    def vision_project(self, x: torch.Tensor, taps: list[torch.Tensor]) -> torch.Tensor:
        x = self.model.vision_model.post_layernorm(x.squeeze(0))
        # Upstream's pixel_shuffle concatenates a window's patches as (hi, wi, D), the offset-major
        # group order, so the projection is one [tokens, D] matmul per group, summed.
        x = x.reshape(self.patches_per_token, -1, x.shape[-1])
        # [scale_factor**2, D, D_out] since process_weights_after_loading.
        proj_weight = cast(torch.Tensor, self.model.connector.modality_projection.proj.weight)
        return torch.matmul(x, proj_weight).sum(dim=0)

