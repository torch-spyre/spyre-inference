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
from vllm.model_executor.models.idefics2_vision_model import Idefics2VisionAttention

from spyre_inference.custom_ops.utils import convert
from spyre_inference.models._vision_encoder import (
    GatherTable,
    SpyreVisionEncoderMixin,
    VisionBuffer,
    fold_conv_patch_embedding,
    offset_major,
    split_weight_by_group,
)
from spyre_inference.multimodal.utils import (
    align_up,
    linear_weight,
    padded_sdpa,
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


def tile_offset_major(t: torch.Tensor, tiles: int, merge: int) -> torch.Tensor:
    """``[tiles * hp, wp, ...]`` -> ``[tiles * hp * wp, ...]``, each tile's patches offset-major.

    A tile is ``[merge**2 offsets, tokens, ...]``, flattened, and tiles follow each other. Tiles
    are stacked along the height, so with ``hp % merge == 0`` a merge window never crosses one.
    """
    groups = offset_major(t, merge)
    groups = groups.reshape(merge**2, tiles, -1, *groups.shape[2:])
    return groups.transpose(0, 1).reshape(-1, *t.shape[2:])


class SpyreIdefics2VisionAttention(Idefics2VisionAttention):
    """Upstream's vision attention with the stick-aligned SDPA in place of ``MMEncoderAttention``.

    Tiles are the batch axis, so each attends to its own patches only and takes no mask.
    """

    def forward(  # ty: ignore[invalid-method-override]
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, patches, _ = hidden_states.shape
        assert attention_mask is None, "tiles are batched, so attention takes no mask"
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = (
            t.reshape(batch, patches, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
            for t in qkv.chunk(3, dim=-1)
        )
        out = padded_sdpa(q, k, v, attention_mask, scale=self.scale)
        out = out.transpose(1, 2).reshape(batch, patches, -1)
        out, _ = self.out_proj(out)
        return out


class SpyreIdefics3ForConditionalGeneration(
    SpyreVisionEncoderMixin, idefics3.Idefics3ForConditionalGeneration
):
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
        self._tile_patches = (vision.image_size // vision.patch_size) ** 2
        self.vision_buffers = {
            "patches": VisionBuffer(align_up(vision.num_channels * vision.patch_size**2)),
            # Index into the learned position embedding.
            "positions": VisionBuffer(None, pad=0, dtype=torch.int64),
        }

        # Switch each attention to the Spyre forward in place: the subclass adds no state, so the
        # stock linears and their weight-loader names stay as built.
        for layer in self.model.vision_model.encoder.layers:
            layer.self_attn.__class__ = SpyreIdefics2VisionAttention

    def process_weights_after_loading(self) -> None:
        """Reshape the vision weights for the graphs, in place, as the Spyre linears do, so one
        copy of each reaches the card. Runs on the host, before the runner moves the model."""
        embeddings = self.model.vision_model.embeddings

        # Patch conv -> [patch width, D], as a linear over (c, py, px)-flattened patches.
        width = cast(int, self.vision_buffers["patches"].width)
        embeddings.patch_embedding = fold_conv_patch_embedding(embeddings.patch_embedding, width)

        # Position embedding -> a row-gathered table.
        table = convert(embeddings.position_embedding.weight.detach(), device="cpu")
        self.spyre_position_table = GatherTable(table.to(torch.float16).contiguous())
        # Nothing runs the embedding now, so the runner must not place a second copy on the card.
        del embeddings.position_embedding

        # Connector projection -> [scale_factor**2, D, D_out], one slice per offset group.
        proj = self.model.connector.modality_projection.proj
        proj.weight = nn.Parameter(
            # ``pixel_shuffle`` concatenates a window's patches as (hi, wi, D): group-major.
            split_weight_by_group(linear_weight(proj), self.patches_per_token, group_major=True),
            requires_grad=False,
        )

    # Whole tiles pad at the end of the budget: tiles attend within themselves.
    @property
    def vision_pad_groups(self) -> int:
        return 1

    @property
    def vision_patch_multiple(self) -> int:
        return self._tile_patches

    # --- SpyreVisionEncoderMixin hooks ---

    def vision_item_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]:
        # An item is an image: ``num_patches`` tiles of ``image_seq_len`` tokens each.
        return [n * self.model.image_seq_len for n in self._get_num_patches_list(mm_kwargs)]

    def vision_inputs(self, mm_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Every tile's patches, each flattened as ``(c, py, px)`` like the conv weight, with
        its position-embedding index. A tile is ``[scale_factor**2 offsets, tokens]``, flattened,
        so ``vision_project``'s merge is a view and the tiles are the attention's batch axis."""
        vision = self.config.vision_config
        p, m = vision.patch_size, self.config.scale_factor
        pixel_values = convert(mm_kwargs["pixel_values"], device="cpu")
        pixel_mask = convert(mm_kwargs["pixel_attention_mask"], device="cpu")
        tiles, c, h, w = pixel_values.shape
        if h % (p * m) or w % (p * m):
            raise ValueError(f"{h}x{w} is not a multiple of {p * m}")
        hp, wp = h // p, w // p

        grids = pixel_values.reshape(tiles, c, hp, p, wp, p).permute(0, 2, 4, 1, 3, 5)
        pos = tile_position_ids(
            self.model.get_patch_attention_mask(pixel_mask), vision.image_size // p
        )
        # Attention has no RoPE and the positions are gathered by index, so a tile's patch order
        # is free: offset-major here, on the host, leaves the graph no regrouping to do.
        patches = tile_offset_major(grids.reshape(tiles * hp, wp, -1), tiles, m)
        positions = tile_offset_major(pos.reshape(tiles * hp, wp), tiles, m)

        flat = patches.to(torch.float16)
        width = cast(int, self.vision_buffers["patches"].width)
        return {
            "patches": F.pad(flat, (0, width - flat.shape[1])),
            "positions": positions.to(torch.int64),
        }

    def vision_embed(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        embeddings = self.model.vision_model.embeddings
        # [patch width, D] since process_weights_after_loading.
        patch_weight_t = cast(torch.Tensor, embeddings.patch_embedding.weight)
        bias = cast(torch.Tensor, embeddings.patch_embedding.bias)
        position = self.spyre_position_table.gather(values["positions"])
        x = torch.matmul(values["patches"], patch_weight_t) + bias + position
        # Tiles are the batch axis, so no pre-norm, no RoPE and no mask.
        return x.view(-1, self._tile_patches, x.shape[-1]), {}

    def vision_blocks(self) -> nn.ModuleList:
        return self.model.vision_model.encoder.layers

    def vision_project(self, x: torch.Tensor, taps: list[torch.Tensor]) -> torch.Tensor:
        x = self.model.vision_model.post_layernorm(x)
        # Upstream's pixel_shuffle concatenates a window's patches as (hi, wi, D), the offset-major
        # group order, so the projection is one [tokens, D] matmul per group, summed. Each tile
        # is [groups, tokens, D]: swap the tile and group axes to put the groups first.
        groups = self.patches_per_token
        tokens = self._tile_patches // groups
        width = x.shape[-1]
        x = x.reshape(-1, groups, tokens, width).transpose(0, 1).reshape(groups, -1, width)
        # [scale_factor**2, D, D_out] since process_weights_after_loading.
        proj_weight = cast(torch.Tensor, self.model.connector.modality_projection.proj.weight)
        return torch.matmul(x, proj_weight).sum(dim=0)
