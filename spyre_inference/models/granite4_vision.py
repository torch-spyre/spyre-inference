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

"""Granite 4 Vision with a compiled, bucketed vision encoder.

``SpyreGranite4VisionForConditionalGeneration`` keeps upstream's processor, weight loading and
decoder, and adds the ``SpyreVisionEncoderMixin`` hooks. The unit of work is a 384x384 tile of
24x24 = 576 patches; an image is its base tile plus its anyres tiles. Per tile:

- SigLIP runs over 576 patches, attending within the tile only, so tiles are the SDPA batch and
  a padded tile (a budget's tail) cannot reach a real one.
- Eight Q-Former projectors (four deepstack layers, four spatial offsets) each turn 576 patches
  into 9 windows of 8x8 patches, then 16 queries per window: 144 image tokens per tile, so
  ``patches_per_token`` is 4.

Patches are laid out **window-major** on the host: ``(window, offset, query)`` instead of raster.
The encoder is permutation-equivariant within a tile once the position embedding and the
projectors' ``image_positions`` follow, and in that layout

- the window split is a free view to ``[tiles * 9, 64, D]``,
- upstream's ``InterpolateDownsampler`` (a 2x2 area mean) is a mean over the 4 offset groups,
- ``SpatialOffsetDownsampler`` offset ``k`` is the ``k``-th offset group,

so none of upstream's 6-D permutes reach the card. The eight levels leave ``vision_project``
packed on the last dimension, as upstream's ``embed_multimodal`` packs them. The window order is
undone, and upstream's unpad-and-newline packing applied, on the host in
``postprocess_encoder_output``, where the output already is.

The deepstack features reach the decoder through upstream's persistent ``_ds_buffers``, which
``embed_input_ids`` fills with a card-to-card copy from a per-level upload.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches
from vllm.model_executor.models import granite4_vision
from vllm.model_executor.models import utils as vllm_utils
from vllm.model_executor.models.utils import StageMissingLayer

from spyre_inference.custom_ops.utils import convert
from spyre_inference.models._vision_encoder import SpyreVisionEncoderMixin, VisionBuffer
from spyre_inference.multimodal import blip2, siglip
from spyre_inference.multimodal.utils import STICK

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_Upstream = granite4_vision.Granite4VisionForConditionalGeneration


def window_major_order(side: int, window: int, query: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices that reorder a raster ``side x side`` patch grid, and one window, window-major.

    A window is ``window x window`` patches that the Q-Former reads together and reduces to
    ``query x query`` queries; ``merge = window // query`` patches of it share each query per
    axis. Returns ``(tile, local)``: ``raster[tile]`` lists the tile's windows in row-major
    order, each as ``(offset_h, offset_w, query_row, query_col)``, and ``local`` is the same
    reorder of a single window (for its ``image_positions``).
    """
    merge, n = window // query, side // window
    grid = torch.arange(side * side).view(n, query, merge, n, query, merge)
    tile = grid.permute(0, 3, 2, 5, 1, 4).reshape(-1)
    local = (
        torch.arange(window * window).view(query, merge, query, merge).permute(1, 3, 0, 2)
    ).reshape(-1)
    return tile, local


def unwindow(x: torch.Tensor, n: int, query: int) -> torch.Tensor:
    """``[tiles * n*n * query*query, C]`` window-major query tokens -> ``[tiles, (n*query)**2, C]``
    in raster order, as upstream's ``WindowQFormerDownsampler._unwin`` leaves them."""
    tiles = x.shape[0] // (n * n * query * query)
    x = x.view(tiles, n, n, query, query, -1).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(tiles, (n * query) ** 2, -1)


def _qformer(qformer: Any, query: torch.Tensor, encoder: torch.Tensor) -> torch.Tensor:
    """Upstream's one-layer ``Blip2QFormerModel`` with the query length equal to the sequence:
    self-attention, cross-attention to ``encoder``, then the query feed-forward. Called layer by
    layer so the graph holds no ``apply_chunking_to_forward``."""
    layer = qformer.encoder.layer[0]
    h = qformer.layernorm(query)
    h = layer.attention(h)
    h = layer.crossattention(h, encoder_hidden_states=encoder)
    return layer.output_query(layer.intermediate_query(h), h)


class SpyreGranite4VisionForConditionalGeneration(
    SpyreVisionEncoderMixin, granite4_vision.Granite4VisionForConditionalGeneration
):
    """Upstream Granite 4 Vision with its vision encoder compiled per token budget."""

    # --- construction ---

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._init_spyre_vision()
        # The Q-Former's attention keeps the global patch it had before this class existed.
        blip2.patch_blip2_qformer_attention()

    def _init_spyre_vision(self) -> None:
        """Set the mixin's attributes. Separate from ``__init__`` so tests can run it without a
        full ``VllmConfig``."""
        cfg = self.config
        vision = cfg.vision_config
        query, window = (int(v) for v in cfg.downsample_rate.split("/"))
        side = vision.image_size // vision.patch_size
        merge = window // query
        self._patch_width = vision.num_channels * vision.patch_size**2
        if (
            window % query
            or merge != 2  # the spatial projectors take one of 2x2 offsets
            or side % window
            or (side * side) % STICK
            or self._patch_width % STICK
            or cfg.vision_feature_select_strategy != "full"
        ):
            raise NotImplementedError(
                f"Granite Vision encoder layout needs a 2x2 merge, stick-aligned tiles and "
                f"strategy 'full'; got downsample_rate={cfg.downsample_rate!r}, "
                f"{side}x{side} patches, strategy {cfg.vision_feature_select_strategy!r}"
            )
        self._side, self._window, self._query = side, window, query
        self._windows = (side // window) ** 2
        self._tile_patches = side * side
        self.patches_per_token = merge * merge
        self._tokens_per_tile = self._tile_patches // self.patches_per_token
        self._levels = len(cfg.deepstack_layer_map) + len(cfg.spatial_target_layers)
        self.out_hidden_size = cfg.text_config.hidden_size * self._levels
        self.vision_buffers = {"patches": VisionBuffer(self._patch_width)}
        self._tile_order, self._window_order = window_major_order(side, window, query)
        max_tiles = max(
            (h // vision.image_size) * (w // vision.image_size) + 1
            for h, w in cfg.image_grid_pinpoints
        )
        # Rounded up to a power of two: on hardware, budgets of 5 and 11 tiles crash the Spyre
        # compiler in the first block (dcc Selector.cpp:272), while 1, 2, 4 and 8 compile. The
        # extra tiles are padding that postprocess drops.
        self._max_tiles = 1 << (max_tiles - 1).bit_length()

    def process_weights_after_loading(self) -> None:
        """Reshape the tower and projector weights for the vision graphs, in place, as the Spyre
        linears do. Runs on the host, after the per-layer finish and before the runner moves the
        model to the card."""
        if isinstance(self.vision_tower, StageMissingLayer):
            return  # an image limit of 0: upstream leaves a placeholder instead of the tower
        tower = self.vision_tower.vision_model
        # Pads each head's q/k/v/out from 72 to 128 and sets MMEncoderAttention's head size, so
        # the blocks run through SpyreMMEncoderAttention.
        siglip.patch_siglip_attention(self.vision_tower)

        emb = tower.embeddings
        conv = emb.patch_embedding
        # The patch conv as a linear on window-major patches. A plain holder, not the conv
        # layer: nothing runs the conv, so no tiled conv copy of the weight goes to the card.
        holder = nn.Module()
        holder.weight = nn.Parameter(conv.weight.detach().flatten(1).t().contiguous())
        holder.bias = nn.Parameter(conv.bias.detach().clone())
        emb.patch_embedding = holder
        position = emb.position_embedding
        position.weight = nn.Parameter(position.weight.detach()[self._tile_order])

        for proj in (*self.layerwise_projectors, *self.spatial_projectors):
            proj.image_positions.data = proj.image_positions.data[:, self._window_order]
        # Plain tensor, so it stays on the host: the packing in postprocess runs there.
        self._image_newline_host = convert(self.image_newline.detach(), device="cpu").clone()

    def _apply(self, fn, recurse=True):
        """Also move ``_ds_buffers``: plain tensors, so ``model.to`` skips them, but forward
        reads them on every step, text-only ones included."""
        super()._apply(fn, recurse)
        dtype = next(self.parameters()).dtype
        self._ds_buffers = [fn(b.to(dtype)) for b in self._ds_buffers]
        return self

    # --- protocol methods upstream Granite 4 Vision does not provide ---

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        return "image"

    def get_max_frames_per_video(self) -> int:
        return 1

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        tiles = self._tiles(mm_kwargs)
        return {
            **mm_kwargs,
            "pixel_values": [tiles[i] for i in indices],
            "image_sizes": self._image_sizes(mm_kwargs)[indices],
        }

    def get_encoder_cudagraph_budget_range(self, vllm_config: VllmConfig) -> tuple[int, int]:
        # The processor's budget counts placeholders (unpadded, newlines and base); the
        # encoder's largest item is the most tiles an anyres grid takes.
        return self._tokens_per_tile, self._max_tiles * self._tokens_per_tile

    # Whole tiles pad at the end of the budget: tiles attend within themselves.
    @property
    def vision_pad_groups(self) -> int:
        return 1

    @property
    def vision_patch_multiple(self) -> int:
        return self._tile_patches

    @property
    def vision_taps(self) -> tuple[int, ...]:  # ty: ignore[invalid-method-override]
        """The deepstack layers, then the one layer the spatial projectors read."""
        cfg = self.config
        return (*(vision for vision, _ in cfg.deepstack_layer_map), cfg.spatial_vision_layer)

    # --- SpyreVisionEncoderMixin hooks ---

    def vision_item_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]:
        return [tiles * self._tokens_per_tile for tiles in self._num_tiles(mm_kwargs)]

    def vision_inputs(self, mm_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Every tile's patches in window-major order, each flattened as ``(c, py, px)`` like
        the conv weight."""
        tiles = [convert(t, device="cpu") for t in self._tiles(mm_kwargs)]
        pixels = torch.cat(tiles)
        count, channels = pixels.shape[:2]
        side, patch = self._side, self.config.vision_config.patch_size
        x = pixels.reshape(count, channels, side, patch, side, patch)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(count, side * side, -1)
        x = x[:, self._tile_order].reshape(count * side * side, -1)
        return {"patches": x.to(torch.float16)}

    def vision_embed(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        emb = self.vision_tower.vision_model.embeddings
        # [patch width, D] and [D] since process_weights_after_loading.
        x = torch.matmul(values["patches"], emb.patch_embedding.weight) + emb.patch_embedding.bias
        x = x.view(-1, self._tile_patches, x.shape[-1]) + emb.position_embedding.weight
        return x, {}

    def vision_blocks(self) -> nn.ModuleList:
        return self.vision_tower.vision_model.encoder.layers

    def vision_call_block(
        self, block: nn.Module, x: torch.Tensor, block_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        return block(x)[0]

    def vision_project(self, x: torch.Tensor, taps: list[torch.Tensor]) -> torch.Tensor:
        """Each level's projector over its tap, packed on the last dimension: the deepstack
        levels, then the spatial offsets, in the order of upstream's ``_ds_layer_indices``."""
        layerwise = len(self.layerwise_projectors)
        levels = [
            self._project(proj, taps[i], None) for i, proj in enumerate(self.layerwise_projectors)
        ]
        levels += [
            self._project(proj, taps[layerwise], offset)
            for offset, proj in enumerate(self.spatial_projectors)
        ]
        return torch.cat(levels, dim=-1)

    def _project(self, proj: nn.Module, tap: torch.Tensor, offset: int | None) -> torch.Tensor:
        """``WindowQFormerDownsampler.forward`` on window-major patches, to ``[tokens, H]``.

        The downsample is the mean over the offset groups (upstream's area interpolation) or
        one group (its spatial offset), both free views of the window.
        """
        merge2, queries = self.patches_per_token, self._query**2
        enc = proj.norm(tap).reshape(-1, self._window**2, tap.shape[-1])
        groups = enc.reshape(enc.shape[0], merge2, queries, -1)
        down = groups.mean(dim=1) if offset is None else groups[:, offset]
        out = _qformer(proj.qformer, proj.query + down, enc + proj.image_positions)
        return proj.out_linear(out).reshape(-1, proj.out_linear.out_features)

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Raster order, then upstream's unpad-and-newline packing, level by level.

        ``outputs["default"]`` is on the host and ends in the budget's padding tiles, which the
        split by ``per_item_out_tokens`` drops. The placeholders are the packed rows, not the
        encoder tokens: ``dest`` gets ``unpadded + newlines + base`` rows per image.
        """
        assert batch_mm_kwargs is not None
        sizes = self._image_sizes(batch_mm_kwargs)
        counts = [per_item_out_tokens[i] for i in indices]
        chunks = outputs["default"][: sum(counts)].split(counts)
        hidden = self.config.text_config.hidden_size
        n = self._side // self._window
        per_level: list[list[torch.Tensor]] = [[] for _ in range(self._levels)]
        for chunk in chunks:
            tiles = unwindow(chunk, n, self._query)  # [tiles, tokens per tile, levels * H]
            for level, feats in enumerate(tiles.split(hidden, dim=-1)):
                per_level[level].append(feats.contiguous())
        # The only state upstream's packing reads, with the newline on the host.
        packer = SimpleNamespace(
            config=self.config,
            _downsample_rate=Fraction(self.config.downsample_rate),
            image_newline=self._image_newline_host,
        )
        packed = [_Upstream._pack_and_unpad_image_features(packer, f, sizes) for f in per_level]
        for position, idx in enumerate(indices):
            dest[idx] = torch.cat([level[position] for level in packed], dim=-1)

    # --- the merge ---

    def vision_merge(  # ty: ignore[invalid-method-override]
        self, text: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Graph 4: zero the image rows of the text embedding, as upstream's ``masked_fill``
        does. The image rows themselves go to the decoder as deepstack features, not here."""
        return torch.where(mask, torch.zeros_like(text), text)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = True,
    ) -> torch.Tensor:
        """Upstream's text embedding (times ``embedding_multiplier``), the image rows zeroed
        through ``vision_merge``, and each deepstack level handed to the decoder's buffers.

        The rows are ``[tokens, levels * H]``. Each level is placed on the host in a
        ``[tokens, H]`` staging tensor and copied into the ``[:tokens]`` prefix of its
        persistent buffer. Rows past ``tokens`` are stale by design: ``forward`` reads only
        ``[:tokens]`` (``_ds_num_tokens``), as upstream's own write does, so the per-call copy
        scales with the prompt rather than the buffer capacity.
        """
        text = _Upstream.embed_input_ids(self, input_ids, (), is_multimodal=is_multimodal)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return text
        assert is_multimodal is not None, "image tokens need is_multimodal"
        mask = convert(is_multimodal, device="cpu")
        rows = convert(vllm_utils._flatten_embeddings(multimodal_embeddings), device="cpu")
        if rows.shape[0] != int(mask.sum()):
            expr = vllm_utils._embedding_count_expression(multimodal_embeddings)
            raise ValueError(
                f"Attempted to assign {expr} = {rows.shape[0]} multimodal tokens to "
                f"{int(mask.sum())} placeholders"
            )
        tokens, hidden = text.shape
        for level, feats in enumerate(rows.to(text.dtype).split(hidden, dim=-1)):
            staged = torch.zeros((tokens, hidden), dtype=text.dtype)
            staged[mask] = feats
            self._ds_buffers[level][:tokens].copy_(convert(staged, device=text.device))
        self._ds_num_tokens = tokens
        merge = getattr(self, "_spyre_vision_merge", self.vision_merge)
        return merge(text, convert(mask[:, None], device=text.device))

    # --- private helpers ---

    @staticmethod
    def _image_sizes(mm_kwargs: dict[str, Any]) -> torch.Tensor:
        """``[images, 2]`` (height, width), which the processor keeps on the host."""
        sizes = mm_kwargs["image_sizes"]
        return torch.stack(list(sizes)) if isinstance(sizes, list) else sizes

    def _num_tiles(self, mm_kwargs: dict[str, Any]) -> list[int]:
        """Tiles of each image, base included."""
        if mm_kwargs.get("image_sizes") is None:
            return []
        return [
            image_size_to_num_patches(
                image_size=size,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for size in self._image_sizes(mm_kwargs)
        ]

    def _tiles(self, mm_kwargs: dict[str, Any]) -> list[torch.Tensor]:
        """Each image's ``[tiles, C, H, W]``, with the processor's tile padding dropped.

        ``pixel_values`` is a padded ``[images, max tiles, ...]`` tensor, a list of per-image
        tiles, or (as upstream's ``embed_multimodal`` also accepts) every image's tiles
        concatenated along dim 0.
        """
        pixels = mm_kwargs.get("pixel_values")
        if pixels is None:
            return []
        counts = self._num_tiles(mm_kwargs)
        if isinstance(pixels, torch.Tensor):
            items = list(pixels.unbind(0)) if pixels.dim() == 5 else list(pixels.split(counts))
        else:
            items = list(pixels)
        return [item[:count] for item, count in zip(items, counts, strict=True)]
