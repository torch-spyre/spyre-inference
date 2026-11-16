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

"""CLIP on Spyre: ``SpyreCLIPEmbeddingModel``, the vision-tower layer that takes the
same-image mask (``SpyreCLIPEncoderLayer``), and the EngineArgs override that must
land before ModelConfig is built.

The vision tower runs on ``SpyreVisionEncoderMixin``, in one of two layouts picked from the
pooling type:

- ``"class"`` (``LAST``, ``CLS``): one image token per image. Each image fills a fixed slot of
  ``align_up(grid**2 + 1, 64)`` patches, class row first and padding at the slot's tail, so
  ``patches_per_token`` is the slot and the token budgets count images.
- ``"full"`` (``MEAN``, ``ALL``): every row is an image token, packed back to back as Pixtral's.
  ``MEAN`` is pooled by the pooler over the image's placeholders, not in the tower.
"""

from __future__ import annotations

from copy import copy
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm.model_executor.models.clip as vllm_clip
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention, MMEncoderAttention
from vllm.model_executor.models.clip import (
    CLIPAttention,
    CLIPEmbeddingModel,
    CLIPEncoderLayer,
    CLIPVisionTransformer,
    _get_vision_feature_select_strategy,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.attention.backend import AttentionBackend, AttentionType, subclass_attention_backend
from vllm.v1.attention.selector import get_attn_backend

from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm, _layer_norm_kernel
from spyre_inference.custom_ops.utils import convert
from spyre_inference.models._vision_encoder import (
    SEGMENT_ITEMS,
    GatherTable,
    SpyreVisionEncoderMixin,
    VisionBuffer,
    one_hot_segments,
    segment_buffer,
    segment_mask,
    unbind_items,
)
from spyre_inference.multimodal.utils import align_up, padded_sdpa
from spyre_inference.v1.pool.spyre_pooler import select_rows
from spyre_inference.v1.worker.spyre_shape_bucketer import encoder_dense_row_indices

if TYPE_CHECKING:
    from transformers import CLIPVisionConfig, PretrainedConfig
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)


def force_disable_chunked_prefill(
    engine_args: EngineArgs, hf_config: PretrainedConfig | None
) -> None:
    """SpyreAllPool (CLIP's token-level pooler) rejects chunked prefill outright,
    but ModelConfig.is_chunked_prefill_supported doesn't exclude ALL-type token
    pooling, so it isn't disabled automatically.
    """
    if engine_args.enable_chunked_prefill is not None:
        return
    if getattr(hf_config, "model_type", None) != "clip":
        return
    engine_args.enable_chunked_prefill = False
    logger.info("CLIP: disabling chunked prefill (unsupported with token-level pooling).")


# Mixed image+text steps: vLLM 0.28 skips the text tower for a step holding any image.
# vllm-project/vllm#53165 (0.29) fixes that, but restores the image rows with the packed
# mask, unsqueezed on the device: wrong on the encoder grid, and a card fault
# (torch-spyre#5230). So SpyreCLIPEmbeddingModel routes them itself on every version.


def _has_mm(multimodal_embeddings) -> bool:
    return multimodal_embeddings is not None and len(multimodal_embeddings) > 0


def has_text_tokens(multimodal_embeddings, is_multimodal: torch.Tensor | None) -> bool:
    """Whether the step still has text rows to run through the text tower."""
    has_mm = _has_mm(multimodal_embeddings)
    if not has_mm or is_multimodal is None:
        return not has_mm
    return bool((~convert(is_multimodal, device="cpu")).any())


def merge_text_and_vision(
    text: torch.Tensor, vision: torch.Tensor, is_multimodal: torch.Tensor | None
) -> torch.Tensor:
    """Text-tower output on text rows, the vision embedding on image rows."""
    if is_multimodal is None:
        return text
    src = convert(is_multimodal, device="cpu")[: text.shape[0]]
    if not bool(src.any()):
        return text
    mask = torch.zeros(text.shape[0], dtype=torch.bool)
    mask[: src.shape[0]] = src
    # Unsqueezed on the host: a device-side [N, 1] view gives the result a layout that
    # index_select faults the card on past row 63 (torch-spyre#5230).
    return torch.where(convert(mask.unsqueeze(-1), device=text.device), vision, text)


def select_class_rows(feats: torch.Tensor) -> torch.Tensor:
    """Each image's class token, ``feats[:, :1, :]``, gathered rather than sliced.

    Past one image the slice is a strided view, and the on-card post-norm reads it as
    contiguous: every image after the first got rows of the first image instead.
    """
    b, s, h = feats.shape
    rows = torch.arange(b, dtype=torch.int64) * s
    return select_rows(feats.reshape(b * s, h), rows).reshape(b, 1, h)


class SpyreCLIPAttention(CLIPAttention):
    """Upstream's vision attention with the stick-aligned SDPA under an additive mask.

    Upstream's ``attn`` takes packed sequence lengths and builds its mask on the host; this
    one takes the same-image mask built once per encoder call in ``vision_embed``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Upstream's host-masked attention: unused by forward, and weightless.
        del self.attn

    def forward(  # ty: ignore[invalid-method-override]
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, patches, _ = x.shape
        qkv, _ = self.qkv_proj(x)
        q, k, v = (
            t.reshape(batch, patches, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
            for t in qkv.chunk(3, dim=-1)
        )
        out = padded_sdpa(q, k, v, mask, scale=self.scale)
        out = out.transpose(1, 2).reshape(batch, patches, -1)
        out, _ = self.out_proj(out)
        return out


class SpyreCLIPEncoderLayer(CLIPEncoderLayer):
    """Upstream's pre-norm block with ``SpyreCLIPAttention``, called as ``(x, mask)``.

    For the vision tower only: the text tower shares ``CLIPEncoderLayer``, so this is swapped
    into ``vision_model.encoder.layers`` rather than patched in globally.
    """

    def __init__(self, config, quant_config=None, *, prefix: str = "", attn_cls) -> None:
        super().__init__(config, quant_config, prefix=prefix, attn_cls=attn_cls)
        self.self_attn = SpyreCLIPAttention(
            config, quant_config, prefix=f"{prefix}.self_attn", attn_cls=attn_cls
        )

    def forward(  # ty: ignore[invalid-method-override]
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        h = x + self.self_attn(self.layer_norm1(x), mask)
        return h + self.mlp(self.layer_norm2(h))


@lru_cache
def _causal_encoder_backend(backend: type[AttentionBackend]) -> type[AttentionBackend]:
    class CausalEncoderOnlyBuilder(backend.get_builder_cls()):
        def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
            common_attn_metadata = copy(common_attn_metadata)
            common_attn_metadata.causal = True
            return super().build(common_prefix_len, common_attn_metadata, *args, **kwargs)

    return subclass_attention_backend(
        "CausalEncoderOnlyAttention_", backend, CausalEncoderOnlyBuilder
    )


class CausalEncoderOnlyAttention(Attention):
    """``EncoderOnlyAttention`` with the opposite builder: no KV cache, and the metadata is
    causal by this layer's say rather than by vLLM's ``causal=True`` default."""

    def __init__(self, num_heads: int, head_size: int, scale: float, **kwargs) -> None:
        backend = get_attn_backend(
            head_size, torch.get_default_dtype(), "auto", attn_type=AttentionType.ENCODER_ONLY
        )
        super().__init__(
            num_heads,
            head_size,
            scale,
            attn_backend=_causal_encoder_backend(backend),
            attn_type=AttentionType.ENCODER_ONLY,
            **kwargs,
        )


def _pixel_values(mm_kwargs: dict[str, Any]) -> list[torch.Tensor]:
    """The ``[C, H, W]`` images in ``mm_kwargs``, whether batched in one tensor or a list."""
    pixels = mm_kwargs.get("pixel_values")
    return [] if pixels is None else unbind_items(pixels)


class SpyreCLIPEmbeddingModel(SpyreVisionEncoderMixin, CLIPEmbeddingModel):
    """CLIP with its text tower on the encoder backend, its vision tower compiled per token
    budget, and correct mixed image and text steps."""

    vision_max_items_per_call = SEGMENT_ITEMS
    # A "class" image is one token, so upstream's smallest budget would allow one per call.
    vision_items_limited_by_tokens = True

    _class_pick: torch.Tensor

    # --- construction ---

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # The text tower takes its layers from the clip module's ``Attention``.
        attention = vllm_clip.Attention
        vllm_clip.Attention = CausalEncoderOnlyAttention  # ty: ignore[invalid-assignment]
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            vllm_clip.Attention = attention
        self._spyre_mm_token_mask: torch.Tensor | None = None
        self._spyre_has_text_tokens = True
        # Built here, so the checkpoint loads into it by name: the boundary norm sits outside
        # every compiled graph, where plain LayerNorm crashes on the card (see SpyreLayerNorm).
        norm = self.text_model.final_layer_norm
        self.text_model.final_layer_norm = SpyreLayerNorm(norm.normalized_shape, eps=norm.eps)
        self._init_spyre_vision(prefix, vllm_config.quant_config)

    def _init_spyre_vision(
        self, prefix: str, quant_config: QuantizationConfig | None = None
    ) -> None:
        """Set the mixin's attributes and swap in the Spyre layers. Separate from ``__init__``
        so tests can run it on upstream modules without a full ``VllmConfig``."""
        vision = cast("CLIPVisionConfig", self.config.vision_config)
        pooling_type = self.pooler_config.seq_pooling_type
        assert pooling_type is not None
        self._class_layout = _get_vision_feature_select_strategy(pooling_type) == "class"
        self._patch_size: int = vision.patch_size
        grid = cast(int, vision.image_size) // self._patch_size
        self._image_rows = grid * grid + 1
        self._slot = align_up(self._image_rows) if self._class_layout else self._image_rows
        self._tokens_per_image = 1 if self._class_layout else self._image_rows
        self.patches_per_token = self._slot if self._class_layout else 1
        self.out_hidden_size = cast(int, self.projection_dim)
        self._patch_width = align_up(vision.num_channels * self._patch_size**2)
        self.vision_buffers = {
            "patches": VisionBuffer(self._patch_width),
            "segments": segment_buffer(),
            "positions": VisionBuffer(None, pad=0, dtype=torch.int64),
        }
        # Weighs a slot's rows down to its class row: a strided slice there is misread on the
        # card (see select_class_rows).
        pick = torch.zeros(1, self._slot, 1)
        pick[0, 0] = 1.0
        self.register_buffer("_class_pick", pick, persistent=False)

        # With an image limit of 0, upstream leaves a placeholder instead of the tower.
        if not isinstance(self.vision_model, CLIPVisionTransformer):
            return
        layers = self.vision_model.encoder.layers
        for i in range(len(layers)):
            layers[i] = SpyreCLIPEncoderLayer(
                vision,
                quant_config,
                prefix=maybe_prefix(prefix, f"vision_model.encoder.layers.{i}"),
                attn_cls=MMEncoderAttention,
            )

    def process_weights_after_loading(self) -> None:
        """Fold the patch conv into a linear and the class embedding into the position table,
        on the host, before the runner moves the model to the card."""
        if not isinstance(self.vision_model, CLIPVisionTransformer):
            return
        emb = self.vision_model.embeddings
        weight = emb.patch_embedding.weight.detach().flatten(1)
        weight_t = F.pad(weight, (0, self._patch_width - weight.shape[1])).t().contiguous()
        # A plain holder, not the conv layer: nothing runs the conv, so the runner must not
        # place a tiled conv copy of the weight on the card as well.
        emb.patch_embedding = nn.Module()
        emb.patch_embedding.weight = nn.Parameter(weight_t, requires_grad=False)
        # Row 0 serves the class row, which has no patch: class embedding plus position 0.
        table = emb.position_embedding.weight.detach().float().clone()
        table[0] += emb.class_embedding.detach().float()
        self.spyre_position_table = GatherTable(table.to(weight.dtype))

    # --- protocol methods upstream CLIP does not provide ---

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        return "image"

    def get_max_frames_per_video(self) -> int:
        return 1

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        images = _pixel_values(mm_kwargs)
        return {**mm_kwargs, "pixel_values": [images[i] for i in indices]}

    def get_encoder_cudagraph_budget_range(self, vllm_config: VllmConfig) -> tuple[int, int]:
        # The default maximum is the largest item, which is one token for "class", so every
        # budget would hold one image; and for "full" the default minimum (64) exceeds an
        # item (50 for ViT-B/32), which upstream rejects. Cover a whole step instead.
        low = self.vision_patch_multiple // self.patches_per_token
        high = align_up(self.max_items_per_step * self._tokens_per_image, low)
        return low, max(high, low)

    # Every image pads as a whole unit at the budget's tail: there is no merge to keep a view.
    @property
    def vision_pad_groups(self) -> int:
        return 1

    # --- SpyreVisionEncoderMixin hooks ---

    def vision_item_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]:
        return [self._tokens_per_image] * len(_pixel_values(mm_kwargs))

    def vision_inputs(self, mm_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Each image's slot: a zero class row, then its patches in raster order, each
        flattened as ``(c, py, px)`` like the conv weight, then the slot's padding rows."""
        p = self._patch_size
        tail = self._slot - self._image_rows
        flat_patches, seg, pos = [], [], []
        for i, img in enumerate(_pixel_values(mm_kwargs)):
            c, h, w = img.shape
            hp, wp = h // p, w // p
            assert hp * wp + 1 == self._image_rows, f"{h}x{w} is not the configured image size"
            patches = convert(img, device="cpu").reshape(c, hp, p, wp, p).permute(1, 3, 0, 2, 4)
            patches = patches.reshape(hp * wp, -1).to(torch.float16)
            row = patches.new_zeros(1, patches.shape[1])
            flat_patches += [row, patches, row.expand(tail, -1)]
            seg += [torch.full((self._image_rows,), i), torch.full((tail,), SEGMENT_ITEMS)]
            pos += [torch.arange(self._image_rows), torch.zeros(tail, dtype=torch.int64)]
        flat = torch.cat(flat_patches)
        return {
            "patches": F.pad(flat, (0, self._patch_width - flat.shape[1])),
            "segments": one_hot_segments(torch.cat(seg)),
            "positions": torch.cat(pos),
        }

    def vision_embed(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        tower = self.vision_model
        # [patch width, D] since process_weights_after_loading.
        patch_weight_t = cast(torch.Tensor, tower.embeddings.patch_embedding.weight)
        x = torch.matmul(values["patches"], patch_weight_t)
        x = x + self.spyre_position_table.gather(values["positions"])
        # The fp32 statistics of SpyreLayerNorm, in-graph: CLIP's boundary norms see
        # activations past fp16's range.
        norm = tower.pre_layrnorm
        x = _layer_norm_kernel(x, norm.weight, norm.bias, norm.eps).unsqueeze(0)
        return x, {"mask": segment_mask(values["segments"], x.dtype)}

    def vision_blocks(self) -> nn.ModuleList:
        return self.vision_model.encoder.layers

    def vision_project(self, x: torch.Tensor, taps: list[torch.Tensor]) -> torch.Tensor:
        x = x.squeeze(0)
        if self._class_layout:
            x = (x.reshape(-1, self._slot, x.shape[-1]) * self._class_pick).sum(dim=1)
        norm = self.vision_model.post_layernorm
        if norm is not None:
            x = _layer_norm_kernel(x, norm.weight, norm.bias, norm.eps)
        return self.visual_projection(x)

    # --- the merge ---

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        """Records the step's image rows for ``forward``, then the mixin's merge: the image
        rows are on the host, CLIP's input embeddings ``projection_dim`` wide."""
        mask = None
        if _has_mm(multimodal_embeddings) and is_multimodal is not None:
            # The one host copy per step: the runner's hook and the merge both read it.
            mask = convert(is_multimodal, device="cpu")
        self._spyre_mm_token_mask = mask
        self._spyre_has_text_tokens = has_text_tokens(multimodal_embeddings, mask)
        return super().embed_input_ids(
            input_ids, multimodal_embeddings, is_multimodal=is_multimodal
        )

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kw):
        # Not super().forward(): see the mixed-step note above.
        if intermediate_tensors is not None:
            raise RuntimeError("PP is not supported for this model")
        assert inputs_embeds is not None
        if not self._spyre_has_text_tokens:
            return inputs_embeds
        text_embeds = inputs_embeds
        if text_embeds.shape[1] > self.text_embed_dim:
            text_embeds = text_embeds[:, : self.text_embed_dim]
        text = self.get_text_features(input_ids, positions, text_embeds)
        return merge_text_and_vision(text, inputs_embeds, self._spyre_mm_token_mask)

    def spyre_align_token_mask(self, num_tokens: int, grid) -> None:
        """Trim the mask to the step's real rows, then lay it out as the grid, if any.

        Padding rows are not text: an image-only step then skips the text tower.
        """
        mask = self._spyre_mm_token_mask
        if mask is None:
            return
        mask = mask[:num_tokens]
        self._spyre_has_text_tokens = bool((~mask).any())
        if grid is not None:
            extent, width, query_lens = grid
            grid_mask = torch.zeros(width * extent, dtype=torch.bool)
            grid_mask[encoder_dense_row_indices(query_lens, extent)] = mask
            mask = grid_mask
        self._spyre_mm_token_mask = mask

    def get_image_features(self, pixel_values, feature_select_strategy=None):
        if feature_select_strategy is None and pixel_values.shape[0] > 1:
            pooling_type = self.pooler_config.seq_pooling_type
            assert pooling_type is not None
            if _get_vision_feature_select_strategy(pooling_type) == "class":
                feature_select_strategy = select_class_rows
        return super().get_image_features(pixel_values, feature_select_strategy)
