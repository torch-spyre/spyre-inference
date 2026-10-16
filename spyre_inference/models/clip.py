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

"""CLIP on Spyre: ``SpyreCLIPEmbeddingModel``, and the EngineArgs override that must
land before ModelConfig is built.

The LayerNorm boundary-norm swap lives in ``multimodal/clip.py`` instead
(an instance-level, post-load patch)."""

from __future__ import annotations

from copy import copy
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import vllm.model_executor.models.clip as vllm_clip
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.clip import (
    CLIPEmbeddingModel,
    _get_vision_feature_select_strategy,
)
from vllm.v1.attention.backend import AttentionBackend, AttentionType, subclass_attention_backend
from vllm.v1.attention.selector import get_attn_backend

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.pool.spyre_pooler import select_rows
from spyre_inference.v1.worker.spyre_shape_bucketer import encoder_dense_row_indices

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)


def force_disable_chunked_prefill(engine_args: EngineArgs) -> None:
    """SpyreAllPool (CLIP's token-level pooler) rejects chunked prefill outright,
    but ModelConfig.is_chunked_prefill_supported doesn't exclude ALL-type token
    pooling, so it isn't disabled automatically. Mirrors
    models.gemma4.force_text_backbone's early get_config() probe.
    """
    if engine_args.enable_chunked_prefill is not None:
        return
    from vllm.transformers_utils.config import get_config

    try:
        hf_config = get_config(
            engine_args.hf_config_path or engine_args.model,
            engine_args.trust_remote_code,
            engine_args.revision,
            engine_args.code_revision,
            engine_args.config_format,
            token=engine_args.hf_token,
        )
    except Exception:
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


class SpyreCLIPEmbeddingModel(CLIPEmbeddingModel):
    """CLIP with its text tower on the encoder backend, and correct mixed and batched
    image steps."""

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

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
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
