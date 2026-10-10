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

"""CLIP boundary-LayerNorm workaround for Spyre.

Only ``vision_model.pre_layrnorm``/``post_layernorm`` and
``text_model.final_layer_norm`` are swapped to ``SpyreLayerNorm``. Those three
sit at the model boundary, outside any per-block compiled graph, which is
what triggers the crash ``SpyreLayerNorm`` works around (see
``spyre_inference.custom_ops.layer_norm``). ``CLIPEncoderLayer.layer_norm1``/
``layer_norm2`` are traced inside the per-block ``torch.compile`` region
already and never hit that crashing path, so they're left as plain
``nn.LayerNorm`` -- swapping them too would be unnecessary.

Applied to the already-loaded model instance (weights included), so the
replacement ``SpyreLayerNorm`` here copies the original's already-loaded
weight/bias explicitly, rather than relying on a later ``load_weights()``
pass to populate them.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.pool.spyre_pooler import select_rows
from spyre_inference.v1.worker.spyre_shape_bucketer import expand_packed_embeds_to_encoder_grid

logger = init_logger(__name__)

# Mixed image+text steps: vLLM 0.28's CLIPEmbeddingModel treats a step holding any
# image as image-only, so its text requests skip the text tower. Backport of
# vllm-project/vllm#53165 (in vLLM >= 0.29). Remove this, and the spyre_align_token_mask
# call in TorchSpyreModelRunner._preprocess, once the vLLM pin reaches 0.29.


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


def patch_mixed_batches() -> None:
    from vllm.model_executor.models.clip import CLIPEmbeddingModel

    if getattr(CLIPEmbeddingModel.forward, "_spyre_patched", False):
        return
    orig_embed_input_ids = CLIPEmbeddingModel.embed_input_ids
    orig_forward = CLIPEmbeddingModel.forward

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        out = orig_embed_input_ids(
            self, input_ids, multimodal_embeddings, is_multimodal=is_multimodal
        )
        self._spyre_mm_token_mask = is_multimodal if _has_mm(multimodal_embeddings) else None
        self._is_text_input = has_text_tokens(multimodal_embeddings, is_multimodal)
        return out

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kw):
        out = orig_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kw)
        mask = getattr(self, "_spyre_mm_token_mask", None)
        if self._is_text_input and mask is not None and inputs_embeds is not None:
            out = merge_text_and_vision(out, inputs_embeds, mask)
        return out

    def spyre_align_token_mask(self, num_tokens: int, grid) -> None:
        """Trim the mask to the step's real rows, then lay it out as the grid, if any.

        Padding rows are not text: an image-only step then skips the text tower.
        """
        mask = getattr(self, "_spyre_mm_token_mask", None)
        if mask is None:
            return
        mask = convert(mask, device="cpu")[:num_tokens]
        self._is_text_input = bool((~mask).any())
        if grid is not None:
            extent, width, query_lens = grid
            grid_mask = expand_packed_embeds_to_encoder_grid(
                mask.unsqueeze(-1), query_lens, width, extent
            )
            mask = grid_mask.squeeze(-1)
        self._spyre_mm_token_mask = mask

    forward._spyre_patched = True
    CLIPEmbeddingModel.spyre_align_token_mask = spyre_align_token_mask
    CLIPEmbeddingModel.embed_input_ids = embed_input_ids  # ty: ignore[invalid-assignment]
    CLIPEmbeddingModel.forward = forward  # ty: ignore[invalid-assignment]


def select_class_rows(feats: torch.Tensor) -> torch.Tensor:
    """Each image's class token, ``feats[:, :1, :]``, gathered rather than sliced.

    Past one image the slice is a strided view, and the on-card post-norm reads it as
    contiguous: every image after the first got rows of the first image instead.
    """
    b, s, h = feats.shape
    rows = torch.arange(b, dtype=torch.int64) * s
    return select_rows(feats.reshape(b * s, h), rows).reshape(b, 1, h)


def patch_class_token_select() -> None:
    from vllm.model_executor.models.clip import (
        CLIPEmbeddingModel,
        _get_vision_feature_select_strategy,
    )

    if getattr(CLIPEmbeddingModel.get_image_features, "_spyre_patched", False):
        return
    orig = CLIPEmbeddingModel.get_image_features

    def get_image_features(self, pixel_values, feature_select_strategy=None):
        pooling_type = self.pooler_config.seq_pooling_type
        if (
            feature_select_strategy is None
            and pixel_values.shape[0] > 1
            and _get_vision_feature_select_strategy(pooling_type) == "class"
        ):
            feature_select_strategy = select_class_rows
        return orig(self, pixel_values, feature_select_strategy)

    get_image_features._spyre_patched = True
    CLIPEmbeddingModel.get_image_features = get_image_features  # ty: ignore[invalid-assignment]


def _to_spyre_layer_norm(ln: torch.nn.LayerNorm, device: torch.device) -> torch.nn.LayerNorm:
    new_ln = SpyreLayerNorm(
        list(ln.normalized_shape),
        eps=ln.eps,
        elementwise_affine=ln.elementwise_affine,
        bias=ln.bias is not None,
    ).to(device=device, dtype=ln.weight.dtype if ln.elementwise_affine else torch.float16)
    if ln.elementwise_affine:
        with torch.no_grad():
            new_ln.weight.copy_(ln.weight)
            if ln.bias is not None:
                new_ln.bias.copy_(ln.bias)
    return new_ln


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Swap CLIP's three boundary LayerNorms for ``SpyreLayerNorm``, in place.

    The ``isinstance`` checks are a second line of defense on top of the
    ``model_type == "clip"`` dispatch gate in ``multimodal/__init__.py``: they
    keep this a no-op (rather than an ``AttributeError`` on ``normalized_shape``)
    for any boundary norm that isn't a plain ``nn.LayerNorm``.
    """
    patch_mixed_batches()
    patch_class_token_select()
    text_model = getattr(model, "text_model", None)
    if text_model is not None:
        ln = getattr(text_model, "final_layer_norm", None)
        if isinstance(ln, torch.nn.LayerNorm):
            text_model.final_layer_norm = _to_spyre_layer_norm(ln, device)

    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None:
        pre_ln = getattr(vision_model, "pre_layrnorm", None)
        if isinstance(pre_ln, torch.nn.LayerNorm):
            vision_model.pre_layrnorm = _to_spyre_layer_norm(pre_ln, device)
        post_ln = getattr(vision_model, "post_layernorm", None)
        if isinstance(post_ln, torch.nn.LayerNorm):
            vision_model.post_layernorm = _to_spyre_layer_norm(post_ln, device)

    logger.info_once(
        "Spyre: CLIP's boundary LayerNorms (pre_layrnorm/post_layernorm/"
        "final_layer_norm) use SpyreLayerNorm; layer_norm1/layer_norm2 inside "
        "encoder blocks are unaffected."
    )
