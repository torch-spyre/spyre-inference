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
"""Spyre adaptation of vLLM's Transformers backend for HF encoder models.

Kept apart from ``transformers_backend`` so the encoder classes are imported only when an
encoder is served: a vLLM without them must not break the decoder path.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.models.transformers import (
    TransformersEmbeddingModel,
    TransformersForSequenceClassification,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _patch_encoder_gather(model: nn.Module) -> None:
    """Replace gather-based token_type_ids in HF encoder embeddings with explicit zeros."""
    patched = 0
    for module in model.modules():
        buf = getattr(module, "token_type_ids", None)
        if isinstance(buf, torch.Tensor) and hasattr(module, "word_embeddings"):
            original_forward = module.forward
            sig = inspect.signature(original_forward)

            def _patched_forward(
                *args,
                _orig=original_forward,
                _sig=sig,
                **kwargs,
            ) -> torch.Tensor:
                try:
                    bound = _sig.bind_partial(*args, **kwargs)
                except TypeError:
                    return _orig(*args, **kwargs)

                if bound.arguments.get("token_type_ids") is None:
                    ref = bound.arguments.get("input_ids")
                    if ref is None:
                        ref = bound.arguments.get("inputs_embeds")
                    if ref is not None:
                        bound.arguments["token_type_ids"] = torch.zeros(
                            ref.shape[:2], dtype=torch.int32, device=ref.device
                        )
                return _orig(*bound.args, **bound.kwargs)

            module.forward = _patched_forward
            patched += 1

    if patched:
        logger.debug(
            "Patched %d HF embedding module(s) to skip aten::gather on token_type_ids.",
            patched,
        )


class _SpyreTransformersEncoderMixin:
    """Mixin for Transformers-backend encoder models with Spyre-specific fixes."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)  # ty: ignore[unknown-argument]
        # The Spyre model runner already applies the RoBERTa position offset on the host
        # (roberta_position_delta = padding_idx + 1). Disable LegacyMixin's second offset
        # to avoid double-shifting position IDs (pos + 4 instead of pos + 2).
        self.is_roberta = False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        result = super().load_weights(weights)
        _patch_encoder_gather(self.model)
        return result


class SpyreTransformersEmbeddingModel(_SpyreTransformersEncoderMixin, TransformersEmbeddingModel):
    """Transformers-backend embedding model with Spyre-specific fixes."""


SpyreTransformersEmbeddingModel.__name__ = "TransformersEmbeddingModel"


class SpyreTransformersForSequenceClassification(
    _SpyreTransformersEncoderMixin, TransformersForSequenceClassification
):
    """Transformers-backend sequence-classification model with Spyre-specific fixes."""


SpyreTransformersForSequenceClassification.__name__ = "TransformersForSequenceClassification"
