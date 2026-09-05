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

"""Spyre adaptations for vLLM BERT-family pooling models.

Every class here exists only to route ``token_type_ids`` around vLLM's
bit-pack transport; see ``spyre_inference.models._token_type``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.model_executor.models.bert import (
    BertEmbedding,
    BertEmbeddingModel,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
    BertSpladeSparseEmbeddingModel,
)

from spyre_inference.models._token_type import (
    SpyreTokenTypeEmbedding,
    SpyreTokenTypeModel,
)

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig


class SpyreBertEmbedding(SpyreTokenTypeEmbedding, BertEmbedding):
    """``BertEmbedding`` reading segment ids from the side buffer."""

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = (
            inputs_embeds
            + self.spyre_token_type_embeddings(input_ids)
            + self.position_embeddings(position_ids)
        )
        return self.LayerNorm(embeddings)


class SpyreBertEmbeddingMixin:
    """Inject the Spyre embedding through ``BertEmbeddingModel._build_model``."""

    def _build_model(self, vllm_config: VllmConfig, prefix: str = "") -> BertModel:
        return BertModel(vllm_config=vllm_config, prefix=prefix, embedding_class=SpyreBertEmbedding)


class SpyreBertEmbeddingModel(SpyreBertEmbeddingMixin, BertEmbeddingModel):
    pass


class SpyreBertSpladeSparseEmbeddingModel(SpyreBertEmbeddingMixin, BertSpladeSparseEmbeddingModel):
    pass


class SpyreBertForSequenceClassification(SpyreTokenTypeModel, BertForSequenceClassification):
    spyre_embedding_class = SpyreBertEmbedding


class SpyreBertForTokenClassification(SpyreTokenTypeModel, BertForTokenClassification):
    spyre_embedding_class = SpyreBertEmbedding


class SpyreBertForMaskedLM(SpyreTokenTypeModel, BertForMaskedLM):
    spyre_embedding_class = SpyreBertEmbedding
