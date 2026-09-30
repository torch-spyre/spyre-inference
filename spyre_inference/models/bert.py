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

These classes route ``token_type_ids`` around vLLM's bit-pack transport;
the embedding subclass also fuses its prologue. See ``spyre_inference.models._token_type``.
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

from spyre_inference.custom_ops.lazy_compile import CompileOutermost, maybe_compile
from spyre_inference.models._token_type import (
    SpyreTokenTypeEmbedding,
    SpyreTokenTypeModel,
)

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig


class SpyreBertEmbedding(CompileOutermost, SpyreTokenTypeEmbedding, BertEmbedding):
    """``BertEmbedding`` reading segment ids from the side buffer.

    One compiled forward: the word, segment, and position tables and the
    layer norm inline into it.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Copy segment ids out of the side buffer and pass that tensor into the compiled gather.
        return self._compiled_forward(
            input_ids,
            position_ids,
            self.spyre_token_type_ids_for(input_ids),
            inputs_embeds,
        )

    @maybe_compile
    def _compiled_forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = (
            inputs_embeds
            + self.token_type_embeddings(token_type_ids)
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
