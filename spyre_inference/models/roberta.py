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

"""Spyre adaptations for vLLM RoBERTa / XLM-R pooling models.

RoBERTa reuses BERT's ``token_type_ids`` bit-pack transport, so these mirror
``spyre_inference.models.bert``; the embedding differs only in RoBERTa's
position offset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.model_executor.models.bert import BertModel
from vllm.model_executor.models.roberta import (
    BgeM3EmbeddingModel,
    RobertaEmbedding,
    RobertaEmbeddingModel,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
)

from spyre_inference.models._token_type import (
    SpyreTokenTypeEmbedding,
    SpyreTokenTypeModel,
)

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.model_executor.models.bert_with_rope import BertWithRope


class SpyreRobertaEmbedding(SpyreTokenTypeEmbedding, RobertaEmbedding):
    """``RobertaEmbedding`` reading segment ids from the side buffer."""

    padding_idx: int

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
            + self.position_embeddings(position_ids + self.padding_idx + 1)
        )
        return self.LayerNorm(embeddings)


class SpyreRobertaEmbeddingMixin:
    """Inject the Spyre embedding through ``RobertaEmbeddingModel._build_model``.

    A mixin rather than an override on the concrete class: ``RobertaEmbedding``
    unpacks the bit-packed segment ids on every ``forward``, so every
    ``RobertaEmbeddingModel`` subclass needs the swap to compile at all.
    """

    def _build_model(self, vllm_config: VllmConfig, prefix: str = "") -> BertModel | BertWithRope:
        hf_config = vllm_config.model_config.hf_config
        if getattr(hf_config, "position_embedding_type", "absolute") != "absolute":
            # Rotary variants (Jina) do not use the bit-pack transport.
            return super()._build_model(vllm_config, prefix)
        return BertModel(
            vllm_config=vllm_config,
            prefix=prefix,
            embedding_class=SpyreRobertaEmbedding,
        )


class SpyreRobertaEmbeddingModel(SpyreRobertaEmbeddingMixin, RobertaEmbeddingModel):
    pass


class SpyreBgeM3EmbeddingModel(SpyreRobertaEmbeddingMixin, BgeM3EmbeddingModel):
    """BGE-M3 keeps its own ``__init__``/``_build_pooler`` (sparse + colbert heads)."""


class SpyreRobertaForSequenceClassification(SpyreTokenTypeModel, RobertaForSequenceClassification):
    spyre_embedding_class = SpyreRobertaEmbedding
    spyre_encoder_attr = "roberta"


class SpyreRobertaForTokenClassification(SpyreTokenTypeModel, RobertaForTokenClassification):
    spyre_embedding_class = SpyreRobertaEmbedding
    spyre_encoder_attr = "roberta"
