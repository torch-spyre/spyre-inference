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
position offset, which runs on CPU (SDSC cannot schedule integer add).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.bert import BertModel
from vllm.model_executor.models.roberta import (
    BgeM3EmbeddingModel,
    RobertaEmbedding,
    RobertaEmbeddingModel,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
)

from spyre_inference.custom_ops.utils import convert
from spyre_inference.models._token_type import (
    SpyreTokenTypeEmbedding,
    SpyreTokenTypeModel,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.models.bert_with_rope import BertWithRope


logger = init_logger(__name__)


def cap_max_model_len_for_position_offset(model_config: Any) -> None:
    """Lower ``max_model_len`` to what the offset position embedding can index.

    ``offset_roberta_position_ids`` gathers ``position_ids + pad_token_id + 1``, so the
    usable context is ``max_position_embeddings - pad_token_id - 1`` -- 512 for the
    514-row table, not the 514 vLLM derives.

    Needed since the encoder grew its rectangular path, which pads every sequence out to
    the declared length: a max-length request's pad rows alone then index two past the
    table on every request. Packed-only, positions never ran past the real prompt length,
    so a prompt had to actually be 514 tokens to notice.

    Called from ``TorchSpyrePlatform.apply_config_platform_defaults`` rather than from
    this module's model classes, which are not imported until load: the cap has to land
    before the encoder shape tables derive from ``max_model_len`` and before vLLM's
    ``SchedulerConfig`` validation. No-op for every other architecture.
    """
    hf_config = model_config.hf_config
    architectures = getattr(hf_config, "architectures", None) or []
    if not any("Roberta" in arch for arch in architectures):
        return
    if getattr(hf_config, "position_embedding_type", "absolute") != "absolute":
        return
    rows = getattr(hf_config, "max_position_embeddings", None)
    pad_token_id = getattr(hf_config, "pad_token_id", None)
    if not isinstance(rows, int) or not isinstance(pad_token_id, int):
        return
    usable = rows - pad_token_id - 1
    if usable < 1 or model_config.max_model_len <= usable:
        return
    logger.warning(
        "Lowering max_model_len %d -> %d: %s offsets positions by pad_token_id+1=%d "
        "into a %d-row position embedding.",
        model_config.max_model_len,
        usable,
        architectures[0],
        pad_token_id + 1,
        rows,
    )
    model_config.max_model_len = usable


@torch.library.custom_op("spyre_inference::roberta_offset_positions", mutates_args=())
def offset_roberta_position_ids(position_ids: torch.Tensor, padding_idx: int) -> torch.Tensor:
    """``position_ids + padding_idx + 1`` on CPU, then H2D as int64.

    Stock torch-spyre cannot schedule SDSC int32 add (warmup crash:
    ``0_add``), and int64 add CPU-falls-back through ``to_dtype``. Keep the
    offset off the device so position embedding is only a gather.

    A custom op, not a plain function, so the host round trip is one opaque node.
    Inlined, the intermediate is a CPU tensor *inside* the graph, and Inductor lowers
    its dtype conversion to ``spyre::to_dtype_cpu``, which has no CPU registration --
    whole-model compile then dies in warmup. ``convert`` is opaque for the same reason;
    it is the arithmetic between two converts that has to be hidden too.
    """
    device = position_ids.device
    pos = convert(position_ids, device="cpu")
    pos = pos + int(padding_idx) + 1
    return convert(pos, device=device, dtype=torch.int64)


@offset_roberta_position_ids.register_fake
def _offset_roberta_position_ids_fake(position_ids: torch.Tensor, padding_idx: int) -> torch.Tensor:
    return torch.empty_like(position_ids, dtype=torch.int64)


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
            + self.position_embeddings(offset_roberta_position_ids(position_ids, self.padding_idx))
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
