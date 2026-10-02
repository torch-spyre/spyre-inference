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
``spyre_inference.models.bert``. The position offset (``padding_idx + 1``) is
applied on the host in ``TorchSpyreModelRunner._preprocess``, before the ids
are copied up. The embedding forward gathers those positions in one program.
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

from spyre_inference.custom_ops.lazy_compile import CompileOutermost, maybe_compile
from spyre_inference.models._token_type import (
    SpyreTokenTypeEmbedding,
    SpyreTokenTypeModel,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.models.bert_with_rope import BertWithRope


logger = init_logger(__name__)


def roberta_position_delta(hf_config: Any) -> int:
    """``padding_idx + 1`` for absolute-position RoBERTa, else ``0``.

    The embedding table is indexed by ``position_ids + padding_idx + 1``. The
    runner adds that on the host before the ids are copied up.
    """
    if hf_config is None:
        return 0
    architectures = getattr(hf_config, "architectures", None) or []
    if not any("Roberta" in arch for arch in architectures):
        return 0
    if getattr(hf_config, "position_embedding_type", "absolute") != "absolute":
        return 0
    pad_token_id = getattr(hf_config, "pad_token_id", None)
    if not isinstance(pad_token_id, int):
        return 0
    return pad_token_id + 1


def cap_max_model_len_for_position_offset(model_config: Any) -> None:
    """Lower ``max_model_len`` to what the offset position embedding can index.

    The runner offsets positions by ``pad_token_id + 1`` before the embedding gathers
    them, so the usable context is ``max_position_embeddings - pad_token_id - 1`` --
    512 for the 514-row table, not the 514 vLLM derives.

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
    delta = roberta_position_delta(hf_config)
    if delta == 0:
        return
    rows = getattr(hf_config, "max_position_embeddings", None)
    if not isinstance(rows, int):
        return
    usable = rows - delta
    if usable < 1 or model_config.max_model_len <= usable:
        return
    architectures = getattr(hf_config, "architectures", None) or ["Roberta"]
    logger.warning(
        "Lowering max_model_len %d -> %d: %s offsets positions by pad_token_id+1=%d "
        "into a %d-row position embedding.",
        model_config.max_model_len,
        usable,
        architectures[0],
        delta,
        rows,
    )
    model_config.max_model_len = usable


def offset_host_positions(positions: torch.Tensor, delta: int) -> torch.Tensor:
    """Add ``delta`` to every position on the host, as one new int64 tensor.

    Python rather than ``aten::add``. This runs on the per-step preprocess path,
    and every extra CPU aten op there lengthens the eager guard chain (#981).
    The rectangular path adds the same offset inside ``expand_packed_to_encoder_grid``.
    """
    if delta == 0:
        return positions
    values = positions.detach().cpu().tolist()
    return torch.tensor([int(value) + delta for value in values], dtype=torch.int64)


class SpyreRobertaEmbedding(CompileOutermost, SpyreTokenTypeEmbedding, RobertaEmbedding):
    """``RobertaEmbedding`` reading segment ids from the side buffer.

    ``position_ids`` already include ``padding_idx + 1``. One compiled forward
    so the three gathers, the two adds, and the layer norm are one program.
    """

    padding_idx: int

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pass side-buffer segment ids, or zeros, into the compiled gather.
        # ``position_ids`` already include the offset.
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
