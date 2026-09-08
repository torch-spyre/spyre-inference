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

"""Out-of-band ``token_type_ids`` transport for BERT-family encoders.

Shared machinery rather than one architecture's adaptation, hence the private
name: the modules beside it mirror the vLLM modules whose architectures they adapt.

vLLM packs segment ids into the high bits of ``input_ids`` (``TOKEN_TYPE_SHIFT``
in ``vllm.model_executor.models.bert``) so torch.compile sees one persistent
tensor, then unpacks them inside the embedding's ``forward``. That packing is a
vLLM transport detail, not a BERT requirement, and Spyre does not lower the
integer bitwise unpack (torch-spyre#3509).

The mixins here carry the segment ids as a side input instead: the top-level
model copies them into a buffer owned by the embedding, and the embedding reads
that buffer rather than unpacking ``input_ids``. Shared by BERT and RoBERTa,
whose embeddings differ only in position handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import nn
    from vllm.sequence import IntermediateTensors


class SpyreTokenTypeEmbedding:
    """Embedding mixin taking ``token_type_ids`` as a side input."""

    spyre_token_type_ids: torch.Tensor | None = None

    def set_spyre_token_type_ids(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None
    ) -> None:
        """Copy segment ids into an ``input_ids``-shaped buffer, zero-padded.

        Right-pad slots are segment 0. The buffer is reused across steps so the
        compiled graph keeps seeing one tensor, and it is re-shaped whenever the
        padded length changes: a batch without ``token_type_ids`` still has to
        leave an ``input_ids``-shaped buffer behind, or the next embedding would
        add a stale one of the previous length.
        """
        buffer = self.spyre_token_type_ids
        if token_type_ids is not None:
            dtype = token_type_ids.dtype
        elif buffer is not None:
            dtype = buffer.dtype
        else:
            # Single-segment model: the embedding falls back to all-zeros.
            return

        if (
            buffer is None
            or buffer.shape != input_ids.shape
            or buffer.device != input_ids.device
            or buffer.dtype != dtype
        ):
            buffer = torch.zeros(input_ids.shape, dtype=dtype, device=input_ids.device)
            self.spyre_token_type_ids = buffer
        else:
            buffer.zero_()

        if token_type_ids is not None:
            n = token_type_ids.shape[0]
            buffer[:n].copy_(token_type_ids[:n])

    def spyre_token_type_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_type_ids = self.spyre_token_type_ids
        if token_type_ids is None or token_type_ids.shape != input_ids.shape:
            # No buffer: single-segment model, every token is segment 0. A
            # mismatched one is unreachable via set_spyre_token_type_ids, and
            # all-zeros beats failing to lower the add.
            token_type_ids = torch.zeros_like(input_ids)
        return self.token_type_embeddings(token_type_ids)


class SpyreTokenTypeModel:
    """Top-level mixin handing ``token_type_ids`` to the embedding out of band.

    The wrapper classes (``*ForSequenceClassification`` and friends) hardcode
    ``embedding_class``, so the already-built embedding is retyped to its Spyre
    subclass: same ``__init__``, same parameters, same module tree — only
    ``forward`` differs. ``super().forward`` is then called with no
    ``token_type_ids`` so upstream skips the bit-pack.
    """

    spyre_embedding_class: type[nn.Module]
    spyre_encoder_attr: str = "bert"

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)  # ty: ignore[unknown-argument]
        embeddings = self.spyre_embeddings()
        upstream_class = self.spyre_embedding_class.__bases__[-1]
        if type(embeddings) is not upstream_class:
            raise RuntimeError(
                f"expected {upstream_class.__name__} embeddings, got "
                f"{type(embeddings).__name__}; the Spyre token_type transport "
                "needs updating for this vLLM version."
            )
        embeddings.__class__ = self.spyre_embedding_class

    def spyre_embeddings(self) -> SpyreTokenTypeEmbedding:
        return getattr(self, self.spyre_encoder_attr).embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            self.spyre_embeddings().set_spyre_token_type_ids(input_ids, token_type_ids)
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
