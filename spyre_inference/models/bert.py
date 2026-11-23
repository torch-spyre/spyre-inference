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
They also split each ``BertSelfAttention``'s fused QKV projection; see ``split_bert_qkv``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.models.bert import (
    BertEmbedding,
    BertEmbeddingModel,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
    BertSelfAttention,
    BertSpladeSparseEmbeddingModel,
)

from spyre_inference.custom_ops.lazy_compile import CompileOutermost
from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod, spyre_linear_t
from spyre_inference.models._token_type import (
    SpyreTokenTypeEmbedding,
    SpyreTokenTypeModel,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_QKV = ("q", "k", "v")


def _split_qkv_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    proj = self.qkv_proj
    q, k, v = (
        spyre_linear_t(hidden_states, getattr(proj, f"{n}_weight_t"), getattr(proj, f"{n}_bias"))
        for n in _QKV
    )
    return self.attn(q, k, v)


class _SplitQKVLinearMethod(SpyreUnquantizedLinearMethod):
    def __init__(self, sizes: list[int]) -> None:
        self.sizes = sizes

    def build_weight_t(self, layer: nn.Module, w: torch.Tensor) -> None:
        # Reload restores the original fused weight, then copies these processed
        # parameters into their existing storage, including compiled graph references.
        # The retained bias can still be on Spyre while the restored weight is on CPU.
        bias = cast(torch.Tensor, layer.bias).data.to(w.device) if layer.bias is not None else None
        biases = bias.split(self.sizes) if bias is not None else (None,) * 3
        for n, shard, b in zip(_QKV, w.split(self.sizes, dim=0), biases):
            layer.register_parameter(f"{n}_weight_t", _frozen(shard.t()))
            layer.register_parameter(f"{n}_bias", None if b is None else _frozen(b))
        del layer.weight


def split_bert_qkv(model: nn.Module) -> None:
    """Replace each ``BertSelfAttention``'s fused QKV projection with three matmuls.

    The fused output has three consumers needing different layouts, so torch-spyre
    writes it to HBM and Q, K and V each read the whole buffer back
    (https://github.com/torch-spyre/torch-spyre/issues/6566). Runs on the host, on
    the transposed weights, before the move to Spyre.
    """
    count = 0
    for module in model.modules():
        if not isinstance(module, BertSelfAttention):
            continue
        proj = getattr(module, "qkv_proj", None)
        if proj is None or type(proj.quant_method) is not SpyreUnquantizedLinearMethod:
            continue
        if proj.weight.device.type != "cpu":
            continue
        sizes = [module.q_size, module.kv_size, module.kv_size]
        proj.quant_method = _SplitQKVLinearMethod(sizes)
        # Initial loading already transposed the weight. Retain the projection
        # and its bias so vLLM's checkpoint mapper and reload metadata stay valid.
        proj.quant_method.build_weight_t(proj, cast(torch.Tensor, proj.weight).data.t())
        module.forward = _split_qkv_forward.__get__(module)  # ty: ignore[invalid-assignment]
        count += 1
    if count:
        logger.info("Split the fused QKV projection of %d BertSelfAttention layers.", count)


def _frozen(t: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(t.contiguous(), requires_grad=False)


class SpyreSplitQKVMixin:
    """Run ``split_bert_qkv`` from vLLM's model-level post-load hook, which follows the
    per-layer weight transpose and precedes the runner's move to Spyre."""

    def process_weights_after_loading(self) -> None:
        split_bert_qkv(cast(nn.Module, self))

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        if loaded is not None:
            for prefix, module in cast(nn.Module, self).named_modules():
                if isinstance(getattr(module, "quant_method", None), _SplitQKVLinearMethod):
                    for source, target in (("weight", "weight_t"), ("bias", "bias")):
                        if f"{prefix}.{source}" in loaded:
                            loaded.update(f"{prefix}.{n}_{target}" for n in _QKV)
        return loaded


class SpyreBertEmbedding(CompileOutermost, SpyreTokenTypeEmbedding, BertEmbedding):
    """``BertEmbedding`` reading segment ids from the side buffer."""


class SpyreBertEmbeddingMixin:
    """Inject the Spyre embedding through ``BertEmbeddingModel._build_model``."""

    def _build_model(self, vllm_config: VllmConfig, prefix: str = "") -> BertModel:
        return BertModel(vllm_config=vllm_config, prefix=prefix, embedding_class=SpyreBertEmbedding)


class SpyreBertEmbeddingModel(SpyreSplitQKVMixin, SpyreBertEmbeddingMixin, BertEmbeddingModel):
    pass


class SpyreBertSpladeSparseEmbeddingModel(
    SpyreSplitQKVMixin, SpyreBertEmbeddingMixin, BertSpladeSparseEmbeddingModel
):
    pass


class SpyreBertForSequenceClassification(
    SpyreSplitQKVMixin, SpyreTokenTypeModel, BertForSequenceClassification
):
    spyre_embedding_class = SpyreBertEmbedding


class SpyreBertForTokenClassification(
    SpyreSplitQKVMixin, SpyreTokenTypeModel, BertForTokenClassification
):
    spyre_embedding_class = SpyreBertEmbedding


class SpyreBertForMaskedLM(SpyreSplitQKVMixin, SpyreTokenTypeModel, BertForMaskedLM):
    spyre_embedding_class = SpyreBertEmbedding
