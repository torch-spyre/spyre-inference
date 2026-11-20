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

"""Splits BertSelfAttention's fused QKV projection into three matmuls before the device move.

One ``[T, q + 2kv]`` output has three consumers in different layouts, so it is written to HBM
and each of Q, K and V re-reads the whole buffer; separate outputs are each read once.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.models.bert import BertSelfAttention

from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod, spyre_linear_t

logger = init_logger(__name__)

_NAMES = ("q", "k", "v")


def _param(t: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(t.contiguous(), requires_grad=False)


def _split_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    q, k, v = (
        spyre_linear_t(hidden_states, getattr(self, f"{n}_weight_t"), getattr(self, f"{n}_bias"))
        for n in _NAMES
    )
    return self.attn(q, k, v)


def split_bert_qkv(model: nn.Module) -> None:
    """Replace each unquantized ``qkv_proj`` of a BertSelfAttention with Q/K/V weights (on CPU)."""
    count = 0
    for module in model.modules():
        if not isinstance(module, BertSelfAttention):
            continue
        proj = module.qkv_proj
        if type(proj.quant_method) is not SpyreUnquantizedLinearMethod:
            continue
        if proj.weight.device.type != "cpu":
            continue
        sizes = [module.q_size, module.kv_size, module.kv_size]
        weights = proj.weight.data.split(sizes, dim=1)  # stored transposed: [in, q + 2kv]
        biases = proj.bias.data.split(sizes) if proj.bias is not None else (None,) * 3
        for n, w, b in zip(_NAMES, weights, biases):
            module.register_parameter(f"{n}_weight_t", _param(w))
            module.register_parameter(f"{n}_bias", None if b is None else _param(b))
        del module.qkv_proj
        module.forward = _split_forward.__get__(module)
        count += 1
    if count:
        logger.info("Split the fused QKV projection of %d BertSelfAttention layers.", count)
