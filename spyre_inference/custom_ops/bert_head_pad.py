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

"""Pads BertSelfAttention/BertAttention to a stick-aligned head_dim at load time."""

from __future__ import annotations

import torch.nn as nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import EncoderOnlyAttention
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.models.bert import BertAttention, BertSelfAttention
from vllm.utils.math_utils import cdiv

from spyre_inference.custom_ops.head_pad import head_padding_active

logger = init_logger(__name__)

# One Spyre stick of fp16 elements. Encoder attention has no RoPE, so unlike the
# RoPE path's 128-multiple (head_pad.py), this is the only alignment that matters.
_STICK = 64

_PATCHED_ATTR = "_spyre_bert_head_pad_patched"


def install_bert_head_pad(model_config) -> None:
    """No-op unless this is a pooling model with a sub-stick head_dim. Safe to call
    once per process: a later model that doesn't need padding is unaffected."""
    hf_config = model_config.hf_config
    if not head_padding_active(hf_config):
        return
    if getattr(model_config, "runner_type", None) != "pooling":
        return
    if getattr(BertSelfAttention, _PATCHED_ATTR, False):
        return

    original_attn_init = BertAttention.__init__

    def patched_self_attn_init(
        self,
        hidden_size,
        num_attention_heads,
        cache_config=None,
        quant_config=None,
        prefix="",
    ):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_attention_heads
        assert self.total_num_heads % tp_size == 0

        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = self.total_num_heads
        orig_head_dim = self.hidden_size // self.total_num_heads
        assert orig_head_dim * self.total_num_heads == self.hidden_size
        self.head_dim = cdiv(orig_head_dim, _STICK) * _STICK

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = orig_head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.attn = EncoderOnlyAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def patched_attn_init(
        self,
        hidden_size,
        num_attention_heads,
        layer_norm_eps,
        cache_config=None,
        quant_config=None,
        prefix="",
    ):
        original_attn_init(
            self,
            hidden_size,
            num_attention_heads,
            layer_norm_eps,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        # self.self.head_dim is already the padded (or unchanged) width.
        padded_width = self.self.total_num_heads * self.self.head_dim
        if padded_width == hidden_size:
            return
        self.output.dense = RowParallelLinear(
            input_size=padded_width,
            output_size=hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.output.dense",
        )

    BertSelfAttention.__init__ = patched_self_attn_init  # ty: ignore[invalid-assignment]
    BertAttention.__init__ = patched_attn_init  # ty: ignore[invalid-assignment]
    setattr(BertSelfAttention, _PATCHED_ATTR, True)
    logger.info(
        "Patched BertSelfAttention/BertAttention to build sub-stick head_dim "
        "pooling models at a stick-aligned width."
    )
