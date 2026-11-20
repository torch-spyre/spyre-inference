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

"""``split_bert_qkv`` replaces the fused QKV projection with three bit-identical ones (CPU-only)."""

import torch
import torch.nn as nn

from spyre_inference.custom_ops.bert_split_qkv import split_bert_qkv


class _Capture(nn.Module):
    def forward(self, q, k, v):
        return q, k, v


def _self_attention(hidden_size, num_heads):
    from vllm.model_executor.models.bert import BertSelfAttention

    attn = BertSelfAttention(hidden_size=hidden_size, num_attention_heads=num_heads)
    proj = attn.qkv_proj
    g = torch.Generator().manual_seed(0)
    proj.weight.data.copy_(torch.randn(proj.weight.shape, generator=g))
    proj.bias.data.copy_(torch.randn(proj.bias.shape, generator=g))
    proj.quant_method.process_weights_after_loading(proj)
    attn.attn = _Capture()
    return attn


def test_split_matches_the_fused_projection(default_vllm_config, tp_group):
    attn = _self_attention(hidden_size=128, num_heads=2)
    x = torch.randn(5, 128)
    qkv, _ = attn.qkv_proj(x)
    fused = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

    split_bert_qkv(nn.Sequential(attn))

    assert not hasattr(attn, "qkv_proj")
    assert attn.q_weight_t.shape == (128, attn.q_size)
    assert attn.k_weight_t.is_contiguous() and attn.v_bias.is_contiguous()
    for got, want in zip(attn(x), fused):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
