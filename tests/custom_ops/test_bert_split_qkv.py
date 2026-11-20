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

"""CPU-only."""

import pytest
import torch
import torch.nn as nn

from spyre_inference.custom_ops import bert_split_qkv
from spyre_inference.custom_ops.bert_split_qkv import grouped_rectangles, split_bert_qkv
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    _encoder_rect_kernel,
    encoder_grid,
    encoder_key_pad_mask,
)

WIDTH, EXTENT, HIDDEN, HEADS = 2, 64, 128, 4


class _Capture(nn.Module):
    def forward(self, q, k, v):
        return q, k, v


class _RectAttention(nn.Module):
    def __init__(self, heads, head_dim):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.impl = type("Impl", (), {"scale": head_dim**-0.5})()

    def forward(self, q, k, v):
        h, d = self.heads, self.head_dim
        mask = encoder_grid().mask
        a = _encoder_rect_kernel(
            q.view(-1, h, d),
            k.view(-1, h, d),
            v.view(-1, h, d),
            mask,
            d**-0.5,
            WIDTH,
            EXTENT,
            h,
            h,
            d,
        )
        return a.reshape(-1, h * d)


def _loaded(module):
    g = torch.Generator().manual_seed(0)
    for proj in (m for m in module.modules() if hasattr(m, "quant_method")):
        proj.weight.data.copy_(torch.randn(proj.weight.shape, generator=g) * 0.1)
        if proj.bias is not None:
            proj.bias.data.copy_(torch.randn(proj.bias.shape, generator=g) * 0.1)
        proj.quant_method.process_weights_after_loading(proj)
    return module


def _self_attention():
    from vllm.model_executor.models.bert import BertSelfAttention

    attn = _loaded(BertSelfAttention(hidden_size=HIDDEN, num_attention_heads=HEADS))
    attn.attn = _Capture()
    return attn


def _attention(prefix):
    from vllm.model_executor.models.bert import BertAttention

    attn = _loaded(
        BertAttention(
            hidden_size=HIDDEN, num_attention_heads=HEADS, layer_norm_eps=1e-5, prefix=prefix
        )
    )
    attn.self.attn = _RectAttention(HEADS, HIDDEN // HEADS)
    return attn


def test_split_matches_the_fused_projection(default_vllm_config, tp_group):
    attn = _self_attention()
    x = torch.randn(5, HIDDEN)
    qkv, _ = attn.qkv_proj(x)
    fused = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

    split_bert_qkv(nn.Sequential(attn))

    assert not hasattr(attn, "qkv_proj")
    assert attn.q_weight_t.shape == (HIDDEN, attn.q_size)
    for got, want in zip(attn(x), fused):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("inline", [False, True], ids=["fallback", "rectangle"])
def test_head_groups_match_one_group(default_vllm_config, tp_group, monkeypatch, inline):
    x = torch.randn(WIDTH * EXTENT, HIDDEN)
    mask = encoder_key_pad_mask(EXTENT, [EXTENT, EXTENT // 2], torch.float32, False)
    encoder_grid().publish(mask if inline else None)
    monkeypatch.setattr(bert_split_qkv, "encoder_inline_active", lambda: inline)
    try:
        one = _attention("one")
        one.spyre_group_rects = frozenset()
        split_bert_qkv(nn.Sequential(one))
        two = _attention("two")
        two.spyre_group_rects = frozenset({(WIDTH, EXTENT)})
        split_bert_qkv(nn.Sequential(two))

        assert hasattr(two.self, "q_weight_t_1") and not hasattr(one.self, "q_weight_t_1")
        torch.testing.assert_close(two(x), one(x), rtol=1e-5, atol=1e-5)
    finally:
        encoder_grid().publish(None)


def test_planner_groups_only_kv_tiled_rectangles():
    rects = [(64, 32), (128, 16), (256, 8), (512, 4)]
    assert grouped_rectangles(rects, 16, 16, 64) == {(4, 512)}  # e5-large, bge-reranker-large
    assert grouped_rectangles(rects, 12, 12, 64) == set()  # granite-embedding
    assert grouped_rectangles([(64, 8), (128, 4)], 16, 16, 64) == set()  # all-roberta, L 128
