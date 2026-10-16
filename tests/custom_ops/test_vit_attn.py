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

"""Tests for `spyre_inference/custom_ops/vit_attn.py`. CPU-only.

The padding/masking math itself is `multimodal.utils.padded_sdpa`, already covered
by `tests/multimodal/test_pixtral.py`; these tests cover this module's own surface:
the OOT `MMEncoderAttention`'s layout handling, that it traces into one graph, the
unmasked key-pad mask, and registration.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention

from spyre_inference.custom_ops.vit_attn import SpyreMMEncoderAttention
from spyre_inference.multimodal.utils import _KEY_PAD_MASKS, _key_pad_mask_key


def _attn(num_heads: int, head_size: int, num_kv_heads: int | None = None):
    """Skips ``MMEncoderAttention.__init__``, which needs a vLLM platform context."""
    attn = object.__new__(SpyreMMEncoderAttention)
    nn.Module.__init__(attn)
    attn.num_heads = num_heads
    attn.num_kv_heads = num_kv_heads or num_heads
    attn.head_size = head_size
    attn.scale = head_size**-0.5
    return attn


def _reference(q, k, v, attn):
    """Unpadded SDPA over `[B, S, H*D]` inputs, as upstream's `_forward_sdpa` computes it."""
    b, s, _ = q.shape
    q, k, v = (x.view(b, s, -1, attn.head_size).transpose(1, 2) for x in (q, k, v))
    out = F.scaled_dot_product_attention(
        q, k, v, scale=attn.scale, enable_gqa=attn.num_heads > attn.num_kv_heads
    )
    return out.transpose(1, 2).reshape(b, s, -1)


def _random_qkv(batch, seq, heads, head_size, kv_heads=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    kv = (kv_heads or heads) * head_size
    q = torch.randn(batch, seq, heads * head_size, generator=g)
    k = torch.randn(batch, seq, kv, generator=g)
    v = torch.randn(batch, seq, kv, generator=g)
    return q, k, v


class TestForwardOotMatchesReference:
    @pytest.mark.parametrize(
        ("batch", "seq", "heads"),
        [(1, 50, 12), (3, 50, 12), (1, 64, 8), (2, 257, 16)],
        ids=["vit_b32", "vit_b32_batch", "aligned", "vit_l14"],
    )
    def test_matches_unpadded_sdpa(self, batch, seq, heads):
        # seq=50 (7x7 patches + CLS) is the shape that crashed; 64 takes the no-pad branch.
        attn = _attn(heads, 64)
        q, k, v = _random_qkv(batch, seq, heads, 64)
        torch.testing.assert_close(
            attn.forward_oot(q, k, v), _reference(q, k, v, attn), atol=1e-5, rtol=1e-5
        )

    def test_grouped_query_heads(self):
        attn = _attn(8, 64, num_kv_heads=2)
        q, k, v = _random_qkv(1, 50, 8, 64, kv_heads=2)
        torch.testing.assert_close(
            attn.forward_oot(q, k, v), _reference(q, k, v, attn), atol=1e-5, rtol=1e-5
        )

    def test_4d_input_keeps_its_rank(self):
        attn = _attn(4, 64)
        q, k, v = (x.view(2, 50, 4, 64) for x in _random_qkv(2, 50, 4, 64))
        out = attn.forward_oot(q, k, v)
        assert out.shape == (2, 50, 4, 64)
        flat = [x.reshape(2, 50, -1) for x in (q, k, v)]
        torch.testing.assert_close(
            out.reshape(2, 50, -1), _reference(*flat, attn), atol=1e-5, rtol=1e-5
        )


def test_traces_into_one_graph():
    """A marked tower's block is compiled `fullgraph=True`; the attention must not
    break it, and SDPA must be in the graph rather than behind an opaque op."""
    graphs: list[torch.fx.GraphModule] = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm

    attn = _attn(12, 64)
    q, k, v = _random_qkv(2, 50, 12, 64)
    compiled = torch.compile(attn.forward_oot, backend=backend, fullgraph=True, dynamic=False)
    torch.testing.assert_close(compiled(q, k, v), _reference(q, k, v, attn), atol=1e-5, rtol=1e-5)

    assert len(graphs) == 1
    targets = {n.target for n in graphs[0].graph.nodes if n.op == "call_function"}
    assert F.scaled_dot_product_attention in targets


def test_rejects_cross_attention():
    """`padded_sdpa` pads both sides to the query's length, so a shorter key would be
    silently wrong; raising is the only safe answer."""
    attn = _attn(4, 64)
    q = torch.randn(1, 50, 4 * 64)
    k = v = torch.randn(1, 64, 4 * 64)
    with pytest.raises(NotImplementedError):
        attn.forward_oot(q, k, v)


def test_packed_images_attend_within_their_own_block():
    """Pixtral's HF tower packs every image into one sequence and passes `cu_seqlens`;
    the result has to match attending over each image alone."""
    attn = _attn(4, 64)
    q, k, v = _random_qkv(1, 50, 4, 64)
    cu_seqlens = torch.tensor([0, 30, 50], dtype=torch.int32)

    expected = torch.cat(
        [_reference(*(x[:, s:e] for x in (q, k, v)), attn) for s, e in [(0, 30), (30, 50)]],
        dim=1,
    )
    torch.testing.assert_close(
        attn.forward_oot(q, k, v, cu_seqlens), expected, atol=1e-5, rtol=1e-5
    )


def test_mm_encoder_attention_instantiates_the_spyre_class():
    assert type(MMEncoderAttention.__new__(MMEncoderAttention)) is SpyreMMEncoderAttention


class TestKeyPadMask:
    def test_masks_only_the_padded_keys(self):
        mask = _KEY_PAD_MASKS[_key_pad_mask_key(50, 64, torch.float16, torch.device("cpu"))]
        assert mask.shape == (1, 1, 1, 64)
        assert (mask[..., :50] == 0).all()
        assert (mask[..., 50:] == torch.finfo(torch.float16).min / 2).all()

    def test_same_shape_returns_the_same_object(self):
        # One upload per sequence length, shared by every layer and every batch size.
        cpu = torch.device("cpu")
        assert (
            _KEY_PAD_MASKS[_key_pad_mask_key(50, 64, torch.float16, cpu)]
            is _KEY_PAD_MASKS[_key_pad_mask_key(50, 64, torch.float16, cpu)]
        )

    def test_unpadded_length_masks_nothing(self):
        row = _KEY_PAD_MASKS[_key_pad_mask_key(64, 64, torch.float16, torch.device("cpu"))]
        assert (row == 0).all()
