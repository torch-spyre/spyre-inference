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

"""Tests for SpyreMMEncoderAttention and its helpers.

All tests run on CPU only: no Spyre card is acquired, no vLLM platform init is
needed.  We bypass MMEncoderAttention.__init__ (which calls get_vit_attn_backend
and touches the platform registry) by using object.__new__ + direct attribute
assignment, exactly as the upstream class documents its own instance state.

What is under test
------------------
- _apply_sdpa   – four-dimensional SDPA wrapper; einops rearrange round-trip.
- _sdpa_forward – batched dispatch: no cu_seqlens → single call, cu_seqlens
                  present → per-image chunked loop with correct reassembly.
- SpyreMMEncoderAttention.forward_oot:
    * tensors are moved to CPU before SDPA and back to the target device after
    * the 3-D input path (is_reshaped=True) produces (bsz, q_len, hidden)
    * the 4-D input path (is_reshaped=False) preserves the 4-D output shape
    * GQA (num_heads > num_kv_heads) is forwarded to F.sdpa via enable_gqa
    * cu_seqlens is transferred to CPU along with Q/K/V
    * scale is passed through to the SDPA call
"""

import math
import unittest.mock as mock

import pytest
import torch
import torch.nn.functional as F

from spyre_inference.custom_ops.mm_encoder_attention import (
    SpyreMMEncoderAttention,
    _apply_sdpa,
    _sdpa_forward,
)

pytestmark = pytest.mark.mm_encoder_attention

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_HEADS = 4
_NUM_KV_HEADS = 4
_HEAD_SIZE = 16
_BSZ = 1
_SEQ_LEN = 6


def _make_attn(
    num_heads: int = _NUM_HEADS,
    head_size: int = _HEAD_SIZE,
    num_kv_heads: int | None = None,
    scale: float | None = None,
) -> SpyreMMEncoderAttention:
    """Instantiate SpyreMMEncoderAttention without triggering the vLLM init path.

    MMEncoderAttention.__init__ calls get_vit_attn_backend() which requires a
    vLLM platform context.  We use object.__new__ and replicate only the
    instance attributes that forward_oot reads.
    """
    obj = object.__new__(SpyreMMEncoderAttention)
    obj.num_heads = num_heads
    obj.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    obj.head_size = head_size
    obj.scale = (1.0 / math.sqrt(head_size)) if scale is None else scale
    return obj


def _qkv(bsz=_BSZ, seq=_SEQ_LEN, num_heads=_NUM_HEADS, head_size=_HEAD_SIZE, num_kv_heads=None):
    """Return (query, key, value) as 3-D tensors (bsz, seq, num_heads * head_size)."""
    nkv = num_kv_heads if num_kv_heads is not None else num_heads
    hidden = num_heads * head_size
    kv_hidden = nkv * head_size
    g = torch.Generator(device="cpu").manual_seed(42)
    q = torch.randn(bsz, seq, hidden, generator=g)
    k = torch.randn(bsz, seq, kv_hidden, generator=g)
    v = torch.randn(bsz, seq, kv_hidden, generator=g)
    return q, k, v


# ---------------------------------------------------------------------------
# _apply_sdpa
# ---------------------------------------------------------------------------


class TestApplySdpa:
    def test_output_shape_matches_input(self):
        """Output (b, s, h, d) must be the same shape as the query."""
        g = torch.Generator(device="cpu").manual_seed(0)
        q = torch.randn(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE, generator=g)
        k = torch.randn(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE, generator=g)
        v = torch.randn(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE, generator=g)

        out = _apply_sdpa(q, k, v)

        assert out.shape == q.shape

    def test_scale_is_forwarded(self):
        """A scale of 0.0 forces all-zero attention weights → output ≈ mean(V)."""
        g = torch.Generator(device="cpu").manual_seed(1)
        q = torch.randn(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE, generator=g)
        k = torch.randn(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE, generator=g)
        v = torch.ones(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE)

        out = _apply_sdpa(q, k, v, scale=0.0)

        # With scale=0 all logits are 0 → uniform attention → output = mean(V) = 1.
        assert torch.allclose(out, torch.ones_like(out), atol=1e-5)

    def test_output_dtype_matches_input(self):
        g = torch.Generator(device="cpu").manual_seed(2)
        q = torch.randn(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE, generator=g).half()
        k = q.clone()
        v = q.clone()

        out = _apply_sdpa(q, k, v)

        assert out.dtype == torch.float16

    def test_gqa_enable_flag(self):
        """GQA (4 query heads, 2 KV heads) must not raise and must reshape correctly."""
        num_heads, num_kv_heads, head_size = 4, 2, 16
        g = torch.Generator(device="cpu").manual_seed(3)
        q = torch.randn(_BSZ, _SEQ_LEN, num_heads, head_size, generator=g)
        k = torch.randn(_BSZ, _SEQ_LEN, num_kv_heads, head_size, generator=g)
        v = torch.randn(_BSZ, _SEQ_LEN, num_kv_heads, head_size, generator=g)

        out = _apply_sdpa(q, k, v, enable_gqa=True)

        assert out.shape == (_BSZ, _SEQ_LEN, num_heads, head_size)


# ---------------------------------------------------------------------------
# _sdpa_forward
# ---------------------------------------------------------------------------


class TestSdpaForward:
    def test_no_cu_seqlens_delegates_to_apply_sdpa(self):
        """Without cu_seqlens a single _apply_sdpa is issued."""
        g = torch.Generator(device="cpu").manual_seed(4)
        q = torch.randn(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE, generator=g)
        k = torch.randn(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE, generator=g)
        v = torch.randn(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE, generator=g)

        with mock.patch(
            "spyre_inference.custom_ops.mm_encoder_attention._apply_sdpa",
            wraps=_apply_sdpa,
        ) as spy:
            _sdpa_forward(q, k, v, cu_seqlens=None)

        spy.assert_called_once()

    def test_cu_seqlens_splits_and_concatenates(self):
        """With two images of lengths [3, 3], output must equal two independent SDPAs."""
        bsz = 1
        lens = [3, 3]
        total = sum(lens)
        g = torch.Generator(device="cpu").manual_seed(5)
        q = torch.randn(bsz, total, _NUM_HEADS, _HEAD_SIZE, generator=g)
        k = torch.randn(bsz, total, _NUM_HEADS, _HEAD_SIZE, generator=g)
        v = torch.randn(bsz, total, _NUM_HEADS, _HEAD_SIZE, generator=g)
        cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)

        out = _sdpa_forward(q, k, v, cu_seqlens=cu_seqlens)

        # Reference: compute each chunk independently and cat.
        ref = torch.cat(
            [
                _apply_sdpa(q[:, :3], k[:, :3], v[:, :3]),
                _apply_sdpa(q[:, 3:], k[:, 3:], v[:, 3:]),
            ],
            dim=1,
        )
        assert torch.allclose(out, ref, atol=1e-5)

    def test_cu_seqlens_output_shape(self):
        """Output shape must equal input shape regardless of chunking."""
        bsz = 1
        total = 9
        g = torch.Generator(device="cpu").manual_seed(6)
        q = torch.randn(bsz, total, _NUM_HEADS, _HEAD_SIZE, generator=g)
        k = q.clone()
        v = q.clone()
        cu_seqlens = torch.tensor([0, 4, 9], dtype=torch.int32)

        out = _sdpa_forward(q, k, v, cu_seqlens=cu_seqlens)

        assert out.shape == q.shape

    def test_single_image_cu_seqlens_matches_no_cu_seqlens(self):
        """One chunk is semantically identical to the no-cu_seqlens path."""
        g = torch.Generator(device="cpu").manual_seed(7)
        q = torch.randn(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE, generator=g)
        k = torch.randn(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE, generator=g)
        v = torch.randn(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE, generator=g)

        cu_seqlens = torch.tensor([0, _SEQ_LEN], dtype=torch.int32)
        out_chunked = _sdpa_forward(q, k, v, cu_seqlens=cu_seqlens)
        out_plain = _sdpa_forward(q, k, v, cu_seqlens=None)

        assert torch.allclose(out_chunked, out_plain, atol=1e-5)


# ---------------------------------------------------------------------------
# SpyreMMEncoderAttention.forward_oot
# ---------------------------------------------------------------------------


class TestForwardOot:
    def test_output_shape_3d_input(self):
        """3-D input (bsz, seq, hidden) must produce (bsz, seq, hidden) output."""
        attn = _make_attn()
        q, k, v = _qkv()

        out = attn.forward_oot(q, k, v)

        assert out.shape == (q.shape[0], q.shape[1], _NUM_HEADS * _HEAD_SIZE)

    def test_output_shape_4d_input(self):
        """4-D input (bsz, seq, heads, head_size) is NOT reshaped on output."""
        attn = _make_attn()
        g = torch.Generator(device="cpu").manual_seed(8)
        q = torch.randn(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE, generator=g)
        k = torch.randn(_BSZ, _SEQ_LEN, _NUM_KV_HEADS, _HEAD_SIZE, generator=g)
        v = torch.randn(_BSZ, _SEQ_LEN, _NUM_KV_HEADS, _HEAD_SIZE, generator=g)

        out = attn.forward_oot(q, k, v)

        assert out.shape == (_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE)

    def test_output_values_match_reference_sdpa(self):
        """forward_oot must produce the same result as a direct F.sdpa call."""
        attn = _make_attn()
        q, k, v = _qkv()

        out = attn.forward_oot(q, k, v)

        # Reference: manual reshape → F.sdpa → reshape back.
        q4 = q.view(_BSZ, _SEQ_LEN, _NUM_HEADS, _HEAD_SIZE)
        k4 = k.view(_BSZ, _SEQ_LEN, _NUM_KV_HEADS, _HEAD_SIZE)
        v4 = v.view(_BSZ, _SEQ_LEN, _NUM_KV_HEADS, _HEAD_SIZE)
        # F.sdpa expects (b, h, s, d).
        q4t = q4.permute(0, 2, 1, 3)
        k4t = k4.permute(0, 2, 1, 3)
        v4t = v4.permute(0, 2, 1, 3)
        ref = F.scaled_dot_product_attention(q4t, k4t, v4t, scale=attn.scale)
        ref = ref.permute(0, 2, 1, 3).reshape(_BSZ, _SEQ_LEN, -1)

        assert torch.allclose(out, ref, atol=1e-5)

    def test_cpu_offload_called_for_cpu_input(self):
        """For CPU tensors target_device == current device so convert is a no-op.

        The important invariant is that forward_oot returns a CPU tensor when
        the inputs live on CPU (target_device == query.device at entry).
        """
        attn = _make_attn()
        q, k, v = _qkv()

        out = attn.forward_oot(q, k, v)

        assert out.device.type == "cpu"

    def test_scale_is_honoured(self):
        """A custom scale must reach the attention computation."""
        attn_default = _make_attn()
        attn_half = _make_attn(scale=attn_default.scale * 0.5)
        q, k, v = _qkv()

        out_default = attn_default.forward_oot(q, k, v)
        out_half = attn_half.forward_oot(q, k, v)

        # Different scales → different outputs (unless the sequence is trivial).
        assert not torch.allclose(out_default, out_half, atol=1e-4)

    def test_gqa_path_does_not_raise(self):
        """num_heads > num_kv_heads should pass enable_gqa=True without error."""
        num_heads, num_kv_heads = 4, 2
        attn = _make_attn(num_heads=num_heads, num_kv_heads=num_kv_heads)
        q, k, v = _qkv(num_heads=num_heads, num_kv_heads=num_kv_heads)

        out = attn.forward_oot(q, k, v)

        assert out.shape == (_BSZ, _SEQ_LEN, num_heads * _HEAD_SIZE)

    def test_cu_seqlens_offloaded_to_cpu(self):
        """cu_seqlens on CPU must be accepted and produce the correct output shape."""
        attn = _make_attn()
        q, k, v = _qkv()
        cu_seqlens = torch.tensor([0, _SEQ_LEN], dtype=torch.int32)

        out = attn.forward_oot(q, k, v, cu_seqlens=cu_seqlens)

        assert out.shape == (_BSZ, _SEQ_LEN, _NUM_HEADS * _HEAD_SIZE)

    def test_cu_seqlens_two_images(self):
        """Two-image batch split by cu_seqlens must not corrupt cross-image attention."""
        attn = _make_attn()
        seq_a, seq_b = 3, 4
        total = seq_a + seq_b
        g = torch.Generator(device="cpu").manual_seed(9)
        hidden = _NUM_HEADS * _HEAD_SIZE
        q = torch.randn(_BSZ, total, hidden, generator=g)
        k = torch.randn(_BSZ, total, hidden, generator=g)
        v = torch.randn(_BSZ, total, hidden, generator=g)
        cu_seqlens = torch.tensor([0, seq_a, total], dtype=torch.int32)

        out = attn.forward_oot(q, k, v, cu_seqlens=cu_seqlens)

        # Reference: process each sub-sequence independently.
        attn_ref = _make_attn()
        ref_a = attn_ref.forward_oot(q[:, :seq_a], k[:, :seq_a], v[:, :seq_a])
        ref_b = attn_ref.forward_oot(q[:, seq_a:], k[:, seq_a:], v[:, seq_a:])
        ref = torch.cat([ref_a, ref_b], dim=1)

        assert torch.allclose(out, ref, atol=1e-5)
