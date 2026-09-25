# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `spyre_inference/multimodal/siglip.py`.


`patch_siglip_vision_embeddings` uses class-level patching for consistency with
the other granite vision modules (patch_blip2_qformer_attention, etc.).  It
replaces `SiglipVisionEmbeddings.forward` once and pins position buffers to CPU
per-instance.  The staleness tripwire, embedding-buffer CPU-pin, and
output-equivalence checks cover the three distinct failure modes:
a vLLM rename (silent no-op), a device leak, and a numeric regression.

Section 4 repeats the numeric check on the card and skips without a device.
"""

import sys

import pytest
import torch
import torch.nn as nn

siglip = pytest.importorskip("vllm.model_executor.models.siglip")

# SigLIP-SO400M/patch-14-384 dimensions (Granite Vision 4.1 default).
PATCH_SIZE = 14
IMAGE_SIZE = 336
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 576
HIDDEN_SIZE = 1152
IN_CHANNELS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_siglip_embeddings(device: torch.device | None = None) -> nn.Module:
    """Instantiate a real SiglipVisionEmbeddings with small deterministic weights.

    Uses the actual class so the patch target is a genuine instance.
    """
    if device is None:
        device = torch.device("cpu")
    from vllm.model_executor.models.siglip import SiglipVisionConfig

    config = SiglipVisionConfig(
        hidden_size=HIDDEN_SIZE,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_channels=IN_CHANNELS,
    )
    emb = siglip.SiglipVisionEmbeddings(config).to(torch.float16).to(device)
    rng = torch.Generator(device="cpu").manual_seed(0)
    for p in emb.parameters():
        p.data.copy_(torch.empty_like(p.data, device="cpu").normal_(std=0.02, generator=rng))
    return emb


def _make_model_with_siglip(device: torch.device | None = None) -> nn.Module:
    """Wrap a SiglipVisionEmbeddings inside a parent module to exercise the
    `model.modules()` traversal in `patch_siglip_vision_embeddings`."""
    model = nn.Module()
    model.embeddings = _make_siglip_embeddings(device)
    return model


# ---------------------------------------------------------------------------
# 1. Staleness tripwires
# ---------------------------------------------------------------------------


@pytest.mark.siglip
@pytest.mark.parametrize(
    "symbol",
    [
        "SiglipVisionEmbeddings",
    ],
)
def test_patch_target_symbols_still_exist(symbol):
    """Every symbol the patch reaches for must still exist in the vLLM module.
    The `try/except ImportError` path returns silently, so this is the only
    place a rename or removal is caught."""
    assert getattr(siglip, symbol, None) is not None, (
        f"vllm.model_executor.models.siglip.{symbol} is gone — the corresponding "
        "Spyre patch in multimodal/siglip.py is now a silent no-op and must be updated"
    )


@pytest.mark.siglip
def test_siglip_vision_embeddings_has_patch_embedding():
    """SiglipVisionEmbeddings must have a `patch_embedding` attribute — the Conv2d
    whose weight dtype the patched forward reads for dtype promotion."""
    cls = siglip.SiglipVisionEmbeddings
    assert hasattr(cls, "__init__"), "SiglipVisionEmbeddings must be a class"
    emb = _make_siglip_embeddings()
    assert hasattr(emb, "patch_embedding"), (
        "SiglipVisionEmbeddings instance has no `patch_embedding` — "
        "the patched forward reads `self.patch_embedding.weight.dtype`"
    )


@pytest.mark.siglip
def test_siglip_vision_embeddings_has_position_embedding_and_ids():
    """SiglipVisionEmbeddings must have `position_embedding` and `position_ids` —
    both are moved to CPU by the patch."""
    emb = _make_siglip_embeddings()
    assert hasattr(emb, "position_embedding"), (
        "SiglipVisionEmbeddings has no `position_embedding` — "
        "the patch calls `module.position_embedding.to('cpu')`"
    )
    assert hasattr(emb, "position_ids"), (
        "SiglipVisionEmbeddings has no `position_ids` — the patch re-registers it as a CPU buffer"
    )


# ---------------------------------------------------------------------------
# 2. Patch application: CPU pin and forward binding
# ---------------------------------------------------------------------------


@pytest.mark.siglip
def test_patch_moves_position_embedding_to_cpu():
    """`patch_siglip_vision_embeddings` must move `position_embedding` to CPU."""
    from spyre_inference.multimodal.siglip import patch_siglip_vision_embeddings

    model = _make_model_with_siglip()
    patch_siglip_vision_embeddings(model, torch.device("cpu"))

    assert model.embeddings.position_embedding.weight.device.type == "cpu", (
        "position_embedding.weight must be on CPU after patch"
    )


@pytest.mark.siglip
def test_patch_moves_position_ids_to_cpu():
    """`patch_siglip_vision_embeddings` must re-register `position_ids` as a CPU buffer."""
    from spyre_inference.multimodal.siglip import patch_siglip_vision_embeddings

    model = _make_model_with_siglip()
    patch_siglip_vision_embeddings(model, torch.device("cpu"))

    assert model.embeddings.position_ids.device.type == "cpu", (
        "position_ids must be on CPU after patch"
    )


@pytest.mark.siglip
def test_patch_binds_instance_forward():
    """`patch_siglip_vision_embeddings` must replace `SiglipVisionEmbeddings.forward`
    with a patched version marked `_spyre_patched=True` (class-level patch, consistent
    with the other granite vision module patches)."""
    from spyre_inference.multimodal.siglip import patch_siglip_vision_embeddings

    model = _make_model_with_siglip()
    patch_siglip_vision_embeddings(model, torch.device("cpu"))

    assert getattr(siglip.SiglipVisionEmbeddings.forward, "_spyre_patched", False), (
        "SiglipVisionEmbeddings.forward must carry _spyre_patched=True after patch"
    )


@pytest.mark.siglip
def test_apply_patches_all_siglip_instances_in_model():
    """The per-instance loop must pin position buffers to CPU on every
    SiglipVisionEmbeddings found in the model, even when there are multiple."""
    from spyre_inference.multimodal.siglip import patch_siglip_vision_embeddings

    model = nn.Module()
    model.emb1 = _make_siglip_embeddings()
    model.emb2 = _make_siglip_embeddings()

    patch_siglip_vision_embeddings(model, torch.device("cpu"))

    assert model.emb1.position_ids.device.type == "cpu", "emb1 position_ids must be on CPU"
    assert model.emb2.position_ids.device.type == "cpu", "emb2 position_ids must be on CPU"


# ---------------------------------------------------------------------------
# 3. Numeric equivalence on CPU
# ---------------------------------------------------------------------------


@pytest.mark.siglip
def test_patched_forward_output_matches_stock():
    """The patched forward must produce the same output as the unpatched forward
    on CPU — the only change is routing the position-embed add through CPU."""
    from spyre_inference.multimodal.siglip import patch_siglip_vision_embeddings

    rng = torch.Generator(device="cpu").manual_seed(1)
    pixel_values = torch.randn(
        1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float16, generator=rng
    )

    # Reference: stock forward on an unpatched instance.
    emb_stock = _make_siglip_embeddings()
    expected = emb_stock(pixel_values)

    # Patched: same weights, patched forward.
    emb_patched = _make_siglip_embeddings()
    model = nn.Module()
    model.embeddings = emb_patched
    patch_siglip_vision_embeddings(model, torch.device("cpu"))
    actual = emb_patched(pixel_values)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.siglip
def test_patched_forward_interpolate_pos_encoding_cpu_roundtrip():
    """The patched forward with interpolate_pos_encoding=True routes embeddings
    through CPU before calling interpolate_pos_encoding, whose
    position_embedding.weight and position_ids are pinned to CPU.

    Granite Vision 4.1 always uses the native 336px image size so this branch
    is not exercised in production.  The test verifies the CPU round-trip is
    present by monkeypatching interpolate_pos_encoding with a stub that asserts
    its input is on CPU — no non-native size needed, so the vLLM reshape bug
    (weight.shape[1] vs weight.shape[0]) is never hit.
    """
    import types

    from spyre_inference.multimodal.siglip import patch_siglip_vision_embeddings

    rng = torch.Generator(device="cpu").manual_seed(7)
    pixel_values = torch.randn(
        1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float16, generator=rng
    )

    emb = _make_siglip_embeddings()
    model = nn.Module()
    model.embeddings = emb
    patch_siglip_vision_embeddings(model, torch.device("cpu"))

    seen_devices = []

    def _stub_interpolate(self, embeddings, height, width):
        seen_devices.append(embeddings.device.type)
        return torch.zeros_like(embeddings)

    emb.interpolate_pos_encoding = types.MethodType(_stub_interpolate, emb)

    emb(pixel_values, interpolate_pos_encoding=True)

    assert seen_devices == ["cpu"], (
        f"interpolate_pos_encoding received embeddings on {seen_devices} — "
        "expected CPU round-trip before the call"
    )


# ---------------------------------------------------------------------------
# 4. On-card equivalence
# ---------------------------------------------------------------------------
# The on-card forward path (convert(embeddings_cpu + pos_emb, device=spyre))
# is covered end-to-end by tests/e2e/test_granite_vision.py.  Unit-testing it
# here is not possible: patch_siglip_vision_embeddings uses class-level
# patching, so the `device` closure is set once and shared across all
# instances.  In an eager unit-test context torch-spyre's dispatcher rejects
# the mixed-device add before the convert op runs.  The e2e test is the
# correct place for on-card validation.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5. patch_siglip_attention — padded_sdpa replaces mm_encoder_attention
# ---------------------------------------------------------------------------
# SigLIP so400m has head_dim=72 (not stick-aligned: 72 % 64 != 0).  The old
# SpyreMMEncoderAttention path has been removed; the fix is now in
# patch_siglip_attention: it zero-pads every head block of qkv_proj and
# out_proj from 72→128 and replaces SiglipAttention.forward to call
# padded_sdpa directly with scale fixed to the original head_dim.
#
# These tests use a fake SiglipAttention built with object.__new__ + direct
# attribute assignment (the same pattern the deleted test_mm_encoder_attention.py
# used) to stay CPU-only and avoid the vLLM platform init in __init__.

_ORIG_HEAD_DIM = 72  # SigLIP so400m native (not stick-aligned)
_PAD_HEAD_DIM = 128  # align_up(72, 64)
_NUM_HEADS_SIGLIP = 16
_HIDDEN_SIGLIP = _NUM_HEADS_SIGLIP * _ORIG_HEAD_DIM  # 1152


class _FakeLinear(nn.Module):
    """Minimal stand-in for a vLLM linear layer: forward returns (output, None)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_features, out_features, dtype=torch.float16))
        self.bias: nn.Parameter | None = (
            nn.Parameter(torch.zeros(out_features, dtype=torch.float16)) if bias else None
        )

    def forward(self, x: torch.Tensor):  # noqa: D401
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out, None


def _make_fake_siglip_attention(
    num_heads: int = _NUM_HEADS_SIGLIP,
    head_dim: int = _ORIG_HEAD_DIM,
    with_bias: bool = False,
) -> nn.Module:
    """Fake SiglipAttention with the attributes patch_siglip_attention reads.

    Uses object.__new__ to skip __init__ (which calls get_vit_attn_backend and
    requires vLLM platform context), then wires up the exact attributes the
    patch and the patched forward read:
      - head_dim, num_heads_per_partition
      - qkv_proj (weight shape: [hidden, 3*H*D], bias shape: [3*H*D])
      - out_proj  (weight shape: [H*D, hidden])
    Weight layout mirrors SpyreTransposedWeightMethod (Wᵀ storage).
    """
    from vllm.model_executor.models.siglip import SiglipAttention

    obj = object.__new__(SiglipAttention)

    nn.Module.__init__(obj)
    obj.head_dim = head_dim
    obj.num_heads_per_partition = num_heads
    hidden = num_heads * head_dim
    rng = torch.Generator(device="cpu").manual_seed(42)
    obj.qkv_proj = _FakeLinear(hidden, 3 * hidden, bias=with_bias)
    obj.out_proj = _FakeLinear(hidden, hidden)
    for p in obj.qkv_proj.parameters():
        p.data.normal_(std=0.02, generator=rng)
    for p in obj.out_proj.parameters():
        p.data.normal_(std=0.02, generator=rng)
    return obj


def _make_model_with_siglip_attention(
    num_heads: int = _NUM_HEADS_SIGLIP,
    head_dim: int = _ORIG_HEAD_DIM,
    with_bias: bool = False,
) -> nn.Module:
    """Wrap a fake SiglipAttention in a parent module for model.modules() traversal."""
    model = nn.Module()
    model.attn = _make_fake_siglip_attention(num_heads, head_dim, with_bias)
    return model


@pytest.mark.siglip
def test_patch_siglip_attention_target_symbol_exists():
    """SiglipAttention must still exist in the vLLM siglip module."""
    assert getattr(siglip, "SiglipAttention", None) is not None, (
        "vllm.model_executor.models.siglip.SiglipAttention is gone — "
        "patch_siglip_attention in multimodal/siglip.py is a silent no-op"
    )


@pytest.fixture(autouse=False)
def _restore_siglip_attention_forward():
    """Save and restore SiglipAttention.forward around each test.

    patch_siglip_attention sets a class-level _spyre_patched guard on
    SiglipAttention.forward. Without restoration, the first test that calls
    patch_siglip_attention poisons the guard for all subsequent tests in the
    same process — they hit the early-return and the weight-padding loop
    never runs.
    """
    original = siglip.SiglipAttention.forward
    yield
    siglip.SiglipAttention.forward = original


@pytest.mark.siglip
@pytest.mark.usefixtures("_restore_siglip_attention_forward")
def test_patch_siglip_attention_is_applied_and_idempotent():
    """`patch_siglip_attention` must set `_spyre_patched` on SiglipAttention.forward
    and a second call must be a no-op (same function in place)."""
    from spyre_inference.multimodal.siglip import patch_siglip_attention

    model = _make_model_with_siglip_attention()
    patch_siglip_attention(model)

    patched = siglip.SiglipAttention.forward
    assert getattr(patched, "_spyre_patched", False) is True, (
        "SiglipAttention.forward must carry _spyre_patched=True after patch"
    )

    patch_siglip_attention(model)
    assert siglip.SiglipAttention.forward is patched, "second call must be a no-op"


@pytest.mark.siglip
@pytest.mark.usefixtures("_restore_siglip_attention_forward")
def test_patch_siglip_attention_noop_for_aligned_head_dim():
    """`patch_siglip_attention` must skip models whose head_dim is already
    stick-aligned (head_dim % 64 == 0) — patching them would corrupt weights.

    Verified via weight shapes: if the patch ran it would widen qkv_proj and
    out_proj, so unchanged shapes prove it was skipped.  This does not depend
    on the class-level forward state (which may already be patched by an
    earlier test in the same process).
    """
    from spyre_inference.multimodal.siglip import patch_siglip_attention

    aligned_head_dim = 64
    num_heads = 4
    model = _make_model_with_siglip_attention(num_heads=num_heads, head_dim=aligned_head_dim)
    attn = model.attn

    orig_qkv_shape = attn.qkv_proj.weight.shape
    orig_out_shape = attn.out_proj.weight.shape

    patch_siglip_attention(model)

    assert attn.qkv_proj.weight.shape == orig_qkv_shape, (
        "patch must not widen qkv_proj.weight for an already stick-aligned head_dim"
    )
    assert attn.out_proj.weight.shape == orig_out_shape, (
        "patch must not widen out_proj.weight for an already stick-aligned head_dim"
    )


@pytest.mark.siglip
@pytest.mark.usefixtures("_restore_siglip_attention_forward")
def test_patch_siglip_attention_pads_qkv_weight_shape():
    """After patching, qkv_proj.weight must be widened from [H, 3*H*orig_d] to [H, 3*H*pad_d]."""
    from spyre_inference.multimodal.siglip import patch_siglip_attention

    model = _make_model_with_siglip_attention()
    attn = model.attn

    expected_orig = (_HIDDEN_SIGLIP, 3 * _NUM_HEADS_SIGLIP * _ORIG_HEAD_DIM)
    assert attn.qkv_proj.weight.shape == torch.Size(expected_orig)

    patch_siglip_attention(model)

    expected_pad = (_HIDDEN_SIGLIP, 3 * _NUM_HEADS_SIGLIP * _PAD_HEAD_DIM)
    assert attn.qkv_proj.weight.shape == torch.Size(expected_pad), (
        f"qkv_proj.weight shape after padding: {attn.qkv_proj.weight.shape}, "
        f"expected {expected_pad}"
    )


@pytest.mark.siglip
@pytest.mark.usefixtures("_restore_siglip_attention_forward")
def test_patch_siglip_attention_pads_out_weight_shape():
    """After patching, out_proj.weight must be widened from [H*orig_d, H] to [H*pad_d, H]."""
    from spyre_inference.multimodal.siglip import patch_siglip_attention

    model = _make_model_with_siglip_attention()
    attn = model.attn

    patch_siglip_attention(model)

    expected_pad = (_NUM_HEADS_SIGLIP * _PAD_HEAD_DIM, _HIDDEN_SIGLIP)
    assert attn.out_proj.weight.shape == torch.Size(expected_pad), (
        f"out_proj.weight shape after padding: {attn.out_proj.weight.shape}, "
        f"expected {expected_pad}"
    )


@pytest.mark.siglip
@pytest.mark.usefixtures("_restore_siglip_attention_forward")
def test_patch_siglip_attention_qkv_weight_preserves_orig_values():
    """The original weights must appear in the correct interleaved positions after padding.

    For each block b in {Q,K,V} and head h, the orig_d columns must be copied to
    offset b*H*pad_d + h*pad_d; the remaining pad_d-orig_d columns must be zero.
    """
    from spyre_inference.multimodal.siglip import patch_siglip_attention

    model = _make_model_with_siglip_attention()
    attn = model.attn
    orig_w = attn.qkv_proj.weight.data.clone()

    patch_siglip_attention(model)
    padded_w = attn.qkv_proj.weight.data

    n, d, pd = _NUM_HEADS_SIGLIP, _ORIG_HEAD_DIM, _PAD_HEAD_DIM
    for b in range(3):
        for h in range(n):
            src_start = b * n * d + h * d
            dst_start = b * n * pd + h * pd
            assert torch.equal(
                padded_w[:, dst_start : dst_start + d],
                orig_w[:, src_start : src_start + d],
            ), f"block {b} head {h}: original values not preserved"
            assert not padded_w[:, dst_start + d : dst_start + pd].any(), (
                f"block {b} head {h}: padding region not zero"
            )


@pytest.mark.siglip
@pytest.mark.usefixtures("_restore_siglip_attention_forward")
def test_patch_siglip_attention_with_bias():
    """qkv_proj bias must also be padded correctly when present."""
    from spyre_inference.multimodal.siglip import patch_siglip_attention

    model = _make_model_with_siglip_attention(with_bias=True)
    attn = model.attn

    orig_bias = attn.qkv_proj.bias.data.clone()
    patch_siglip_attention(model)

    padded_bias = attn.qkv_proj.bias.data
    assert padded_bias.shape == torch.Size([3 * _NUM_HEADS_SIGLIP * _PAD_HEAD_DIM])

    n, d, pd = _NUM_HEADS_SIGLIP, _ORIG_HEAD_DIM, _PAD_HEAD_DIM
    for b in range(3):
        for h in range(n):
            src = b * n * d + h * d
            dst = b * n * pd + h * pd
            assert torch.equal(padded_bias[dst : dst + d], orig_bias[src : src + d])
            assert not padded_bias[dst + d : dst + pd].any()


@pytest.mark.siglip
@pytest.mark.usefixtures("_restore_siglip_attention_forward")
def test_patch_siglip_attention_marks_instance_padded():
    """Each patched SiglipAttention instance must carry `_spyre_head_dim_padded=True`
    so a second `patch_siglip_attention` call does not re-pad already-widened weights."""
    from spyre_inference.multimodal.siglip import patch_siglip_attention

    model = _make_model_with_siglip_attention()
    patch_siglip_attention(model)

    assert getattr(model.attn, "_spyre_head_dim_padded", False) is True


@pytest.mark.siglip
@pytest.mark.usefixtures("_restore_siglip_attention_forward")
def test_patch_siglip_attention_forward_output_shape():
    """The patched forward must return `(output, None)` where output is `[B, S, hidden]`."""
    from spyre_inference.multimodal.siglip import patch_siglip_attention

    model = _make_model_with_siglip_attention()
    patch_siglip_attention(model)

    bsz, seq = 1, 64  # seq must be stick-aligned for padded_sdpa to not pad seq dim
    hidden = _HIDDEN_SIGLIP
    rng = torch.Generator(device="cpu").manual_seed(5)
    hidden_states = torch.randn(bsz, seq, hidden, dtype=torch.float16, generator=rng)

    attn_out, attn_weights = model.attn(hidden_states)

    assert attn_out.shape == (bsz, seq, hidden), (
        f"patched forward output shape {attn_out.shape} != expected ({bsz}, {seq}, {hidden})"
    )
    assert attn_weights is None


@pytest.mark.siglip
@pytest.mark.usefixtures("_restore_siglip_attention_forward")
def test_patch_siglip_attention_scale_uses_orig_head_dim():
    """The attention scale must be fixed to orig_head_dim**-0.5, not pad_head_dim**-0.5.

    A doubled scale (1/sqrt(128) instead of 1/sqrt(72)) would flatten the softmax
    and corrupt the encoder's attention pattern.  Verified indirectly: running the
    patched forward with a manually-computed reference at the correct scale.
    """
    from spyre_inference.custom_ops.vit_attn import _full_attend_mask
    from spyre_inference.multimodal.siglip import (
        _pad_out_weight,
        _pad_qkv_weight,
        patch_siglip_attention,
    )
    from spyre_inference.multimodal.utils import padded_sdpa

    model = _make_model_with_siglip_attention()
    attn = model.attn

    # Save pre-patch weights for the reference computation.
    orig_qkv_w = attn.qkv_proj.weight.data.clone()
    orig_out_w = attn.out_proj.weight.data.clone()

    patch_siglip_attention(model)

    bsz, seq = 1, 64
    rng = torch.Generator(device="cpu").manual_seed(6)
    hidden_states = torch.randn(bsz, seq, _HIDDEN_SIGLIP, dtype=torch.float16, generator=rng)

    actual, _ = model.attn(hidden_states)

    # Reference: pad weights manually, run padded_sdpa with scale=orig**-0.5.
    pad_qkv_w = _pad_qkv_weight(orig_qkv_w, _NUM_HEADS_SIGLIP, _ORIG_HEAD_DIM, _PAD_HEAD_DIM)
    pad_out_w = _pad_out_weight(orig_out_w, _NUM_HEADS_SIGLIP, _ORIG_HEAD_DIM, _PAD_HEAD_DIM)

    qkv = hidden_states @ pad_qkv_w
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(bsz, seq, _NUM_HEADS_SIGLIP, _PAD_HEAD_DIM).transpose(1, 2)
    k = k.view(bsz, seq, _NUM_HEADS_SIGLIP, _PAD_HEAD_DIM).transpose(1, 2)
    v = v.view(bsz, seq, _NUM_HEADS_SIGLIP, _PAD_HEAD_DIM).transpose(1, 2)
    ref_out = padded_sdpa(q, k, v, _full_attend_mask(seq), scale=_ORIG_HEAD_DIM**-0.5)
    ref_out = ref_out.transpose(1, 2).reshape(bsz, seq, -1)
    expected, _ = ref_out @ pad_out_w, None

    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
