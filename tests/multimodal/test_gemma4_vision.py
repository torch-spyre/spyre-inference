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

"""Tests for `spyre_inference/multimodal/gemma4_vision.py`.

The head-dim padding, its channel repacking, and the rope built on top of it can be
wrong while still producing plausible numbers, so the load-bearing test is the
equivalence check against stock `apply_multidimensional_rope`.

The patches are `getattr`-guarded, so a transformers rename would silently no-op one
-- hence the staleness tripwires. All host-side math; no card needed.
"""

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
configuration_gemma4 = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")

pytestmark = [pytest.mark.gemma4_vision]

# 26B-A4B's real vision head_dim; 72 is neither a 64-multiple nor evenly halvable
# onto the stick, which is the whole reason the padding exists.
ORIG_HEAD_DIM = 72
PADDED_HEAD_DIM = 128
NUM_HEADS = 4
NUM_PATCHES = 50


def _vision_config(head_dim: int = ORIG_HEAD_DIM, num_heads: int = NUM_HEADS):
    return configuration_gemma4.Gemma4VisionConfig(
        hidden_size=num_heads * head_dim,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=head_dim,
    )


def _position_ids(num_patches: int = NUM_PATCHES) -> torch.Tensor:
    """`[1, num_patches, 2]` (x, y) patch coordinates, as the encoder feeds them."""
    side = 8
    xs = torch.arange(side).repeat(side)[:num_patches]
    ys = torch.arange(side).repeat_interleave(side)[:num_patches]
    return torch.stack([xs, ys], dim=-1).unsqueeze(0)


def _pad_activation_quarters(x: torch.Tensor, orig: int, padded: int) -> torch.Tensor:
    """The `_pad_qk_linear` channel remap, applied to activations instead of weights.

    Kept in the test rather than imported: it mirrors what the padded q_proj weight
    does to its output, so writing it out independently is what makes the rope
    equivalence check below meaningful.
    """
    quarter = orig // 4
    half = padded // 2
    out = torch.zeros((*x.shape[:-1], padded), dtype=x.dtype)
    out[..., :quarter] = x[..., :quarter]
    out[..., quarter : 2 * quarter] = x[..., 2 * quarter : 3 * quarter]
    out[..., half : half + quarter] = x[..., quarter : 2 * quarter]
    out[..., half + quarter : half + 2 * quarter] = x[..., 3 * quarter :]
    return out


def _unpad_activation_quarters(padded: torch.Tensor, orig: int, padded_dim: int) -> torch.Tensor:
    quarter = orig // 4
    half = padded_dim // 2
    out = torch.zeros((*padded.shape[:-1], orig), dtype=padded.dtype)
    out[..., :quarter] = padded[..., :quarter]
    out[..., 2 * quarter : 3 * quarter] = padded[..., quarter : 2 * quarter]
    out[..., quarter : 2 * quarter] = padded[..., half : half + quarter]
    out[..., 3 * quarter :] = padded[..., half + quarter : half + 2 * quarter]
    return out


# ---------------------------------------------------------------------------
# Staleness tripwires: the upstream symbols the patches reach for must exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "Gemma4RMSNorm",
        "Gemma4VisionEncoder",
        "Gemma4VisionPatchEmbedder",
        "apply_multidimensional_rope",
        "Gemma4VisionRotaryEmbedding",
    ],
)
def test_patched_upstream_symbols_still_exist(name):
    """Every patch is `getattr`-guarded, so a rename would silently no-op it."""
    assert getattr(modeling_gemma4, name, None) is not None, (
        f"transformers.models.gemma4.modeling_gemma4.{name} is gone; "
        "spyre_inference/multimodal/gemma4_vision.py needs updating."
    )


# ---------------------------------------------------------------------------
# Head-dim padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("orig", "expected"),
    [(64, 128), (72, 128), (128, 128), (129, 256), (256, 256)],
)
def test_padded_head_dim_rounds_to_a_double_stick(orig, expected):
    """Rope needs each half on a whole stick, so head_dim pads to 2*64, not 64."""
    from spyre_inference.multimodal.gemma4_vision import _padded_head_dim

    assert _padded_head_dim(orig) == expected


def test_pad_qk_linear_preserves_every_original_channel():
    """The quarter-interleave must be a permutation of the real channels into the
    padded layout -- no value dropped, no value duplicated."""
    from spyre_inference.multimodal.gemma4_vision import _pad_qk_linear

    torch.manual_seed(0)
    in_features = 32
    linear = torch.nn.Linear(in_features, NUM_HEADS * ORIG_HEAD_DIM, bias=False)
    linear.weight.data.normal_()
    proj = type("_Proj", (), {"linear": linear})()

    padded = _pad_qk_linear(proj, NUM_HEADS, ORIG_HEAD_DIM, PADDED_HEAD_DIM)

    assert padded.weight.shape == (NUM_HEADS * PADDED_HEAD_DIM, in_features)
    orig_w = linear.weight.detach().view(NUM_HEADS, ORIG_HEAD_DIM, in_features)
    new_w = padded.weight.detach().view(NUM_HEADS, PADDED_HEAD_DIM, in_features)
    quarter = ORIG_HEAD_DIM // 4
    half = PADDED_HEAD_DIM // 2
    # X-axis quarters land in the first slot of each half, Y-axis in the second.
    torch.testing.assert_close(new_w[:, :quarter], orig_w[:, :quarter])
    torch.testing.assert_close(
        new_w[:, quarter : 2 * quarter], orig_w[:, 2 * quarter : 3 * quarter]
    )
    torch.testing.assert_close(new_w[:, half : half + quarter], orig_w[:, quarter : 2 * quarter])
    torch.testing.assert_close(
        new_w[:, half + quarter : half + 2 * quarter], orig_w[:, 3 * quarter :]
    )
    # Everything else is the zero padding.
    assert torch.all(new_w[:, 2 * quarter : half] == 0)
    assert torch.all(new_w[:, half + 2 * quarter :] == 0)


def test_pad_norm_weight_fills_padding_lanes_with_ones():
    """A norm weight is multiplicative, so padding lanes must be 1.0 (not 0.0) --
    a zero there would be harmless today but wrong if the lane ever carried data."""
    from spyre_inference.multimodal.gemma4_vision import _pad_norm_weight

    norm = modeling_gemma4.Gemma4RMSNorm(dim=ORIG_HEAD_DIM, eps=1e-6, with_scale=True)
    norm.weight.data.normal_()

    padded = _pad_norm_weight(norm, ORIG_HEAD_DIM, PADDED_HEAD_DIM)

    assert padded.shape == (PADDED_HEAD_DIM,)
    quarter = ORIG_HEAD_DIM // 4
    half = PADDED_HEAD_DIM // 2
    assert torch.all(padded[2 * quarter : half] == 1.0)
    assert torch.all(padded[half + 2 * quarter :] == 1.0)


def test_pad_mlp_is_numerically_transparent():
    """26B-A4B's vision MLP is 4304 wide (not a 64-multiple), so it is zero-extended
    onto the stick. The zero lanes must contribute nothing: `gelu(0) * 0 == 0`, and
    the matching zero down-projection columns ignore whatever lands there."""
    from spyre_inference.multimodal.gemma4_vision import _pad_mlp

    torch.manual_seed(0)
    orig, padded = 200, 256
    config = configuration_gemma4.Gemma4VisionConfig(
        hidden_size=128,
        intermediate_size=orig,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        num_hidden_layers=1,
        use_clipped_linears=False,
    )
    layer = modeling_gemma4.Gemma4VisionEncoderLayer(config=config, layer_idx=0).to(torch.float32)
    x = torch.randn(1, 5, 128)

    want = layer.mlp(x)
    _pad_mlp(layer, orig, padded)
    got = layer.mlp(x)

    assert layer.mlp.gate_proj.linear.out_features == padded
    assert layer.mlp.up_proj.linear.out_features == padded
    assert layer.mlp.down_proj.linear.in_features == padded
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    # Idempotent: apply() runs per load, and re-padding would double the width.
    _pad_mlp(layer, orig, padded)
    assert layer.mlp.gate_proj.linear.out_features == padded


def test_pad_mlp_is_a_noop_when_already_stick_aligned():
    """E2B's 3072-wide MLP needs no padding; don't rebuild its linears."""
    from spyre_inference.multimodal.gemma4_vision import _pad_mlp

    config = configuration_gemma4.Gemma4VisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        num_hidden_layers=1,
        use_clipped_linears=False,
    )
    layer = modeling_gemma4.Gemma4VisionEncoderLayer(config=config, layer_idx=0)
    before = layer.mlp.gate_proj.linear

    _pad_mlp(layer, 256, 256)

    assert layer.mlp.gate_proj.linear is before


# ---------------------------------------------------------------------------
# RMSNorm: no fp32 promotion (torch-spyre gap), padding-corrected denominator
# ---------------------------------------------------------------------------


def test_padded_rms_norm_does_not_promote_via_an_fp32_weight():
    """These weights are fp32, so an unguarded `x * weight` promotes the activation --
    which torch-spyre cannot lower. Stock hid it behind a trailing `.type_as`."""
    from spyre_inference.multimodal.gemma4_vision import _padded_rms_norm

    x = torch.randn(1, 4, NUM_HEADS, PADDED_HEAD_DIM, dtype=torch.float16)
    fp32_weight = torch.randn(PADDED_HEAD_DIM, dtype=torch.float32)
    out = _padded_rms_norm(x, fp32_weight, 1e-6, ORIG_HEAD_DIM)
    assert out.dtype == torch.float16


def test_patched_rms_norm_does_not_promote_via_its_fp32_weight():
    """Same hazard through the patched module: transformers builds these weights at
    the default (fp32) dtype, and the vision tower feeds fp16 activations."""
    from spyre_inference.multimodal import gemma4_vision

    norm = modeling_gemma4.Gemma4RMSNorm(dim=16, eps=1e-6, with_scale=True)
    assert norm.weight.dtype == torch.float32, "premise of this test changed upstream"

    gemma4_vision.patch_rms_norm()
    out = norm(torch.randn(2, 3, 16, dtype=torch.float16))
    assert out.dtype == torch.float16


def test_padded_rms_norm_ignores_zero_padding_lanes():
    """Normalizing the padded activation must match normalizing the real channels
    alone -- that is what the `padded/orig` variance rescale buys."""
    from spyre_inference.multimodal.gemma4_vision import _padded_rms_norm

    torch.manual_seed(0)
    real = torch.randn(1, 4, NUM_HEADS, ORIG_HEAD_DIM, dtype=torch.float32)
    padded = _pad_activation_quarters(real, ORIG_HEAD_DIM, PADDED_HEAD_DIM)

    got = _padded_rms_norm(padded, None, 1e-6, ORIG_HEAD_DIM)
    got_real_channels = _unpad_activation_quarters(got, ORIG_HEAD_DIM, PADDED_HEAD_DIM)

    variance = (real * real).mean(-1, keepdim=True)
    want = real * torch.rsqrt(variance + 1e-6)

    torch.testing.assert_close(got_real_channels, want, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# The load-bearing check: padded two-axis rope == the transformers reference
# ---------------------------------------------------------------------------


@pytest.mark.rotary
def test_padded_rope_matches_transformers_reference():
    """Rope over the padded layout must reproduce stock `apply_multidimensional_rope`
    on the real channels. Catches an off-by-a-quarter interleave, a swapped axis, or a
    sin sign error -- each of which still produces plausible magnitudes.
    """
    from spyre_inference.multimodal.gemma4_vision import _apply_rope, _gemma4_rope_cos_sin

    torch.manual_seed(0)
    config = _vision_config()
    position_ids = _position_ids()
    x = torch.randn(1, NUM_PATCHES, NUM_HEADS, ORIG_HEAD_DIM, dtype=torch.float32)

    rope = modeling_gemma4.Gemma4VisionRotaryEmbedding(config)
    cos, sin = rope(x, position_ids)
    want = modeling_gemma4.apply_multidimensional_rope(x, cos, sin, position_ids, unsqueeze_dim=2)

    x_padded = _pad_activation_quarters(x, ORIG_HEAD_DIM, PADDED_HEAD_DIM)
    cos_full, sin_full = _gemma4_rope_cos_sin(
        rope.inv_freq, position_ids, PADDED_HEAD_DIM, torch.float32
    )
    got_padded = _apply_rope(x_padded, cos_full, sin_full)
    got = _unpad_activation_quarters(got_padded, ORIG_HEAD_DIM, PADDED_HEAD_DIM)

    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.rotary
def test_rope_cos_sin_padding_lanes_are_the_identity_rotation():
    """Padding lanes must rotate by nothing (cos=1, sin=0) so a padded channel that
    is not exactly zero still cannot leak into a real one."""
    from spyre_inference.multimodal.gemma4_vision import _gemma4_rope_cos_sin

    config = _vision_config()
    rope = modeling_gemma4.Gemma4VisionRotaryEmbedding(config)
    cos, sin = _gemma4_rope_cos_sin(rope.inv_freq, _position_ids(), PADDED_HEAD_DIM, torch.float32)

    assert cos.shape == (1, NUM_PATCHES, 1, PADDED_HEAD_DIM)
    assert sin.shape == (1, NUM_PATCHES, 1, PADDED_HEAD_DIM)
    quarter = ORIG_HEAD_DIM // 4
    half = PADDED_HEAD_DIM // 2
    for lanes in (slice(2 * quarter, half), slice(half + 2 * quarter, PADDED_HEAD_DIM)):
        assert torch.all(cos[..., lanes] == 1.0)
        assert torch.all(sin[..., lanes] == 0.0)


def test_apply_rope_swaps_halves_and_keeps_each_half_stick_aligned():
    """cos=0, sin=1 isolates the swap term, so this pins the half-swap itself."""
    from spyre_inference.multimodal.gemma4_vision import _apply_rope

    head_dim = 8
    x = torch.arange(head_dim, dtype=torch.float32).view(1, 1, 1, head_dim)
    cos = torch.zeros(1, 1, 1, head_dim)
    sin = torch.ones(1, 1, 1, head_dim)
    # cos=0, sin=1 isolates the swap term.
    got = _apply_rope(x, cos, sin)
    want = torch.cat([x[..., head_dim // 2 :], x[..., : head_dim // 2]], dim=-1)
    torch.testing.assert_close(got, want)


# ---------------------------------------------------------------------------
# Patch application / dispatch
# ---------------------------------------------------------------------------


def test_patches_are_idempotent():
    """`apply()` runs per model load, so every patch must be re-entrant."""
    from spyre_inference.multimodal import gemma4_vision

    gemma4_vision.patch_rms_norm()
    first = modeling_gemma4.Gemma4RMSNorm.forward
    gemma4_vision.patch_rms_norm()
    assert modeling_gemma4.Gemma4RMSNorm.forward is first

    gemma4_vision.patch_vision_encoder()
    first_encoder = modeling_gemma4.Gemma4VisionEncoder.forward
    gemma4_vision.patch_vision_encoder()
    assert modeling_gemma4.Gemma4VisionEncoder.forward is first_encoder


def test_patched_rms_norm_stays_in_input_dtype_and_matches_reference():
    """The patched forward drops stock's fp32 round trip; in fp32 (where the
    promotion is a no-op) it must still agree with stock to tight tolerance."""
    from spyre_inference.multimodal import gemma4_vision

    torch.manual_seed(0)
    dim = 16
    norm = modeling_gemma4.Gemma4RMSNorm(dim=dim, eps=1e-6, with_scale=True)
    norm.weight.data.normal_()
    x = torch.randn(2, 3, dim, dtype=torch.float32)

    mean_squared = x.pow(2).mean(-1, keepdim=True) + norm.eps
    want = x * torch.pow(mean_squared, -0.5) * norm.weight

    gemma4_vision.patch_rms_norm()
    got = norm(x)

    assert got.dtype == torch.float32
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_dispatch_routes_gemma4_tower_away_from_pixtral(monkeypatch):
    """`apply_multimodal_patches` keys on the tower's class name, since Pixtral and
    Gemma 4 both hang their tower off a `vision_tower` attribute."""
    from spyre_inference import multimodal

    called = []
    monkeypatch.setattr(
        multimodal.gemma4_vision, "apply", lambda *a: called.append("gemma4"), raising=True
    )
    monkeypatch.setattr(
        multimodal.pixtral, "apply", lambda *a: called.append("pixtral"), raising=True
    )

    # The name is the dispatch key, so it has to match upstream's exactly.
    class Gemma4VisionModel(torch.nn.Module):
        pass

    class _PixtralTower(torch.nn.Module):
        pass

    gemma4_model = torch.nn.Module()
    gemma4_model.vision_tower = Gemma4VisionModel()
    multimodal.apply_multimodal_patches(gemma4_model, torch.device("cpu"))
    assert called == ["gemma4"]

    called.clear()
    pixtral_model = torch.nn.Module()
    pixtral_model.vision_tower = _PixtralTower()
    multimodal.apply_multimodal_patches(pixtral_model, torch.device("cpu"))
    assert called == ["pixtral"]

    called.clear()
    multimodal.apply_multimodal_patches(torch.nn.Module(), torch.device("cpu"))
    assert called == [], "a text-only model must get no vision patches"


def test_accelerator_memory_info_falls_back_to_host_ram(monkeypatch):
    """Spyre registers no accelerator memory-info hook, so the native call raises;
    Gemma 4's encoder chunking needs a real number back."""
    from spyre_inference.multimodal import gemma4_vision

    def _unimplemented(*args, **kwargs):
        raise NotImplementedError("getMemoryInfo is not implemented for this allocator yet.")

    monkeypatch.setattr(torch.accelerator, "get_memory_info", _unimplemented, raising=True)
    gemma4_vision.patch_accelerator_memory_info()

    free, total = torch.accelerator.get_memory_info()
    assert free > 0
    assert total >= free


def test_accelerator_memory_info_passes_through_when_native_call_works(monkeypatch):
    from spyre_inference.multimodal import gemma4_vision

    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *a, **k: (123, 456))
    gemma4_vision.patch_accelerator_memory_info()

    assert torch.accelerator.get_memory_info() == (123, 456)


# ---------------------------------------------------------------------------
# Correctness regressions
# ---------------------------------------------------------------------------


def test_fp32_inv_freq_is_recomputed_not_read_off_a_downcast_buffer():
    """`model.to(bfloat16)` downcasts `inv_freq`, and these frequencies span a range
    where bf16 costs real precision. Recomputing must recover full fp32."""
    from spyre_inference.multimodal.gemma4_vision import _fp32_inv_freq

    config = _vision_config()
    rope = modeling_gemma4.Gemma4VisionRotaryEmbedding(config)
    exact = rope.inv_freq.float().clone()

    # Simulate the bf16 cast the real model applies to the whole tower.
    rope.inv_freq = rope.inv_freq.to(torch.bfloat16)
    assert not torch.allclose(rope.inv_freq.float(), exact), "premise: bf16 loses bits"

    recovered = _fp32_inv_freq(rope, config)
    assert recovered.dtype == torch.float32
    torch.testing.assert_close(recovered, exact, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# Guards on assumptions the padding makes about the projections it rewrites
# ---------------------------------------------------------------------------


def _clipped_proj(output_min: float, output_max: float):
    """A `Gemma4ClippableLinear` with real (finite) output clipping bounds."""
    config = configuration_gemma4.Gemma4VisionConfig(
        hidden_size=NUM_HEADS * ORIG_HEAD_DIM,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        head_dim=ORIG_HEAD_DIM,
        use_clipped_linears=True,
    )
    proj = modeling_gemma4.Gemma4ClippableLinear(config, 32, NUM_HEADS * ORIG_HEAD_DIM)
    proj.output_min = torch.nn.Buffer(torch.tensor(output_min))
    proj.output_max = torch.nn.Buffer(torch.tensor(output_max))
    return proj


@pytest.mark.parametrize(
    ("lo", "hi"),
    [(-6.0, 6.0), (0.0, 6.0), (-6.0, 0.0), (-float("inf"), float("inf"))],
)
def test_clipping_that_includes_zero_is_accepted(lo, hi):
    """A range containing zero keeps the padding lanes at zero; a zero bound is fine,
    since clamp(0) is still 0."""
    from spyre_inference.multimodal.gemma4_vision import _assert_clipping_preserves_zero

    _assert_clipping_preserves_zero(_clipped_proj(lo, hi), "test proj")


@pytest.mark.parametrize(("lo", "hi"), [(0.5, 6.0), (-6.0, -0.5)])
def test_clipping_that_excludes_zero_is_rejected(lo, hi):
    """A range excluding zero maps the padding lanes to a nonzero bound, invalidating
    `_padded_rms_norm`'s variance correction."""
    from spyre_inference.multimodal.gemma4_vision import _assert_clipping_preserves_zero

    with pytest.raises(NotImplementedError, match="must include zero"):
        _assert_clipping_preserves_zero(_clipped_proj(lo, hi), "test proj")


def test_unclipped_projection_skips_the_clipping_check():
    """Unclipped is the common case, so the check must be inert and must not touch
    bounds buffers that do not exist."""
    from spyre_inference.multimodal.gemma4_vision import _assert_clipping_preserves_zero

    config = _vision_config()
    assert not config.use_clipped_linears, "premise: this config is unclipped"
    proj = modeling_gemma4.Gemma4ClippableLinear(config, 32, NUM_HEADS * ORIG_HEAD_DIM)
    assert not hasattr(proj, "output_min")

    _assert_clipping_preserves_zero(proj, "test proj")  # must not raise


def test_quantized_projection_is_rejected_rather_than_dequantized():
    """Rebuilding as a plain nn.Linear drops the quantization method, so a quantized
    projection must fail rather than be silently dequantized."""
    from spyre_inference.multimodal.gemma4_vision import _as_plain_linear

    class _FakeQuantMethod:
        pass

    class _FakeQuantLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(32, 8))
            self.bias = None
            self.quant_method = _FakeQuantMethod()

    with pytest.raises(NotImplementedError, match="quantization method"):
        _as_plain_linear(_FakeQuantLinear())


def test_unquantized_vllm_linear_is_bounced_to_a_plain_linear():
    """A vLLM linear stores `Wᵀ`, so the bounce must transpose back to `[out, in]`."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from spyre_inference.multimodal.gemma4_vision import _as_plain_linear

    class _FakeReplicatedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.weight = torch.nn.Parameter(torch.randn(8, 32))  # [in, out]
            self.bias = None
            self.quant_method = UnquantizedLinearMethod()

    src = _FakeReplicatedLinear()
    plain = _as_plain_linear(src)

    assert isinstance(plain, torch.nn.Linear)
    assert plain.weight.shape == (32, 8)  # [out, in]
    torch.testing.assert_close(plain.weight, src.weight.t().contiguous())


# ---------------------------------------------------------------------------
# On-card: the padding must not be computed on device-resident weights
# ---------------------------------------------------------------------------


def test_padding_on_device_weights_matches_padding_on_host():
    """Padding a layer already on Spyre must give the same weights as padding it on the
    host and moving afterwards.

    This is the regression `_host` exists for: composed on device-resident weights the
    helpers' strided slice-assignments lower incorrectly, producing finite but wrong
    padded weights and costing most of the encoder's accuracy. Individual assignments are
    fine, so only the composite is affected and nothing raises -- and vLLM moves the
    model before our patches run, so the bad ordering is the one that happens in
    production.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    import copy

    from spyre_inference.multimodal import gemma4_vision as gv

    # Real 26B-A4B vision proportions: head_dim 72 -> 128, intermediate 4304 -> 4352.
    config = configuration_gemma4.Gemma4VisionConfig(
        hidden_size=NUM_HEADS * 72,
        intermediate_size=4304,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        head_dim=72,
        num_hidden_layers=1,
        use_clipped_linears=False,
    )
    orig_hd = config.head_dim
    padded_hd = gv._padded_head_dim(orig_hd)
    padded_inter = 4352

    torch.manual_seed(0)
    layer = modeling_gemma4.Gemma4VisionEncoderLayer(config=config, layer_idx=0)
    layer = layer.to(torch.bfloat16).eval()

    # Reference: pad on the host, then move.
    on_host = copy.deepcopy(layer)
    gv._prepare_attention(on_host.self_attn, NUM_HEADS, orig_hd, padded_hd)
    gv._pad_mlp(on_host, config.intermediate_size, padded_inter)
    on_host = on_host.to(torch.device("spyre"))

    # Production ordering: move first, then pad.
    on_device = copy.deepcopy(layer).to(torch.device("spyre"))
    gv._prepare_attention(on_device.self_attn, NUM_HEADS, orig_hd, padded_hd)
    gv._pad_mlp(on_device, config.intermediate_size, padded_inter)

    checked = 0
    for name, want in on_host.named_parameters():
        got = dict(on_device.named_parameters())[name]
        torch.testing.assert_close(
            got.detach().to("cpu").float(),
            want.detach().to("cpu").float(),
            msg=lambda m, name=name: f"{name} differs when padded on device:\n{m}",
        )
        checked += 1
    assert checked, "no parameters compared -- the layer walk found nothing"


# ---------------------------------------------------------------------------
# Pooling: the average runs on device as one bf16 matmul
# ---------------------------------------------------------------------------


def _pool_setup(num_patches=90, hidden=32, k=3, pad_last=0):
    """A k-divisible patch grid, which is what the processor emits."""
    length = num_patches // (k * k)
    side_x = 3 * 2
    side_y = num_patches // side_x
    assert side_y % k == 0 and (side_x // k) * (side_y // k) == length
    xs = torch.arange(side_x).repeat(side_y)[:num_patches]
    ys = torch.arange(side_y).repeat_interleave(side_x)[:num_patches]
    pos = torch.stack([xs, ys], dim=-1).unsqueeze(0)
    if pad_last:
        pos[:, -pad_last:] = -1
    torch.manual_seed(0)
    hidden_states = torch.randn(1, num_patches, hidden, dtype=torch.float32)
    return pos, (pos == -1).all(dim=-1), hidden_states, length, k


@pytest.mark.parametrize("pad_last", [0, 9], ids=["no_padding", "padding_patches"])
def test_pool_weights_reproduce_the_stock_average(pad_last):
    """The host-built weight matrix must give stock's pooled values and stock's mask.

    Padding patches are handled by zeroing their weight rows instead of `masked_fill`
    on the hidden states, so this checks that substitution is exact -- including that
    the mask still comes from the *unzeroed* weights, as stock derives it.
    """
    from spyre_inference.multimodal.gemma4_vision import _pool_weights

    pos, padding, hidden_states, length, k = _pool_setup(pad_last=pad_last)
    config = configuration_gemma4.Gemma4VisionConfig(
        hidden_size=hidden_states.shape[-1],
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        pooling_kernel_size=k,
    )
    want, want_mask = modeling_gemma4.Gemma4VisionPooler(config)(
        hidden_states=hidden_states,
        pixel_position_ids=pos,
        padding_positions=padding,
        output_length=length,
    )

    weights, mask = _pool_weights(pos, padding, length, k)
    got = (weights.transpose(1, 2) @ hidden_states) * (hidden_states.shape[-1] ** 0.5)

    assert torch.equal(mask, want_mask)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_pool_weights_rows_sum_to_one_over_each_output_cell():
    """Each output cell is a mean of exactly k^2 patches, so its column must sum to 1
    -- the property that makes the matmul an average rather than an arbitrary GEMM."""
    from spyre_inference.multimodal.gemma4_vision import _pool_weights

    pos, padding, _, length, k = _pool_setup()
    weights, _ = _pool_weights(pos, padding, length, k)

    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1, length))
    assert weights.count_nonzero() == length * k * k


def test_patched_pooler_returns_stock_dtype_and_stays_on_host():
    """The caller indexes the result with the mask and then runs an fp32 affine, so
    the patch must preserve stock's fp32 return and host placement even though the
    pool itself now happens on device in bf16."""
    from spyre_inference.multimodal import gemma4_vision

    pos, padding, hidden_states, length, k = _pool_setup()
    config = configuration_gemma4.Gemma4VisionConfig(
        hidden_size=hidden_states.shape[-1],
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        pooling_kernel_size=k,
    )
    gemma4_vision.patch_pooler()
    pooled, mask = modeling_gemma4.Gemma4VisionPooler(config)(
        hidden_states=hidden_states,
        pixel_position_ids=pos,
        padding_positions=padding,
        output_length=length,
    )

    assert pooled.dtype == torch.float32
    assert pooled.device.type == "cpu" and mask.device.type == "cpu"
    assert pooled.shape == (1, length, hidden_states.shape[-1])


def test_patched_pooler_delegates_when_there_is_nothing_to_pool():
    """At equal lengths stock skips pooling and returns `padding_positions` as the
    mask; taking the pooling path there would hand back a differently-derived one."""
    from spyre_inference.multimodal import gemma4_vision

    pos, padding, hidden_states, _, k = _pool_setup()
    num_patches = hidden_states.shape[1]
    config = configuration_gemma4.Gemma4VisionConfig(
        hidden_size=hidden_states.shape[-1],
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        pooling_kernel_size=k,
    )
    gemma4_vision.patch_pooler()
    _, mask = modeling_gemma4.Gemma4VisionPooler(config)(
        hidden_states=hidden_states,
        pixel_position_ids=pos,
        padding_positions=padding,
        output_length=num_patches,
    )

    assert torch.equal(mask, padding)


def test_pooling_matmul_matches_the_host_average_on_device():
    """The bf16 device matmul against the fp32 host average.

    Stock computes this matmul in fp32 but rounds the result straight back to the
    input dtype, so bf16 here concedes only accumulation precision.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.custom_ops.utils import convert
    from spyre_inference.multimodal.gemma4_vision import _pool_weights

    pos, padding, hidden_states, length, k = _pool_setup(num_patches=576, hidden=128)
    weights, _ = _pool_weights(pos, padding, length, k)
    want = weights.transpose(1, 2) @ hidden_states

    device = torch.device("spyre")
    got = torch.matmul(
        convert(weights.to(torch.bfloat16), device=device).transpose(1, 2),
        convert(hidden_states.to(torch.bfloat16), device=device),
    )

    torch.testing.assert_close(convert(got, device="cpu").float(), want, rtol=2e-2, atol=2e-2)
