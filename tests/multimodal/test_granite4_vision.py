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

"""Tests for `spyre_inference/multimodal/granite4_vision.py`.

Two class-level patches, each guarded by a `_spyre_patched` flag:

- `patch_interpolate_downsampler`: offloads `InterpolateDownsampler.__call__`
  to CPU (F.adaptive_avg_pool2d not supported on Spyre).
- `patch_pack_and_unpad_image_features`: offloads
  `Granite4VisionForConditionalGeneration._pack_and_unpad_image_features` to CPU
  (5-D permute produces a stick expression Spyre's work_division cannot lower).

The staleness tripwires catch silent no-ops from vLLM renames. The equivalence
tests confirm the CPU offload does not change values. Section 4 repeats the
numeric checks on the card and skips without a device.
"""

import sys

import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available

granite4_vision = pytest.importorskip("vllm.model_executor.models.granite4_vision")

# InterpolateDownsampler config matching Granite Vision 4.1-4B defaults.
# The SigLIP encoder outputs a 24×24 grid (336px / 14px patch = 24 patches/side)
# which the downsampler reduces to 12×12 = 144 tokens.
ORIG_IMAGE_SIDE = 24
NEW_IMAGE_SIDE = 12
FEATURE_DIM = 1152  # SigLIP hidden size

# Matches Granite Vision 4.1-4B: image_size=336, patch_size=14, downsample_rate=1/2
# orig_image_side = 336 // 14 = 24; new_image_side = 24 * (1/2) = 12
_VISION_IMAGE_SIZE = 336
_VISION_PATCH_SIZE = 14
_DOWNSAMPLE_RATE = "1/2"


class _MinimalVisionConfig:
    image_size = _VISION_IMAGE_SIZE
    patch_size = _VISION_PATCH_SIZE


class _MinimalDownsamplerConfig:
    vision_config = _MinimalVisionConfig()
    downsample_rate = _DOWNSAMPLE_RATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_interpolate_downsampler() -> nn.Module:
    """Instantiate a real InterpolateDownsampler using a minimal config."""
    cls = getattr(granite4_vision, "InterpolateDownsampler", None)
    if cls is None:
        pytest.skip("InterpolateDownsampler not present in this vLLM version")
    return cls(_MinimalDownsamplerConfig())


def _make_image_features(batch: int = 1) -> torch.Tensor:
    """Flat image feature tensor `[batch, orig_side^2, dim]` as produced by
    the SigLIP encoder before the downsampler."""
    rng = torch.Generator(device="cpu").manual_seed(0)
    return torch.randn(batch, ORIG_IMAGE_SIDE**2, FEATURE_DIM, dtype=torch.float16, generator=rng)


def _make_dummy_model() -> nn.Module:
    """Minimal stand-in with the attributes `apply_multimodal_patches` checks."""
    m = nn.Module()
    m.vision_tower = nn.Module()
    return m


class _MinimalGranite4VisionModel(nn.Module):
    """Wraps `Granite4VisionForConditionalGeneration._pack_and_unpad_image_features`
    as a bare instance call without instantiating the full model.

    Uses single-patch inputs (image_feature.shape[0] == 1) so only
    `self.image_newline` is needed — the multi-patch branch additionally
    reads `self.config`, `self._downsample_rate`, etc.
    """

    def __init__(self):
        super().__init__()
        self.image_newline = None  # single-patch branch: only checked for None
        # _pack_and_unpad_image_features reads self.config and self._downsample_rate
        # unconditionally on entry, before branching on image_feature.shape[0].
        self.config = _MinimalDownsamplerConfig()
        from fractions import Fraction

        self._downsample_rate = float(Fraction(_DOWNSAMPLE_RATE))  # 0.5

    def pack_and_unpad(self, image_features, image_sizes):
        cls = granite4_vision.Granite4VisionForConditionalGeneration
        return cls._pack_and_unpad_image_features(self, image_features, image_sizes)


# ---------------------------------------------------------------------------
# 1. Staleness tripwires
# ---------------------------------------------------------------------------


@pytest.mark.granite4_vision
@pytest.mark.parametrize(
    "symbol",
    [
        "Granite4VisionForConditionalGeneration",
        "InterpolateDownsampler",
    ],
)
def test_patch_target_symbols_still_exist(symbol):
    """Every symbol the patches reach for must still exist in the vLLM module.
    Both patches use `try/except ImportError` and return silently, so this is
    the only place a rename or removal is caught."""
    assert getattr(granite4_vision, symbol, None) is not None, (
        f"vllm.model_executor.models.granite4_vision.{symbol} is gone — "
        "the corresponding Spyre patch in multimodal/granite4_vision.py is "
        "now a silent no-op and must be updated"
    )


@pytest.mark.granite4_vision
def test_granite4_vision_model_has_pack_and_unpad_method():
    """Granite4VisionForConditionalGeneration must have
    `_pack_and_unpad_image_features` — the method the patch targets."""
    cls = getattr(granite4_vision, "Granite4VisionForConditionalGeneration", None)
    if cls is None:
        pytest.skip("Granite4VisionForConditionalGeneration not present")
    assert hasattr(cls, "_pack_and_unpad_image_features"), (
        "Granite4VisionForConditionalGeneration._pack_and_unpad_image_features "
        "is gone — the patch in multimodal/granite4_vision.py targets that method"
    )


# ---------------------------------------------------------------------------
# 2. Patch application and idempotency
# ---------------------------------------------------------------------------


@pytest.mark.granite4_vision
def test_patch_interpolate_downsampler_is_applied_and_idempotent():
    """`patch_interpolate_downsampler` must mark `__call__` with `_spyre_patched`
    and a second call must leave the same function in place."""
    from spyre_inference.multimodal.granite4_vision import patch_interpolate_downsampler

    patch_interpolate_downsampler()
    cls = getattr(granite4_vision, "InterpolateDownsampler", None)
    if cls is None:
        pytest.skip("InterpolateDownsampler not present")

    patched = cls.__call__
    assert getattr(patched, "_spyre_patched", False) is True

    patch_interpolate_downsampler()
    assert cls.__call__ is patched, "second call must be a no-op"


@pytest.mark.granite4_vision
def test_patch_pack_and_unpad_is_applied_and_idempotent():
    """`patch_pack_and_unpad_image_features` must mark the method with
    `_spyre_patched` and a second call must leave the same function in place."""
    from spyre_inference.multimodal.granite4_vision import patch_pack_and_unpad_image_features

    patch_pack_and_unpad_image_features()
    cls = getattr(granite4_vision, "Granite4VisionForConditionalGeneration", None)
    if cls is None:
        pytest.skip("Granite4VisionForConditionalGeneration not present")

    patched = cls._pack_and_unpad_image_features
    assert getattr(patched, "_spyre_patched", False) is True

    patch_pack_and_unpad_image_features()
    assert cls._pack_and_unpad_image_features is patched, "second call must be a no-op"


@pytest.mark.granite4_vision
def test_apply_is_idempotent():
    """`apply(model, device)` must be safe to call twice without double-wrapping."""
    from spyre_inference.multimodal import granite4_vision as spyre_gv

    model = _make_dummy_model()
    device = torch.device("cpu")

    spyre_gv.apply(model, device)
    spyre_gv.apply(model, device)  # must not raise or double-wrap


# ---------------------------------------------------------------------------
# 3. Numeric equivalence: InterpolateDownsampler on CPU
# ---------------------------------------------------------------------------


@pytest.mark.granite4_vision
def test_interpolate_downsampler_patched_matches_stock():
    """The patched `InterpolateDownsampler.__call__` must produce the same output
    as the unpatched version on CPU — the only change is an explicit CPU round-trip."""
    from spyre_inference.multimodal.granite4_vision import patch_interpolate_downsampler

    ds_stock = _make_interpolate_downsampler()
    image_features = _make_image_features()
    expected = ds_stock(image_features)

    patch_interpolate_downsampler()
    ds_patched = _make_interpolate_downsampler()
    actual = ds_patched(image_features)

    assert actual.shape == expected.shape, (
        f"shape mismatch: got {actual.shape}, expected {expected.shape}"
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.granite4_vision
@pytest.mark.parametrize("batch_size", [1, 2])
def test_interpolate_downsampler_output_shape(batch_size):
    """The downsampler must reduce `[B, orig^2, D]` → `[B, new^2, D]`."""
    from spyre_inference.multimodal.granite4_vision import patch_interpolate_downsampler

    patch_interpolate_downsampler()
    ds = _make_interpolate_downsampler()
    image_features = _make_image_features(batch=batch_size)
    out = ds(image_features)

    expected_shape = (batch_size, NEW_IMAGE_SIDE**2, FEATURE_DIM)
    assert out.shape == torch.Size(expected_shape), (
        f"downsampler output shape {out.shape} != expected {expected_shape}"
    )


# ---------------------------------------------------------------------------
# 3b. Numeric equivalence: _pack_and_unpad_image_features on CPU
# ---------------------------------------------------------------------------


def _make_pack_and_unpad_inputs(num_images: int = 1):
    """Build minimal `image_features` and `image_sizes` for
    `_pack_and_unpad_image_features`.

    Each entry has shape `[1, NEW_IMAGE_SIDE^2, FEATURE_DIM]` — the leading
    dimension of 1 selects the single-patch branch in the method, which only
    reads `self.image_newline` (set to None in `_MinimalGranite4VisionModel`).
    The multi-patch branch additionally requires `self.config`,
    `self._downsample_rate`, etc., which a minimal stub cannot provide.
    """
    image_features = [
        torch.randn(1, NEW_IMAGE_SIDE**2, FEATURE_DIM, dtype=torch.float16)
        for _ in range(num_images)
    ]
    # image_sizes: (H, W) in original pixels — only read in the multi-patch branch.
    image_sizes = torch.tensor([[336, 336]] * num_images, dtype=torch.long)
    return image_features, image_sizes


@pytest.mark.granite4_vision
def test_pack_and_unpad_patched_matches_stock():
    """The patched `_pack_and_unpad_image_features` must produce the same output
    as the unpatched version on CPU."""
    from spyre_inference.multimodal.granite4_vision import patch_pack_and_unpad_image_features

    image_features, image_sizes = _make_pack_and_unpad_inputs(num_images=1)
    obj = _MinimalGranite4VisionModel()

    # Stock output (before patch).
    expected = obj.pack_and_unpad(image_features, image_sizes)

    patch_pack_and_unpad_image_features()
    actual = obj.pack_and_unpad(image_features, image_sizes)

    assert len(actual) == len(expected), "result list length must match"
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a.shape == e.shape, f"item {i} shape mismatch: {a.shape} vs {e.shape}"
        torch.testing.assert_close(a.float(), e.float(), atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 4. On-card equivalence (skipped without a Spyre device)
# ---------------------------------------------------------------------------


@pytest.mark.granite4_vision
def test_interpolate_downsampler_matches_cpu_on_spyre():
    """The patched downsampler on-card must equal the same call on CPU.

    At `ORIG_IMAGE_SIDE=24` the flat token count is 576, which is a multiple
    of the 64-wide stick. This exercises the CPU round-trip path rather than
    a stick-alignment workaround.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.multimodal.granite4_vision import patch_interpolate_downsampler

    patch_interpolate_downsampler()

    image_features_cpu = _make_image_features()
    ds = _make_interpolate_downsampler()
    expected = ds(image_features_cpu)

    device = torch.device("spyre")
    image_features_dev = image_features_cpu.to(device)
    actual = ds(image_features_dev)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.granite4_vision
def test_pack_and_unpad_result_on_correct_device_on_spyre():
    """After the patched `_pack_and_unpad_image_features` call with Spyre inputs,
    the results must land back on Spyre (not stay stranded on CPU)."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.multimodal.granite4_vision import patch_pack_and_unpad_image_features

    patch_pack_and_unpad_image_features()

    device = torch.device("spyre")
    image_features_cpu, image_sizes = _make_pack_and_unpad_inputs(num_images=1)
    image_features_dev = [f.to(device) for f in image_features_cpu]

    obj = _MinimalGranite4VisionModel()
    result = obj.pack_and_unpad(image_features_dev, image_sizes.to(device))

    for i, r in enumerate(result):
        assert r.device.type == "spyre", (
            f"result[{i}] is on {r.device} — patched _pack_and_unpad must move "
            "results back to Spyre when inputs were on Spyre"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
