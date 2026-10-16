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

# Capture unpatched methods at import time, before any test can trigger the
# process-wide class patches via patch_interpolate_downsampler() /
# patch_pack_and_unpad_image_features().
_STOCK_INTERPOLATE_CALL = granite4_vision.InterpolateDownsampler.__call__
_STOCK_PACK_AND_UNPAD = (
    granite4_vision.Granite4VisionForConditionalGeneration._pack_and_unpad_image_features
)

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

    image_features = _make_image_features()

    # Use the __call__ captured at import time — guards against earlier tests
    # having already applied the process-wide class patch.
    expected = _STOCK_INTERPOLATE_CALL(_make_interpolate_downsampler(), image_features)

    patch_interpolate_downsampler()
    actual = _make_interpolate_downsampler()(image_features)

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
    rng = torch.Generator(device="cpu").manual_seed(1)
    image_features = [
        torch.randn(1, NEW_IMAGE_SIDE**2, FEATURE_DIM, dtype=torch.float16, generator=rng)
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

    # Use the method captured at import time — guards against earlier tests
    # having already applied the process-wide class patch.
    expected = _STOCK_PACK_AND_UNPAD(_MinimalGranite4VisionModel(), image_features, image_sizes)

    patch_pack_and_unpad_image_features()
    actual = _MinimalGranite4VisionModel().pack_and_unpad(image_features, image_sizes)

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


# ---------------------------------------------------------------------------
# 3c. patch_embed_input_ids — replaces spyre_model_runner index_put logic
# ---------------------------------------------------------------------------
# Upstream embed_input_ids uses two boolean-mask index_puts that Spyre cannot
# execute:
#   1. text_embeds[is_multimodal] = 0.0
#   2. target[is_multimodal] = level_features[level_idx]
#
# The patch replaces (1) with torch.where and (2) with a CPU scatter + copy_,
# keeping the embedding lookup on-card.
#
# The fake model only needs:
#   - self.language_model.model.embed_input_ids(input_ids) → text embeddings
#   - self.language_model.model.config.embedding_multiplier
#   - self._ds_buffers   list of [max_tokens, lm_hidden] tensors (one per level)
#   - self._ds_layer_indices  list of level indices (len = num_levels)
#   - self._ds_num_tokens     written by the patch

_LM_HIDDEN = 32  # small but realistic; must be divisible by the split
_MAX_TOKENS = 16
_NUM_LEVELS = 2
_EMBEDDING_MULTIPLIER = 0.5


class _MinimalLMInner:
    """Stub for self.language_model.model."""

    class _Config:
        embedding_multiplier = _EMBEDDING_MULTIPLIER

    config = _Config()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Deterministic: token index → float embedding of width _LM_HIDDEN.
        rng = torch.Generator(device="cpu").manual_seed(int(input_ids.sum().item()))
        return torch.randn(len(input_ids), _LM_HIDDEN, dtype=torch.float16, generator=rng)


class _MinimalLanguageModel:
    model = _MinimalLMInner()


class _MinimalGranite4VisionEmbedModel:
    """Minimal stub for Granite4VisionForConditionalGeneration.

    Only the attributes embed_input_ids reads are provided.
    """

    def __init__(self, num_levels: int = _NUM_LEVELS):
        self.language_model = _MinimalLanguageModel()
        self._ds_buffers = [
            torch.zeros(_MAX_TOKENS, _LM_HIDDEN, dtype=torch.float16) for _ in range(num_levels)
        ]
        self._ds_layer_indices = list(range(num_levels))
        self._ds_num_tokens = -1  # sentinel; will be overwritten by the patch

    def embed_input_ids(
        self, input_ids, multimodal_embeddings=None, *, is_multimodal=None, handle_oov_mm_token=True
    ):
        cls = granite4_vision.Granite4VisionForConditionalGeneration
        return cls.embed_input_ids(
            self,
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )


@pytest.mark.granite4_vision
def test_patch_embed_input_ids_target_symbol_exists():
    """Granite4VisionForConditionalGeneration must have `embed_input_ids`."""
    cls = getattr(granite4_vision, "Granite4VisionForConditionalGeneration", None)
    assert cls is not None
    assert hasattr(cls, "embed_input_ids"), (
        "Granite4VisionForConditionalGeneration.embed_input_ids is gone — "
        "patch_embed_input_ids in multimodal/granite4_vision.py is a silent no-op"
    )


@pytest.mark.granite4_vision
def test_patch_embed_input_ids_is_applied_and_idempotent():
    """`patch_embed_input_ids` must mark `embed_input_ids` with `_spyre_patched`
    and a second call must be a no-op."""
    from spyre_inference.multimodal.granite4_vision import patch_embed_input_ids

    patch_embed_input_ids()
    cls = granite4_vision.Granite4VisionForConditionalGeneration
    patched = cls.embed_input_ids
    assert getattr(patched, "_spyre_patched", False) is True

    patch_embed_input_ids()
    assert cls.embed_input_ids is patched, "second call must be a no-op"


@pytest.mark.granite4_vision
def test_patch_embed_input_ids_text_only_path():
    """With no multimodal embeddings, the patch must:
    - return text_embeds * embedding_multiplier
    - set _ds_num_tokens = 0
    """
    from spyre_inference.multimodal.granite4_vision import patch_embed_input_ids

    patch_embed_input_ids()
    obj = _MinimalGranite4VisionEmbedModel()

    N = 8
    input_ids = torch.arange(N)

    result = obj.embed_input_ids(input_ids, multimodal_embeddings=None, is_multimodal=None)

    assert result.shape == (N, _LM_HIDDEN)
    assert obj._ds_num_tokens == 0

    # Values must equal text_embeds * embedding_multiplier.
    expected_embeds = obj.language_model.model.embed_input_ids(input_ids)
    torch.testing.assert_close(
        result.float(),
        (expected_embeds * _EMBEDDING_MULTIPLIER).float(),
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.granite4_vision
def test_patch_embed_input_ids_text_only_path_ds_num_tokens_zero():
    """_ds_num_tokens must be 0 even when is_multimodal is all-False."""
    from spyre_inference.multimodal.granite4_vision import patch_embed_input_ids

    patch_embed_input_ids()
    obj = _MinimalGranite4VisionEmbedModel()

    N = 6
    input_ids = torch.arange(N)
    is_multimodal = torch.zeros(N, dtype=torch.bool)

    obj.embed_input_ids(input_ids, multimodal_embeddings=[], is_multimodal=is_multimodal)

    assert obj._ds_num_tokens == 0


@pytest.mark.granite4_vision
def test_patch_embed_input_ids_vision_path_zeros_image_positions():
    """Image-token positions in the output must be zero (not the raw text embedding).

    The patch uses torch.where(mask, zeros, text_embeds); the text-token positions
    must be text_embeds * embedding_multiplier and the image positions must be 0.
    """
    from spyre_inference.multimodal.granite4_vision import patch_embed_input_ids

    patch_embed_input_ids()
    obj = _MinimalGranite4VisionEmbedModel(num_levels=_NUM_LEVELS)

    N = 8
    num_img_tokens = 2
    # positions 3 and 5 are image tokens
    is_multimodal = torch.zeros(N, dtype=torch.bool)
    is_multimodal[3] = True
    is_multimodal[5] = True

    # multimodal_embeddings: one tensor of shape [num_img_tokens, lm_hidden * num_levels]
    rng = torch.Generator(device="cpu").manual_seed(10)
    mm_emb = torch.randn(
        num_img_tokens, _LM_HIDDEN * _NUM_LEVELS, dtype=torch.float16, generator=rng
    )

    input_ids = torch.arange(N)
    result = obj.embed_input_ids(input_ids, [mm_emb], is_multimodal=is_multimodal)

    assert result.shape == (N, _LM_HIDDEN)

    # Image positions must be zero.
    assert not result[is_multimodal].any(), (
        "image-token positions in inputs_embeds must be zeroed by torch.where"
    )

    # Text positions must be non-zero (text_embeds * multiplier).
    text_mask = ~is_multimodal
    assert result[text_mask].any(), "text positions must carry the text embeddings"


@pytest.mark.granite4_vision
def test_patch_embed_input_ids_vision_path_fills_ds_buffers():
    """_ds_buffers must be filled with the scattered multimodal features.

    For each level l, _ds_buffers[l][is_multimodal] must equal the l-th chunk
    of the packed multimodal tensor (split along the last dim by lm_hidden).
    """
    from spyre_inference.multimodal.granite4_vision import patch_embed_input_ids

    patch_embed_input_ids()
    obj = _MinimalGranite4VisionEmbedModel(num_levels=_NUM_LEVELS)

    N = 8
    num_img_tokens = 3
    is_multimodal = torch.zeros(N, dtype=torch.bool)
    is_multimodal[1] = True
    is_multimodal[4] = True
    is_multimodal[6] = True

    rng = torch.Generator(device="cpu").manual_seed(20)
    mm_emb = torch.randn(
        num_img_tokens, _LM_HIDDEN * _NUM_LEVELS, dtype=torch.float16, generator=rng
    )

    input_ids = torch.arange(N)
    obj.embed_input_ids(input_ids, [mm_emb], is_multimodal=is_multimodal)

    assert obj._ds_num_tokens == N

    # Each level's buffer slice must equal the corresponding level features.
    level_features = mm_emb.split(_LM_HIDDEN, dim=-1)
    for lvl in range(_NUM_LEVELS):
        buf_slice = obj._ds_buffers[lvl][:N]
        # Only image-token rows are filled; text rows stay zero.
        torch.testing.assert_close(
            buf_slice[is_multimodal].float(),
            level_features[lvl].float(),
            atol=1e-4,
            rtol=1e-4,
            msg=f"level {lvl}: ds_buffer image rows do not match packed features",
        )
        assert not buf_slice[~is_multimodal].any(), (
            f"level {lvl}: text-token rows in ds_buffer must be zero"
        )


@pytest.mark.granite4_vision
def test_patch_embed_input_ids_ds_buffers_migrated_to_correct_device():
    """_ds_buffers must be migrated to match text_embeds device/dtype on first call.

    On CPU the migration is a no-op (same device), but the dtype check fires when
    _ds_buffers are float32 and text_embeds are float16.
    """
    from spyre_inference.multimodal.granite4_vision import patch_embed_input_ids

    patch_embed_input_ids()
    obj = _MinimalGranite4VisionEmbedModel()

    # Deliberately initialise buffers in float32 to trigger the dtype migration.
    obj._ds_buffers = [
        torch.zeros(_MAX_TOKENS, _LM_HIDDEN, dtype=torch.float32) for _ in range(_NUM_LEVELS)
    ]

    input_ids = torch.arange(4)
    obj.embed_input_ids(input_ids)  # text-only path still runs the migration

    for i, buf in enumerate(obj._ds_buffers):
        assert buf.dtype == torch.float16, (
            f"_ds_buffers[{i}].dtype must be float16 after migration; got {buf.dtype}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
