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

"""Tests for `spyre_inference/multimodal/clip.py`.

No Spyre hardware: `apply()` is exercised against a minimal stand-in with real
`nn.LayerNorm` instances (mirroring `CLIPEmbeddingModel`'s `text_model`/
`vision_model` shape), on `device="cpu"` -- SpyreLayerNorm's CPU path is a
plain `nn.LayerNorm`, so this checks the swap and weight-copy without needing
a device.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm
from spyre_inference.multimodal import apply_multimodal_patches
from spyre_inference.multimodal.clip import apply as apply_clip_patches


def _fake_clip_model(hidden_size: int = 64, with_post_norm: bool = True):
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    text_model = types.SimpleNamespace(final_layer_norm=torch.nn.LayerNorm(hidden_size))
    text_model.final_layer_norm.weight.data.normal_(generator=g)
    text_model.final_layer_norm.bias.data.normal_(generator=g)

    vision_model = types.SimpleNamespace(
        pre_layrnorm=torch.nn.LayerNorm(hidden_size),
        post_layernorm=(torch.nn.LayerNorm(hidden_size) if with_post_norm else None),
    )
    vision_model.pre_layrnorm.weight.data.normal_(generator=g)
    vision_model.pre_layrnorm.bias.data.normal_(generator=g)
    if with_post_norm:
        vision_model.post_layernorm.weight.data.normal_(generator=g)
        vision_model.post_layernorm.bias.data.normal_(generator=g)

    return types.SimpleNamespace(text_model=text_model, vision_model=vision_model)


def test_apply_swaps_boundary_norms_and_preserves_weights():
    model = _fake_clip_model()
    orig_text_w = model.text_model.final_layer_norm.weight.clone()
    orig_text_b = model.text_model.final_layer_norm.bias.clone()
    orig_pre_w = model.vision_model.pre_layrnorm.weight.clone()
    orig_post_w = model.vision_model.post_layernorm.weight.clone()

    apply_clip_patches(model, torch.device("cpu"))

    assert isinstance(model.text_model.final_layer_norm, SpyreLayerNorm)
    assert isinstance(model.vision_model.pre_layrnorm, SpyreLayerNorm)
    assert isinstance(model.vision_model.post_layernorm, SpyreLayerNorm)
    # Not swapped: layer_norm1/layer_norm2 (inside encoder blocks) aren't touched,
    # and this stand-in doesn't even define them -- swapping only the three
    # boundary norms is the whole point.
    torch.testing.assert_close(model.text_model.final_layer_norm.weight, orig_text_w)
    torch.testing.assert_close(model.text_model.final_layer_norm.bias, orig_text_b)
    torch.testing.assert_close(model.vision_model.pre_layrnorm.weight, orig_pre_w)
    torch.testing.assert_close(model.vision_model.post_layernorm.weight, orig_post_w)


def test_apply_skips_missing_post_layernorm():
    """CLIPVisionTransformer.post_layernorm can be None (require_post_norm=False);
    apply() must not crash trying to swap a missing/None attribute."""
    model = _fake_clip_model(with_post_norm=False)

    apply_clip_patches(model, torch.device("cpu"))

    assert isinstance(model.vision_model.pre_layrnorm, SpyreLayerNorm)
    assert model.vision_model.post_layernorm is None


def test_apply_skips_a_non_layernorm_boundary_norm():
    """A vision tower using RMSNorm/nn.Identity for its boundary norm (not CLIP's
    plain LayerNorm) must be left alone, not crash on `normalized_shape`."""
    model = types.SimpleNamespace(
        text_model=types.SimpleNamespace(final_layer_norm=torch.nn.Identity()),
        vision_model=types.SimpleNamespace(
            pre_layrnorm=torch.nn.Identity(), post_layernorm=torch.nn.Identity()
        ),
    )

    apply_clip_patches(model, torch.device("cpu"))

    assert isinstance(model.text_model.final_layer_norm, torch.nn.Identity)
    assert isinstance(model.vision_model.pre_layrnorm, torch.nn.Identity)
    assert isinstance(model.vision_model.post_layernorm, torch.nn.Identity)


def test_apply_tolerates_a_model_with_neither_tower():
    """apply_multimodal_patches gates on hasattr(text_model/vision_model); apply()
    itself must also tolerate a model missing both (defensive, not load-bearing)."""
    apply_clip_patches(types.SimpleNamespace(), torch.device("cpu"))


@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_apply_preserves_shape_eps_affine_bias(elementwise_affine, bias):
    hidden_size, eps = 128, 1e-6
    ln = torch.nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=elementwise_affine, bias=bias)
    model = types.SimpleNamespace(
        text_model=types.SimpleNamespace(final_layer_norm=ln),
        vision_model=types.SimpleNamespace(),
    )

    apply_clip_patches(model, torch.device("cpu"))

    patched = model.text_model.final_layer_norm
    assert isinstance(patched, SpyreLayerNorm)
    assert patched.normalized_shape == ln.normalized_shape
    assert patched.eps == eps
    assert patched.elementwise_affine == elementwise_affine


def test_apply_multimodal_patches_dispatches_to_clip():
    """`model.config.model_type == "clip"` is the gate `apply_multimodal_patches` uses
    to route to clip.apply() -- a rename would silently stop dispatching, so pin the
    exact check."""
    model = _fake_clip_model()
    model.config = types.SimpleNamespace(model_type="clip")

    apply_multimodal_patches(model, torch.device("cpu"))

    assert isinstance(model.text_model.final_layer_norm, SpyreLayerNorm)


def test_apply_multimodal_patches_skips_a_non_clip_model_with_the_same_shape():
    """A model_type mismatch (e.g. BLIP-2, which also sets vision_model) must not
    dispatch to clip.apply() just because the attributes happen to be present."""
    model = _fake_clip_model()
    model.config = types.SimpleNamespace(model_type="blip-2")

    apply_multimodal_patches(model, torch.device("cpu"))

    assert not isinstance(model.text_model.final_layer_norm, SpyreLayerNorm)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
