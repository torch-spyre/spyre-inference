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

"""Tests for `SpyreConv2d` (custom_ops/conv.py), the Pixtral patch-embed conv.

`SpyreConv2d` is registered OOT for every `Conv2dLayer`, but its tiled layouts only
suit a patch embed, so `_layouts_supported` — the gate keeping other convs on the
stock path — is the test that matters most. The layout and numeric tests need a card.
"""

import sys

import pytest
import torch
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available

# Pixtral patch embed: 1x3xHxW image, 16x16 patches, 1024 out-channels.
PATCH = 16
OUT_CHANNELS = 1024


def _layer(in_ch=3, out_ch=OUT_CHANNELS, kernel=PATCH, stride=PATCH, bias=False):
    """A `Conv2dLayer` with deterministic weights (its weight is `torch.empty`)."""
    from vllm.model_executor.layers.conv import Conv2dLayer

    layer = Conv2dLayer(
        in_ch,
        out_ch,
        kernel,
        stride=stride,
        bias=bias,
        params_dtype=torch.float16,
    )
    torch.manual_seed(0)
    layer.weight.data.normal_(std=0.02)
    if bias:
        layer.bias.data.normal_(std=0.02)
    return layer


# ---------------------------------------------------------------------------
# OOT dispatch
# ---------------------------------------------------------------------------


@pytest.mark.conv
def test_conv2d_oot_dispatch():
    """`Conv2dLayer(...)` instantiates `SpyreConv2d` and selects `forward_oot`."""
    from spyre_inference.custom_ops.conv import SpyreConv2d

    layer = _layer()
    assert isinstance(layer, SpyreConv2d)
    assert layer._forward_method == layer.forward_oot


# ---------------------------------------------------------------------------
# _layouts_supported — the fallback gate
# ---------------------------------------------------------------------------


@pytest.mark.conv
def test_layouts_supported_accepts_a_patch_embed():
    from spyre_inference.custom_ops.conv import _layouts_supported

    x = torch.randn(1, 3, 64, 64, dtype=torch.float16)
    weight = torch.randn(OUT_CHANNELS, 3, PATCH, PATCH, dtype=torch.float16)
    assert _layouts_supported(x, weight) is True


@pytest.mark.conv
@pytest.mark.parametrize(
    "x_shape,w_shape,reason",
    [
        ((2, 3, 64, 64), (OUT_CHANNELS, 3, PATCH, PATCH), "batch > 1"),
        ((1, 65, 64, 64), (OUT_CHANNELS, 65, PATCH, PATCH), "in_channels > 64"),
        ((1, 3, 64, 64), (100, 3, PATCH, PATCH), "out_channels not a multiple of 64"),
        ((1, 3, 64), (OUT_CHANNELS, 3, PATCH, PATCH), "input not 4-D"),
        ((1, 3, 64, 64), (OUT_CHANNELS, 3, PATCH), "weight not 4-D"),
    ],
)
def test_layouts_supported_rejects_non_patch_shapes(x_shape, w_shape, reason):
    """Anything outside the tiled-layout assumptions must fall back to vLLM's
    stock path instead of building a layout that would silently tile it wrong."""
    from spyre_inference.custom_ops.conv import _layouts_supported

    x = torch.randn(*x_shape, dtype=torch.float16)
    weight = torch.randn(*w_shape, dtype=torch.float16)
    assert _layouts_supported(x, weight) is False, reason


@pytest.mark.conv
def test_unsupported_shape_falls_back_to_forward_native():
    """A non-patch-shaped conv routes through `forward_native` and matches it
    exactly — `SpyreConv2d` must be transparent for every other `Conv2dLayer`."""
    from spyre_inference.custom_ops.conv import SpyreConv2d

    # groups=1, kernel != stride -> enable_linear False, and out_channels 100
    # is not a multiple of 64, so the tiled layouts do not apply.
    layer = _layer(in_ch=3, out_ch=100, kernel=3, stride=1)
    assert isinstance(layer, SpyreConv2d)

    x = torch.randn(1, 3, 32, 32, dtype=torch.float16)
    torch.testing.assert_close(layer.forward_oot(x), layer.forward_native(x))


@pytest.mark.conv
def test_unsupported_patch_shape_falls_back_to_conv_not_mulmat():
    """`forward_native` would route a patch-embed fallback into `_forward_mulmat`. Both
    paths are numerically equal, so this pins the route by making the wrong one raise."""
    from spyre_inference.custom_ops.conv import SpyreConv2d, _layouts_supported

    # kernel == stride and no padding -> enable_linear; out_channels 100 is not a
    # multiple of 64, so the tiled layouts do not apply and it must fall back.
    layer = _layer(in_ch=3, out_ch=100, kernel=16, stride=16)
    assert isinstance(layer, SpyreConv2d)
    assert layer.enable_linear is True
    x = torch.randn(1, 3, 32, 32, dtype=torch.float16)
    assert _layouts_supported(x, layer.weight) is False

    def _boom(_x):
        raise AssertionError("fallback must not use the im2col/GEMM path")

    layer._forward_mulmat = _boom
    torch.testing.assert_close(layer.forward_oot(x), layer._forward_conv(x))


# ---------------------------------------------------------------------------
# Layout derivation (needs torch-spyre, not necessarily a card)
# ---------------------------------------------------------------------------


@pytest.mark.conv
@pytest.mark.parametrize("out_ch", [64, 128, OUT_CHANNELS])
@pytest.mark.parametrize("hw", [(64, 64), (48, 80)])
def test_layouts_build_for_valid_shapes(out_ch, hw):
    """Layouts are derived from tensor shape, not hardcoded: any 64-aligned
    out-channel count and any image size must build without raising."""
    pytest.importorskip("torch_spyre")
    from spyre_inference.custom_ops.conv import _input_layout, _weight_layout

    assert _weight_layout(torch.randn(out_ch, 3, PATCH, PATCH, dtype=torch.float16)) is not None
    assert _input_layout(torch.randn(1, 3, *hw, dtype=torch.float16)) is not None


# ---------------------------------------------------------------------------
# Numeric correctness on-card
# ---------------------------------------------------------------------------


@pytest.mark.conv
@pytest.mark.parametrize(
    "patch,height,width",
    [
        (16, 64, 64),  # stick-aligned patch grid (4x4 patches)
        (16, 272, 272),  # 17x17 patches — coprime with the stick; stock im2col fails
        (14, 336, 308),  # 24x22 at Ministral-3's patch size. NON-SQUARE on purpose:
        # `_input_layout` derives strides from (h, w), so an H/W swap is invisible
        # while height == width.
    ],
)
@pytest.mark.parametrize("use_bias", [False, True])
def test_patch_conv_matches_cpu_reference(patch, height, width, use_bias):
    """On-card `F.conv2d` with tiled layouts matches a plain CPU `F.conv2d`."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    layer = _layer(kernel=patch, stride=patch, bias=use_bias)

    torch.manual_seed(3)
    x = torch.randn(1, 3, height, width, dtype=torch.float16)
    expected = F.conv2d(
        x,
        layer.weight.data,
        layer.bias.data if use_bias else None,
        stride=patch,
    )

    layer = layer.to("spyre")
    actual = layer.forward_oot(x.to("spyre"))

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
