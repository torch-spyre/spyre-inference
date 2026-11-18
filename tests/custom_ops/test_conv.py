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

"""Tests for `SpyreConv2d` (custom_ops/conv.py), the vision patch-embed conv.

`SpyreConv2d` is registered OOT for every `Conv2dLayer`, so it must stay transparent on
the host and must never take upstream's unfold path on the card. The numeric test needs
a card.
"""

import sys

import pytest
import torch
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available

# A patch embed: Bx3xHxW images, 16x16 patches, 1024 out-channels.
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


@pytest.mark.conv
def test_host_input_matches_forward_native():
    """On the host, `SpyreConv2d` is transparent: it matches stock `forward_native`."""
    from spyre_inference.custom_ops.conv import SpyreConv2d

    layer = _layer(in_ch=3, out_ch=100, kernel=3, stride=1)
    assert isinstance(layer, SpyreConv2d)

    x = torch.randn(1, 3, 32, 32, dtype=torch.float16)
    torch.testing.assert_close(layer.forward_oot(x), layer.forward_native(x))


@pytest.mark.conv
def test_on_card_path_never_uses_mulmat():
    """A patch embed sets `enable_linear`, so `forward_native` would pick upstream's
    unfold/GEMM path, which torch-spyre cannot lay out. Every shape must take
    `F.conv2d` instead, including one with a non-64 out-channel count."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from spyre_inference.custom_ops.conv import SpyreConv2d

    layer = _layer(in_ch=3, out_ch=100, kernel=16, stride=16)
    assert isinstance(layer, SpyreConv2d)
    assert layer.enable_linear is True
    x = torch.randn(1, 3, 32, 32, dtype=torch.float16)
    expected = layer._forward_conv(x)

    def _boom(_x):
        raise AssertionError("SpyreConv2d must not use the im2col/GEMM path")

    layer = layer.to("spyre")
    layer._forward_mulmat = _boom
    actual = layer.forward_oot(x.to("spyre"))
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Numeric correctness on-card
# ---------------------------------------------------------------------------


@pytest.mark.conv
@pytest.mark.parametrize(
    "patch,height,width",
    [
        (16, 64, 64),  # stick-aligned patch grid (4x4 patches)
        (16, 272, 272),  # 17x17 patches — coprime with the stick; stock im2col fails
        (14, 336, 308),  # 24x22 at Ministral-3's patch size, non-square to catch an H/W swap
    ],
)
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("batch", [1, 2, 5])
def test_patch_conv_matches_cpu_reference(patch, height, width, use_bias, batch):
    """`F.conv2d` through `SpyreConv2d` matches a plain CPU `F.conv2d` at any batch."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    layer = _layer(kernel=patch, stride=patch, bias=use_bias)

    torch.manual_seed(3)
    x = torch.randn(batch, 3, height, width, dtype=torch.float16)
    expected = F.conv2d(
        x.float(),
        layer.weight.data.float(),
        layer.bias.data.float() if use_bias else None,
        stride=patch,
    )

    layer = layer.to("spyre")
    actual = layer.forward_oot(x.to("spyre"))

    assert actual.shape == expected.shape
    # torch-spyre's on-card im2col conv (torch-spyre#5642) rounds a few elements past 1e-2.
    torch.testing.assert_close(actual.cpu().float(), expected, atol=2e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
