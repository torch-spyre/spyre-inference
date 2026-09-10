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

"""Spyre-specific Conv2d implementation (Pixtral/Ministral vision patch embed).

vLLM lowers a patch conv to im2col + GEMM, whose on-device reshape produces a
sub-stick `copy_from_d2d` expression torch-spyre cannot lay out for patch grids
coprime with the 64-wide stick. So run the real `F.conv2d` on-card instead, with
the weight and input placed into explicit `SpyreTensorLayout`s. Layout tuples are
derived from shapes, so any out-channel count and image size work.
"""

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.conv import Conv2dLayer

from .lazy_compile import CompileOutermost, compile_when_outermost

logger = init_logger(__name__)


def _layouts_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the layout tuples below apply: one image, in-channels within a stick,
    out-channels a whole number of sticks.

    True for a Pixtral patch embed, not for convs in general — and this class is
    registered OOT for *every* `Conv2dLayer`, so fall back rather than assert.
    """
    if x.dim() != 4 or weight.dim() != 4:
        return False
    b, c = x.shape[0], x.shape[1]
    return b == 1 and c <= 64 and weight.shape[0] % 64 == 0


def _weight_layout(weight: torch.Tensor):
    """SpyreTensorLayout for a conv weight (O, C, K1, K2), sticked on out-channels.

    The stick walks the out-channel dim (host stride C*K1*K2), tiling it into O//64.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    o, c, k1, k2 = weight.shape
    assert o % 64 == 0, f"conv out_channels {o} must be a multiple of the 64-wide stick"
    return SpyreTensorLayout(
        [k2, k1, o // 64, c, 64],
        [1, k2, c * k1 * k2 * 64, k1 * k2, c * k1 * k2],
        get_device_dtype(weight.dtype),
    )


def _input_layout(x: torch.Tensor):
    """SpyreTensorLayout for a conv input (1, C, H, W), sticked on in-channels.

    The stick walks the channel dim (host stride H*W), padding C up to a full stick.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    b, c, h, w = x.shape
    assert b == 1, f"conv input batch {b} != 1 (Pixtral feeds one image at a time)"
    assert c <= 64, f"conv in_channels {c} must fit in one 64-wide stick"
    return SpyreTensorLayout(
        [w, h, 1, 1, 64],
        [1, w, -1, c * h * w, h * w],
        get_device_dtype(x.dtype),
    )


@Conv2dLayer.register_oot(name="Conv2dLayer")
class SpyreConv2d(CompileOutermost, Conv2dLayer):
    """Out-of-tree Conv2d for Spyre: `F.conv2d` on-card with explicit tiled layouts.

    Spyre needs static shapes, so the kernel recompiles per distinct (H, W). Past
    ``torch._dynamo.config.cache_size_limit`` (default 8) dynamo falls back to
    eager, which is unvalidated for pre-laid-out tensors — bucket or resize images
    if a workload uses many resolutions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._w_dev: torch.Tensor | None = None

    @compile_when_outermost
    def _conv_native(self, x: torch.Tensor, w: torch.Tensor, bias) -> torch.Tensor:
        return F.conv2d(
            x,
            w,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def _weight_on_device(self) -> torch.Tensor:
        """Place the conv weight into its tiled layout once, then cache."""
        if self._w_dev is None:
            w_cpu = self.weight.detach().to("cpu")
            self._w_dev = w_cpu.to(  # ty: ignore[no-matching-overload]
                "spyre", device_layout=_weight_layout(w_cpu)
            )
        return self._w_dev

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        # `_forward_conv`, not `forward_native`: a patch embed sets `enable_linear`, so
        # `forward_native` picks the unfold/reshape path this class exists to avoid.
        if x.device.type != "spyre":
            # The tiled layouts move a tensor onto the card; applying them to a
            # CPU input is the opposite of what the caller asked for.
            return self._forward_conv(x)
        if not _layouts_supported(x, self.weight):
            logger.warning_once(
                "Spyre conv2d: shape %s (weight %s) outside the tiled-layout "
                "assumptions (batch 1, in_channels <= 64, out_channels %% 64 == 0); "
                "falling back to F.conv2d without them.",
                tuple(x.shape),
                tuple(self.weight.shape),
            )
            return self._forward_conv(x)
        logger.info_once("Spyre conv2d: on-card F.conv2d with tiled layouts")
        # Via CPU: CPU->spyre is the tested entry path, and a device-side
        # restickify would hit the same unsupported layout.
        x_cpu = x.to("cpu")
        x_dev = x_cpu.to(  # ty: ignore[no-matching-overload]
            "spyre", device_layout=_input_layout(x_cpu)
        )
        return self._conv_native(x_dev, self._weight_on_device(), self.bias)
