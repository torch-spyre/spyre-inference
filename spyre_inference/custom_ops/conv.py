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
coprime with the 64-wide stick. So call the real `F.conv2d` and let torch-spyre
lower it.
"""

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.conv import Conv2dLayer

from .lazy_compile import CompileOutermost, maybe_compile

logger = init_logger(__name__)


def _layouts_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether this call is a vision patch embed: in-channels within a stick,
    out-channels a whole number of sticks, any batch.

    True for a patch embed, not for convs in general — and this class is
    registered OOT for *every* `Conv2dLayer`, so fall back rather than assert.
    """
    if x.dim() != 4 or weight.dim() != 4:
        return False
    c = x.shape[1]
    return c <= 64 and weight.shape[0] % 64 == 0


@Conv2dLayer.register_oot(name="Conv2dLayer")
class SpyreConv2d(CompileOutermost, Conv2dLayer):
    """Out-of-tree Conv2d for Spyre: `F.conv2d` on-card.

    Spyre needs static shapes, so the kernel recompiles per distinct (H, W). The
    platform lifts dynamo's recompile limits, so there is no eager fallback, but every
    new resolution costs a full compile mid-request — bucket or resize images if a
    workload uses many resolutions.
    """

    # Per-(H, W) recompiles are this layer's contract, so the compile guard must not
    # report them; warmup cannot enumerate every image resolution.
    allow_inference_recompiles = True

    @maybe_compile
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

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        # `_forward_conv`, not `forward_native`: a patch embed sets `enable_linear`, so
        # `forward_native` picks the unfold/reshape path this class exists to avoid.
        if x.device.type != "spyre":
            return self._forward_conv(x)
        if not _layouts_supported(x, self.weight):
            logger.warning_once(
                "Spyre conv2d: shape %s (weight %s) outside the patch-embed "
                "assumptions (in_channels <= 64, out_channels %% 64 == 0); "
                "falling back to the stock conv path.",
                tuple(x.shape),
                tuple(self.weight.shape),
            )
            return self._forward_conv(x)
        logger.info_once("Spyre conv2d: on-card F.conv2d")
        return self._conv_native(x, self.weight, self.bias)
