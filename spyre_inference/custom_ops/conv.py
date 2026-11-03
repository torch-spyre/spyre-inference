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

"""Spyre-specific Conv2d implementation (vision patch embeds such as CLIP and SigLIP).

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


@Conv2dLayer.register_oot(name="Conv2dLayer")
class SpyreConv2d(CompileOutermost, Conv2dLayer):
    """Out-of-tree Conv2d for Spyre: `F.conv2d` on-card.

    Spyre needs static shapes, so the kernel recompiles per distinct (B, H, W). Warmup
    covers the batch widths up to the image limit at the dummy resolution; every other
    resolution costs a full compile mid-request — bucket or resize images if a workload
    uses many resolutions.
    """

    # Per-(B, H, W) recompiles are this layer's contract, so the compile guard must not
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
        logger.info_once("Spyre conv2d: on-card F.conv2d")
        return self._conv_native(x, self.weight, self.bias)
