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

"""Spyre OOT replacement for Conv2dLayer.

References:
    - Upstream: vllm/model_executor/layers/conv.py
"""

import torch
from vllm.model_executor.layers.conv import Conv2dLayer


@Conv2dLayer.register_oot(name="Conv2dLayer")
class SpyreConv2dLayer(Conv2dLayer):
    """Always use F.conv2d, never the unfold-as-matmul `_forward_mulmat` path.

    `_forward_mulmat`'s `unfold().unfold().permute().reshape()` forces an
    on-device relayout whose stick expression mixes two independent kernel-
    window iterations whenever kernel_size < 64 (Spyre's fp16 stick width),
    e.g. patch_size=32 for clip-vit-base-patch32. torch-spyre's Inductor
    restickify pass has no legalization rule for that pattern and raises
    `Unsupported: ... stick expression`. `F.conv2d` decomposes through
    torch-spyre's own hardened `aten.unfold`/reshape ops, which already
    CPU-fall-back for unsupported layouts, so it works under both eager and
    `STOCK_TORCH_COMPILE`.
    """

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        return self._forward_conv(x)
