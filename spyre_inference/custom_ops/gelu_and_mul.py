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

"""Spyre-specific GeluAndMul implementation (GeGLU).

Gemma models use `gelu_pytorch_tanh` gated MLPs -> vLLM's `GeluAndMul`. The stock
`forward_native` slices the fused `[..., 2*d]` tensor on-device, which corrupts
Spyre memory (same hazard `SpyreSiluAndMul` works around). This mirrors that
override with GELU instead of SiLU.
"""

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.activation import GeluAndMul

from .utils import convert


@GeluAndMul.register_oot(name="GeluAndMul")
class SpyreGeluAndMul(GeluAndMul):
    """Out-of-tree (OOT) GeluAndMul implementation for IBM's Spyre device."""

    def forward_oot(self, x) -> torch.Tensor:
        """GeGLU: gelu(gate) * up, output shape [..., d].

        `x` is either a pre-split gate/up pair (a SplitGateUp from an un-fused
        gate_up_proj, see unfuse.py) or a fused [..., 2*d] tensor. The fused
        path only runs for layers unfuse left alone (e.g. quantized); slicing a
        Spyre tensor corrupts memory, so it slices on CPU (mirrors
        SpyreSiluAndMul).
        """
        if not isinstance(x, torch.Tensor):
            # Unfused path: gate/up already split, both contiguous on device.
            x1, x2 = x
        else:
            # Fused path: slice on CPU — slicing a Spyre tensor corrupts memory,
            # and non-contiguous slices corrupt again on transfer back.
            original_device = x.device
            x = convert(x, device="cpu")
            d = x.shape[-1] // 2
            x1 = convert(x[..., :d].contiguous(), device=original_device)
            x2 = convert(x[..., d:].contiguous(), device=original_device)
        return F.gelu(x1, approximate=self.approximate) * x2
