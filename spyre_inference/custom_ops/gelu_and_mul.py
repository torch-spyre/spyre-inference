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
`forward_native` slices the fused `[..., 2*d]` tensor on the last dim; on Spyre
that slice now works in eager mode (torch-spyre#3578), so `forward_oot` simply
calls it directly. Mirrors `SpyreSiluAndMul` with GELU instead of SiLU.
"""

import torch

from vllm.model_executor.layers.activation import GeluAndMul


@GeluAndMul.register_oot(name="GeluAndMul")
class SpyreGeluAndMul(GeluAndMul):
    """Out-of-tree (OOT) GeluAndMul implementation for IBM's Spyre device."""

    def forward_oot(self, x) -> torch.Tensor:
        """GeGLU: gelu(gate) * up, output shape [..., d]."""

        return self.forward_native(x)
