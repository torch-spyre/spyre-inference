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

"""Spyre OOT replacement for GemmaRMSNorm.

Gemma models (1/2/3) use GemmaRMSNorm for every normalization (input/post-attn/
pre-post-feedforward layernorms and gemma-3's per-head q_norm/k_norm).

The fp16->fp32 upcast is correct only through torch-spyre's compile-time EA
propagation (PR #2927), broken in eager. Hence force compiling here.
"""

import torch

from vllm.model_executor.layers.layernorm import GemmaRMSNorm


@GemmaRMSNorm.register_oot(name="GemmaRMSNorm")
class SpyreGemmaRMSNorm(GemmaRMSNorm):
    """Out-of-tree (OOT) GemmaRMSNorm implementation for IBM's Spyre."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # With fullgraph compile enabled, forward_native is compiled anyway.
        self._forward = self.forward_native
        if not torch.compiler.is_dynamo_compiling():
            self._forward = torch.compile(self.forward_native, dynamic=False)

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return self._forward(x, residual)
        return self._forward(x)
