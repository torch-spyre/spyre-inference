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

"""Compile upstream FP32 GemmaRMSNorm when it is outside a block graph.

Gemma models (1/2/3) use GemmaRMSNorm for every normalization (input/post-attn/
pre-post-feedforward layernorms and gemma-3's per-head q_norm/k_norm).

References:
    - Upstream GemmaRMSNorm: vllm/model_executor/layers/layernorm.py
"""

import torch
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.models.transformers.fusers.rms_norm import TPAwareGemmaRMSNorm

from .lazy_compile import CompileOutermost, compile_when_outermost


@GemmaRMSNorm.register_oot(name="GemmaRMSNorm")
class SpyreGemmaRMSNorm(CompileOutermost, GemmaRMSNorm):
    """Out-of-tree (OOT) GemmaRMSNorm implementation for IBM's Spyre."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @compile_when_outermost(force_compile=True)
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the unchanged vLLM native implementation."""
        return super().forward_native(x, residual)


# See the SpyreTPAwareRMSNorm note in rms_norm.py.
@GemmaRMSNorm.register_oot(name="TPAwareGemmaRMSNorm")
class SpyreTPAwareGemmaRMSNorm(TPAwareGemmaRMSNorm, SpyreGemmaRMSNorm):
    """Spyre GemmaRMSNorm that reconstructs a TP-sharded input before normalizing."""
