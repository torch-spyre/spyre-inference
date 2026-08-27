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

"""Spyre OOT replacement for RMSNorm.

The fp16->fp32 upcast is correct only through torch-spyre's compile-time EA
propagation (PR #2927), broken in eager. Hence force compiling here.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.transformers.fusers.rms_norm import TPAwareRMSNorm

from .lazy_compile import CompileOutermost, compile_when_outermost

logger = init_logger(__name__)


@RMSNorm.register_oot(name="RMSNorm")
class SpyreRMSNorm(CompileOutermost, RMSNorm):
    """Out-of-tree (OOT) RMSNorm implementation for IBM's Spyre."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # With fullgraph compile enabled, forward_native is compiled anyway.
        self._forward = self.forward_native
        if not torch.compiler.is_dynamo_compiling():
            self._forward = torch.compile(self.forward_native, dynamic=False)

        logger.warning_once(
            "SpyreRMSNorm: no dtype promotion is performed, "
            "expect numerical differences to upstream vLLM."
        )

    @compile_when_outermost
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._forward(x, residual)


# The norm fuser instantiates TPAwareRMSNorm and OOT dispatch keys on the concrete class
# name, so the fused norm needs its own entry.
@RMSNorm.register_oot(name="TPAwareRMSNorm")
class SpyreTPAwareRMSNorm(TPAwareRMSNorm, SpyreRMSNorm):
    """Spyre RMSNorm that reconstructs a TP-sharded input before normalizing."""
