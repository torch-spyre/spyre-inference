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

"""Compile upstream RMSNorm when it is outside a block graph, as the final norm is."""

import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.transformers.fusers.rms_norm import TPAwareRMSNorm

from .lazy_compile import CompileOutermost, maybe_compile


@RMSNorm.register_oot(name="RMSNorm")
class SpyreRMSNorm(CompileOutermost, RMSNorm):
    @maybe_compile
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return super().forward_native(x, residual)


# The norm fuser instantiates TPAwareRMSNorm and OOT dispatch keys on the concrete class
# name, so the fused norm needs its own entry.
@RMSNorm.register_oot(name="TPAwareRMSNorm")
class SpyreTPAwareRMSNorm(TPAwareRMSNorm, SpyreRMSNorm):
    pass
