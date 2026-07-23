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

Spyre constraints:
    - No dtype promotion to float32 (not yet supported in torch-spyre)

References:
    - Upstream RMSNorm: vllm/model_executor/layers/layernorm.py
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm

logger = init_logger(__name__)


@RMSNorm.register_oot(name="RMSNorm")
class SpyreRMSNorm(RMSNorm):
    """Out-of-tree (OOT) RMSNorm implementation for IBM's Spyre."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.warning_once(
            "SpyreRMSNorm: no dtype promotion is performed, "
            "expect numerical differences to upstream vLLM."
        )

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm kernel for Spyre."""

        if self.variance_size_override is not None:
            raise NotImplementedError("TODO: variance_size_override not yet implemented")

        if residual is not None:
            x = x + residual
            residual = x

        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size to be {self.hidden_size}, but found: {x.shape[-1]}"
            )

        variance = x.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.variance_epsilon)

        if self.has_weight:
            x = x * self.weight
        if residual is None:
            return x
        else:
            return x, residual
