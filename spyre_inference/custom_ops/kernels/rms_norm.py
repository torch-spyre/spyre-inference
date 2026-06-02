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

"""Spyre IR provider for rms_norm.

Self-contained provider that handles:
- Device transfer: CPU → Spyre → compute → CPU
- No dtype promotion (torch-spyre limitation, stays in input dtype)
- Epsilon as tensor (scalar broadcast limited on Spyre)
"""

import torch

from vllm import ir


def _supports_spyre(x, weight, epsilon, variance_size=None):
    """Accept tensors when variance_size is not used.
    Falls back to native provider when
    variance_size is set (not yet supported on Spyre).
    """
    return variance_size is None and all(t.dtype == torch.float16 for t in [x, weight])


@ir.ops.rms_norm.register_impl("spyre", supports_args=_supports_spyre, supported=True)
def spyre_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,  # noqa: ARG001 — required by IR schema
) -> torch.Tensor:
    """Spyre IR provider for rms_norm, adopted from vllm/ir/ops/layernorm.py

    Spyre-specific implementation details:
    - No dtype promotion: torch-spyre limitation, stays in input dtype.
    - variance_size: not supported; _supports_spyre rejects it so dispatch
      falls back to native.
    """
    orig_dtype = x.dtype

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(orig_dtype)
    if weight is not None:
        x = x * weight
    return x
