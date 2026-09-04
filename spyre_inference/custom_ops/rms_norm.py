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
    - fp16->fp32 ops are not registered as Spyre kernels (torch_spyre/ops/eager.py),
      so the promotion runs on CPU behind an opaque op and copies back.

References:
    - Upstream RMSNorm: vllm/model_executor/layers/layernorm.py
"""

from functools import lru_cache

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.transformers.fusers.rms_norm import TPAwareRMSNorm
from vllm.utils.torch_utils import direct_register_custom_op

from .lazy_compile import CompileOutermost, compile_when_outermost
from .utils import convert

logger = init_logger(__name__)


def _rms_norm_fp32_op(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Opaque-op body: the fp32 CPU core of RMSNorm (variance + rsqrt normalize).

    torch-spyre registers no fp16->fp32 ops, so the promotion runs on CPU; as one
    opaque node its ``mean`` never reaches Inductor's cpp backend, which cannot
    codegen it. Single-output on purpose: a multi-output op lowers to a
    ``MultiOutput`` buffer that torch-spyre's restickify pass asserts on.
    """
    xf = convert(x, device="cpu", dtype=torch.float32)
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    return convert(xf * torch.rsqrt(variance + eps), device=x.device, dtype=x.dtype)


def _rms_norm_fp32_fake(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


@lru_cache(maxsize=1)
def register():
    """Register ``torch.ops.vllm.spyre_rms_norm_fp32`` once.

    CompositeExplicitAutograd so it dispatches on Spyre tensors while the body runs
    on CPU (mirrors ``spyre_convert``)."""
    direct_register_custom_op(
        op_name="spyre_rms_norm_fp32",
        op_func=_rms_norm_fp32_op,
        fake_impl=_rms_norm_fp32_fake,
        dispatch_key="CompositeExplicitAutograd",
    )
    logger.debug_once("Registered custom op: spyre_rms_norm_fp32")


@RMSNorm.register_oot(name="RMSNorm")
class SpyreRMSNorm(CompileOutermost, RMSNorm):
    """Out-of-tree (OOT) RMSNorm implementation for IBM's Spyre."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.info_once("SpyreRMSNorm: fp32 promotion runs on CPU via spyre_rms_norm_fp32.")

    @compile_when_outermost
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm kernel for Spyre."""

        if self.variance_size_override is not None:
            raise NotImplementedError("TODO: variance_size_override not yet implemented")

        # Residual add stays on-device in fp16: upstream rounds the sum back to
        # orig_dtype anyway, so the extra precision would reach only the variance.
        if residual is not None:
            x = x + residual
            residual = x

        # fp32 for the variance, matching upstream: x**2 overflows fp16 above |x| ~ 256.
        x = torch.ops.vllm.spyre_rms_norm_fp32(
            x,  # ty: ignore[invalid-argument-type]
            self.variance_epsilon,  # ty: ignore[invalid-argument-type]
        )
        if self.has_weight:
            x = x * self.weight

        return x if residual is None else (x, residual)


# The norm fuser instantiates TPAwareRMSNorm and OOT dispatch keys on the concrete class
# name, so the fused norm needs its own entry.
@RMSNorm.register_oot(name="TPAwareRMSNorm")
class SpyreTPAwareRMSNorm(TPAwareRMSNorm, SpyreRMSNorm):
    """Spyre RMSNorm that reconstructs a TP-sharded input before normalizing."""
