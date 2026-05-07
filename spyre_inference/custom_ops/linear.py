# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific linear layer implementations using out-of-tree (OOT) registration.

This module provides Spyre-device-specific replacements for the parallel linear
layer classes used inside MLP blocks:

    - SpyreMergedColumnParallelLinear  — replaces MergedColumnParallelLinear
      (vllm/model_executor/layers/linear.py)
    - SpyreQKVParallelLinear          — replaces QKVParallelLinear
      (vllm/model_executor/layers/linear.py)
    - SpyreRowParallelLinear          — replaces RowParallelLinear
      (vllm/model_executor/layers/linear.py)

At TP=1, the upstream forward() methods reduce to quant_method.apply() + bias
handling.  We inject a custom quant_method (SpyreUnquantizedLinearMethod) that
performs F.linear directly, QKV and RowParallel still override forward()
for device placement (D2H after GEMM, H2D before GEMM).

Spyre Device Constraints:
    - Computations performed in torch.float16:
      Input (dtype defined by model / user) converted to torch.float16 for
      operations on spyre and then converted back to original dtype for cpu.
    - Tensor parallelism: TP=1 assumed (single Spyre device)

References:
    - Upstream linear layers:   vllm/model_executor/layers/linear.py
"""

import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)

from .utils import convert

logger = init_logger(__name__)


class SpyreUnquantizedLinearMethod(UnquantizedLinearMethod):
    """Spyre-specific linear method: F.linear without platform GEMM dispatch.

    Replaces the default UnquantizedLinearMethod so that upstream forward()
    methods work unchanged on Spyre at TP=1.

    - create_weights() is inherited — standard ModelWeightParameter works.
    - apply() does F.linear directly (no platform-specific GEMM dispatch).
    - process_weights_after_loading() is a no-op (skips CPU GEMM dispatch).
    """

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight.data, bias)

    def process_weights_after_loading(self, layer):
        pass


class SpyreLinearBase:
    """Shared initialization for Spyre linear layers at TP=1."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tp_size > 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} only supports TP=1, got TP={self.tp_size}"
            )

        if isinstance(self.quant_method, UnquantizedLinearMethod):
            self.quant_method = SpyreUnquantizedLinearMethod()


@MergedColumnParallelLinear.register_oot(name="MergedColumnParallelLinear")
class SpyreMergedColumnParallelLinear(SpyreLinearBase, MergedColumnParallelLinear):
    """Spyre MergedColumnParallelLinear (TP=1 only)."""


@QKVParallelLinear.register_oot(name="QKVParallelLinear")
class SpyreQKVParallelLinear(SpyreLinearBase, QKVParallelLinear):
    """Spyre QKVParallelLinear (TP=1 only)."""

    def forward(self, input_):
        result = super().forward(input_)
        # D2H before downstream .split() — Spyre can't handle strided views
        if self.return_bias:
            return convert(result[0], device="cpu"), result[1]
        return convert(result, device="cpu")


@RowParallelLinear.register_oot(name="RowParallelLinear")
class SpyreRowParallelLinear(SpyreLinearBase, RowParallelLinear):
    """Spyre RowParallelLinear (TP=1 only).
    RowParallelLinear is currently invoked from `GraniteAttention` where
    `input_` is on `cpu` and from `GraniteMLP` where `input_` is on spyre.
    Thus, we always convert the `input_` to `spyre`, which is a NoOp in
    case of `GraniteMLP`.
    """

    def forward(self, input_):
        return super().forward(convert(input_, device=self.weight.device))
