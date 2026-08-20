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

"""Spyre OOT linear layers and the transposed-weight fast path.

`SpyreUnquantizedLinearMethod` stores each 2-D linear
weight physically transposed as `Wᵀ` (shape `[in, out]`, contiguous) in
`process_weights_after_loading`, which is more efficient on Spyre.
"""

from typing import cast

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)

logger = init_logger(__name__)


def spyre_linear_t(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor | None):
    """Linear forward with a pre-transposed weight: `x @ Wᵀ (+ bias)`.

    `weight_t` is the physically-transposed weight of shape `[in, out]`, so the
    matmul is a plain `x @ A` (the Spyre-fast layout), not `F.linear`'s `x @ Aᵀ`.
    """
    out = torch.matmul(x, weight_t)
    if bias is not None:
        out = out + bias
    return out


class SpyreUnquantizedLinearMethod(UnquantizedLinearMethod):
    """Unquantized linear method that stores `Wᵀ` and matmuls it directly.

    Overrides only the two hooks that matter: `process_weights_after_loading`
    transposes the loaded weight to `[in, out]`, and `apply` runs the Spyre-fast
    `x @ Wᵀ` instead of `F.linear`'s `x @ Aᵀ` (torch-spyre #3512).
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        # Store the transposed layout (`[in, out]`, contiguous) so the forward
        # GEMM is the Spyre-fast `x @ A`.
        weight = cast(torch.Tensor, layer.weight)
        layer.weight = Parameter(weight.data.t().contiguous(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return spyre_linear_t(x, cast(torch.Tensor, layer.weight), bias)


class _SpyreTransposedLinearMixin:
    """Swaps in `SpyreUnquantizedLinearMethod` for unquantized linear layers.

    Mixed in before a concrete vLLM linear class so `super().__init__` builds the
    layer normally; we then replace the unquantized method with the transposed
    one. Quantized layers keep their own method (and the slow `F.linear` path):
    the transpose fast path only applies to unquantized weights.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.quant_method, UnquantizedLinearMethod):
            self.quant_method = SpyreUnquantizedLinearMethod()


@ColumnParallelLinear.register_oot(name="ColumnParallelLinear")
class SpyreColumnParallelLinear(_SpyreTransposedLinearMixin, ColumnParallelLinear):
    """OOT ColumnParallelLinear storing `Wᵀ` for the Spyre-fast GEMM."""


@MergedColumnParallelLinear.register_oot(name="MergedColumnParallelLinear")
class SpyreMergedColumnParallelLinear(_SpyreTransposedLinearMixin, MergedColumnParallelLinear):
    """OOT MergedColumnParallelLinear (e.g. gate_up_proj) storing `Wᵀ`."""


@RowParallelLinear.register_oot(name="RowParallelLinear")
class SpyreRowParallelLinear(_SpyreTransposedLinearMixin, RowParallelLinear):
    """OOT RowParallelLinear (e.g. o_proj, down_proj) storing `Wᵀ`."""


@ReplicatedLinear.register_oot(name="ReplicatedLinear")
class SpyreReplicatedLinear(_SpyreTransposedLinearMixin, ReplicatedLinear):
    """OOT ReplicatedLinear storing `Wᵀ` for the Spyre-fast GEMM."""


@QKVParallelLinear.register_oot(name="QKVParallelLinear")
class SpyreQKVParallelLinear(_SpyreTransposedLinearMixin, QKVParallelLinear):
    """OOT QKVParallelLinear for IBM's Spyre device.

    The fused QKV output is returned whole; the model splits it on-device with
    the unmodified `qkv.split(...)` idiom (no CPU-side unfusing).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.gather_output, (
            f"{self.__class__.__name__} requires gather_output=False; "
            "all_gather is not yet supported on Spyre"
        )
