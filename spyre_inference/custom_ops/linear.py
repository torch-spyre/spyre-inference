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

"""Spyre OOT linear layers and the shared transposed-weight fast path.

`SpyreTransposedWeightMethod` is the common base for the Spyre linear and
lm-head quant methods: it stores each 2-D weight physically transposed as `Wᵀ`
(shape `[in, out]`, contiguous) in `process_weights_after_loading` and runs the
Spyre-fast `x @ Wᵀ` in `apply`. Subclasses parameterize the destination
attribute and an optional output-row padding (needed by the lm-head).
"""

from typing import cast

import torch
import torch.nn.functional as F
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


class SpyreTransposedWeightMethod:
    """Shared Spyre weight handler: store `Wᵀ` (optionally row-padded) and matmul it.

    A mixin combined *before* a concrete vLLM `Unquantized*Method` (which supplies
    `create_weights` and the `QuantizeMethodBase` lineage), so `super()` calls in
    the methods below reach that concrete method via the MRO. Subclasses set:

    - `WEIGHT_T_ATTR`: layer attribute that holds the transposed weight.
      `"weight"` replaces it in place; a distinct name (e.g. `"padded_weight_t"`)
      preserves the original `weight` — required for a tied lm-head whose
      `weight` IS `embed_tokens.weight` and must keep its gather layout.
    - `ROW_ALIGN`: pad the output (row) dim up to a multiple of this before
      transposing (torch-spyre matmul work-division limit); `None` = no padding.
    """

    WEIGHT_T_ATTR: str = "weight"
    ROW_ALIGN: int | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        w = cast(torch.Tensor, layer.weight).data
        padding = (-w.shape[0]) % self.ROW_ALIGN if self.ROW_ALIGN else 0
        layer.spyre_row_padding = padding
        if padding:
            padded = F.pad(w, (0, 0, 0, padding))
            logger.warning_once(
                "%s: weights padded from %d to %d (torch-spyre limitation) "
                "expect numerical differences to upstream vLLM.",
                layer.__class__.__name__,
                w.shape[0],
                padded.shape[0],
            )
            w = padded

        # Store transposed (`[in, out]`, contiguous) so the forward GEMM is the
        # Spyre-fast `x @ A`. `.t().contiguous()` gives INDEPENDENT storage, so a
        # distinct WEIGHT_T_ATTR leaves the source `weight` untouched.
        setattr(layer, self.WEIGHT_T_ATTR, Parameter(w.t().contiguous(), requires_grad=False))

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = spyre_linear_t(x, getattr(layer, self.WEIGHT_T_ATTR), bias)
        padding = cast(int, layer.spyre_row_padding)
        if padding:
            # Drop the trailing pad columns; the slice lowers on-device eagerly
            # (torch-spyre #3578 honors the storage offset).
            out = out[:, :-padding]
        return out


class SpyreUnquantizedLinearMethod(SpyreTransposedWeightMethod, UnquantizedLinearMethod):
    """Unquantized linear method: store `Wᵀ` in place and matmul it (torch-spyre #3512).

    Uses the shared base defaults (`WEIGHT_T_ATTR="weight"`, no padding), so the
    forward GEMM is the Spyre-fast `x @ Wᵀ` instead of `F.linear`'s `x @ Aᵀ`.
    """


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
