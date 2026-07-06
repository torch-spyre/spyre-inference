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

"""Spyre-specific linear layer implementations using out-of-tree (OOT) registration.

This module provides Spyre-device-specific replacements for the parallel linear
layer classes used inside MLP blocks:

    - SpyreQKVParallelLinear          — replaces QKVParallelLinear
      (vllm/model_executor/layers/linear.py)
    - SpyreRowParallelLinear          — replaces RowParallelLinear
      (vllm/model_executor/layers/linear.py)

MergedColumnParallelLinear is left unmodified: upstream UnquantizedLinearMethod
already dispatches to F.linear on the Spyre platform, and no Spyre-specific
device transfers are needed.

Spyre Device Constraints:
    - Computations performed in torch.float16.
    - Tensor parallelism: TP>=1 supported with all_reduce collectives

"""

import torch.nn.functional as F

from vllm.logger import init_logger

from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)

from .utils import convert

logger = init_logger(__name__)


class SpyreUnquantizedLinearMethod(UnquantizedLinearMethod):
    """Spyre-specific linear method: F.linear without platform GEMM dispatch."""

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight.data, bias)

    def process_weights_after_loading(self, layer):
        pass


class SpyreLinearBase:
    """Shared initialization for Spyre linear layers supporting TP>=1."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(self.quant_method, UnquantizedLinearMethod):
            self.quant_method = SpyreUnquantizedLinearMethod()

        logger.debug_once(
            "Initialized %s with TP=%d, rank=%d",
            self.__class__.__name__,
            self.tp_size,
            self.tp_rank,
        )


@QKVParallelLinear.register_oot(name="QKVParallelLinear")
class SpyreQKVParallelLinear(SpyreLinearBase, QKVParallelLinear):
    """Spyre QKVParallelLinear with TP support.

    Supports TP>=1 with weight sharding for Q, K, V projections.
    Performs device transfers (D2H) after F.linear since downstream .split()
    cannot handle strided views on Spyre.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # QKVParallelLinear hardcodes gather_output=False; all_gather is not yet
        # supported on Spyre, so we rely on this invariant holding.
        assert not self.gather_output, (
            f"{self.__class__.__name__} requires gather_output=False; "
            "all_gather is not yet supported on Spyre"
        )

    def forward(self, input_):
        result = super().forward(input_)
        # D2H so that GraniteAttention's qkv.split() and the subsequent
        # v.view() + kv_cache scatter-write run on CPU. Spyre rejects a
        # non-contiguous tensor as a scatter source; see
        # test_spyre_strided_scatter_source for the minimal reproduction.
        if self.return_bias:
            return convert(result[0], device="cpu"), result[1]
        return convert(result, device="cpu")


@RowParallelLinear.register_oot(name="RowParallelLinear")
class SpyreRowParallelLinear(SpyreLinearBase, RowParallelLinear):
    """Spyre RowParallelLinear with TP support."""

    def forward(self, input_):
        return super().forward(convert(input_, device=self.weight.device))
