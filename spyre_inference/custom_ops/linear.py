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

`F.linear(x, W)` computes `x @ Wᵀ` regardless of how `W` is laid out, and on
Spyre the transposed matmul is ~3.5x slower than a plain `x @ A`
(torch-spyre issue #3512). `transpose_linear_weights_for_spyre` stores each 2-D
linear weight physically transposed as `Wᵀ` (shape `[in, out]`, contiguous) at
load time and swaps the GEMM to `x @ Wᵀ`, which is the fast path. It is the
pure-PyTorch equivalent of torch-spyre's `[1,0]` weight layout, which only fires
for `nn.Linear` and so misses every vLLM parallel-linear.

The pass runs on CPU before the move to Spyre. QKV projections keep their fused
weight and are handled by ``_patch_attention_qkv_splits`` in the model runner
(compiled slice+clone). The LM head transposes ``padded_weight`` in
``parallel_lm_head.py``. Both reuse ``spyre_linear_t`` here so the fast path is
defined once.
"""

import types

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    QKVParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

logger = init_logger(__name__)


@QKVParallelLinear.register_oot(name="QKVParallelLinear")
class SpyreQKVParallelLinear(QKVParallelLinear):
    """Out-of-tree (OOT) QKVParallelLinear implementation for IBM's Spyre device."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.gather_output, (
            f"{self.__class__.__name__} requires gather_output=False; "
            "all_gather is not yet supported on Spyre"
        )


def spyre_linear_t(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor | None):
    """Linear forward with a pre-transposed weight: `x @ Wᵀ (+ bias)`.

    `weight_t` is the physically-transposed weight of shape `[in, out]`, so the
    matmul is a plain `x @ A` (the Spyre-fast layout), not `F.linear`'s `x @ Aᵀ`.
    """
    out = torch.matmul(x, weight_t)
    if bias is not None:
        out = out + bias
    return out


def _transposed_apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None):
    """Replacement for `UnquantizedLinearMethod.apply` using the transposed weight."""
    return spyre_linear_t(x, layer.weight_t.data, bias)  # ty: ignore[invalid-argument-type]


def _is_unquantized(module: nn.Module) -> bool:
    return isinstance(getattr(module, "quant_method", None), UnquantizedLinearMethod)


def _transpose_weight(module: nn.Module, name: str) -> None:
    """Replace `module.<name>` (a 2-D `[out, in]` Parameter) with `<name>_t` (`[in, out]`)."""
    w = getattr(module, name).data
    setattr(module, f"{name}_t", Parameter(w.t().contiguous(), requires_grad=False))
    setattr(module, name, None)


def transpose_linear_weights_for_spyre(model: nn.Module) -> None:
    """Transpose 2-D linear weights so the forward GEMM is `x @ A` (Spyre-fast).

    Runs after the checkpoint is loaded (weights on CPU). Covers generic
    unquantized `LinearBase` layers; fused QKV (handled by compiled
    slice+clone in the model runner) and the LM head are transposed by
    their own modules.
    """
    n_linear = 0
    n_quantized = 0
    for module in model.modules():
        if not isinstance(module, LinearBase):
            continue
        if not _is_unquantized(module):
            # Quantized LinearBase keeps the slow `F.linear` (`x @ Aᵀ`) path: we
            # only transpose unquantized weights. Count it so we can warn — the
            # embedding / LM-head paths reject quantization outright, but generic
            # linears would otherwise skip silently.
            n_quantized += 1
            continue
        w = getattr(module, "weight", None)
        # Skip QKV (fused weight transposed by the model runner's compiled
        # slice+clone path) and anything without a plain 2-D weight.
        if w is None or w.dim() != 2 or w.device.type != "cpu":
            continue
        _transpose_weight(module, "weight")
        module.quant_method.apply = types.MethodType(  # ty: ignore[invalid-assignment]
            _transposed_apply, module.quant_method
        )
        n_linear += 1

    if n_quantized > 0:
        logger.warning_once(
            "Spyre linear transpose: %d quantized LinearBase layer(s) left on the "
            "slow `F.linear` (x @ Aᵀ) path — the transpose fast path (torch-spyre "
            "#3512) only applies to unquantized weights.",
            n_quantized,
        )

    n_head = sum(1 for m in model.modules() if isinstance(m, ParallelLMHead))
    logger.debug(
        "Spyre linear transpose: transposed %d linear weights (%d LM head(s) "
        "handled in parallel_lm_head).",
        n_linear,
        n_head,
    )
