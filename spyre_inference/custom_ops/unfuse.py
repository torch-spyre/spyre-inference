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

"""Model-agnostic weight-unfusing pass for the Spyre backend.

Fused projections force a Spyre->CPU->Spyre roundtrip: splitting a fused
weight's output on Spyre yields strided views that corrupt on transfer. This
pass splits the fused weight into contiguous Parameters at load time (on CPU),
each stored transposed ([in, out]) so the forward GEMM is the Spyre-fast
`x @ A` (see linear.py / torch-spyre #3512).

The un-fused forward returns a `SplitQKV` (a `SplitProjection`) holding the
pre-split parts. It mimics `Tensor.split`/`.chunk`, so the unmodified attention
idiom `q, k, v = qkv.split(...)` keeps working.
"""

import types
from typing import cast

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    UnquantizedLinearMethod,
)

from .linear import spyre_linear_t


logger = init_logger(__name__)


class SplitProjection:
    """Base container for the un-fused projection."""

    __slots__ = ("_parts",)

    def __init__(self, *parts: torch.Tensor):
        self._parts = parts


class SplitQKV(SplitProjection):
    """QKV split (q, k, v) that mimic `Tensor.split` / `Tensor.chunk`."""

    __slots__ = ()

    def split(self, sizes, dim=-1):
        q, k, v = self._parts
        # We only pre-split along the last (feature) dim; anything else is not
        # the qkv idiom we assumed.
        assert dim in (-1, q.ndim - 1), f"split expected the last dim, got {dim}"
        expected = [q.shape[-1], k.shape[-1], v.shape[-1]]
        assert list(sizes) == expected, f"split expected q/k/v sizes {expected}, got {list(sizes)}"
        return q, k, v

    def chunk(self, chunks, dim=-1):
        q, k, v = self._parts
        # Only a 3-way chunk maps onto (q, k, v), and only when q/k/v are
        # equal-sized (non-GQA); anything else is not the idiom, so fail closed.
        assert chunks == 3, f"chunk expected chunks=3 (q/k/v), got {chunks}"
        assert dim in (-1, q.ndim - 1), f"chunk expected the last dim, got {dim}"
        assert q.shape[-1] == k.shape[-1] == v.shape[-1], (
            "chunk(3) requires equal-sized q/k/v (non-GQA); "
            f"got {q.shape[-1]}/{k.shape[-1]}/{v.shape[-1]}"
        )
        return q, k, v


def _bias_data(layer: nn.Module, name: str):
    b = getattr(layer, name, None)
    return b.data if b is not None else None


def _assert_cpu(module: nn.Module, name: str) -> None:
    assert module.weight.device.type == "cpu", (
        f"analyze_and_unfuse: expected fused weight of {name} on CPU at "
        f"unfuse time, got {module.weight.device}. Splitting on Spyre "
        f"would hit the non-contiguous-view corruption bug."
    )


def _is_unquantized(module: nn.Module) -> bool:
    return isinstance(getattr(module, "quant_method", None), UnquantizedLinearMethod)


def _unfusable(module: nn.Module) -> bool:
    """True if `module` carries a fused, unquantized weight to split."""
    return getattr(module, "weight", None) is not None and _is_unquantized(module)


def _split_into_params(module: nn.Module, names: list[str], sizes: list[int]) -> None:
    """Split `module.weight` (and bias) row-wise into named per-part Parameters.

    Adds `<name>_weight` Parameters (transposed to `[in, out]`, contiguous, on
    CPU) and `<name>_bias` Parameters, then clears the fused `weight`/`bias` to
    None.
    """
    w = cast(torch.Tensor, module.weight.data)
    assert sum(sizes) == w.shape[0], (
        f"analyze_and_unfuse: part sizes {sizes} != fused weight rows "
        f"{w.shape[0]}; refusing to split."
    )

    # Store each part transposed ([in, out]) so the forward GEMM is the
    # Spyre-fast `x @ A` instead of `F.linear`'s `x @ Aᵀ` (torch-spyre #3512).
    for name, part in zip(names, torch.split(w, sizes, dim=0)):
        setattr(
            module,
            f"{name}_weight",
            Parameter(part.contiguous().t().contiguous(), requires_grad=False),
        )

    if getattr(module, "bias", None) is not None:
        bias_data = cast(torch.Tensor, module.bias.data)
        for name, part in zip(names, torch.split(bias_data, sizes, dim=0)):
            setattr(module, f"{name}_bias", Parameter(part.contiguous(), requires_grad=False))
        module.bias = None

    module.weight = None


def _fused_bias(module: nn.Module, names: list[str]):
    """Re-concatenate the per-part biases into the original fused bias tensor.

    Mirrors the fused bias that `skip_bias_add` returns upstream.
    """
    parts = [getattr(module, f"{n}_bias", None) for n in names]
    if any(p is None for p in parts):
        return None
    return torch.cat([cast(torch.Tensor, p).data for p in parts])


def _split_forward(output, module: nn.Module, names: list[str]):
    """Apply the return_bias / skip_bias_add contract to a split `output`."""
    if not module.return_bias:
        return output
    output_bias = _fused_bias(module, names) if module.skip_bias_add else None
    return output, output_bias


def _qkv_forward(self, x: torch.Tensor):
    """Rebound QKVParallelLinear.forward -> SplitQKV (+ optional bias)."""
    fold_bias = not self.skip_bias_add
    q = spyre_linear_t(x, self.q_weight.data, _bias_data(self, "q_bias") if fold_bias else None)
    k = spyre_linear_t(x, self.k_weight.data, _bias_data(self, "k_bias") if fold_bias else None)
    v = spyre_linear_t(x, self.v_weight.data, _bias_data(self, "v_bias") if fold_bias else None)
    return _split_forward(SplitQKV(q, k, v), self, ["q", "k", "v"])


def _unfuse_qkv(module: QKVParallelLinear) -> None:
    """Split the fused QKV weight into q/k/v Parameters and rebind forward."""
    _assert_cpu(module, "qkv_proj")
    sizes = [
        module.num_heads * module.head_size,
        module.num_kv_heads * module.head_size,
        module.num_kv_heads * module.v_head_size,
    ]
    _split_into_params(module, ["q", "k", "v"], sizes)
    module.forward = types.MethodType(_qkv_forward, module)  # ty: ignore[invalid-assignment]


def analyze_and_unfuse(model: nn.Module) -> None:
    """Analyze the model after the checkpoint is loaded (weights on CPU).

    Every unquantized QKVParallelLinear is un-fused.
    """
    n_qkv = 0
    for module in model.modules():
        if isinstance(module, QKVParallelLinear) and _unfusable(module):
            _unfuse_qkv(module)
            n_qkv += 1

    logger.debug("Spyre weight-unfusing: unfused %d QKV projections.", n_qkv)
