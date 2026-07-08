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
pass splits the fused weight into contiguous Parameters at load time
(on CPU).

Each un-fused forward returns a `SplitProjection` holding the pre-split parts.
The subclass matches its downstream consumer:

  * SplitQKV        (QKVParallelLinear): mimics `Tensor.split`/`.chunk`, so the
    unmodified attention idiom `q, k, v = qkv.split(...)` keeps working.
  * SplitSiluAndMul (MergedColumnParallelLinear feeding SiluAndMul): iterable,
    so `gate, up = proj` unpacks the two parts for SpyreSiluAndMul.
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    UnquantizedLinearMethod,
)

from .silu_and_mul import SpyreSiluAndMul

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


class SplitSiluAndMul(SplitProjection):
    """Gate/up parts, unpackable as `gate, up = proj` by SpyreSiluAndMul."""

    __slots__ = ()

    def __iter__(self):
        return iter(self._parts)


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

    Adds `<name>_weight`/`<name>_bias` Parameters (contiguous, on CPU) and then
    clears the fused `weight`/`bias` to None.
    """
    w = module.weight.data
    assert sum(sizes) == w.shape[0], (
        f"analyze_and_unfuse: part sizes {sizes} != fused weight rows "
        f"{w.shape[0]}; refusing to split."
    )

    for name, part in zip(names, torch.split(w, sizes, dim=0)):
        setattr(module, f"{name}_weight", Parameter(part.contiguous(), requires_grad=False))

    if getattr(module, "bias", None) is not None:
        for name, part in zip(names, torch.split(module.bias.data, sizes, dim=0)):
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
    return torch.cat([p.data for p in parts])


def _split_forward(output, module: nn.Module, names: list[str]):
    """Apply the return_bias / skip_bias_add contract to a split `output`."""
    if not module.return_bias:
        return output
    output_bias = _fused_bias(module, names) if module.skip_bias_add else None
    return output, output_bias


def _qkv_forward(self, x: torch.Tensor):
    """Rebound QKVParallelLinear.forward -> SplitQKV (+ optional bias)."""
    fold_bias = not self.skip_bias_add
    q = F.linear(x, self.q_weight.data, _bias_data(self, "q_bias") if fold_bias else None)
    k = F.linear(x, self.k_weight.data, _bias_data(self, "k_bias") if fold_bias else None)
    v = F.linear(x, self.v_weight.data, _bias_data(self, "v_bias") if fold_bias else None)
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
    module.forward = types.MethodType(_qkv_forward, module)


def _silu_and_mul_forward(self, x: torch.Tensor):
    """Rebound MergedColumnParallelLinear.forward -> SplitSiluAndMul (+ optional bias)."""
    fold_bias = not self.skip_bias_add
    gate = F.linear(x, self.gate_weight.data, _bias_data(self, "gate_bias") if fold_bias else None)
    up = F.linear(x, self.up_weight.data, _bias_data(self, "up_bias") if fold_bias else None)
    return _split_forward(SplitSiluAndMul(gate, up), self, ["gate", "up"])


def _unfuse_silu_and_mul(module: MergedColumnParallelLinear) -> None:
    """Split a gate_up_proj weight into gate/up Parameters, rebind forward."""
    _assert_cpu(module, "gate_up_proj")
    sizes = list(module.output_partition_sizes)  # per-rank, TP-correct
    _split_into_params(module, ["gate", "up"], sizes)
    module.forward = types.MethodType(_silu_and_mul_forward, module)


def _gate_up_sibling(act_fn: nn.Module, parent_of: dict[int, nn.Module]):
    """The gate_up_proj MergedColumnParallelLinear feeding `act_fn`, if any."""
    parent = parent_of.get(id(act_fn))
    if parent is None:
        return None
    for child in parent.children():
        if isinstance(child, MergedColumnParallelLinear) and len(child.output_partition_sizes) == 2:
            return child
    return None


def analyze_and_unfuse(model: nn.Module) -> None:
    """Analyze the model after the checkpoint is loaded (weights on CPU).

    Cases currently handled:
      * QKV: every unquantized QKVParallelLinear is un-fused.
      * SiluAndMul: driven from each SpyreSiluAndMul activation, un-fusing its
        sibling gate_up_proj — the only projection with a part-consuming
        consumer.
    """
    parent_of = {id(model): model}
    for parent in model.modules():
        for child in parent.children():
            parent_of[id(child)] = parent

    n_qkv = 0
    n_silu_and_mul = 0
    for module in model.modules():
        # QKV projections.
        if isinstance(module, QKVParallelLinear) and _unfusable(module):
            _unfuse_qkv(module)
            n_qkv += 1
        # Gate/up projections feeding SiluAndMul.
        if isinstance(module, SpyreSiluAndMul):
            gate_up = _gate_up_sibling(module, parent_of)
            if gate_up is not None and _unfusable(gate_up):
                _unfuse_silu_and_mul(gate_up)
                n_silu_and_mul += 1

    logger.debug(
        "Spyre weight-unfusing: unfused %d QKV and %d gate/up projections.",
        n_qkv,
        n_silu_and_mul,
    )
