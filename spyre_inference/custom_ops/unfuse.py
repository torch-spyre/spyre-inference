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
pass splits the fused weight into contiguous per-projection Parameters at load
time (on CPU) and rebinds forward to run one F.linear per slab, so no split of
a Spyre tensor ever happens. It handles, model-agnostically:

  * QKVParallelLinear -> QKVSplit(q, k, v):
    consumed by Attention via `qkv.split(...)` or `qkv.chunk(3)`.
  * MergedColumnParallelLinear -> [gate, up]:
    consumed by SpyreSiluAndMul, which accepts the pre-split list directly.
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


class QKVSplit:
    """Holds pre-split (q, k, v) and mimics `Tensor.split`/`Tensor.chunk`.

    NOTE: Exposes ONLY split/chunk: any other access raises AttributeError.
    Both entry points fail closed if the caller's request does not match the
    (q, k, v) partition we pre-split, so a mismatched idiom can never silently
    return the wrong tensors.
    """

    __slots__ = ("_q", "_k", "_v")

    def __init__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self._q = q
        self._k = k
        self._v = v

    def split(self, sizes, dim=-1):
        # We only pre-split along the last (feature) dim; anything else is not
        # the qkv idiom we assumed.
        assert dim in (-1, self._q.ndim - 1), f"QKVSplit.split expected the last dim, got dim={dim}"
        expected = [self._q.shape[-1], self._k.shape[-1], self._v.shape[-1]]
        assert list(sizes) == expected, (
            f"QKVSplit.split expected q/k/v sizes {expected}, got {list(sizes)}"
        )
        return self._q, self._k, self._v

    def chunk(self, chunks, dim=-1):
        # Only a 3-way chunk maps onto (q, k, v), and only when q/k/v are
        # equal-sized (non-GQA); anything else is not the idiom, so fail closed.
        assert chunks == 3, f"QKVSplit.chunk expected chunks=3 (q/k/v), got {chunks}"
        assert dim in (-1, self._q.ndim - 1), f"QKVSplit.chunk expected the last dim, got dim={dim}"
        assert self._q.shape[-1] == self._k.shape[-1] == self._v.shape[-1], (
            "QKVSplit.chunk(3) requires equal-sized q/k/v (non-GQA); "
            f"got {self._q.shape[-1]}/{self._k.shape[-1]}/{self._v.shape[-1]}"
        )
        return self._q, self._k, self._v


def _qkv_forward(self, x: torch.Tensor):
    """Rebound QKVParallelLinear.forward -> (QKVSplit, bias=None)."""
    q = F.linear(x, self.q_weight.data, _bias_data(self, "q_bias"))
    k = F.linear(x, self.k_weight.data, _bias_data(self, "k_bias"))
    v = F.linear(x, self.v_weight.data, _bias_data(self, "v_bias"))
    return QKVSplit(q, k, v), None


def _merged_forward(self, x: torch.Tensor):
    """Rebound MergedColumnParallelLinear.forward -> ([gate, up], bias=None)."""
    gate = F.linear(x, self.gate_weight.data, _bias_data(self, "gate_bias"))
    up = F.linear(x, self.up_weight.data, _bias_data(self, "up_bias"))
    return [gate, up], None


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


def _unfuse_qkv(module: QKVParallelLinear) -> None:
    """Split the fused QKV weight into q/k/v Parameters and rebind forward."""
    _assert_cpu(module, "qkv_proj")

    q_size = module.num_heads * module.head_size
    k_size = module.num_kv_heads * module.head_size
    v_size = module.num_kv_heads * module.v_head_size

    w = module.weight.data  # [q_size + k_size + v_size, hidden] on CPU
    assert q_size + k_size + v_size == w.shape[0], (
        f"analyze_and_unfuse: QKV slab sizes {q_size}+{k_size}+{v_size} "
        f"!= fused weight rows {w.shape[0]}; refusing to split."
    )

    q, k, v = torch.split(w, [q_size, k_size, v_size], dim=0)
    module.q_weight = Parameter(q.contiguous(), requires_grad=False)
    module.k_weight = Parameter(k.contiguous(), requires_grad=False)
    module.v_weight = Parameter(v.contiguous(), requires_grad=False)

    if getattr(module, "bias", None) is not None:
        qb, kb, vb = torch.split(module.bias.data, [q_size, k_size, v_size], dim=0)
        module.q_bias = Parameter(qb.contiguous(), requires_grad=False)
        module.k_bias = Parameter(kb.contiguous(), requires_grad=False)
        module.v_bias = Parameter(vb.contiguous(), requires_grad=False)
        del module.bias

    del module.weight  # so model.to(spyre) doesn't carry the fused copy

    module.forward = types.MethodType(_qkv_forward, module)


def _unfuse_merged(module: MergedColumnParallelLinear) -> None:
    """Split a 2-slab gate/up weight into gate/up Parameters, rebind forward."""
    _assert_cpu(module, "gate_up_proj")

    sizes = list(module.output_partition_sizes)  # per-rank, TP-correct
    w = module.weight.data
    assert sum(sizes) == w.shape[0], (
        f"analyze_and_unfuse: merged slab sizes {sizes} != fused weight "
        f"rows {w.shape[0]}; refusing to split."
    )

    gate, up = torch.split(w, sizes, dim=0)
    module.gate_weight = Parameter(gate.contiguous(), requires_grad=False)
    module.up_weight = Parameter(up.contiguous(), requires_grad=False)

    if getattr(module, "bias", None) is not None:
        gb, ub = torch.split(module.bias.data, sizes, dim=0)
        module.gate_bias = Parameter(gb.contiguous(), requires_grad=False)
        module.up_bias = Parameter(ub.contiguous(), requires_grad=False)
        del module.bias

    del module.weight

    module.forward = types.MethodType(_merged_forward, module)


def _has_silu_and_mul_sibling(parent: nn.Module) -> bool:
    """True if `parent` has a direct SpyreSiluAndMul child (the MLP act_fn)."""
    return any(isinstance(child, SpyreSiluAndMul) for child in parent.children())


def analyze_and_unfuse(model: nn.Module) -> None:
    """Analyze the model after the checkpoint is loaded (weights on CPU)."""
    n_qkv = 0
    n_merged = 0
    n_skipped_quant = 0
    n_skipped_merged_no_silu = 0

    parent_of = {id(model): model}
    for parent in model.modules():
        for child in parent.children():
            parent_of[id(child)] = parent

    for module in model.modules():
        if getattr(module, "weight", None) is None:
            # Already unfused (or weightless) — skip.
            continue

        if isinstance(module, QKVParallelLinear):
            if not _is_unquantized(module):
                n_skipped_quant += 1
                continue
            _unfuse_qkv(module)
            n_qkv += 1

        elif isinstance(module, MergedColumnParallelLinear):
            # Only a 2-slab gate/up projection feeding SiluAndMul is in scope.
            if len(module.output_partition_sizes) != 2:
                continue
            if not _is_unquantized(module):
                n_skipped_quant += 1
                continue
            parent = parent_of.get(id(module), model)
            if not _has_silu_and_mul_sibling(parent):
                n_skipped_merged_no_silu += 1
                continue
            _unfuse_merged(module)
            n_merged += 1

    logger.info(
        "Spyre weight-unfusing: unfused %d QKV and %d gate/up projections "
        "(skipped %d quantized, %d merged without SiluAndMul sibling).",
        n_qkv,
        n_merged,
        n_skipped_quant,
        n_skipped_merged_no_silu,
    )
