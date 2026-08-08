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

"""Tests for the generic linear-transpose pass (custom_ops/linear.py).

These run on CPU (no Spyre device needed): the pass is a pure host-side weight
mutation and `spyre_linear_t` is a plain `torch.matmul`, arithmetically identical
to `F.linear` on any device. QKV projections keep their fused weight (handled by
compiled slice+clone in the model runner); the LM head has its own tests in
test_parallel_lm_head.py. This file covers the generic `LinearBase` path
(down_proj, gate_up_proj, ...).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp_module(hidden, inter, bias=False):
    """A minimal MLP: a fused gate_up_proj (MergedColumn) + a down_proj (Row)."""
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        RowParallelLinear,
    )

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = MergedColumnParallelLinear(
                input_size=hidden,
                output_sizes=[inter, inter],
                bias=bias,
                params_dtype=torch.float16,
                quant_config=None,
                disable_tp=True,
                prefix="gate_up_proj",
            )
            self.down_proj = RowParallelLinear(
                input_size=inter,
                output_size=hidden,
                bias=bias,
                params_dtype=torch.float16,
                quant_config=None,
                disable_tp=True,
                prefix="down_proj",
            )

    return MLP()


def _forward(layer, x):
    """Run the (possibly rebound) unquantized apply and drop the bias tuple."""
    out = layer.quant_method.apply(layer, x, layer.bias if layer.bias is not None else None)
    return out


@pytest.mark.mlp
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
@pytest.mark.parametrize("use_bias", [False, True])
def test_transposed_linear_matches_reference(tp_group, num_tokens, use_bias):
    """After the pass, the GEMM `x @ Wᵀ (+bias)` matches the upstream F.linear.

    The core correctness guard: a wrong transpose or axis swap would change the
    output here. Runs entirely on CPU.
    """
    from spyre_inference.custom_ops.linear import (
        transpose_linear_weights_for_spyre,
    )

    hidden, inter = 128, 256
    torch.manual_seed(0)
    mlp = _make_mlp_module(hidden, inter, bias=use_bias)
    for layer in (mlp.gate_up_proj, mlp.down_proj):
        layer.weight.data.normal_(std=0.02)
        if layer.bias is not None:
            layer.bias.data.normal_(std=0.02)

    torch.manual_seed(1)
    x_gate = torch.randn(num_tokens, hidden, dtype=torch.float16)
    x_down = torch.randn(num_tokens, inter, dtype=torch.float16)

    exp_gate = F.linear(x_gate, mlp.gate_up_proj.weight.data, _bias(mlp.gate_up_proj))
    exp_down = F.linear(x_down, mlp.down_proj.weight.data, _bias(mlp.down_proj))

    transpose_linear_weights_for_spyre(mlp)

    torch.testing.assert_close(
        _forward(mlp.gate_up_proj, x_gate).float(), exp_gate.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        _forward(mlp.down_proj, x_down).float(), exp_down.float(), atol=1e-2, rtol=1e-2
    )


def _bias(layer):
    return layer.bias.data if layer.bias is not None else None


@pytest.mark.mlp
def test_weight_replaced_with_transpose(tp_group):
    """The pass sets `weight = None` and stores `weight_t` == `weightᵀ` bit-for-bit."""
    from spyre_inference.custom_ops.linear import (
        transpose_linear_weights_for_spyre,
    )

    hidden, inter = 128, 256
    torch.manual_seed(0)
    mlp = _make_mlp_module(hidden, inter)
    # torch.empty() leaves memory uninitialised (NaN in fp16 compares unequal to
    # itself); fill so the bit-for-bit check is meaningful.
    mlp.down_proj.weight.data.normal_(std=0.02)
    original = mlp.down_proj.weight.data.clone()
    orig_shape = original.shape  # [out, in] == [hidden, inter]

    transpose_linear_weights_for_spyre(mlp)

    assert mlp.down_proj.weight is None
    wt = mlp.down_proj.weight_t
    assert wt.shape == (orig_shape[1], orig_shape[0])  # [in, out]
    assert wt.data.is_contiguous()
    torch.testing.assert_close(wt.data, original.t(), atol=0.0, rtol=0.0)


@pytest.mark.mlp
def test_quantized_layer_skipped_without_cross_contamination(tp_group):
    """A quantized LinearBase is left untouched; a sibling unquantized one is not.

    Guards the per-module `quant_method.apply` patch: because vLLM builds a fresh
    UnquantizedLinearMethod per layer, patching one must not affect the other.
    """
    from spyre_inference.custom_ops.linear import (
        transpose_linear_weights_for_spyre,
    )

    hidden, inter = 128, 256
    torch.manual_seed(0)
    mlp = _make_mlp_module(hidden, inter)
    # Simulate a quantized layer: any non-UnquantizedLinearMethod trips the guard.
    mlp.gate_up_proj.quant_method = object()

    transpose_linear_weights_for_spyre(mlp)

    # Quantized layer: fully untouched.
    assert mlp.gate_up_proj.weight is not None
    assert not hasattr(mlp.gate_up_proj, "weight_t")
    # Unquantized sibling: transposed.
    assert mlp.down_proj.weight is None
    assert hasattr(mlp.down_proj, "weight_t")


@pytest.mark.mlp
def test_qkv_none_weight_skipped_by_generic_pass(tp_group):
    """A layer whose `weight` is None is skipped by the generic transpose pass."""
    from vllm.model_executor.layers.linear import QKVParallelLinear
    from spyre_inference.custom_ops.linear import (
        transpose_linear_weights_for_spyre,
    )

    torch.manual_seed(0)
    qkv = QKVParallelLinear(
        hidden_size=8 * 64,
        head_size=64,
        total_num_heads=8,
        total_num_kv_heads=2,
        params_dtype=torch.float16,
        quant_config=None,
        disable_tp=True,
        prefix="qkv_proj",
    )
    # Edge case: weight cleared (e.g. by a quantizer or other pass).
    qkv.weight = None

    # Must not raise and must not create a weight_t.
    transpose_linear_weights_for_spyre(qkv)
    assert not hasattr(qkv, "weight_t")
