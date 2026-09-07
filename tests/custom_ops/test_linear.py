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

"""Tests for the transposed-weight linear method (custom_ops/linear.py).

These run on CPU (no Spyre device needed): `process_weights_after_loading` is a
pure host-side weight mutation and `spyre_linear_t` is a plain `torch.matmul`,
arithmetically identical to `F.linear` on any device. The generic
`LinearBase` path (gate_up_proj, down_proj, ...) is covered here; QKV and the
LM head have their own tests (test_mlp.py, test_parallel_lm_head.py).
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


def _bias(layer):
    return layer.bias.data if layer.bias is not None else None


def _forward(layer, x):
    """Run the transposed apply and drop the bias tuple."""
    return layer.quant_method.apply(layer, x, _bias(layer))


@pytest.mark.mlp
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
@pytest.mark.parametrize("use_bias", [False, True])
def test_transposed_linear_matches_reference(tp_group, num_tokens, use_bias):
    """After processing weights, `x @ Wᵀ (+bias)` matches the upstream F.linear.

    The core correctness guard: a wrong transpose or axis swap would change the
    output here. Runs entirely on CPU.
    """
    from spyre_inference.custom_ops.linear import (
        SpyreMergedColumnParallelLinear,
        SpyreRowParallelLinear,
    )

    hidden, inter = 128, 256
    torch.manual_seed(0)
    mlp = _make_mlp_module(hidden, inter, bias=use_bias)
    assert isinstance(mlp.gate_up_proj, SpyreMergedColumnParallelLinear)
    assert isinstance(mlp.down_proj, SpyreRowParallelLinear)
    for layer in (mlp.gate_up_proj, mlp.down_proj):
        layer.weight.data.normal_(std=0.02)
        if layer.bias is not None:
            layer.bias.data.normal_(std=0.02)

    torch.manual_seed(1)
    x_gate = torch.randn(num_tokens, hidden, dtype=torch.float16)
    x_down = torch.randn(num_tokens, inter, dtype=torch.float16)

    # Capture the reference BEFORE process_weights_after_loading transposes.
    exp_gate = F.linear(x_gate, mlp.gate_up_proj.weight.data, _bias(mlp.gate_up_proj))
    exp_down = F.linear(x_down, mlp.down_proj.weight.data, _bias(mlp.down_proj))

    for layer in (mlp.gate_up_proj, mlp.down_proj):
        layer.quant_method.process_weights_after_loading(layer)

    torch.testing.assert_close(
        _forward(mlp.gate_up_proj, x_gate).float(), exp_gate.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        _forward(mlp.down_proj, x_down).float(), exp_down.float(), atol=1e-2, rtol=1e-2
    )


@pytest.mark.mlp
def test_weight_stored_transposed(tp_group):
    """process_weights_after_loading replaces `weight` with `weightᵀ`, contiguous."""
    hidden, inter = 128, 256
    torch.manual_seed(0)
    mlp = _make_mlp_module(hidden, inter)
    # torch.empty() leaves memory uninitialised (NaN in fp16 compares unequal to
    # itself); fill so the bit-for-bit check is meaningful.
    mlp.down_proj.weight.data.normal_(std=0.02)
    original = mlp.down_proj.weight.data.clone()
    orig_shape = original.shape  # [out, in] == [hidden, inter]

    mlp.down_proj.quant_method.process_weights_after_loading(mlp.down_proj)

    w = mlp.down_proj.weight
    assert w.shape == (orig_shape[1], orig_shape[0])  # [in, out]
    assert w.data.is_contiguous()
    torch.testing.assert_close(w.data, original.t(), atol=0.0, rtol=0.0)


@pytest.mark.mlp
def test_unquantized_layers_get_spyre_method(tp_group):
    """Unquantized linears get `SpyreUnquantizedLinearMethod` swapped in.

    The mixin only replaces the method when it is an `UnquantizedLinearMethod`
    (`quant_config=None` here), so a quantized layer would keep its own
    slow-but-correct F.linear method untouched.
    """
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod

    hidden, inter = 128, 256
    torch.manual_seed(0)
    mlp = _make_mlp_module(hidden, inter)

    assert isinstance(mlp.gate_up_proj.quant_method, SpyreUnquantizedLinearMethod)
    assert isinstance(mlp.down_proj.quant_method, SpyreUnquantizedLinearMethod)
    # SpyreUnquantizedLinearMethod is itself an UnquantizedLinearMethod subclass.
    assert isinstance(mlp.down_proj.quant_method, UnquantizedLinearMethod)


def _rows_reaching_gemm(gate_up, x, monkeypatch):
    """Rows reaching the GEMM: on CPU `out[:m]` is identical whether or not padding fired."""
    from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod

    seen = []
    real_apply = SpyreUnquantizedLinearMethod.apply

    def spy(self, layer, activations, bias=None):
        seen.append(activations.shape[0])
        return real_apply(self, layer, activations, bias)

    monkeypatch.setattr(SpyreUnquantizedLinearMethod, "apply", spy)
    out = _forward(gate_up, x)
    assert len(seen) == 1
    return seen[0], out


@pytest.mark.mlp
@pytest.mark.parametrize(
    ("num_tokens", "expected_rows"),
    [(1, 8), (4, 8), (7, 8), (8, 8), (9, 9), (64, 64)],
)
def test_short_rows_padded_on_gate_up(tp_group, monkeypatch, num_tokens, expected_rows):
    """A partial row block reaches the GEMM padded to `_PAD_ROWS`; a full one is untouched."""
    from spyre_inference.custom_ops.linear import (
        SpyrePaddedRowsLinearMethod,
        SpyreUnquantizedLinearMethod,
    )

    hidden, inter = 128, 256
    torch.manual_seed(0)
    mlp = _make_mlp_module(hidden, inter)
    assert isinstance(mlp.gate_up_proj.quant_method, SpyrePaddedRowsLinearMethod)
    assert not isinstance(mlp.down_proj.quant_method, SpyrePaddedRowsLinearMethod)

    gate_up = mlp.gate_up_proj
    gate_up.weight.data.normal_(std=0.02)
    gate_up.quant_method.process_weights_after_loading(gate_up)

    torch.manual_seed(1)
    x = torch.randn(num_tokens, hidden, dtype=torch.float16)
    reference = SpyreUnquantizedLinearMethod().apply(gate_up, x, None)

    rows, out = _rows_reaching_gemm(gate_up, x, monkeypatch)
    assert rows == expected_rows
    assert out.shape == (num_tokens, 2 * inter)
    torch.testing.assert_close(out.float(), reference.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
def test_large_weights_not_padded(tp_group, monkeypatch):
    """The weight bound disables padding, since past it the pad rows cost more."""
    from spyre_inference.custom_ops import linear as linear_mod

    hidden, inter = 128, 256
    torch.manual_seed(0)
    gate_up = _make_mlp_module(hidden, inter).gate_up_proj
    gate_up.weight.data.normal_(std=0.02)
    gate_up.quant_method.process_weights_after_loading(gate_up)

    x = torch.randn(1, hidden, dtype=torch.float16)
    monkeypatch.setattr(linear_mod, "_MAX_PAD_WEIGHT", gate_up.weight.numel() - 1)
    rows, _ = _rows_reaching_gemm(gate_up, x, monkeypatch)
    assert rows == 1
