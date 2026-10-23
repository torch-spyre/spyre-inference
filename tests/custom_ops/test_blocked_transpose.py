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

"""SPYRE_BLOCKED_TRANSPOSE: the row-blocked host transpose in `build_weight_t`.

CPU-only: `build_weight_t` is a host-side weight mutation, no Spyre device needed.
The blocked copy must be bit-identical to `w.t().contiguous()`.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

SHAPES = [(1, 64), (128, 64), (300, 96), (512, 256), (1000, 128)]


def _method(weight_t_attr="weight", row_align=None):
    from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod

    class Method(SpyreUnquantizedLinearMethod):
        WEIGHT_T_ATTR = weight_t_attr
        ROW_ALIGN = row_align

    return Method()


def _layer(w):
    layer = nn.Module()
    layer.weight = nn.Parameter(w, requires_grad=False)
    return layer


def _assert_bit_identical(got, expected):
    assert got.shape == expected.shape
    assert got.stride() == expected.stride()
    assert got.dtype == expected.dtype
    assert got.is_contiguous()
    assert torch.equal(got, expected)


@pytest.fixture(params=["0", "1"], ids=["off", "on"])
def blocked_transpose(request, monkeypatch):
    monkeypatch.setenv("SPYRE_BLOCKED_TRANSPOSE", request.param)
    return request.param == "1"


@pytest.mark.mlp
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", SHAPES)
def test_transpose_contiguous_bit_identical(blocked_transpose, shape, dtype):
    from spyre_inference.custom_ops.linear import _transpose_contiguous

    torch.manual_seed(0)
    w = torch.randn(shape, dtype=torch.float32).to(dtype)
    _assert_bit_identical(_transpose_contiguous(w), w.t().contiguous())


@pytest.mark.mlp
@pytest.mark.parametrize("shape", SHAPES)
def test_build_weight_t_in_place(blocked_transpose, shape):
    torch.manual_seed(0)
    w = torch.randn(shape, dtype=torch.float16)
    expected = w.t().contiguous()
    layer = _layer(w)

    _method().build_weight_t(layer, layer.weight.data)

    assert layer.spyre_row_padding == 0
    _assert_bit_identical(layer.weight.data, expected)


@pytest.mark.mlp
@pytest.mark.parametrize("shape", SHAPES)
def test_build_weight_t_padded_keeps_source(blocked_transpose, shape):
    """Padded rows into a distinct attr: matches the default build, `weight` untouched."""
    row_align = 64
    torch.manual_seed(0)
    w = torch.randn(shape, dtype=torch.float16)
    original = w.clone()
    padding = (-shape[0]) % row_align
    expected = F.pad(w, (0, 0, 0, padding)).t().contiguous()
    layer = _layer(w)

    _method("padded_weight_t", row_align).build_weight_t(layer, layer.weight.data)

    assert layer.spyre_row_padding == padding
    _assert_bit_identical(layer.padded_weight_t.data, expected)
    assert layer.weight.data.data_ptr() == w.data_ptr()
    assert layer.padded_weight_t.data.data_ptr() != w.data_ptr()
    _assert_bit_identical(layer.weight.data, original)


@pytest.mark.mlp
def test_non_2d_falls_back(blocked_transpose):
    from spyre_inference.custom_ops.linear import _transpose_contiguous

    w = torch.randn(7, dtype=torch.float16)
    _assert_bit_identical(_transpose_contiguous(w), w.t().contiguous())


@pytest.mark.mlp
@pytest.mark.parametrize("shape", SHAPES)
def test_non_contiguous_input(blocked_transpose, shape):
    """A transposed view (`w.t()` already contiguous) keeps the old aliasing; a
    column slice (neither layout contiguous) goes through the blocked copy."""
    from spyre_inference.custom_ops.linear import _transpose_contiguous

    torch.manual_seed(0)
    rows, cols = shape
    w = torch.randn(cols, rows, dtype=torch.float16).t()
    got = _transpose_contiguous(w)
    _assert_bit_identical(got, w.t().contiguous())
    assert got.data_ptr() == w.data_ptr()

    w = torch.randn(rows, 2 * cols, dtype=torch.float16)[:, 1::2]
    _assert_bit_identical(_transpose_contiguous(w), w.t().contiguous())
