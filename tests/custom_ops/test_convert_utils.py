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

"""Test the convert helpers in spyre_inference.custom_ops.utils.

Every model-runner, encoder-graph, spec-decode and pooling path moves tensors
through `convert`/`convert_tensor_tree`, and the paged-KV and sampling gathers
place their sources via `place_row_gathered`/`row_outermost_layout`, yet none
of that had direct tests.
"""

import sys

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available


def test_convert_noop_returns_same_tensor():
    """A tensor already on the target device and dtype is returned as-is (the
    short-circuit keeps a same-device spyre_convert out of Inductor graphs),
    None passes through, and a str device is accepted."""
    from spyre_inference.custom_ops.utils import convert

    t = torch.randn(2, 64, dtype=torch.float16)
    assert convert(t) is t
    assert convert(t, device="cpu") is t
    assert convert(t, device=torch.device("cpu"), dtype=torch.float16) is t
    assert convert(None, device="cpu", dtype=torch.float32) is None


def test_convert_dtype_change_on_cpu(default_vllm_config):
    """A dtype-only conversion routes through the opaque spyre_convert op
    (registered by the fixture) and keeps the values."""
    from spyre_inference.custom_ops.utils import convert

    t = torch.randn(2, 64, dtype=torch.float16)
    out = convert(t, dtype=torch.float32)
    assert out.device.type == "cpu" and out.dtype == torch.float32
    torch.testing.assert_close(out, t.float())


def test_convert_op_rejects_same_device_and_dtype(default_vllm_config):
    """The op body treats a true no-op as a bug (the Python boundary is
    responsible for short-circuiting it) and must say so."""
    t = torch.randn(2, 64, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="same device"):
        torch.ops.vllm.spyre_convert(t, None, None)


def test_convert_tensor_tree_walks_and_filters(default_vllm_config):
    """The tree walk converts every tensor in a nested structure, passes
    non-tensors and None through untouched, and the predicate excludes
    non-matching tensors."""
    from spyre_inference.custom_ops.utils import convert_tensor_tree

    fp16 = torch.randn(2, 64, dtype=torch.float16)
    fp32 = torch.randn(2, 64, dtype=torch.float32)
    tree = {"a": fp16, "b": [fp32, "keep", None], "c": 5}

    out = convert_tensor_tree(tree, dtype=torch.float32)
    assert out["a"].dtype == torch.float32 and out["b"][0].dtype == torch.float32
    assert out["b"][1] == "keep" and out["b"][2] is None and out["c"] == 5
    torch.testing.assert_close(out["a"], fp16.float())

    out = convert_tensor_tree(
        tree, dtype=torch.float32, predicate=lambda t: t.dtype == torch.float16
    )
    assert out["a"].dtype == torch.float32
    assert out["b"][0] is fp32  # excluded by the predicate


@pytest.mark.parametrize(
    "shape,expected_size,expected_strides",
    [
        ((2, 128), [2, 2, 64], [128, 64, 1]),
        ((3, 2, 128), [3, 2, 2, 64], [256, 128, 64, 1]),
    ],
)
def test_row_outermost_layout_describes_gather_source(shape, expected_size, expected_strides):
    """The layout puts the indexed (row) dim outermost and the last dim as the
    stick axis, with valid elements covering exactly the logical tensor."""
    from spyre_inference.custom_ops.utils import row_outermost_layout

    layout = row_outermost_layout(shape, torch.float16)
    assert list(layout.device_size) == expected_size
    assert list(layout.stride_map) == expected_strides
    valid = dict(layout.valid_elements)
    if len(shape) == 2:
        assert valid == {(0,): shape[0], (1, 2): shape[1]}
    else:
        assert valid == {(0,): shape[0], (1,): shape[1], (2, 3): shape[2]}


@pytest.mark.parametrize("shape", [(2, 32), (2, 96)])
def test_row_outermost_layout_rejects_partial_sticks(shape):
    """A last dim that is not a whole number of sticks has no exact padded
    tiling, so no layout is returned."""
    from spyre_inference.custom_ops.utils import row_outermost_layout

    assert row_outermost_layout(shape, torch.float16) is None


def test_place_row_gathered_off_device_fn_uses_plain_path():
    """A fn whose probe lands off Spyre needs no layout: the source goes
    through fn unchanged, with exactly one call."""
    from spyre_inference.custom_ops.utils import place_row_gathered

    src = torch.randn(4, 128, dtype=torch.float16)
    calls = []

    def fn(t):
        calls.append(t.shape)
        return t * 2

    out = place_row_gathered(src, fn, "test")
    assert calls == [(1, 128), (4, 128)]  # one-row probe, then the plain call
    torch.testing.assert_close(out, src * 2)


def test_convert_moves_cpu_to_spyre_and_back():
    """cpu->spyre and spyre->cpu transfers round-trip the values (within fp16
    ULPs: the device transfer is not bit-exact)."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from spyre_inference.custom_ops.utils import convert

    t = torch.randn(4, 128, dtype=torch.float16)
    on_spyre = convert(t, device="spyre")
    assert on_spyre.device.type == "spyre" and on_spyre.dtype == torch.float16
    torch.testing.assert_close(on_spyre.cpu(), t, atol=1e-2, rtol=1e-2)

    back = convert(on_spyre, device="cpu")
    assert back.device.type == "cpu" and back.dtype == torch.float16
    torch.testing.assert_close(back, t, atol=1e-2, rtol=1e-2)


def test_convert_spyre_dtype_change_detours_through_cpu():
    """Spyre cannot change dtype on device, so a dtype change from a Spyre
    tensor hops via the host before landing on the target device."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from spyre_inference.custom_ops.utils import convert

    t = torch.randn(4, 128, dtype=torch.float16)
    out = convert(t.to("spyre"), device="cpu", dtype=torch.float32)
    assert out.device.type == "cpu" and out.dtype == torch.float32
    torch.testing.assert_close(out, t.float(), atol=1e-2, rtol=1e-2)


def test_place_row_gathered_puts_rows_outermost_on_spyre():
    """A stick-aligned 2D source is placed on the device with its rows
    outermost (probe learns the destination from fn) and survives the move."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from spyre_inference.custom_ops.utils import place_row_gathered

    src = torch.randn(4, 128, dtype=torch.float16)
    calls = []

    def fn(t):
        calls.append(tuple(t.shape))
        return t.to("spyre")

    out = place_row_gathered(src, fn, "test")
    # fn is only probed on one row; the full source moves through the
    # layout-aware convert instead of through fn again.
    assert calls == [(1, 128)]
    assert out.device.type == "spyre" and out.dtype == torch.float16
    torch.testing.assert_close(out.cpu(), src, atol=1e-2, rtol=1e-2)


def test_place_row_gathered_narrow_rows_keep_default_layout():
    """A row width that is not a whole stick cannot be laid out row-outermost,
    so the source falls back to the plain fn path (still correct values)."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from spyre_inference.custom_ops.utils import place_row_gathered

    src = torch.randn(4, 32, dtype=torch.float16)

    def fn(t):
        return t.to("spyre")

    out = place_row_gathered(src, fn, "test")
    assert out.device.type == "spyre" and out.dtype == torch.float16
    torch.testing.assert_close(out.cpu(), src, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
