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

"""Tests for the Spyre conversion helpers (custom_ops/utils.py).

`convert` and friends sit under every device transfer the plugin makes (the
model runner's argument plumbing, the attention KV-cache relayout, the
vocab-embedding table placement), but were only covered indirectly through
those layers. The host-side logic (no-op short-circuit, tree mapping, layout
arithmetic) runs on CPU; the transfer, same-device relayout and row-gathered
placement paths need a card and are guarded below.

A host<->device round trip is not bit-exact on Spyre (1 fp16 ulp), so value
checks use the same atol/rtol=1e-3 as the vocab-embedding tests.
"""

import math

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.custom_ops.utils import (
    convert,
    convert_tensor_tree,
    place_row_gathered,
    register,
    row_outermost_layout,
)

register()


def test_convert_passes_none_through():
    assert convert(None, device="spyre", dtype=torch.float32) is None


def test_convert_noop_returns_the_same_tensor():
    t = torch.randn(3, 64, dtype=torch.float16)
    assert convert(t, device="cpu") is t
    assert convert(t) is t


def test_convert_op_rejects_a_true_noop():
    # The op body must never see a same-device, same-dtype call: `convert`
    # short-circuits it so Inductor never emits a spyre_convert FallbackKernel.
    t = torch.randn(3, 64, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="should never happen"):
        torch.ops.vllm.spyre_convert(t, None, None)


def test_convert_cpu_dtype_change():
    t = torch.randn(4, 64, dtype=torch.float16)
    out = convert(t, dtype=torch.float32)
    assert out.dtype is torch.float32
    assert out.device.type == "cpu"
    torch.testing.assert_close(out, t.float())


def test_convert_tensor_tree_maps_structure_and_predicate():
    t16 = torch.randn(2, 64, dtype=torch.float16)
    t32 = torch.randn(2, 64, dtype=torch.float32)
    tree = {"ids": [3, None], "tensors": (t16, t32)}

    out = convert_tensor_tree(tree, dtype=torch.float32)
    assert isinstance(out, dict)
    assert isinstance(out["tensors"], tuple)
    assert out["ids"] == [3, None]
    assert out["tensors"][0].dtype is torch.float32
    torch.testing.assert_close(out["tensors"][0], t16.float())
    assert out["tensors"][1] is t32

    def only_fp16(t):
        return t.dtype is torch.float16

    out = convert_tensor_tree(tree, dtype=torch.float32, predicate=only_fp16)
    assert out["tensors"][0].dtype is torch.float32
    assert out["tensors"][1] is t32


@pytest.mark.parametrize("shape", [(4, 128), (2, 3, 256)])
def test_row_outermost_layout_places_rows_outermost(shape):
    from torch_spyre._C import get_elem_in_stick

    eps = get_elem_in_stick(torch.float16)
    *lead, width = shape
    assert width % eps == 0, "shape must be stick-aligned"

    layout = row_outermost_layout(shape, torch.float16)
    assert layout.device_size == [*lead, width // eps, eps]
    assert layout.stride_map[-1] == 1
    assert layout.stride_map[0] == math.prod(shape) // shape[0]
    assert layout.valid_elements == {
        **{(i,): d for i, d in enumerate(lead)},
        (len(shape) - 1, len(shape)): width,
    }


def test_row_outermost_layout_rejects_unaligned_width():
    from torch_spyre._C import get_elem_in_stick

    width = 32
    assert width % get_elem_in_stick(torch.float16), "width must be stick-unaligned"
    assert row_outermost_layout((4, width), torch.float16) is None


def test_place_row_gathered_cpu_fn_returns_plain_result():
    t = torch.randn(4, 128, dtype=torch.float16)

    def fn(x):
        return x

    assert place_row_gathered(t, fn, "probe table") is t


def test_convert_roundtrip_and_dtype_detour_on_spyre():
    if not spyre_available():
        pytest.skip("Spyre device not available")

    torch.manual_seed(0)
    t = torch.randn(4, 128, dtype=torch.float16)

    on_device = convert(t, device="spyre")
    assert on_device.device.type == "spyre"
    torch.testing.assert_close(on_device.cpu().float(), t.float(), atol=1e-3, rtol=1e-3)

    back = convert(on_device, device="cpu")
    assert back.device.type == "cpu"
    assert back.dtype is torch.float16
    torch.testing.assert_close(back.float(), t.float(), atol=1e-3, rtol=1e-3)

    # Spyre requires CPU for dtype changes; the detour must stay transparent.
    widened = convert(on_device, dtype=torch.float32)
    assert widened.device.type == "spyre"
    assert widened.dtype is torch.float32
    torch.testing.assert_close(widened.cpu(), t.float(), atol=1e-3, rtol=1e-3)


def test_convert_relayouts_on_device_without_a_transfer():
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from torch_spyre._C import get_spyre_tensor_layout

    torch.manual_seed(1)
    t = torch.randn(4, 128, dtype=torch.float16)
    layout = row_outermost_layout(t.shape, torch.float16)

    default = convert(t, device="spyre")
    assert get_spyre_tensor_layout(default).device_size[0] != 4, (
        "the default tiled layout must not already be row-outermost"
    )

    # device=None: the tensor is already on Spyre, only the layout changes.
    relaid = convert(default, device_layout=layout)
    assert relaid.device.type == "spyre"
    assert get_spyre_tensor_layout(relaid) == layout
    torch.testing.assert_close(relaid.cpu().float(), t.float(), atol=1e-3, rtol=1e-3)


def test_place_row_gathered_puts_rows_outermost():
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from torch_spyre._C import get_spyre_tensor_layout

    torch.manual_seed(2)
    t = torch.randn(8, 128, dtype=torch.float16)

    moved = []

    def fn(x):
        moved.append(tuple(x.shape))
        return x.to("spyre")

    placed = place_row_gathered(t, fn, "probe table")
    assert moved == [(1, 128)], f"expected only the one-row probe, got {moved}"
    assert placed.device.type == "spyre"
    assert get_spyre_tensor_layout(placed) == row_outermost_layout(t.shape, torch.float16)

    rows = torch.tensor([7, 0, 3])
    gathered = placed.index_select(0, rows.to("spyre"))
    torch.testing.assert_close(
        gathered.cpu().float(), t.index_select(0, rows).float(), atol=1e-3, rtol=1e-3
    )


def test_place_row_gathered_unaligned_width_falls_back(monkeypatch):
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from torch_spyre._C import get_elem_in_stick, get_spyre_tensor_layout

    from spyre_inference.custom_ops import utils

    width = 96
    assert width % get_elem_in_stick(torch.float16), "width must be stick-unaligned"

    torch.manual_seed(3)
    t = torch.randn(8, width, dtype=torch.float16)

    warned: list[str] = []
    monkeypatch.setattr(utils.logger, "warning_once", lambda msg, *a: warned.append(msg % a))

    placed = place_row_gathered(t, lambda x: x.to("spyre"), "probe table")
    assert placed.device.type == "spyre", "the table must still move to device"
    assert get_spyre_tensor_layout(placed).device_size[0] != 8, (
        "unaligned rows cannot be outermost"
    )
    assert len(warned) == 1 and "probe table" in warned[0], warned
    torch.testing.assert_close(placed.cpu().float(), t.float(), atol=1e-3, rtol=1e-3)
