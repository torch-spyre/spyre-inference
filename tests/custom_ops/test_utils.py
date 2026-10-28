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

"""custom_ops/utils.py: the device/dtype conversion helpers every layer funnels through.

``convert`` short-circuits a true no-op at the Python boundary so Inductor never emits a
same-device/dtype ``spyre_convert`` fallback, routes a real transfer through the opaque op,
and detours an on-device dtype change via CPU. ``row_outermost_layout`` and
``place_row_gathered`` describe a gather's indexed rows at device position 0, declining the
layout when a row is not a whole number of fp16 sticks (64 elements). ``convert_tensor_tree``
maps the conversion across a nested value under an optional predicate.

A host<->device relayout is not bit-exact: it costs up to one fp16 ULP, so the device
comparisons use a tolerance rather than ``equal``. The layout and no-op-boundary checks are
host-side and need no card.
"""

import sys

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.custom_ops import utils

# A host<->device relayout converts each fp16 element independently; one ULP at unit scale.
_ATOL, _RTOL = 1e-2, 2e-3


@pytest.fixture(autouse=True)
def _register_convert_op():
    """``convert`` routes real transfers through ``torch.ops.vllm.spyre_convert``.

    Production registers it at plugin load; ``register`` is ``lru_cache``-d, so this is a
    no-op once anything in the session has called it.
    """
    utils.register()


@pytest.fixture()
def spyre_device():
    if not spyre_available():
        pytest.skip("Spyre device not available")
    return torch.device("spyre")


# ---------------------------------------------------------------------------
# convert: Python-boundary behaviour (no card)
# ---------------------------------------------------------------------------


def test_convert_none_passthrough():
    """None short-circuits before the op, which cannot return Optional[Tensor]."""
    assert utils.convert(None) is None
    assert utils.convert(None, device="cpu", dtype=torch.float32) is None


def test_convert_true_noop_returns_same_object():
    """A same-device/same-dtype request returns the input untouched, emitting no op."""
    x = torch.randn(4, 64, dtype=torch.float16)
    assert utils.convert(x) is x
    assert utils.convert(x, device="cpu") is x
    assert utils.convert(x, dtype=torch.float16) is x
    assert utils.convert(x, device="cpu", dtype=torch.float16) is x


def test_convert_op_func_rejects_a_true_noop():
    """The opaque body guards the invariant the call site upholds: it is never a no-op."""
    x = torch.randn(2, 64, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="same device"):
        utils._convert_op_func(x, device=torch.device("cpu"), dtype=torch.float16)


def test_convert_cpu_dtype_change_no_card():
    """A CPU-only dtype change still routes through the op and promotes correctly."""
    x = torch.randn(4, 64, dtype=torch.float16)
    out = utils.convert(x, dtype=torch.float32)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"
    torch.testing.assert_close(out, x.float(), atol=_ATOL, rtol=_RTOL)


# ---------------------------------------------------------------------------
# row_outermost_layout: host-side layout description (no card)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,expected_dims,expected_strides",
    [
        ([4, 64], [4, 1, 64], [64, 64, 1]),
        ([4, 128], [4, 2, 64], [128, 64, 1]),
        ([2, 3, 256], [2, 3, 4, 64], [768, 256, 64, 1]),
    ],
)
def test_row_outermost_layout_aligned(shape, expected_dims, expected_strides):
    """A stick-aligned last dim splits into (sticks, 64) with dim 0 kept outermost."""
    layout = utils.row_outermost_layout(shape, torch.float16)
    assert layout is not None
    assert list(layout.device_size) == expected_dims
    assert list(layout.stride_map) == expected_strides


@pytest.mark.parametrize("width", [32, 96, 100])
def test_row_outermost_layout_unaligned_is_none(width):
    """A last dim that is not a whole number of fp16 sticks has no padded tiling."""
    assert utils.row_outermost_layout([4, width], torch.float16) is None


# ---------------------------------------------------------------------------
# convert_tensor_tree: nested mapping (no card)
# ---------------------------------------------------------------------------


def test_convert_tensor_tree_predicate_and_passthrough():
    """Only predicate-matching tensors convert; non-tensors and non-matches pass through."""
    tree = {
        "a": torch.randn(2, 64, dtype=torch.float16),
        "b": 5,
        "c": torch.randn(2, 64, dtype=torch.float32),
        "d": [torch.randn(1, 64, dtype=torch.float16)],
    }
    out = utils.convert_tensor_tree(
        tree, dtype=torch.float32, predicate=lambda t: t.dtype == torch.float16
    )
    assert out["a"].dtype == torch.float32
    assert out["b"] == 5
    assert out["c"].dtype == torch.float32  # already matched the target, so untouched
    assert out["d"][0].dtype == torch.float32


def test_convert_tensor_tree_no_predicate_converts_every_tensor():
    tree = (torch.randn(2, 64, dtype=torch.float16), "x", 7)
    out = utils.convert_tensor_tree(tree, dtype=torch.float32)
    assert out[0].dtype == torch.float32
    assert out[1] == "x"
    assert out[2] == 7


# ---------------------------------------------------------------------------
# convert: on-device transfers (needs a card)
# ---------------------------------------------------------------------------


def test_convert_host_device_roundtrip(spyre_device):
    """A host->device->host roundtrip preserves values within one relayout ULP."""
    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.float16)

    xs = utils.convert(x, device="spyre")
    assert xs.device.type == "spyre"

    back = utils.convert(xs, device="cpu")
    assert back.device.type == "cpu"
    assert back.dtype == torch.float16
    torch.testing.assert_close(back.float(), x.float(), atol=_ATOL, rtol=_RTOL)


def test_convert_on_device_dtype_change_detours_via_cpu(spyre_device):
    """An on-device dtype change keeps the device but promotes through the CPU detour."""
    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.float16)
    xs = utils.convert(x, device="spyre")

    xf = utils.convert(xs, dtype=torch.float32)
    assert xf.dtype == torch.float32
    assert xf.device.type == "spyre"
    torch.testing.assert_close(
        utils.convert(xf, device="cpu"), x.float(), atol=_ATOL, rtol=_RTOL
    )


# ---------------------------------------------------------------------------
# place_row_gathered: probe + row-outermost placement (needs a card)
# ---------------------------------------------------------------------------


def test_place_row_gathered_aligned_matches_plain(spyre_device):
    """A stick-aligned gather source lands on device with the same values as a plain move."""
    torch.manual_seed(0)
    src = torch.randn(8, 128, dtype=torch.float16)
    out = utils.place_row_gathered(src, lambda t: utils.convert(t, device="spyre"), "aligned")
    assert out.device.type == "spyre"
    torch.testing.assert_close(out.cpu().float(), src.float(), atol=_ATOL, rtol=_RTOL)


def test_place_row_gathered_unaligned_falls_back(spyre_device):
    """A row that is not a whole number of sticks keeps the default layout, still on device."""
    torch.manual_seed(0)
    src = torch.randn(4, 96, dtype=torch.float16)
    out = utils.place_row_gathered(src, lambda t: utils.convert(t, device="spyre"), "unaligned")
    assert out.device.type == "spyre"
    torch.testing.assert_close(out.cpu().float(), src.float(), atol=_ATOL, rtol=_RTOL)


def test_place_row_gathered_cpu_probe_returns_plain(spyre_device):
    """When the gather fn keeps the source on CPU, the layout path is skipped entirely."""
    torch.manual_seed(0)
    src = torch.randn(8, 128, dtype=torch.float16)
    out = utils.place_row_gathered(src, lambda t: t, "cpu")
    assert out.device.type == "cpu"
    assert torch.equal(out, src)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
