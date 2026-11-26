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

"""Host-side unit tests for SpyreCommunicator.

The comms probes (tests/probes/) cover the raw device collectives on two cards;
these tests pin the host logic around them: the short-circuits, the eager
all_gather padding workaround for libspyre_comms' 64-element chunked transfer,
the compiled all_gather's rank-axis folding, and its alignment guard. The
collectives themselves are faked, so each test runs in one process.
"""

import math

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.distributed.spyre_communicator import SpyreCommunicator

_GROUP_NAME = "tp:0"


def _int_valued(shape) -> torch.Tensor:
    """fp16 integers in [-3, 3]: exact through every fp16 device roundtrip, so
    a layout or padding bug shows up as a wrong value, not a tolerance."""
    return (torch.arange(math.prod(shape), dtype=torch.float16).reshape(shape) % 7) - 3


def _communicator(world_size: int = 2, group_name: str | None = _GROUP_NAME):
    """A communicator without a live device group: every collective in these
    tests is faked, so only the attributes the host logic reads are needed."""
    comm = SpyreCommunicator.__new__(SpyreCommunicator)
    comm.device = torch.device("spyre")
    comm.device_group = None
    comm.world_size = world_size
    comm._group_name = group_name
    return comm


def test_collectives_pass_through_at_world_size_1():
    x = torch.randn(4, 8)

    comm = _communicator(world_size=1, group_name=None)

    assert comm.all_reduce(x) is x
    assert comm.all_gather(x) is x
    assert comm.reduce_scatter(x) is x


def test_reduce_scatter_raises_above_world_size_1():
    comm = _communicator()

    with pytest.raises(NotImplementedError, match="reduce_scatter"):
        comm.reduce_scatter(torch.randn(4, 8))


def test_init_resolves_the_group_name_once(tp_group):
    device_group = tp_group.cpu_group

    comm = SpyreCommunicator(device_group, torch.device("spyre"), device_group)

    assert comm._group_name is not None
    assert comm.world_size == 1

    comm = SpyreCommunicator(device_group, torch.device("spyre"), None)

    assert comm._group_name is None


def test_all_reduce_uses_the_out_of_place_functional_form(monkeypatch):
    """The graph contract: the reduction result is a new tensor, the input is
    left untouched, and the op sees ("sum", the resolved group name)."""
    if not spyre_available():
        pytest.skip("requires Spyre hardware")

    calls = []
    device_inputs = []

    def fake_all_reduce(tensor, reduce_op, group_name):
        calls.append((reduce_op, group_name))
        device_inputs.append(tensor)
        return tensor.cpu() * 2

    monkeypatch.setattr(torch.ops._c10d_functional, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch.ops._c10d_functional, "wait_tensor", lambda tensor: tensor)

    x = _int_valued((4, 64))
    out = _communicator().all_reduce(x.to("spyre"))

    assert calls == [("sum", _GROUP_NAME)]
    assert len(device_inputs) == 1
    assert out is not device_inputs[0]
    torch.testing.assert_close(out, x * 2)
    torch.testing.assert_close(device_inputs[0].cpu(), x)


@pytest.mark.parametrize(
    ("shape", "dim"),
    [((33, 64), 0), ((100, 64), 0), ((33, 2, 64), 0), ((3, 96), -1)],
)
def test_eager_all_gather_pads_a_short_shard_and_strips_it_back(monkeypatch, shape, dim):
    """A shard the comms library would truncate to 64-element chunks is padded
    to a whole multiple of 64 for the transfer and stripped on the way back."""
    if not spyre_available():
        pytest.skip("requires Spyre hardware")

    x = _int_valued(shape)
    gathered = []

    def fake_all_gather(output_list, tensor, group=None):
        gathered.append(tensor.cpu())
        for rank in range(len(output_list)):
            output_list[rank] = (tensor.cpu() + rank).to(tensor.device)

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)

    out = _communicator().all_gather(x.to("spyre"), dim=dim)

    axis = dim % x.dim()
    pad_spec = [0, 0] * (x.dim() - axis - 1) + [0, -shape[axis] % SpyreCommunicator._GATHER_ALIGN]
    torch.testing.assert_close(gathered[0], F.pad(x, pad_spec))
    expected = torch.cat([x, x + 1], dim=axis)
    assert out.shape == expected.shape
    assert out.device.type == "spyre"
    torch.testing.assert_close(out.cpu(), expected)


def test_eager_all_gather_concatenates_an_aligned_shard(monkeypatch):
    """A 64-aligned shard skips the padding detour and is gathered as-is."""
    if not spyre_available():
        pytest.skip("requires Spyre hardware")

    x = _int_valued((3, 128))
    gathered = []

    def fake_all_gather(output_list, tensor, group=None):
        gathered.append(tensor.cpu())
        for rank in range(len(output_list)):
            output_list[rank] = (tensor.cpu() + rank).to(tensor.device)

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)

    out = _communicator().all_gather(x.to("spyre"), dim=-1)

    torch.testing.assert_close(gathered[0], x)
    expected = torch.cat([x, x + 1], dim=-1)
    assert out.shape == expected.shape
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.parametrize(
    ("shape", "dim"),
    [((4, 128), -1), ((4, 128), 0), ((2, 3, 64), 1), ((2, 3, 64), -2)],
)
def test_compiled_all_gather_folds_the_rank_axis_into_the_requested_dim(monkeypatch, shape, dim):
    """`all_gather_into_tensor` concatenates the ranks on dim 0; the folded
    reshape must move that axis to the caller's `dim` whatever it is."""

    def fake_all_gather_into_tensor(tensor, world_size, group_name):
        assert world_size == 2
        assert group_name == _GROUP_NAME
        return tensor.reshape(-1).repeat(world_size)

    monkeypatch.setattr(
        torch.ops._c10d_functional, "all_gather_into_tensor", fake_all_gather_into_tensor
    )
    monkeypatch.setattr(torch.ops._c10d_functional, "wait_tensor", lambda tensor: tensor)

    x = torch.randn(shape)
    out = _communicator()._all_gather_compiled(x, dim=dim)

    expected = torch.cat([x, x], dim=dim)
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected)


def test_compiled_all_gather_rejects_an_unaligned_shard():
    """An unaligned shard would fault the card in `spyre::all_gather_async`;
    the guard must raise before the collective is issued."""
    with pytest.raises(ValueError, match="divisible by 64"):
        _communicator()._all_gather_compiled(torch.randn(3, 33), dim=-1)


def test_all_gather_dispatches_to_the_compiled_path_when_compiling(monkeypatch):
    """Inside a compiled graph the functional form runs, never the eager
    list-form collective."""
    if not spyre_available():
        pytest.skip("requires Spyre hardware")

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    calls = []

    def fake_all_gather_into_tensor(tensor, world_size, group_name):
        calls.append(world_size)
        return tensor.cpu().reshape(-1).repeat(world_size)

    monkeypatch.setattr(
        torch.ops._c10d_functional, "all_gather_into_tensor", fake_all_gather_into_tensor
    )
    monkeypatch.setattr(torch.ops._c10d_functional, "wait_tensor", lambda tensor: tensor)

    def refuse(*args, **kwargs):
        raise AssertionError("the eager list-form path must not run while compiling")

    monkeypatch.setattr(dist, "all_gather", refuse)

    x = _int_valued((4, 128))
    out = _communicator().all_gather(x.to("spyre"), dim=-1)

    assert calls == [2]
    torch.testing.assert_close(out, torch.cat([x, x], dim=-1))
