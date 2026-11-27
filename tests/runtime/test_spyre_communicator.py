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

"""SpyreCommunicator's host-side branches, which the TP=2 device probes
(tests/probes/test_spyre_comms_native_probes.py) never reach on their own: the
world-size-1 pass-throughs, the CPU and unresolved-group delegations to the base
class, reduce_scatter's refusal, the compiled all_gather's rank-axis fold and its
64-alignment guard, and the eager all_gather's list-form gather with its
pad-and-strip arithmetic. The collectives themselves are faked; only the eager
all_gather paths touch the card, because their pad and strip are host round-trips
around a real device tensor."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

from spyre_inference.distributed.spyre_communicator import SpyreCommunicator


def _comm(world_size: int = 2, group_name: str | None = "group0") -> SpyreCommunicator:
    """A communicator without a process group: only the attributes its methods read."""
    comm = SpyreCommunicator.__new__(SpyreCommunicator)
    comm.world_size = world_size
    comm.device_group = object()
    comm._group_name = group_name
    return comm


@pytest.fixture
def fake_functional_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    """``all_gather_into_tensor`` with rank ``r`` contributing ``t + r``, concatenated
    on dim 0 as the real op does, so the axis fold is checked against torch.cat."""

    def gather(t: torch.Tensor, world_size: int, group_name: str) -> torch.Tensor:
        return torch.cat([t + r for r in range(world_size)], dim=0)

    monkeypatch.setattr(torch.ops._c10d_functional, "all_gather_into_tensor", gather)
    monkeypatch.setattr(torch.ops._c10d_functional, "wait_tensor", lambda t: t)


def test_world_size_one_collectives_are_identity() -> None:
    x = torch.arange(12, dtype=torch.float16).view(3, 4)
    comm = _comm(world_size=1)

    assert comm.all_reduce(x) is x
    assert comm.all_gather(x) is x
    assert comm.reduce_scatter(x) is x


def test_reduce_scatter_refuses_loudly_beyond_one_rank() -> None:
    with pytest.raises(NotImplementedError, match="no Spyre implementation"):
        _comm().reduce_scatter(torch.zeros(8, dtype=torch.float16))


@pytest.mark.parametrize("compiling", [False, True])
@pytest.mark.parametrize("group_name", ["group0", None])
@pytest.mark.parametrize("method", ["all_reduce", "all_gather"])
def test_a_cpu_tensor_goes_to_the_base_class_whatever_the_group(
    monkeypatch: pytest.MonkeyPatch, method: str, group_name: str | None, compiling: bool
) -> None:
    """A CPU tensor has no device collective to lower, and an unresolved group name
    cannot be traced, so both take the base class's eager path: the CPU check wins
    even under torch.compile, whose branch the group name otherwise gates."""
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    calls: list[tuple[str, bool]] = []
    sentinel = torch.zeros(1, dtype=torch.float16)

    def record(self, input_, dim=-1):
        calls.append((method, input_ is x))
        return sentinel

    monkeypatch.setattr(DeviceCommunicatorBase, method, record)
    x = torch.zeros(4, dtype=torch.float16)

    out = getattr(_comm(group_name=group_name), method)(x)

    assert calls == [(method, True)]
    assert out is sentinel


@pytest.mark.parametrize("dim", [-1, 0, 1, 2])
def test_compiled_all_gather_folds_the_rank_axis_into_dim(fake_functional_gather, dim: int) -> None:
    """``all_gather_into_tensor`` concatenates the ranks on dim 0; the compiled gather
    must fold that axis out to the caller's dim, matching torch.cat on every dim."""
    t = torch.arange(2 * 3 * 64, dtype=torch.float16).view(2, 3, 64)

    out = _comm()._all_gather_compiled(t, dim)

    torch.testing.assert_close(out, torch.cat([t + r for r in range(2)], dim=dim % t.dim()))


def test_compiled_all_gather_rejects_an_unaligned_shard() -> None:
    """A shard whose numel is not a 64 multiple faults the card in the lowering, so the
    guard must fail loudly before the collective runs."""
    t = torch.zeros(2, 3, dtype=torch.float16)  # numel 6

    with pytest.raises(ValueError, match="divisible by 64"):
        _comm()._all_gather_compiled(t, 0)


def test_a_compiling_gather_takes_the_functional_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under torch.compile the functional collective is preferred (it lowers to a native
    device op); the eager list-form path below would fail here, having no process group."""
    from spyre_testing_plugin.pytest_plugin import spyre_available

    if not spyre_available():
        pytest.skip("requires Spyre hardware")

    from spyre_inference.custom_ops.utils import convert

    def gather(t: torch.Tensor, world_size: int, group_name: str) -> torch.Tensor:
        host = t if t.device.type == "cpu" else convert(t, device="cpu")
        return torch.cat([host + r for r in range(world_size)], dim=0)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(torch.ops._c10d_functional, "all_gather_into_tensor", gather)
    monkeypatch.setattr(torch.ops._c10d_functional, "wait_tensor", lambda t: t)
    host = torch.arange(4 * 64, dtype=torch.float16).view(4, 64)

    out = _comm().all_gather(convert(host, device="spyre"), dim=-1)

    torch.testing.assert_close(out.cpu(), torch.cat([host + r for r in range(2)], dim=-1))


def test_eager_all_gather_gathers_the_list_and_concats(monkeypatch: pytest.MonkeyPatch) -> None:
    """The eager path keeps the list-form ``dist.all_gather`` (the functional one routes
    to a coalesced entry point spyreccl rejects) and concatenates its results."""
    from spyre_testing_plugin.pytest_plugin import spyre_available

    if not spyre_available():
        pytest.skip("requires Spyre hardware")

    from spyre_inference.custom_ops.utils import convert

    host = torch.arange(4 * 64, dtype=torch.float16).view(4, 64)
    device_tensor = convert(host, device="spyre")
    seen: list[bool] = []

    def fake_all_gather(output_list, tensor, group=None, **kwargs):
        seen.append(tensor.is_contiguous())
        host_shard = convert(tensor, device="cpu")
        for rank, out in enumerate(output_list):
            out.copy_(host_shard + rank)

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)

    out = _comm().all_gather(device_tensor, dim=-1)

    assert seen == [True]
    torch.testing.assert_close(out.cpu(), torch.cat([host + r for r in range(2)], dim=-1))


@pytest.mark.parametrize("dim", [-1, 0])
def test_eager_all_gather_pads_and_strips_on_the_host(
    monkeypatch: pytest.MonkeyPatch, dim: int
) -> None:
    """The eager gather pads the gathered dim to the 64-element alignment
    libspyre_comms transfers in (an unaligned gather faults the card and needs a device
    recovery), then strips the pad from every rank's shard before concatenating."""
    from spyre_testing_plugin.pytest_plugin import spyre_available

    if not spyre_available():
        pytest.skip("requires Spyre hardware")

    from spyre_inference.custom_ops.utils import convert

    align = SpyreCommunicator._GATHER_ALIGN
    orig = 3 * align + 10  # not a multiple of the alignment
    shape = (orig, 5) if dim == 0 else (5, orig)
    host = torch.arange(shape[0] * shape[1], dtype=torch.float16).view(*shape)
    device_tensor = convert(host, device="spyre")
    gathered: list[torch.Size] = []

    def fake_all_gather(output_list, tensor, group=None, **kwargs):
        gathered.append(tensor.shape)
        host_shard = convert(tensor, device="cpu")
        for rank, out in enumerate(output_list):
            out.copy_(host_shard + rank)

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)

    out = _comm().all_gather(device_tensor, dim=dim)

    # The device saw a whole number of 64-element chunks along the gathered dim, and
    # the pad was stripped from the result: the values match an unpadded gather.
    padded_shape = list(shape)
    padded_shape[dim % 2] += (-orig) % align
    assert gathered == [torch.Size(padded_shape)]
    torch.testing.assert_close(out.cpu(), torch.cat([host + r for r in range(2)], dim=dim % 2))
