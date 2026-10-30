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

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

from spyre_inference.distributed.spyre_communicator import SpyreCommunicator


class _Tensor:
    def __init__(self, values, device="spyre"):
        self.values = values
        self.device = SimpleNamespace(type=device)

    @property
    def shape(self):
        return self.values.shape

    def dim(self):
        return self.values.dim()

    def reshape(self, *shape):
        return _Tensor(self.values.reshape(*shape))


@pytest.fixture
def communicator():
    result = SpyreCommunicator.__new__(SpyreCommunicator)
    result.world_size = 4
    result._group_name = "test-tp"
    return result


@pytest.mark.parametrize("compiling", [False, True])
@pytest.mark.parametrize("shape", [(64,), (3, 64), (1, 528, 1024)])
def test_all_reduce_preserves_rank_when_compiling(communicator, monkeypatch, compiling, shape):
    value = _Tensor(torch.arange(torch.tensor(shape).prod().item()).reshape(shape))
    before = value.values.clone()
    calls = []

    def reduce(tensor, operation, group):
        calls.append(("reduce", tensor, operation, group))
        return _Tensor(tensor.values.clone())

    def wait(tensor):
        calls.append(("wait", tensor))
        return tensor

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    monkeypatch.setattr(torch.ops._c10d_functional, "all_reduce", reduce)
    monkeypatch.setattr(torch.ops._c10d_functional, "wait_tensor", wait)
    result = communicator.all_reduce(value)
    assert len(calls) == 2
    assert calls[0][0] == "reduce" and calls[0][2:] == ("sum", "test-tp")
    assert calls[1][0] == "wait"
    sent = calls[0][1]
    if compiling or len(shape) == 1:
        assert sent is value
    else:
        assert sent.shape == (before.numel(),)
    assert result.shape == value.shape
    assert torch.equal(result.values, before)
    assert torch.equal(value.values, before)


def test_compiled_all_reduce_retains_real_view(communicator, monkeypatch):
    value = _Tensor(torch.arange(256).reshape(4, 64).t())
    assert not value.values.is_contiguous()
    sent = []

    def reduce(tensor, operation, group):
        sent.append(tensor)
        return tensor

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(torch.ops._c10d_functional, "all_reduce", reduce)
    monkeypatch.setattr(torch.ops._c10d_functional, "wait_tensor", lambda tensor: tensor)
    assert communicator.all_reduce(value) is value
    assert sent == [value]


@pytest.mark.parametrize("fallback", ["single_rank", "cpu", "missing_group"])
def test_all_reduce_retains_fallbacks(communicator, monkeypatch, fallback):
    value = _Tensor(torch.arange(64), "cpu" if fallback == "cpu" else "spyre")
    if fallback == "single_rank":
        communicator.world_size = 1
    elif fallback == "missing_group":
        communicator._group_name = None
    calls = []

    def base_reduce(self, tensor):
        calls.append(tensor)
        return tensor

    monkeypatch.setattr(DeviceCommunicatorBase, "all_reduce", base_reduce)
    assert communicator.all_reduce(value) is value
    assert calls == ([] if fallback == "single_rank" else [value])
