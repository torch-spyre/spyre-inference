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

import sys
from types import ModuleType

import pytest
import torch
import torch_spyre._inductor
from torch._dynamo.testing import CompileCounter
from torch.utils._config_module import install_config_module

from spyre_inference import envs
from spyre_inference.v1.attention.backends.spyre_head_major_attn import _run_with_capped_cores


def _core_count() -> int:
    from torch_spyre._inductor import config as spyre_config

    return spyre_config.sencores


@pytest.fixture(params=[None, 16], ids=["default", "overridden"])
def core_config(request, monkeypatch):
    config = ModuleType("spyre_test_core_config")
    config.__file__ = __file__
    config.sencores = 32
    install_config_module(config)
    if request.param is not None:
        config.sencores = request.param
    monkeypatch.setitem(sys.modules, config.__name__, config)
    monkeypatch.setattr(torch_spyre._inductor, "config", config)
    monkeypatch.setattr(envs, "SPYRE_ATTN_MAX_CORES", 8)
    torch._dynamo.reset()
    yield config
    torch._dynamo.reset()


def test_attention_core_cap_preserves_a_compiled_config_read(core_config):
    counter = CompileCounter()

    @torch.compile(backend=counter, fullgraph=True)
    def model(x):
        return x + _core_count()

    x = torch.zeros(64)
    expected = x + core_config.sencores
    torch.testing.assert_close(model(x), expected)
    assert counter.frame_count == 1

    def attention(y):
        return y + core_config.sencores

    torch.testing.assert_close(_run_with_capped_cores(1, attention, x), x + 8)

    torch.testing.assert_close(model(x), expected)
    assert counter.frame_count == 1

    core_config.sencores = 4
    torch.testing.assert_close(model(x), x + 4)
    assert counter.frame_count == 2


def test_attention_core_cap_restores_nested_scopes_after_an_exception(core_config, monkeypatch):
    original = core_config.sencores

    def inner():
        assert core_config.sencores == 4
        raise ValueError("inner scope")

    def outer():
        assert core_config.sencores == 8
        monkeypatch.setattr(envs, "SPYRE_ATTN_MAX_CORES", 4)
        try:
            _run_with_capped_cores(1, inner)
        finally:
            assert core_config.sencores == 8

    with pytest.raises(ValueError, match="inner scope"):
        _run_with_capped_cores(1, outer)
    assert core_config.sencores == original


def test_attention_core_cap_leaves_uncapped_calls_unchanged(core_config, monkeypatch):
    monkeypatch.setattr(envs, "SPYRE_ATTN_MAX_CORES", 0)

    def attention(x):
        return x + core_config.sencores

    x = torch.zeros(64)
    expected = x + core_config.sencores
    torch.testing.assert_close(_run_with_capped_cores(32, attention, x), expected)
