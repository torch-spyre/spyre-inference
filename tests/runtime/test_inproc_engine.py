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

"""The worker's thread settings when the offline engine runs in the calling process."""

from types import SimpleNamespace

import pytest
import torch
import torch._inductor.config as inductor_config
import vllm.envs as vllm_envs

from spyre_inference.threading_config import apply_in_process_threading


@pytest.fixture
def thread_calls(monkeypatch):
    """torch.set_num_threads calls, with OMP_NUM_THREADS=8 and the Inductor env vars
    vLLM sets for a spawned engine."""
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("TORCHINDUCTOR_CPP_DYNAMIC_THREADS", "1")
    calls = []
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    return calls


def _multiprocessing(monkeypatch, on: bool) -> None:
    # A module attribute, so vLLM's env cache cannot hide it.
    monkeypatch.setattr(vllm_envs, "VLLM_ENABLE_V1_MULTIPROCESSING", on, raising=False)


@pytest.mark.parametrize(
    "multiprocessing, backend, in_process",
    [(False, "uni", True), (True, "uni", False), (False, "mp", False)],
)
def test_applies_only_in_process(monkeypatch, thread_calls, multiprocessing, backend, in_process):
    _multiprocessing(monkeypatch, multiprocessing)
    with inductor_config.patch({"cpp.dynamic_threads": False}):
        apply_in_process_threading(SimpleNamespace(distributed_executor_backend=backend))
        dynamic_threads = inductor_config.cpp.dynamic_threads
    assert thread_calls == ([8] if in_process else [])
    assert dynamic_threads is in_process


@pytest.mark.parametrize("omp", [None, "", "abc", "0"])
def test_unusable_omp_num_threads_is_ignored(monkeypatch, thread_calls, omp):
    _multiprocessing(monkeypatch, False)
    if omp is None:
        monkeypatch.delenv("OMP_NUM_THREADS")
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", omp)
    with inductor_config.patch({"cpp.dynamic_threads": False}):
        apply_in_process_threading(SimpleNamespace(distributed_executor_backend="uni"))
    assert thread_calls == []


def test_dynamic_threads_follows_the_env_var(monkeypatch, thread_calls):
    _multiprocessing(monkeypatch, False)
    monkeypatch.delenv("TORCHINDUCTOR_CPP_DYNAMIC_THREADS")
    with inductor_config.patch({"cpp.dynamic_threads": False}):
        apply_in_process_threading(SimpleNamespace(distributed_executor_backend="uni"))
        assert inductor_config.cpp.dynamic_threads is False
