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

"""Routed-expert layers prepared on threads during load (no device needed).

The pool is scoped to one load. ``_prepare_layer`` is mocked, so these cover who runs
which layer and what leaves the context; the device tests in ``test_gemma4_moe.py``
check that thread-prepared and chunk-streamed stacks equal the one-shot ones.
"""

import contextlib
import threading
from types import SimpleNamespace

import pytest
import torch

from spyre_inference import envs, moe
from spyre_inference.moe import prepare_moe_layers_on_threads


@pytest.fixture(autouse=True)
def fake_spyre(monkeypatch):
    """A started runtime with no device: the threads only need these calls."""
    monkeypatch.setattr(
        torch,
        "spyre",
        SimpleNamespace(
            current_device=lambda: 0,
            set_device=lambda idx: None,
            Stream=lambda: "stream",
            stream=lambda s: contextlib.nullcontext(),
        ),
        raising=False,
    )


def _prep_threads_alive() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("spyre-moe-prep")]


def _setenv(monkeypatch, name, value):
    """envs.py caches each value on first read; drop it so the new value is seen."""
    monkeypatch.setenv(name, value)
    envs.clear_env_cache()


def _prepare_all(monkeypatch, layers, *, threads=4, fail=None):
    """Prepare ``layers`` inside the context; returns {layer: thread name}."""
    _setenv(monkeypatch, "SPYRE_MOE_PREP_THREADS", str(threads))
    done = {}

    def prepare(layer):
        if layer == fail:
            raise ValueError(f"layer {layer} failed")
        done[layer] = threading.current_thread().name

    monkeypatch.setattr(moe, "_prepare_layer", prepare)
    with prepare_moe_layers_on_threads():
        for layer in layers:
            moe._prepare_layer_soon(layer)
    return done


def test_layers_after_the_first_run_on_threads(monkeypatch):
    done = _prepare_all(monkeypatch, range(6))
    assert sorted(done) == list(range(6))
    assert done[0] == threading.current_thread().name
    assert all(done[i].startswith("spyre-moe-prep") for i in range(1, 6))
    assert not _prep_threads_alive(), "the pool must not outlive the load"


def test_a_failed_layer_is_raised_when_the_load_ends(monkeypatch):
    with pytest.raises(ValueError, match="layer 3 failed"):
        _prepare_all(monkeypatch, range(6), fail=3)
    assert not _prep_threads_alive()


def test_one_thread_prepares_in_order_on_the_caller(monkeypatch):
    done = _prepare_all(monkeypatch, range(4), threads=1)
    assert list(done) == list(range(4))
    assert set(done.values()) == {threading.current_thread().name}
    assert not _prep_threads_alive()


def test_outside_a_load_every_layer_runs_on_the_caller(monkeypatch):
    """A direct call, as a unit test or a reload path, keeps the synchronous behaviour."""
    _setenv(monkeypatch, "SPYRE_MOE_PREP_THREADS", "4")
    done = {}
    monkeypatch.setattr(
        moe, "_prepare_layer", lambda layer: done.setdefault(layer, threading.current_thread().name)
    )
    for layer in range(3):
        moe._prepare_layer_soon(layer)
    assert list(done) == [0, 1, 2]
    assert set(done.values()) == {threading.current_thread().name}
    assert not _prep_threads_alive()


def test_a_load_that_raises_leaves_no_threads_and_no_state_behind(monkeypatch):
    _setenv(monkeypatch, "SPYRE_MOE_PREP_THREADS", "4")
    started = threading.Event()
    release = threading.Event()

    def prepare(layer):
        if layer:
            started.set()
            release.wait(5)

    monkeypatch.setattr(moe, "_prepare_layer", prepare)
    with pytest.raises(RuntimeError, match="load failed"), prepare_moe_layers_on_threads():
        for layer in range(3):
            moe._prepare_layer_soon(layer)
        assert started.wait(5)
        release.set()
        raise RuntimeError("load failed")
    assert not _prep_threads_alive()
    assert moe._active_prep.get() is None
    # The next load starts afresh: its first layer runs on the caller again.
    done = _prepare_all(monkeypatch, range(2))
    assert done[0] == threading.current_thread().name


def test_two_loads_in_one_process_each_get_their_own_pool(monkeypatch):
    """A script that builds several ``vllm.LLM`` objects loads several models in turn."""
    for _ in range(2):
        done = _prepare_all(monkeypatch, range(3))
        assert done[0] == threading.current_thread().name
        assert all(done[i].startswith("spyre-moe-prep") for i in (1, 2))
        assert not _prep_threads_alive()


def test_the_thread_count_is_clamped_and_a_negative_one_rejected(monkeypatch):
    _setenv(monkeypatch, "SPYRE_MOE_PREP_THREADS", "64")
    assert moe._prep_thread_count() == 16
    _setenv(monkeypatch, "SPYRE_MOE_PREP_THREADS", "0")
    assert moe._prep_thread_count() == max(1, min(8, torch.get_num_threads()))
    _setenv(monkeypatch, "SPYRE_MOE_PREP_THREADS", "-1")
    with pytest.raises(ValueError, match="SPYRE_MOE_PREP_THREADS"):
        moe._prep_thread_count()
