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

"""Routed-expert layers prepared on threads during load (no device needed)."""

import contextlib
import threading
from types import SimpleNamespace

import pytest
import torch

from spyre_inference import moe


@pytest.fixture
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


def _prepare_all(monkeypatch, threads, layers, fail=None):
    monkeypatch.setenv("SPYRE_MOE_PREP_THREADS", str(threads))
    done = {}

    def prepare(layer):
        if layer == fail:
            raise ValueError(f"layer {layer} failed")
        done[layer] = threading.current_thread().name

    monkeypatch.setattr(moe, "_prepare_layer", prepare)
    for layer in layers:
        moe._prepare_layer_soon(layer)
    return done


def test_layers_after_the_first_run_on_threads(monkeypatch, fake_spyre):
    done = _prepare_all(monkeypatch, 4, range(6))
    moe.wait_for_moe_layers()
    assert sorted(done) == list(range(6))
    assert done[0] == threading.current_thread().name
    assert all(done[i].startswith("spyre-moe-prep") for i in range(1, 6))
    assert moe._prep_pool is None and not moe._prep_futures


def test_a_failed_layer_is_raised_by_the_wait(monkeypatch, fake_spyre):
    _prepare_all(monkeypatch, 4, range(6), fail=3)
    with pytest.raises(ValueError, match="layer 3 failed"):
        moe.wait_for_moe_layers()
    assert moe._prep_pool is None and not moe._prep_futures


def test_one_thread_prepares_in_order_on_the_caller(monkeypatch, fake_spyre):
    done = _prepare_all(monkeypatch, 1, range(4))
    assert list(done) == list(range(4))
    assert set(done.values()) == {threading.current_thread().name}
    assert moe._prep_pool is None
    moe.wait_for_moe_layers()
