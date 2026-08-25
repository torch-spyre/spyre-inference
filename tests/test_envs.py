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

"""Unit tests for ``spyre_inference.envs`` (no torch/vLLM required)."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _fresh_envs(monkeypatch: pytest.MonkeyPatch):
    """Reload envs with a clean cache for each test."""
    import spyre_inference.envs as envs

    envs.disable_envs_cache()
    monkeypatch.delenv("SPYRE_USE_SPYRE_SAMPLER", raising=False)
    monkeypatch.delenv("SPYRE_ASYNC_NOISE_SCALE", raising=False)
    importlib.reload(envs)
    yield envs
    envs.disable_envs_cache()


def test_use_spyre_sampler_default_on(monkeypatch: pytest.MonkeyPatch):
    import spyre_inference.envs as envs

    monkeypatch.delenv("SPYRE_USE_SPYRE_SAMPLER", raising=False)
    assert envs.SPYRE_USE_SPYRE_SAMPLER is True
    assert envs.is_set("SPYRE_USE_SPYRE_SAMPLER") is False


def test_use_spyre_sampler_override_off(monkeypatch: pytest.MonkeyPatch):
    import spyre_inference.envs as envs

    monkeypatch.setenv("SPYRE_USE_SPYRE_SAMPLER", "0")
    assert envs.SPYRE_USE_SPYRE_SAMPLER is False
    assert envs.is_set("SPYRE_USE_SPYRE_SAMPLER") is True


def test_async_noise_scale_default(monkeypatch: pytest.MonkeyPatch):
    import spyre_inference.envs as envs

    monkeypatch.delenv("SPYRE_ASYNC_NOISE_SCALE", raising=False)
    assert envs.SPYRE_ASYNC_NOISE_SCALE == 4


def test_async_noise_scale_override(monkeypatch: pytest.MonkeyPatch):
    import spyre_inference.envs as envs

    monkeypatch.setenv("SPYRE_ASYNC_NOISE_SCALE", "8")
    assert envs.SPYRE_ASYNC_NOISE_SCALE == 8


def test_async_noise_scale_rejects_below_two(monkeypatch: pytest.MonkeyPatch):
    import spyre_inference.envs as envs

    monkeypatch.setenv("SPYRE_ASYNC_NOISE_SCALE", "1")
    with pytest.raises(ValueError, match="must be >= 2"):
        _ = envs.SPYRE_ASYNC_NOISE_SCALE


def test_envs_cache_freezes_value(monkeypatch: pytest.MonkeyPatch):
    import spyre_inference.envs as envs

    monkeypatch.setenv("SPYRE_ASYNC_NOISE_SCALE", "6")
    envs.enable_envs_cache()
    assert envs.SPYRE_ASYNC_NOISE_SCALE == 6
    monkeypatch.setenv("SPYRE_ASYNC_NOISE_SCALE", "10")
    assert envs.SPYRE_ASYNC_NOISE_SCALE == 6
