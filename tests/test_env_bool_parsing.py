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

import importlib.util
from pathlib import Path

import pytest

# Load envs.py directly to avoid hardware-specific dependencies.
ENV_PATH = Path(__file__).resolve().parents[1] / "spyre_inference" / "envs.py"

spec = importlib.util.spec_from_file_location("spyre_envs_test", ENV_PATH)
envs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(envs)


@pytest.mark.parametrize(
    "variable",
    [
        "SPYRE_ATTN_PROFILING",
        "SPYRE_ATTN_RECORD",
        "SPYRE_ATTN_FOR_EACH_TILE",
        "SPYRE_BATCHED_DECODE",
        "SPYRE_JAGGED_DECODE",
        "SPYRE_BLOCKED_TRANSPOSE",
        "SPYRE_KERNEL_CACHE",
        "SPYRE_LOGITS_CPU_GATHER",
        "SPYRE_UPDATE_THREAD_CONFIG",
    ],
)
@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("off", False),
    ],
)
def test_boolean_env_parsing(monkeypatch, variable, value, expected):
    monkeypatch.setenv(variable, value)

    # Clear cached values so each test reads the patched environment.
    envs.clear_env_cache()

    assert getattr(envs, variable) is expected


DEFAULTS = {
    "SPYRE_ATTN_PROFILING": False,
    "SPYRE_ATTN_RECORD": True,
    "SPYRE_ATTN_FOR_EACH_TILE": True,
    "SPYRE_BATCHED_DECODE": True,
    "SPYRE_JAGGED_DECODE": True,
    "SPYRE_BLOCKED_TRANSPOSE": True,
    "SPYRE_KERNEL_CACHE": False,
    "SPYRE_LOGITS_CPU_GATHER": False,
    "SPYRE_UPDATE_THREAD_CONFIG": True,
}


@pytest.mark.parametrize("variable, expected", DEFAULTS.items())
def test_boolean_env_defaults(monkeypatch, variable, expected):
    monkeypatch.delenv(variable, raising=False)
    envs.clear_env_cache()

    assert getattr(envs, variable) is expected


@pytest.mark.parametrize("variable", DEFAULTS)
def test_boolean_env_invalid_value(monkeypatch, variable):
    monkeypatch.setenv(variable, "not_a_boolean")
    envs.clear_env_cache()

    with pytest.raises(ValueError, match="Invalid boolean value"):
        getattr(envs, variable)


@pytest.mark.parametrize("variable", DEFAULTS)
@pytest.mark.parametrize("value", ["2", "-1"])
def test_boolean_env_nonzero_integer(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    envs.clear_env_cache()

    assert getattr(envs, variable) is True
