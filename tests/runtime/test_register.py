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

"""register() turns on flex's and torch-spyre's hazard tracking unless the user set it."""

import os

import pytest

import spyre_inference
from spyre_inference.v1.worker.spyre import triton_gate

HAZARD_VARS = ("SPYRE_HAZARD_TRACKER", "SPYRE_HAZARD_COMPLETED_READS")


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(triton_gate, "close", lambda: None)
    monkeypatch.setattr(triton_gate, "open", lambda: None)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "")


def test_register_enables_hazard_tracking_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in HAZARD_VARS:
        # setenv first so monkeypatch also undoes what register() sets.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)

    spyre_inference.register()

    assert [os.environ[name] for name in HAZARD_VARS] == ["1", "1"]


def test_register_keeps_an_explicit_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPYRE_HAZARD_TRACKER", "0")
    monkeypatch.setenv("SPYRE_HAZARD_COMPLETED_READS", "0")

    spyre_inference.register()

    assert [os.environ[name] for name in HAZARD_VARS] == ["0", "0"]
