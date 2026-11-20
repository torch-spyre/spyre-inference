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

"""_test_leg.yaml retries unless the attempt proved it passed, so a dead runner retries (#1454)."""

from __future__ import annotations

import pathlib

import yaml

_WORKFLOW = pathlib.Path(__file__).resolve().parents[1] / "workflows" / "_test_leg.yaml"


def _jobs() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text())["jobs"]


def test_attempt_sets_passed_in_its_last_unconditional_step():
    attempt = _jobs()["attempt"]
    last = attempt["steps"][-1]
    assert "if" not in last
    assert last["id"] == "done"
    assert "passed=true" in last["run"]
    assert attempt["outputs"] == {"passed": "${{ steps.done.outputs.passed }}"}


def test_retry_gates_only_on_the_missing_passed_output():
    gate = _jobs()["retry"]["if"]
    assert "needs.attempt.outputs.passed != 'true'" in gate
    assert ".result" not in gate
    assert "failure" not in gate


def test_job_names_carry_the_cfg():
    for job in _jobs().values():
        assert "${{ inputs.cfg }}" in job["name"]
