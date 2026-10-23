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

"""resolve_durations_run.py pins the durations artifact matching the compile cache state."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import resolve_durations_run as rdr  # noqa: E402


def _artifact(run_id):
    return {"created_at": f"2026-10-23T00:00:{run_id:02d}Z", "branch": "main", "run_id": run_id}


@pytest.fixture
def resolve(monkeypatch, tmp_path):
    def run(available, warm):
        monkeypatch.setattr(rdr, "list_durations_artifacts", lambda repo, name: available[name])
        monkeypatch.setattr(rdr, "run_conclusion", lambda repo, run_id: "success")
        out = tmp_path / "out"
        out.write_text("")
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))
        monkeypatch.setenv("REPO", "o/r")
        monkeypatch.setenv("BASE_REF", "main")
        monkeypatch.setenv("COMPILE_CACHE_WARM", warm)
        rdr.main()
        return dict(line.split("=", 1) for line in out.read_text().splitlines())

    return run


BOTH = {rdr.WARM_ARTIFACT: [_artifact(1)], rdr.COLD_ARTIFACT: [_artifact(2)]}


@pytest.mark.parametrize(
    "warm,expected", [("true", ("1", rdr.WARM_ARTIFACT)), ("false", ("2", rdr.COLD_ARTIFACT))]
)
def test_prefers_artifact_matching_cache_state(resolve, warm, expected):
    out = resolve(BOTH, warm)
    assert (out["durations_run_id"], out["durations_artifact"]) == expected


def test_skipped_probe_counts_as_cold(resolve):
    assert resolve(BOTH, "")["durations_artifact"] == rdr.COLD_ARTIFACT


def test_falls_back_to_the_other_artifact(resolve):
    out = resolve({rdr.WARM_ARTIFACT: [_artifact(1)], rdr.COLD_ARTIFACT: []}, "false")
    assert (out["durations_run_id"], out["durations_artifact"]) == ("1", rdr.WARM_ARTIFACT)


def test_nothing_found_leaves_pin_empty(resolve):
    assert resolve({rdr.WARM_ARTIFACT: [], rdr.COLD_ARTIFACT: []}, "true")["durations_run_id"] == ""
