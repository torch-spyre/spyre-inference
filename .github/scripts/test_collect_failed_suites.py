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

"""collect_failed_suites.py: a leg whose runner died must be retried, and fail the run if it isn't."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import collect_failed_suites as cfs  # noqa: E402

INCLUDE = [
    {"cfg": "Spyre probe tests (shard 1/2)", "test_target": "test-probes-shard-0"},
    {"cfg": "Spyre probe tests (shard 2/2)", "test_target": "test-probes-shard-1"},
    {"cfg": "Encoder attention tests", "test_target": "test-encoder-attention"},
]


def _job(cfg: str, conclusion: str | None, retry: bool = False, prefix: str = "test") -> dict:
    name = f"{prefix} / build and run tests ({cfg})" + (cfs.RETRY_SUFFIX if retry else "")
    status = "completed" if conclusion else "in_progress"
    return {"name": name, "status": status, "conclusion": conclusion}


def _all_passed() -> list[dict]:
    return [_job(e["cfg"], "success") for e in INCLUDE] + [
        _job("${{ matrix.cfg }}", "skipped", retry=True),
        {"name": "test / Test matrix result", "status": "in_progress", "conclusion": None},
    ]


def test_static_matrix_cfgs_are_unique():
    cfgs = [e["cfg"] for e in cfs.matrix_include()]
    assert len(cfgs) == len(set(cfgs))


def test_clean_run_retries_nothing_and_verifies():
    assert cfs.build_retry_matrix(_all_passed(), INCLUDE) == {"include": []}
    assert cfs.verify(_all_passed(), INCLUDE) == []


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out", None])
def test_any_unsuccessful_leg_is_retried_with_its_full_entry(conclusion):
    jobs = _all_passed()
    jobs[1] = _job(INCLUDE[1]["cfg"], conclusion, prefix="run-tests")
    assert cfs.build_retry_matrix(jobs, INCLUDE) == {"include": [INCLUDE[1]]}


def test_unretried_failure_fails_verify():
    jobs = _all_passed()
    jobs[0] = _job(INCLUDE[0]["cfg"], "failure")
    assert cfs.verify(jobs, INCLUDE) == [f"{INCLUDE[0]['cfg']}: failed and was never retried"]


def test_retry_outcome_decides_verify():
    jobs = _all_passed()
    jobs[0] = _job(INCLUDE[0]["cfg"], "failure")
    assert cfs.verify([*jobs, _job(INCLUDE[0]["cfg"], "success", retry=True)], INCLUDE) == []
    assert cfs.verify([*jobs, _job(INCLUDE[0]["cfg"], "failure", retry=True)], INCLUDE)


def test_missing_leg_fails_verify():
    jobs = [j for j in _all_passed() if INCLUDE[2]["cfg"] not in j["name"]]
    assert cfs.verify(jobs, INCLUDE) == [f"{INCLUDE[2]['cfg']}: no `test` leg ran"]


def test_failed_leg_outside_static_matrix_is_an_error():
    jobs = [*_all_passed(), _job("Renamed suite", "failure")]
    with pytest.raises(SystemExit):
        cfs.build_retry_matrix(jobs, INCLUDE)
