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

"""collect_failed_suites.py: a leg whose runner died is retried, and fails the run if it isn't."""

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
A, B, C = (e["cfg"] for e in INCLUDE)
PLACEHOLDER = "${{ matrix.cfg }}"
SUFFIX = {0: "", 1: " (pod-level retry)", 2: " (pod-level retry 2)"}


def _job(cfg: str, conclusion: str | None, rnd: int = 0, tests_failed: bool = False) -> dict:
    steps = [{"name": "Set up job", "conclusion": "success"}]
    if conclusion is not None:
        step = "failure" if tests_failed else ("success" if conclusion == "success" else None)
        steps.append({"name": "Run matrix config" + SUFFIX[rnd], "conclusion": step})
    return {
        "name": f"test / build and run tests ({cfg}){SUFFIX[rnd]}",
        "status": "completed" if conclusion else "in_progress",
        "conclusion": conclusion,
        "steps": steps,
    }


def _passed(*extra: dict) -> cfs.Legs:
    return cfs.parse(
        [
            *(_job(cfg, "success") for cfg in (A, B, C)),
            _job(PLACEHOLDER, "skipped", rnd=1),
            _job(PLACEHOLDER, "skipped", rnd=2),
            {"name": "test / Test matrix result", "status": "in_progress", "conclusion": None},
            *extra,
        ]
    )


def _retried(legs: cfs.Legs, rnd: int) -> list[str]:
    return [e["cfg"] for e in cfs.retry_matrix(legs, INCLUDE, rnd)["include"]]


def test_static_matrix_cfgs_are_unique():
    cfgs = [e["cfg"] for e in cfs.matrix_include()]
    assert len(cfgs) == len(set(cfgs))


def test_clean_run_retries_nothing_and_verifies():
    assert _retried(_passed(), 1) == []
    assert _retried(_passed(), 2) == []
    assert cfs.verify(_passed(), INCLUDE) == []


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out", None])
def test_any_unsuccessful_leg_is_retried_with_its_full_entry(conclusion):
    legs = _passed(_job(B, conclusion))
    assert cfs.retry_matrix(legs, INCLUDE, 1) == {"include": [INCLUDE[1]]}


def test_leg_missing_from_job_list_is_retried():
    legs = _passed()
    del legs.test[C]
    assert _retried(legs, 1) == [C]
    assert cfs.verify(legs, INCLUDE) == [f"{C}: did not pass (test, then each retry round: [])"]


def test_unretried_failure_fails_verify():
    errors = cfs.verify(_passed(_job(A, "failure")), INCLUDE)
    assert errors == [f"{A}: did not pass (test, then each retry round: ['failure'])"]


def test_retry_outcome_decides_verify():
    assert cfs.verify(_passed(_job(A, "failure"), _job(A, "success", rnd=1)), INCLUDE) == []
    assert cfs.verify(_passed(_job(A, "failure"), _job(A, "failure", rnd=1)), INCLUDE)


def test_second_round_only_reruns_infra_deaths():
    legs = _passed(
        _job(A, "failure", tests_failed=True),
        _job(A, "failure", rnd=1, tests_failed=True),
        _job(B, "failure"),
        _job(B, "failure", rnd=1),
        _job(C, "failure"),
        _job(C, "success", rnd=1),
    )
    assert _retried(legs, 2) == [B]


def test_second_round_reruns_a_leg_round_one_never_reached():
    legs = _passed(_job(A, "failure"))
    assert _retried(legs, 2) == [A]


def test_second_round_success_passes_verify():
    legs = _passed(_job(B, "failure"), _job(B, "failure", rnd=1), _job(B, "success", rnd=2))
    assert cfs.verify(legs, INCLUDE) == []


def test_unexpanded_matrix_fails_verify_without_crashing_collect():
    legs = cfs.parse([_job(PLACEHOLDER, "cancelled"), _job(PLACEHOLDER, "skipped", rnd=1)])
    assert _retried(legs, 1) == []
    assert cfs.verify(legs, INCLUDE) == ["the `test` matrix never expanded, so no suite ran"]


def test_failed_leg_outside_static_matrix_warns_and_fails_verify():
    legs = _passed(_job("Renamed suite", "failure"))
    assert _retried(legs, 1) == []
    assert cfs.verify(legs, INCLUDE) == ["Renamed suite: failed, and is not in the static matrix"]


def test_partial_rerun_falls_back_to_earlier_attempt(monkeypatch):
    attempt1 = [_job(A, "success"), _job(B, "success"), _job(C, "failure")]
    attempt2 = [_job(C, "success")]
    monkeypatch.setattr(cfs, "list_jobs", lambda repo, run, a: {1: attempt1, 2: attempt2}[a])
    legs = cfs.load("r", "1", 2, {A, B, C})
    assert {c: j["conclusion"] for c, j in legs.test.items()} == dict.fromkeys((A, B, C), "success")


def test_pending_legs_are_polled_until_complete(monkeypatch):
    calls = iter([[_job(A, None), _job(B, "success"), _job(C, "success")], _passed_jobs()])
    monkeypatch.setattr(cfs, "list_jobs", lambda *_: next(calls))
    monkeypatch.setattr(cfs.time, "sleep", lambda _: None)
    assert cfs.load("r", "1", 1, {A, B, C}).test[A]["conclusion"] == "success"


def _passed_jobs() -> list[dict]:
    return [_job(cfg, "success") for cfg in (A, B, C)]
