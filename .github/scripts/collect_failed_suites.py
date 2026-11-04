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

"""
Pod-level retry for _test_matrix.yaml, driven by job conclusions so a leg whose runner died
before uploading anything is still retried (#1454).

    collect 1 -- retry matrix of every `test` leg that did not pass.
    collect 2 -- retry matrix of round-1 retries that died on infra, not on a test failure.
    verify    -- fail unless every matrix leg passed directly or on a retry.
"""

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "_test_matrix.yaml"
ROUNDS = 2
# The caller's job id prefixes the name ("test / ...", "run-tests / ..."); cfg itself has parens.
LEG_NAME = re.compile(
    r"build and run tests \((?P<cfg>.+?)\)(?P<retry> \(pod-level retry(?: (?P<round>\d+))?\))?$"
)
PASSED = ("success", "skipped")


def matrix_include() -> list[dict]:
    doc = yaml.safe_load(WORKFLOW.read_text())
    return doc["jobs"]["test"]["strategy"]["matrix"]["include"]


def list_jobs(repo: str, run_id: str, attempt: int) -> list[dict]:
    cmd = [
        "gh",
        "api",
        "--paginate",
        f"/repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100",
        "--jq",
        ".jobs[] | {name, status, conclusion, steps: [.steps[]? | {name, conclusion}]}",
    ]
    delays = (5, 15, 30, 60, 120)
    for i in range(len(delays) + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        print(result.stderr, file=sys.stderr)
        if i < len(delays):
            time.sleep(delays[i])
    raise SystemExit(f"::error::Could not list the jobs of run attempt {attempt}.")


@dataclass
class Legs:
    """Matrix jobs keyed by cfg: `test` legs, and retry legs per round (0 = the `test` leg)."""

    rounds: list[dict[str, dict]] = field(default_factory=lambda: [{} for _ in range(ROUNDS + 1)])

    @property
    def test(self) -> dict[str, dict]:
        return self.rounds[0]

    def add(self, job: dict) -> None:
        m = LEG_NAME.search(job["name"])
        # An unexpanded matrix reports one job literally named "... (${{ matrix.cfg }})".
        if m and "${{" not in m["cfg"]:
            rnd = int(m["round"] or 1) if m["retry"] else 0
            self.rounds[rnd][m["cfg"]] = job


def parse(jobs: list[dict]) -> Legs:
    legs = Legs()
    for job in jobs:
        legs.add(job)
    return legs


def died_on_infra(job: dict) -> bool:
    """Did not pass, but its test step never concluded with a failure (runner lost, setup)."""
    if job["conclusion"] in PASSED:
        return False
    return not any(
        s["name"].startswith("Run matrix config") and s["conclusion"] == "failure"
        for s in job.get("steps", [])
    )


def needs_round(legs: Legs, cfgs: list[str], rnd: int) -> list[str]:
    if not legs.test:
        return []
    if rnd == 1:
        # A cfg missing from the job list, or with no conclusion yet, counts as failed.
        return [c for c in cfgs if legs.test.get(c, {}).get("conclusion") not in PASSED]
    prev = legs.rounds[rnd - 1]
    return [c for c in needs_round(legs, cfgs, rnd - 1) if c not in prev or died_on_infra(prev[c])]


def retry_matrix(legs: Legs, include: list[dict], rnd: int) -> dict:
    cfgs = [e["cfg"] for e in include]
    unknown = sorted(
        c for c, j in legs.test.items() if c not in cfgs and j["conclusion"] not in PASSED
    )
    if unknown:
        print(f"::warning::Failed legs not in the static matrix, not retried: {unknown}")
    if not legs.test:
        print("::warning::The `test` matrix never expanded; nothing to retry.")
    retry = needs_round(legs, cfgs, rnd)
    print(f"Round {rnd} suites to retry: {retry}")
    return {"include": [e for e in include if e["cfg"] in retry]}


def verify(legs: Legs, include: list[dict]) -> list[str]:
    if not legs.test:
        return ["the `test` matrix never expanded, so no suite ran"]
    cfgs = [e["cfg"] for e in include]
    errors = [
        f"{c}: failed, and is not in the static matrix"
        for c, j in sorted(legs.test.items())
        if c not in cfgs and j["conclusion"] not in PASSED
    ]
    for cfg in cfgs:
        history = [r[cfg]["conclusion"] if cfg in r else "missing" for r in legs.rounds]
        while history and history[-1] == "missing":
            history.pop()
        if not history or history[-1] not in PASSED:
            errors.append(f"{cfg}: did not pass (test, then each retry round: {history})")
    return errors


def load(repo: str, run_id: str, attempt: int, cfgs: set[str]) -> Legs:
    # `needs` has already waited for these jobs; this only rides out API lag on their conclusions.
    polls = 18
    for i in range(polls):
        legs = parse(list_jobs(repo, run_id, attempt))
        pending = sorted(
            c for r in legs.rounds for c, j in r.items() if c in cfgs and j["status"] != "completed"
        )
        if not pending:
            break
        if i < polls - 1:
            time.sleep(10)
    else:
        print(f"::warning::Still not completed after waiting, counted as failed: {pending}")

    # A partial re-run lists carried-over jobs today; fall back to earlier attempts if it stops.
    for earlier in range(attempt - 1, 0, -1):
        missing = cfgs - legs.test.keys()
        if not missing or not legs.test:
            break
        for cfg, job in parse(list_jobs(repo, run_id, earlier)).test.items():
            if cfg in missing:
                print(f"{cfg}: not listed in attempt {attempt}, using attempt {earlier}")
                legs.test[cfg] = job
    return legs


def set_output(name: str, value: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"{name}={value}\n")


def main() -> None:
    mode, *rest = sys.argv[1:] or ["collect"]
    include = matrix_include()
    legs = load(
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_RUN_ID"],
        int(os.environ["GITHUB_RUN_ATTEMPT"]),
        {e["cfg"] for e in include},
    )

    if mode == "collect":
        matrix = retry_matrix(legs, include, int(rest[0]) if rest else 1)
        set_output("matrix", json.dumps(matrix))
        set_output("has_failed", "true" if matrix["include"] else "false")
    elif mode == "verify":
        errors = verify(legs, include)
        for err in errors:
            print(f"::error::{err}")
        if errors:
            sys.exit(1)
        print(f"All {len(include)} matrix configs passed (directly or after pod-level retry).")
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
