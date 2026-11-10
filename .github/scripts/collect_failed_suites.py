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

    collect -- write the retry matrix and has_failed to GITHUB_OUTPUT.
    verify  -- fail unless every matrix leg passed directly or on retry.
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
# The caller's job id prefixes the name ("test / ...", "run-tests / ..."); cfg itself has parens.
LEG_NAME = re.compile(r"build and run tests \((?P<cfg>.+?)\)(?P<retry> \(pod-level retry\))?$")
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
        ".jobs[] | {name, status, conclusion}",
    ]
    delays = (5, 15, 30, 60, 120)
    for i in range(len(delays) + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        print(result.stderr, file=sys.stderr)
        # A 4xx other than a rate limit (missing `actions: read`, bad run id) won't clear on retry.
        if re.search(r"HTTP 4(?!29)\d\d", result.stderr) and "rate limit" not in result.stderr:
            break
        if i < len(delays):
            time.sleep(delays[i])
    raise SystemExit(f"::error::Could not list the jobs of run attempt {attempt}.")


@dataclass
class Legs:
    """Matrix jobs keyed by cfg."""

    test: dict[str, dict] = field(default_factory=dict)
    retry: dict[str, dict] = field(default_factory=dict)


def parse(jobs: list[dict]) -> Legs:
    legs = Legs()
    for job in jobs:
        m = LEG_NAME.search(job["name"])
        # An unexpanded matrix reports one job literally named "... (${{ matrix.cfg }})".
        if m and "${{" not in m["cfg"]:
            (legs.retry if m["retry"] else legs.test)[m["cfg"]] = job
    return legs


def failed(legs: Legs, cfgs: list[str]) -> list[str]:
    # A cfg missing from the job list, or with no conclusion yet, counts as failed.
    return [c for c in cfgs if legs.test.get(c, {}).get("conclusion") not in PASSED]


def unknown_failures(legs: Legs, cfgs: list[str]) -> list[str]:
    return sorted(
        c for c, j in legs.test.items() if c not in cfgs and j["conclusion"] not in PASSED
    )


def retry_matrix(legs: Legs, include: list[dict]) -> dict:
    cfgs = [e["cfg"] for e in include]
    if unknown := unknown_failures(legs, cfgs):
        print(f"::warning::Failed legs not in the static matrix, not retried: {unknown}")
    if not legs.test:
        print("::warning::The `test` matrix never expanded; nothing to retry.")
        return {"include": []}
    retry = failed(legs, cfgs)
    print(f"Suites to retry: {retry}")
    return {"include": [e for e in include if e["cfg"] in retry]}


def verify(legs: Legs, include: list[dict]) -> list[str]:
    if not legs.test:
        return ["the `test` matrix never expanded, so no suite ran"]
    cfgs = [e["cfg"] for e in include]
    errors = [f"{c}: failed, and is not in the static matrix" for c in unknown_failures(legs, cfgs)]
    for cfg in failed(legs, cfgs):
        test = legs.test.get(cfg, {}).get("conclusion", "missing")
        retry = legs.retry.get(cfg, {}).get("conclusion", "missing")
        if retry != "success":
            errors.append(f"{cfg}: did not pass (test: {test}, pod-level retry: {retry})")
    return errors


def load(repo: str, run_id: str, attempt: int, cfgs: set[str]) -> Legs:
    # `needs` has already waited for these jobs; this only rides out API lag on their conclusions.
    polls = 18
    for i in range(polls):
        legs = parse(list_jobs(repo, run_id, attempt))
        pending = sorted(
            c
            for side in (legs.test, legs.retry)
            for c, j in side.items()
            if c in cfgs and j["status"] != "completed"
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
        if not missing:
            break
        earlier_legs = parse(list_jobs(repo, run_id, earlier))
        for cfg, job in earlier_legs.test.items():
            if cfg in missing:
                print(f"{cfg}: not listed in attempt {attempt}, using attempt {earlier}")
                legs.test[cfg] = job
                if cfg not in legs.retry and cfg in earlier_legs.retry:
                    legs.retry[cfg] = earlier_legs.retry[cfg]
    return legs


def set_output(name: str, value: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"{name}={value}\n")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "collect"
    include = matrix_include()
    legs = load(
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_RUN_ID"],
        int(os.environ["GITHUB_RUN_ATTEMPT"]),
        {e["cfg"] for e in include},
    )

    if mode == "collect":
        matrix = retry_matrix(legs, include)
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
