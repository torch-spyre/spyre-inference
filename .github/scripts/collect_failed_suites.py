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
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "_test_matrix.yaml"
RETRY_SUFFIX = " (pod-level retry)"
# The caller's job id prefixes the name ("test / ...", "run-tests / ..."); cfg itself has parens.
LEG_NAME = re.compile(r"build and run tests \((?P<cfg>.+)\)$")
PASSED = ("success", "skipped")


def matrix_include() -> list[dict]:
    doc = yaml.safe_load(WORKFLOW.read_text())
    return doc["jobs"]["test"]["strategy"]["matrix"]["include"]


def list_jobs(repo: str, run_id: str, attempt: str) -> list[dict]:
    cmd = [
        "gh",
        "api",
        "--paginate",
        f"/repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100",
        "--jq",
        ".jobs[] | {name, status, conclusion}",
    ]
    for delay in (5, 15, 45, 0):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        print(result.stderr, file=sys.stderr)
        time.sleep(delay)
    raise SystemExit("::error::Could not list this run attempt's jobs.")


def legs(jobs: list[dict]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    test: dict[str, list[dict]] = {}
    retry: dict[str, list[dict]] = {}
    for job in jobs:
        name = job["name"]
        is_retry = name.endswith(RETRY_SUFFIX)
        if is_retry:
            name = name[: -len(RETRY_SUFFIX)]
        m = LEG_NAME.search(name)
        if m:
            (retry if is_retry else test).setdefault(m["cfg"], []).append(job)
    return test, retry


def failed_cfgs(test: dict[str, list[dict]]) -> list[str]:
    # No conclusion yet (runner lost, API lag) counts as failed.
    return [cfg for cfg, jobs in test.items() if any(j["conclusion"] not in PASSED for j in jobs)]


def build_retry_matrix(jobs: list[dict], include: list[dict]) -> dict:
    test, _ = legs(jobs)
    by_cfg = {entry["cfg"]: entry for entry in include}
    failed = failed_cfgs(test)
    unknown = [cfg for cfg in failed if cfg not in by_cfg]
    if unknown:
        raise SystemExit(f"::error::Failed legs not in the static matrix: {unknown}")
    print(f"Suites to retry: {failed}")
    return {"include": [by_cfg[cfg] for cfg in failed]}


def verify(jobs: list[dict], include: list[dict]) -> list[str]:
    test, retry = legs(jobs)
    errors = [f"{e['cfg']}: no `test` leg ran" for e in include if e["cfg"] not in test]
    for cfg in failed_cfgs(test):
        conclusions = [j["conclusion"] for j in retry.get(cfg, [])]
        if not conclusions:
            errors.append(f"{cfg}: failed and was never retried")
        elif any(c != "success" for c in conclusions):
            errors.append(f"{cfg}: failed, and its pod-level retry ended {conclusions}")
    return errors


def set_output(name: str, value: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"{name}={value}\n")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "collect"
    repo = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["GITHUB_RUN_ID"]
    attempt = os.environ["GITHUB_RUN_ATTEMPT"]
    include = matrix_include()

    cfgs = {entry["cfg"] for entry in include}
    # `needs` has already waited for these jobs; this only rides out API lag on their conclusions.
    for _ in range(6):
        jobs = list_jobs(repo, run_id, attempt)
        pending = [
            cfg
            for side in legs(jobs)
            for cfg, js in side.items()
            if cfg in cfgs and any(j["status"] != "completed" for j in js)
        ]
        if not pending:
            break
        time.sleep(10)

    if mode == "collect":
        matrix = build_retry_matrix(jobs, include)
        set_output("matrix", json.dumps(matrix))
        set_output("has_failed", "true" if matrix["include"] else "false")
    elif mode == "verify":
        errors = verify(jobs, include)
        for err in errors:
            print(f"::error::{err}")
        if errors:
            sys.exit(1)
        print(f"All {len(include)} matrix configs passed (directly or after pod-level retry).")
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
