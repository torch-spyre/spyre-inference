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
Pin the `test-durations` artifact (uploaded by the `durations` job of an earlier
run) that this run's shard jobs weight their partition by. One id is resolved
here so every shard of a suite reads the SAME durations -- shards computing
different weights would drop or double-run tests.

Prefers the newest artifact on this PR's head branch (so a PR with heavy test
changes self-balances against its own last run) and falls back to the base
branch. Writes `durations_run_id=<id>` (empty when none found) to GITHUB_OUTPUT;
an empty pin makes the plugin fall back to its static heuristic weights.

Usage (called by _test_matrix.yaml's resolve_durations job):
    GH_TOKEN=... REPO=... HEAD_REF=... BASE_REF=... python3 resolve_durations_run.py
"""

import json
import os
import subprocess

# Cap on serial per-run conclusion lookups in pick_run_id (one gh api call each), so a
# long failure streak on a branch can't sit unboundedly in front of the whole matrix.
_MAX_CONCLUSION_LOOKUPS = 10


def list_durations_artifacts(repo: str) -> list[dict]:
    # The artifacts REST API's workflow_run object exposes only id/head_branch/
    # head_sha (no conclusion), so run status is looked up separately per run.
    result = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{repo}/actions/artifacts?name=test-durations&per_page=100",
            "--jq",
            ".artifacts[] | select(.expired == false) "
            "| {created_at: .created_at, branch: .workflow_run.head_branch, "
            "run_id: .workflow_run.id}",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def run_conclusion(repo: str, run_id: int) -> str:
    # "" on any lookup failure -- treated as non-passing, never blocks the resolve.
    try:
        result = subprocess.run(
            ["gh", "api", f"/repos/{repo}/actions/runs/{run_id}", "--jq", ".conclusion"],
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to read conclusion for run {run_id} ({e}).")
        return ""
    return result.stdout.strip()


def pick_run_id(repo: str, artifacts: list[dict], head_ref: str, base_ref: str) -> str:
    # Head branch first so a PR self-balances against its own last run; else base.
    # Within a branch prefer a passing run: the `durations` job is `if: always()`,
    # so a failed run uploads a partial, skewed file; fall back to newest if none
    # passed.
    for branch in (head_ref, base_ref):
        if not branch:
            continue
        matches = sorted(
            (a for a in artifacts if a["branch"] == branch),
            key=lambda a: a["created_at"],
            reverse=True,
        )
        if not matches:
            continue
        for a in matches[:_MAX_CONCLUSION_LOOKUPS]:
            if run_conclusion(repo, a["run_id"]) == "success":
                print(f"Pinning durations to passing run {a['run_id']} (branch {branch}).")
                return str(a["run_id"])
        newest = matches[0]
        print(
            f"Pinning durations to most recent run {newest['run_id']} "
            f"(branch {branch}); no passing run yet."
        )
        return str(newest["run_id"])
    print("No test-durations artifact found; shards will use heuristic weights.")
    return ""


def main() -> None:
    repo = os.environ["REPO"]
    head_ref = os.environ.get("HEAD_REF", "")
    base_ref = os.environ.get("BASE_REF", "")

    try:
        artifacts = list_durations_artifacts(repo)
    except subprocess.CalledProcessError as e:
        # Never block the run on a resolve failure: fall back to heuristic weights.
        print(f"Failed to list artifacts ({e}); shards will use heuristic weights.")
        artifacts = []

    run_id = pick_run_id(repo, artifacts, head_ref, base_ref)
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"durations_run_id={run_id}\n")


if __name__ == "__main__":
    main()
