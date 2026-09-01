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
Downloads this run-attempt's failed-suite descriptor artifacts (uploaded by
_test_matrix.yaml's `test` job on a failed matrix leg) and rebuilds the
{"include": [...]} retry matrix that the `test_retry` job consumes via
fromJSON.

Usage (called by the GHA workflow):
    GH_TOKEN=... python3 collect_failed_suites.py
Reads GITHUB_REPOSITORY, GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT, GITHUB_OUTPUT
from the environment (all set by default on GitHub-hosted/self-hosted runners).
"""

import glob
import json
import os
import subprocess
from pathlib import Path

FAILED_SUITES_DIR = Path("failed_suites")


def list_artifacts(repo: str, run_id: str, attempt: str) -> list[dict]:
    # --paginate (30/page default); attempt-scoped so a rerun skips already-passed suites.
    result = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
            "--jq",
            f'.artifacts[] | select(.name | startswith("failed-suite-{attempt}-")) '
            "| {id: .id, name: .name}",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def download_and_extract(repo: str, artifacts: list[dict]) -> None:
    FAILED_SUITES_DIR.mkdir(exist_ok=True)
    for artifact in artifacts:
        out_dir = FAILED_SUITES_DIR / artifact["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir.with_suffix(".zip")
        result = subprocess.run(
            ["gh", "api", f"/repos/{repo}/actions/artifacts/{artifact['id']}/zip"],
            capture_output=True,
            check=True,
        )
        zip_path.write_bytes(result.stdout)
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(out_dir)], check=True)
    print(f"Downloaded {len(artifacts)} failed-suite descriptor(s).")


def build_retry_matrix() -> dict:
    suites = []
    for path in sorted(glob.glob(str(FAILED_SUITES_DIR / "failed-suite-*/suite.json"))):
        with open(path) as f:
            suites.append(json.load(f))
    print(f"Suites to retry: {[s.get('cfg') for s in suites]}")
    return {"include": suites}


def set_output(name: str, value: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"{name}={value}\n")


def main() -> None:
    repo = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["GITHUB_RUN_ID"]
    attempt = os.environ["GITHUB_RUN_ATTEMPT"]

    artifacts = list_artifacts(repo, run_id, attempt)
    download_and_extract(repo, artifacts)
    matrix = build_retry_matrix()

    set_output("matrix", json.dumps(matrix))
    set_output("has_failed", "true" if matrix["include"] else "false")


if __name__ == "__main__":
    main()
