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

"""Lists a GHA run's artifacts (paginated: the default 30/page silently dropped the rest)."""

import argparse
import json
import os
import subprocess


def list_artifacts(repo: str, run_id: str) -> list[dict]:
    result = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
            "--jq",
            ".artifacts[] | {id: .id, name: .name}",
        ],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="GHA run to list")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/repo the run belongs to (default: $GITHUB_REPOSITORY)",
    )
    args = parser.parse_args()
    if not args.repo:
        parser.error("--repo is required when $GITHUB_REPOSITORY is unset")

    artifacts = list_artifacts(args.repo, args.run_id)
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"artifacts={json.dumps(artifacts)}\n")
    print("Found artifacts:")
    for a in artifacts:
        print(" •", a["name"])


if __name__ == "__main__":
    main()
