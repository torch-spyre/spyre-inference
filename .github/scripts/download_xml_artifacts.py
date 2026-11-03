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

"""Downloads+unzips a run's junit-* and id-sidecar artifacts, retrying and skipping bad ones."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ID_ARTIFACTS = {"run-id", "artifact-id"}


def safe_segment(name: str) -> bool:
    """True iff `name` is a single, non-traversing path component.

    Artifact names come from the GHA artifacts API and are attacker-controlled:
    any run that uploads artifacts (including a contributor's pull_request run)
    picks the name. `wanted()` only checks the `junit-` prefix, and
    `junit-../../x` keeps that prefix, so without this guard `out_dir / name`
    would escape `out_dir` and `unzip -d <dest>` would write attacker-chosen
    files outside it — on a runner that holds ClickHouse credentials. Reject
    anything that is not one plain segment.
    """
    return (
        name not in ("", ".", "..")
        and "/" not in name
        and "\\" not in name
        and os.sep not in name
        and (os.altsep is None or os.altsep not in name)
        and not name.startswith(".")
    )


def wanted(artifact: dict) -> bool:
    name = artifact["name"]
    return safe_segment(name) and (name.startswith("junit-") or name in ID_ARTIFACTS)


def fetch(repo: str, artifact: dict, zip_path: Path, dest: Path, attempts: int) -> bool:
    """Download and unzip one artifact; retried whole, since a truncated zip only fails at unzip."""
    for attempt in range(1, attempts + 1):
        shutil.rmtree(dest, ignore_errors=True)
        for cmd in (
            ["gh", "api", f"/repos/{repo}/actions/artifacts/{artifact['id']}/zip"],
            ["unzip", "-q", "-o", str(zip_path), "-d", str(dest)],
        ):
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                err = (result.stderr or result.stdout[-500:]).decode(errors="replace").strip()
                print(
                    f"[retry] {artifact['name']}: {cmd[0]} attempt {attempt}/{attempts} "
                    f"exited {result.returncode}: {err}"
                )
                break
            if cmd[0] == "gh":
                zip_path.write_bytes(result.stdout)
        else:
            return True
        if attempt < attempts:
            time.sleep(2**attempt)
    shutil.rmtree(dest, ignore_errors=True)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        default=os.environ.get("ARTIFACTS", "[]"),
        help='JSON array of {"id", "name"} objects (default: $ARTIFACTS)',
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/repo the artifacts belong to (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument("--out-dir", default="xml_artifacts", type=Path)
    parser.add_argument("--attempts", default=4, type=int)
    args = parser.parse_args()
    if not args.repo:
        parser.error("--repo is required when $GITHUB_REPOSITORY is unset")

    all_artifacts = json.loads(args.artifacts)
    for a in all_artifacts:
        # Surface — rather than silently drop — a name that would escape out_dir,
        # so a traversal attempt leaves a trace in the ingest log.
        if a["name"].startswith("junit-") and not safe_segment(a["name"]):
            print(
                f"::warning::Rejecting artifact {a['name']!r} (id={a['id']}): "
                "name is not a single safe path segment; refusing to use it as a path."
            )
    artifacts = [a for a in all_artifacts if wanted(a)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_root = args.out_dir.resolve()
    failed = []
    for a in artifacts:
        print(f"Downloading: {a['name']} (id={a['id']})")
        # Own subdir per artifact so same-basename XMLs never clobber; ingest globs recursively.
        dest = args.out_dir / a["name"]
        # Defence in depth: wanted() already requires a safe single segment, so
        # this never fires for a benign name — but keep the containment check
        # right next to the write so no future caller can slip a path out.
        resolved = dest.resolve()
        if resolved.parent != out_root:
            print(
                f"::warning::Rejecting artifact {a['name']!r} (id={a['id']}): "
                "resolved destination escapes the output directory."
            )
            failed.append(a["name"])
            continue
        if not fetch(args.repo, a, args.out_dir / f"{a['name']}.zip", dest, args.attempts):
            print(
                f"::warning::Skipping artifact {a['name']} (id={a['id']}): "
                f"download/unzip failed {args.attempts} times; its results are not ingested."
            )
            failed.append(a["name"])
    print(f"Fetched {len(artifacts) - len(failed)}/{len(artifacts)} artifact(s).")
    if artifacts and len(failed) == len(artifacts):
        print("::error::No artifact could be downloaded.")
        sys.exit(1)


if __name__ == "__main__":
    main()
