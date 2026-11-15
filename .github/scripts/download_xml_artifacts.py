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
import zipfile
from pathlib import Path

ID_ARTIFACTS = {"run-id", "artifact-id"}


class UnsafeZipMember(Exception):
    """A zip member resolves outside the extraction directory."""


def assert_zip_confined(zip_path: Path, dest: Path) -> None:
    """Refuse a zip whose members would escape `dest`.

    `safe_segment()` only guards the artifact name used as the extraction
    subdir; it says nothing about the member paths inside the downloaded zip.
    An artifact zip is attacker-controlled (any PR's CI can upload one), so a
    member with a `..` component or an absolute path could land outside `dest`
    even though the subdir name was safe. Validate every member before we hand
    the archive to `unzip`.
    """
    base = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            # A drive letter / UNC / absolute member, or one that climbs out
            # via `..`, resolves outside `base`.
            target = (dest / name).resolve()
            if target != base and base not in target.parents:
                raise UnsafeZipMember(f"{name!r} escapes {base}")


def safe_segment(name: str) -> bool:
    # The name is attacker-chosen and `wanted()` keeps `junit-../x`.
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
        dl = subprocess.run(
            ["gh", "api", f"/repos/{repo}/actions/artifacts/{artifact['id']}/zip"],
            capture_output=True,
        )
        if dl.returncode != 0:
            err = (dl.stderr or dl.stdout[-500:]).decode(errors="replace").strip()
            print(
                f"[retry] {artifact['name']}: gh attempt {attempt}/{attempts} "
                f"exited {dl.returncode}: {err}"
            )
        else:
            zip_path.write_bytes(dl.stdout)
            try:
                assert_zip_confined(zip_path, dest)
            except (UnsafeZipMember, zipfile.BadZipFile) as exc:
                # A traversal member is not a transient failure: do not retry it.
                print(
                    f"::warning::Rejecting artifact {artifact['name']!r} "
                    f"(id={artifact['id']}): {exc}"
                )
                shutil.rmtree(dest, ignore_errors=True)
                return False
            unz = subprocess.run(
                ["unzip", "-q", "-o", str(zip_path), "-d", str(dest)],
                capture_output=True,
            )
            if unz.returncode == 0:
                return True
            err = (unz.stderr or unz.stdout[-500:]).decode(errors="replace").strip()
            print(
                f"[retry] {artifact['name']}: unzip attempt {attempt}/{attempts} "
                f"exited {unz.returncode}: {err}"
            )
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
