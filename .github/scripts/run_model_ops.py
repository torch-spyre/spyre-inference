#!/usr/bin/env python3
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

"""Run the model-ops YAMLs in tests/model_ops/ with torch-spyre's OOT test runner.

Each YAML lists the torch ops a model executes through this plugin; torch-spyre's
tests/models/test_model_ops_v2.py runs every case on Spyre and records its capability
verdict in the JUnit report, which the v2 ingest reads. That test, its conftest and
tests/run_test.sh are not in the torch_spyre wheel, so they come from a torch-spyre
source tree at the installed commit: the install itself when editable (the prebaked
image), else a sparse checkout of that commit, cached per commit. TORCH_DEVICE_ROOT,
when set, names the tree instead.

Each YAML runs in its own pytest session, since the runner's cross-case dedupe would
otherwise drop one model's case that another model already ran; the per-YAML reports
are combined into --junit-xml.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import NoReturn

SENTINEL = "tests/models/test_model_ops_v2.py"
# pyproject.toml makes the checkout pytest's rootdir, so this repo's pytest config stays
# out; run_test.sh calls mark_retried.py after a crashed test file.
SPARSE_PATHS = (
    "/tests/",
    "/pyproject.toml",
    "/extensions/clickhouse-ingest/spyre_clickhouse_ingest/mark_retried.py",
)


def fail(msg: str) -> NoReturn:
    sys.exit(f"run_model_ops: {msg}")


def checkout(url: str, commit: str, dest: Path) -> None:
    tmp = dest.with_name(f"{dest.name}.tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp), *args], check=True)

    tmp.mkdir(parents=True)
    git("init", "-q")
    git("remote", "add", "origin", url)
    git("sparse-checkout", "set", "--no-cone", *SPARSE_PATHS)
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
    git("checkout", "-q", "FETCH_HEAD")
    try:
        tmp.rename(dest)
    except OSError:
        # A concurrent run finished the same commit first.
        shutil.rmtree(tmp, ignore_errors=True)


def torch_spyre_source() -> Path:
    override = os.environ.get("TORCH_DEVICE_ROOT")
    if override:
        if not (Path(override) / SENTINEL).is_file():
            fail(f"TORCH_DEVICE_ROOT={override} has no {SENTINEL}")
        return Path(override)

    try:
        text = distribution("torch_spyre").read_text("direct_url.json")
    except PackageNotFoundError:
        fail("torch_spyre is not installed in this environment")
    if not text:
        fail("torch_spyre was not installed from git or a local tree; set TORCH_DEVICE_ROOT")
    info = json.loads(text)
    url = info.get("url", "")

    if url.startswith("file://"):
        root = Path(url.removeprefix("file://"))
        if not (root / SENTINEL).is_file():
            fail(f"torch_spyre is installed from {root}, which has no {SENTINEL}")
        return root

    commit = info.get("vcs_info", {}).get("commit_id")
    if not commit:
        fail(f"cannot tell which torch-spyre commit {url} is; set TORCH_DEVICE_ROOT")
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    dest = cache / "spyre-inference" / "torch-spyre-src" / commit
    if not (dest / SENTINEL).is_file():
        checkout(url, commit, dest)
    return dest


def combine(reports: list[Path], out: Path) -> None:
    root = ET.Element("testsuites")
    for report in reports:
        tree = ET.parse(report).getroot()
        root.extend([tree] if tree.tag == "testsuite" else list(tree))
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="*", type=Path, help="default: tests/model_ops/*.yaml")
    parser.add_argument("--junit-xml", type=Path, help="combined JUnit report")
    parser.add_argument("--pytest-args", default="-v", help="passed to run_test.sh")
    args = parser.parse_args()

    configs = args.configs or sorted(Path("tests/model_ops").glob("*.yaml"))
    if not configs:
        print("run_model_ops: no model-ops YAML in tests/model_ops/, nothing to run")
        return
    root = torch_spyre_source()
    print(f"run_model_ops: torch-spyre tests from {root}", flush=True)
    env = {**os.environ, "TORCH_DEVICE_ROOT": str(root)}

    rc = 0
    reports = []
    with tempfile.TemporaryDirectory() as tmp:
        for config in configs:
            report = Path(tmp) / f"{config.stem}.xml"
            cmd = [
                "bash",
                str(root / "tests" / "run_test.sh"),
                str(config.resolve()),
                *args.pytest_args.split(),
                f"--junit-xml={report}",
            ]
            print(f"run_model_ops: {config}", flush=True)
            if subprocess.run(cmd, env=env).returncode:
                rc = 1
            if report.is_file():
                reports.append(report)
        if args.junit_xml and reports:
            combine(reports, args.junit_xml)
    sys.exit(rc)


if __name__ == "__main__":
    main()
