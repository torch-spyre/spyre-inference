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

"""Run the model-ops configs in tests/model_ops/ with torch-spyre's OOT test runner.

Each config lists the torch ops a model executes through this plugin;
torch-spyre's tests/models/test_model_ops_v2.py runs every case on Spyre and
records its capability verdict in the JUnit report, which the v2 ingest reads.

That test, its conftest and the OOT framework are not in the torch_spyre wheel,
so they come from a torch-spyre source tree: TORCH_DEVICE_ROOT when set, else the
tree baked into the runner image.

One pytest session PER config, deliberately: the runner keys its test ids on the
op signature, so a second config's case for an op another config already ran is
dropped as a duplicate within a single session. The per-config reports are merged
into --junit-xml.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NoReturn

SENTINEL = "tests/models/test_model_ops_v2.py"
# The dev images bake the whole torch-spyre checkout here (tests/ included).
DEFAULT_ROOTS = ("/home/senuser/torch-spyre",)


def fail(msg: str) -> NoReturn:
    sys.exit(f"run_model_ops: {msg}")


def torch_spyre_source() -> Path:
    """The torch-spyre tree holding the OOT runner."""
    override = os.environ.get("TORCH_DEVICE_ROOT")
    candidates = [Path(override)] if override else [Path(p) for p in DEFAULT_ROOTS]
    for root in candidates:
        if (root / SENTINEL).is_file():
            return root
    looked = ", ".join(str(c) for c in candidates)
    fail(
        f"no torch-spyre source tree with {SENTINEL} (looked in: {looked}). "
        "Set TORCH_DEVICE_ROOT to a torch-spyre checkout, or run on an image that bakes one."
    )


def merge_reports(parts: list[Path], out: Path) -> None:
    """Combine per-config JUnit files into one, summing the testsuite counters."""
    root = ET.Element("testsuites")
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    time = 0.0
    for part in parts:
        if not part.is_file():
            continue
        tree = ET.parse(part)
        node = tree.getroot()
        suites = [node] if node.tag == "testsuite" else list(node)
        for suite in suites:
            for key in totals:
                totals[key] += int(suite.get(key, 0) or 0)
            time += float(suite.get("time", 0) or 0)
            root.append(suite)
    for key, value in totals.items():
        root.set(key, str(value))
    root.set("time", f"{time:.3f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config-dir",
        default="tests/model_ops",
        help="directory of model-ops configs (default: tests/model_ops)",
    )
    ap.add_argument(
        "--config",
        action="append",
        default=[],
        help="run only this config (repeatable); default is every *.yaml/*.yml in --config-dir",
    )
    ap.add_argument(
        "--junit-xml", type=Path, default=None, help="write a combined JUnit report here"
    )
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="exit 0 when --config-dir holds no configs, instead of failing",
    )
    args, pytest_args = ap.parse_known_args()

    config_dir = Path(args.config_dir)
    if args.config:
        configs = [Path(c) if os.sep in c else config_dir / c for c in args.config]
        missing = [str(c) for c in configs if not c.is_file()]
        if missing:
            fail(f"config not found: {', '.join(missing)}")
    else:
        configs = sorted(
            p for p in config_dir.glob("*") if p.suffix in (".yaml", ".yml") and p.is_file()
        )

    if not configs:
        # A suite that declares itself and then runs nothing reports PASS while testing
        # nothing, so an empty config dir is an error unless the caller opts in.
        if args.allow_empty:
            print(f"run_model_ops: no configs in {config_dir}, nothing to run (--allow-empty)")
            return 0
        fail(f"no configs in {config_dir}. Pass --allow-empty if running nothing is intended.")

    source = torch_spyre_source()
    tests_dir = source / "tests"
    print(f"run_model_ops: {len(configs)} config(s), runner from {source}")

    rc = 0
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for config in configs:
            part = Path(tmp) / f"{config.stem}.xml"
            env = {
                **os.environ,
                # How the OOT framework receives its config; the whole point of
                # running an out-of-tree config through an in-image runner.
                "PYTORCH_TEST_CONFIG": str(config.resolve()),
                # A config may reference the runner tree as ${TORCH_DEVICE_ROOT}
                # in its files[].path, so export it even when it was discovered
                # rather than passed in.
                "TORCH_DEVICE_ROOT": str(source),
                "PYTHONPATH": os.pathsep.join(
                    p for p in (str(tests_dir), os.environ.get("PYTHONPATH", "")) if p
                ),
            }
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(tests_dir / "models" / SENTINEL.split("/")[-1]),
                "-v",
                f"--junit-xml={part}",
                *pytest_args,
            ]
            print(f"\n=== {config.name} ===", flush=True)
            # cwd=tests_dir so the runner resolves its conftest and helper imports.
            result = subprocess.run(cmd, cwd=tests_dir, env=env)
            if result.returncode != 0:
                rc = result.returncode
            parts.append(part)

        if args.junit_xml:
            merge_reports(parts, args.junit_xml)
            print(f"\nrun_model_ops: combined report -> {args.junit_xml}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
