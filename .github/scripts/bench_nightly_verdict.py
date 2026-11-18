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

"""Read the verdict of a nightly leg and fail loudly when there is none.

No stored run id, a verdict command that fails, or zero verdict rows for the ids is
`no verdict: <reason>` (exit 1, also in the step summary). Only words and counts are printed.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def _run_ids():
    spec = importlib.util.spec_from_file_location(
        "bench_nightly_run_ids", Path(__file__).with_name("bench_nightly_run_ids.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fail(reason: str) -> int:
    line = f"no verdict: {reason}"
    print(f"::error::{line}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(f"\n{line}\n")
    return 1


def verdict(bench: str, results_dir: Path, gha_run_id: str = "", run_id: str = "") -> int:
    try:
        ids = _run_ids().stored_run_ids(results_dir, gha_run_id, orchestrator_run_id=run_id)
    except Exception as exc:  # noqa: BLE001 - any failure to derive an id is "no verdict"
        return fail(f"could not determine the stored run id ({type(exc).__name__})")
    if not ids:
        return fail("no stored run id")
    total = 0
    for run_id in ids:
        proc = subprocess.run(
            [bench, "verdict", "--run-id", run_id, "--store", "clickhouse", "--quiet", "--json"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return fail(f"verdict failed for run {run_id} (exit {proc.returncode})")
        try:
            summary = json.loads(proc.stdout)["summary"]
        except (ValueError, KeyError):
            return fail(f"unreadable verdict output for run {run_id}")
        total += int(summary.get("total", 0))
        print(f"run {run_id}: " + ", ".join(f"{k} {v}" for k, v in summary.items() if k != "worst"))
    if total == 0:
        return fail("zero verdict rows for the stored run id(s): rows not landed or wrong id")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="spyre-bench")
    ap.add_argument("--results-dir", default="results/bench")
    ap.add_argument("--gha-run-id", default="")
    ap.add_argument(
        "--run-id", default="", help="the orchestrator uuid the ingest was given (Jenkins)"
    )
    args = ap.parse_args(argv)
    return verdict(args.bench, Path(args.results_dir), args.gha_run_id, args.run_id)


if __name__ == "__main__":
    sys.exit(main())
