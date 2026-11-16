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

"""Gather an engine run's per-case vLLM JSON into one directory for the ingest step.

Copies what exists, then fails naming every `ok` case that left no result or no `.pytorch.json`
sibling (the ingest rejects latency and throughput results without it), so a half-written leg
is never ingested silently.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

_JSON = ".json"
_SIBLING = ".pytorch.json"


def collect(results_dir: Path, native_dir: Path) -> list[str]:
    """Copy every ok case's raw/*.json to native_dir; return the problems found."""
    run = json.loads((results_dir / "run.json").read_text())
    native_dir.mkdir(parents=True, exist_ok=True)
    problems = []
    for case in run.get("cases", []):
        if case.get("status") != "ok":
            continue
        raw = results_dir / case["dir"] / "raw"
        files = sorted(raw.glob("*.json")) if raw.is_dir() else []
        mains = {f.name[: -len(_JSON)].lower(): f for f in files if not f.name.endswith(_SIBLING)}
        main = mains.get(case["name"].lower())
        if main is None:
            problems.append(f"{case['name']}: no result JSON")
            continue
        if not main.with_name(main.name[: -len(_JSON)] + _SIBLING).exists():
            problems.append(f"{case['name']}: no .pytorch.json sibling")
        for f in files:
            target = native_dir / f.name
            if target.exists():
                problems.append(f"{case['name']}: {f.name} collides with another case")
                continue
            shutil.copy2(f, target)
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results/bench")
    ap.add_argument("--native-dir", default="results/bench/native")
    args = ap.parse_args(argv)
    problems = collect(Path(args.results_dir), Path(args.native_dir))
    for p in problems:
        print(f"collect: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
