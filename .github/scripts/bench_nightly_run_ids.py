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

"""The stored run ids a nightly leg's verdict should read, one per line.

An engine run that ingested itself records them in run.json; otherwise the rows were written by the
ingest action, which derives the run id from the GitHub run id.
"""

import argparse
import json
import sys
import uuid
from pathlib import Path


def stored_run_ids(
    results_dir: Path, gha_run_id: str = "", arch: str = "x86_64", orchestrator_run_id: str = ""
) -> list[str]:
    """The orchestrator uuid (Jenkins, verbatim), else run.json's, else the GitHub derivation."""
    if orchestrator_run_id:
        return [str(uuid.UUID(orchestrator_run_id))]
    path = results_dir / "run.json"
    if path.exists():
        ids = json.loads(path.read_text()).get("stored_run_ids") or []
        if ids:
            return list(dict.fromkeys(ids))
    from spyre_clickhouse_ingest import run_id_of

    if not gha_run_id:
        return []
    return [str(run_id_of("gha", gha_run_id, arch, "perf"))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results/bench")
    ap.add_argument("--gha-run-id", default="")
    ap.add_argument(
        "--run-id", default="", help="the orchestrator uuid the ingest was given (Jenkins)"
    )
    ap.add_argument("--arch", default="x86_64")
    args = ap.parse_args(argv)
    print(
        "\n".join(stored_run_ids(Path(args.results_dir), args.gha_run_id, args.arch, args.run_id))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
