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
Merge the per-leg {nodeid: seconds} durations files each test job wrote (via the
plugin's pytest_sessionfinish, SPYRE_TEST_DURATIONS_OUT) into the single
`test-durations.json` a later run pins and weights its shard partition by (see
_load_durations / _apply_shard in the spyre_testing_plugin).

Each sharded leg records only its own slice and no test runs twice in a run, so
the union across legs is the whole suite -- a plain dict merge. When a nodeid
appears in more than one leg (a pod-level retry re-ran it) the larger time wins,
biasing the next run's partition toward the worst observed case so a shard stays
under budget. Keys are exact nodeids, so no reconstruction is involved.

Usage:
    python3 merge_test_durations.py durations/*.json --out test-durations.json
"""

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Per-leg durations JSON files")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    merged: dict[str, float] = {}
    for path in args.inputs:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            print(f"Skipping unreadable {path}: {e}")
            continue
        for nodeid, seconds in data.items():
            try:
                seconds = float(seconds)
            except (TypeError, ValueError):
                continue
            if seconds >= 0 and seconds > merged.get(nodeid, -1.0):
                merged[nodeid] = seconds

    with open(args.out, "w") as f:
        json.dump(merged, f, indent=0, sort_keys=True)
    print(f"Merged {len(merged)} test durations into {args.out}")


if __name__ == "__main__":
    main()
