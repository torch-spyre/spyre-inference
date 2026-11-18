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

"""The nightly bench route's cases: the `*-tests-nightly.yaml` entries of one model and TP leg.

Prints the lower-cased test names, comma separated, which are the case names in benchmarks/decoder/.
"""

import argparse
import sys
from pathlib import Path

import yaml

KINDS = ("latency", "throughput")


def load_cases(configs_dir: Path, suite: str = "nightly") -> list[dict]:
    """Every entry of `<kind>-tests-<suite>.yaml` with its model and tensor-parallel size."""
    cases = []
    for kind in KINDS:
        path = configs_dir / f"{kind}-tests-{suite}.yaml"
        for entry in yaml.safe_load(path.read_text()) or []:
            params = entry.get("parameters") or {}
            cases.append(
                {
                    "kind": kind,
                    "test_name": entry["test_name"],
                    "case": entry["test_name"].lower(),
                    "model": params.get("model", ""),
                    "tp": int(params.get("tensor_parallel_size", 1)),
                }
            )
    return cases


def leg_cases(cases: list[dict], model: str, tp: int) -> list[str]:
    return [c["case"] for c in cases if c["model"].lower() == model.lower() and c["tp"] == tp]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs-dir", default="vllm-benchmarks/benchmarks/spyre")
    ap.add_argument("--suite", default="nightly")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, required=True)
    args = ap.parse_args(argv)
    names = leg_cases(load_cases(Path(args.configs_dir), args.suite), args.model, args.tp)
    if not names:
        print(f"no {args.suite} case for {args.model} tp{args.tp}", file=sys.stderr)
        return 2
    print(",".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
