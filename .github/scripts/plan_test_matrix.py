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

"""Write the legs _test_matrix.yaml schedules for one test type to GITHUB_OUTPUT."""

import json
import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
TEST_MATRIX = REPO / ".github" / "test-matrix.yaml"
BENCH_CONFIGS = REPO / "vllm-benchmarks" / "benchmarks" / "spyre"

# The card pools runners come in; a leg takes the smallest that fits its TP.
CARD_POOLS = (1, 2, 4)


def _runs_on(cards: int) -> list[str]:
    return ["x86_64", f"spyre_pf_x{cards}", "linux"]


def makefile_shard_counts() -> dict[str, int]:
    out = subprocess.run(
        ["make", "-s", "-C", str(REPO), "print-shards"], capture_output=True, text=True, check=True
    ).stdout
    return {target: int(n) for target, n in (line.split() for line in out.splitlines())}


def load_test_configs(path: Path = TEST_MATRIX) -> list[dict]:
    """The matrix as per-job legs: each sharded suite becomes one leg per Makefile shard."""
    counts = makefile_shard_counts()
    legs = []
    for suite in yaml.safe_load(path.read_text()):
        leg = {
            "cfg": suite["cfg"],
            "test_types": suite["test_types"],
            "runs_on": _runs_on(suite["cards"]),
            "image_label": "image_torch_spyre",
            "test_target": suite["test_target"],
        }
        if not suite.get("sharded"):
            legs.append(leg)
            continue
        n = counts.get(leg["test_target"])
        if not n:
            raise SystemExit(f"::error::`make print-shards` has no count for {leg['test_target']}")
        legs.extend(
            {
                **leg,
                "cfg": f"{leg['cfg']} (shard {i + 1}/{n})",
                "test_target": f"{leg['test_target']}-{i}",
            }
            for i in range(n)
        )
    return legs


def select_test_configs(configs: list[dict], test_type: str) -> list[dict]:
    return [c for c in configs if test_type in c["test_types"].split()]


def _bench_tests(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text())
    if isinstance(raw, list):
        return raw
    defaults = raw.get("defaults") or {}
    tests = []
    for test in raw.get("tests") or []:
        merged = dict(test)
        for section in ("parameters", "server_parameters"):
            merged[section] = {**(defaults.get(section) or {}), **(test.get(section) or {})}
        tests.append(merged)
    return tests


def _model_and_tp(test: dict) -> tuple[str | None, int]:
    model = None
    tp = 1
    for section in ("parameters", "server_parameters"):
        params = test.get(section) or {}
        model = model or params.get("model")
        for key in ("tensor-parallel-size", "tensor_parallel_size"):
            if key in params:
                tp = max(tp, int(params[key]))
    return (model.lower() if model else None), tp


def plan_perf(configs_dir: Path = BENCH_CONFIGS, models: set[str] | None = None) -> list[dict]:
    tp_by_model: dict[str, int] = {}
    # *-tests-<suite>.yaml files are for manual runs.
    for path in sorted(configs_dir.glob("*-tests.yaml")):
        for test in _bench_tests(path):
            model, tp = _model_and_tp(test)
            if model is None or (models and model not in models):
                continue
            tp_by_model[model] = max(tp_by_model.get(model, 1), tp)
    legs = []
    for model, tp in tp_by_model.items():
        cards = next(n for n in CARD_POOLS if n >= tp)
        legs.append({"model": model, "runs_on": _runs_on(cards)})
    return legs


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--test-type", required=True)
    parser.add_argument("--models", default="", help="perf only: comma-separated subset")
    parser.add_argument("--no-changes", action="store_true")
    args = parser.parse_args()

    if args.no_changes:
        print("No source files changed; nothing to run.")
        tests, perf = [], []
    elif args.test_type == "perf":
        models = {m.strip().lower() for m in args.models.split(",") if m.strip()}
        tests: list[dict] = []
        perf = plan_perf(models=models)
        if not perf:
            raise SystemExit(f"::error::no benchmark config matches models {sorted(models)}")
    else:
        tests = select_test_configs(load_test_configs(), args.test_type)
        perf = []

    print(f"test legs: {[t['cfg'] for t in tests]}")
    print(f"perf legs: {[p['model'] for p in perf]}")
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"test_matrix={json.dumps({'include': tests})}\n")
        f.write(f"perf_matrix={json.dumps({'include': perf})}\n")
        f.write(f"has_tests={'true' if tests else 'false'}\n")
        f.write(f"has_perf={'true' if perf else 'false'}\n")


if __name__ == "__main__":
    main()
