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

"""Generate a simplified vLLM benchmark CI matrix for Spyre runners."""

import importlib.util
import json
import logging
import os
import sys
from argparse import Action, ArgumentParser, Namespace
from logging import warning
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)

# The scripts are not a package, so load the runner by path: the matrix uses
# the same suite selection as the run.
_RUNNER_SCRIPT = Path(__file__).resolve().parent / "run_vllm_benchmarks.py"
_runner_spec = importlib.util.spec_from_file_location("run_vllm_benchmarks", _RUNNER_SCRIPT)
assert _runner_spec and _runner_spec.loader
runner = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(runner)


class ValidateDir(Action):
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        if os.path.isdir(values):
            setattr(namespace, self.dest, values)
            return

        parser.error(f"{values} is not a valid directory")


def parse_args() -> Any:
    parser = ArgumentParser("Generate vLLM benchmark CI matrix for Spyre")

    parser.add_argument(
        "--benchmark-configs-dir",
        type=str,
        default="vllm-benchmarks/benchmarks",
        action=ValidateDir,
        help="the directory searched for *-runs.yaml benchmark configs",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=runner.DEFAULT_SUITE,
        choices=runner.SUITES,
        help=f"suite whose models make up the matrix (default: {runner.DEFAULT_SUITE})",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="comma-separated list of models to benchmark (empty = all)",
    )

    return parser.parse_args()


def set_output(name: str, val: Any) -> None:
    github_output = os.getenv("GITHUB_OUTPUT")

    if not github_output:
        print(f"::set-output name={name}::{val}")
        return

    with open(github_output, "a") as env:
        env.write(f"{name}={val}\n")


def _suite_models(benchmark_configs_dir: str, suite: str) -> list[str]:
    """Models with a config in `suite`, in file order."""
    selected = runner.collect_configs(
        Path(benchmark_configs_dir), suite, set(runner.VALID_BENCH_TYPES)
    )
    models: list[str] = []
    for configs in selected.values():
        for config in configs:
            model = runner._config_model(config)
            if model is None:
                warning("Model name not set in %s, skipping...", config.get("test_name"))
                continue
            if model.lower() not in models:
                models.append(model.lower())
    return models


def generate_benchmark_matrix(
    benchmark_configs_dir: str, models: list[str], suite: str = runner.DEFAULT_SUITE
) -> dict[str, Any]:
    """One matrix entry per model, with the suite it runs in.

    With no `models`, every model of `suite`. Otherwise each model runs in the
    first suite, from `suite` on in SUITES order, that has a config for it.
    """
    if not models:
        return {
            "include": [
                {"models": m, "suite": suite} for m in _suite_models(benchmark_configs_dir, suite)
            ]
        }

    candidates = runner.SUITES[runner.SUITES.index(suite) :]
    suite_models = {s: _suite_models(benchmark_configs_dir, s) for s in candidates}
    include = []
    for model in dict.fromkeys(models):
        model_suite = next((s for s in candidates if model in suite_models[s]), None)
        if model_suite is None:
            logging.error("%s has no config in suites %s", model, list(candidates))
            sys.exit(2)
        include.append({"models": model, "suite": model_suite})
    return {"include": include}


def main() -> None:
    args = parse_args()
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    benchmark_matrix = generate_benchmark_matrix(
        args.benchmark_configs_dir,
        models,
        args.suite,
    )
    print(json.dumps(benchmark_matrix))
    set_output("benchmark_matrix", json.dumps(benchmark_matrix))


if __name__ == "__main__":
    main()
