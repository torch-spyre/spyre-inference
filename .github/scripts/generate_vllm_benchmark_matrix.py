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


def generate_benchmark_matrix(
    benchmark_configs_dir: str, models: list[str], suite: str = runner.DEFAULT_SUITE
) -> dict[str, Any]:
    """Build a matrix with one entry per model that has a config in `suite`."""
    benchmark_matrix: dict[str, Any] = {
        "include": [],
    }

    selected_models = []

    selected = runner.collect_configs(
        Path(benchmark_configs_dir), suite, set(runner.VALID_BENCH_TYPES)
    )
    for configs in selected.values():
        for config in configs:
            model = runner._config_model(config)
            if model is None:
                warning("Model name not set in %s, skipping...", config.get("test_name"))
                continue
            model = model.lower()

            # Dedup
            if model in selected_models:
                continue
            # Filter to selected models if specified
            if models and model not in models:
                continue
            selected_models.append(model)

            benchmark_matrix["include"].append(
                {
                    "models": model,
                }
            )

    return benchmark_matrix


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
