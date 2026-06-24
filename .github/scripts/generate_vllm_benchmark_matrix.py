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

import glob
import json
import logging
import os
from argparse import Action, ArgumentParser, Namespace
from logging import warning

import yaml
from typing import Any

logging.basicConfig(level=logging.INFO)

# All the different names vLLM uses to refer to their benchmark configs
VLLM_BENCHMARK_CONFIGS_PARAMETER = set(
    [
        "parameters",
        "server_parameters",
        "common_parameters",
    ]
)


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
        help="the directory containing vLLM benchmark configs",
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


def generate_benchmark_matrix(benchmark_configs_dir: str, models: list[str]) -> dict[str, Any]:
    """
    Parse serving config JSON files in the spyre benchmark configs directory
    to build a matrix of models to benchmark.
    """
    benchmark_matrix: dict[str, Any] = {
        "include": [],
    }

    selected_models = []

    for file in glob.glob(f"{benchmark_configs_dir}/spyre/*.yaml"):
        with open(file) as f:
            try:
                configs = yaml.safe_load(f)
            except yaml.YAMLError as e:
                warning("Failed to load %s: %s", file, e)
                continue

        for config in configs:
            param = list(VLLM_BENCHMARK_CONFIGS_PARAMETER & set(config.keys()))
            if not param:
                warning("No recognized parameter key in config: %s", config)
                continue

            benchmark_config = config[param[0]]
            if "model" not in benchmark_config:
                warning("Model name not set in %s, skipping...", benchmark_config)
                continue
            model = benchmark_config["model"].lower()

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
    )
    print(json.dumps(benchmark_matrix))
    set_output("benchmark_matrix", json.dumps(benchmark_matrix))


if __name__ == "__main__":
    main()
