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
Validate vLLM benchmark results.

Checks that benchmark result JSON files are non-empty and contain non-zero values.
"""

import glob
import json
import logging
import os
import sys
from argparse import Action, ArgumentParser, Namespace
from json.decoder import JSONDecodeError
from logging import info, warning
from typing import Any

logging.basicConfig(level=logging.INFO)


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
    parser = ArgumentParser("Check benchmark results")

    parser.add_argument(
        "--benchmark-results",
        type=str,
        required=True,
        action=ValidateDir,
        help="the directory with the benchmark results",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="exit with code 1 when all benchmark results are zeroed",
    )

    return parser.parse_args()


def read_benchmark_results(filepath: str) -> list[dict[str, Any]]:
    results = []
    with open(filepath) as f:
        try:
            r = json.load(f)
            if isinstance(r, dict):
                results.append(r)
            elif isinstance(r, list):
                results = r

        except JSONDecodeError:
            f.seek(0)

            # Try JSONEachRow format
            for line in f:
                try:
                    r = json.loads(line)
                    if isinstance(r, dict):
                        results.append(r)
                    elif isinstance(r, list):
                        results.extend(r)
                    else:
                        warning("Not a JSON dict or list %s, skipping", line)
                        continue

                except JSONDecodeError:
                    warning("Invalid JSON %s, skipping", line)

    return results


def check_benchmark_results(benchmark_results_dir: str, strict: bool = False) -> dict[str, list]:
    all_results = {}

    for file in glob.glob(f"{benchmark_results_dir}/*.json"):
        filename = os.path.basename(file)
        results = read_benchmark_results(file)

        if not results or not isinstance(results, list):
            warning("%s is empty", file)
            continue

        values = []
        for r in results:
            if (
                "benchmark" not in r
                or "metric" not in r
                or "benchmark_values" not in r["metric"]
                or type(r["metric"]["benchmark_values"]) is not list
            ):
                continue
            values.extend(r["metric"]["benchmark_values"])

        if not values:
            warning("Found no PyTorch benchmark results in %s", file)
            continue

        if all(v == 0 for v in values):
            warning("All PyTorch benchmark results in %s are zeroed", file)
            if strict:
                sys.exit(1)
            continue

        info("Loading benchmark results from %s", file)
        all_results[filename] = results

    return all_results


def main() -> None:
    args = parse_args()

    if not check_benchmark_results(args.benchmark_results, strict=args.strict):
        warning("Found no benchmark results in %s", args.benchmark_results)
        sys.exit(1)


if __name__ == "__main__":
    main()
