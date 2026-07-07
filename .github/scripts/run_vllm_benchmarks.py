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

"""Run vLLM benchmarks from JSON config files.

Reads benchmark configs from the specified directory, builds the appropriate
`vllm bench latency` / `vllm bench throughput` commands, and executes them.
"""

import logging
import os
import re
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Valid environment variable name pattern
ENV_VAR_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")


def parse_args():
    parser = ArgumentParser(description="Run vLLM benchmarks from JSON configs")
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="vllm-benchmarks/benchmarks/spyre",
        help="directory containing benchmark JSON config files",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmark-results",
        help="directory to write benchmark result JSON files",
    )
    parser.add_argument(
        "--spyre-devices",
        type=str,
        default=os.environ.get("SPYRE_DEVICES", "0"),
        help="SPYRE_DEVICES value (default: from env or '0')",
    )
    parser.add_argument(
        "--aiu-world-size",
        type=str,
        default=os.environ.get("AIU_WORLD_SIZE", "1"),
        help="AIU_WORLD_SIZE value (default: from env or '1')",
    )
    return parser.parse_args()


def build_command_args(parameters: dict) -> list[str]:
    """Convert a parameters dict to CLI arguments for vllm bench."""
    args = []
    for key, value in parameters.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            args.append(flag)
        elif value is False:
            continue
        else:
            args.append(flag)
            args.append(str(value))
    return args


def build_env_vars(env_config: dict) -> dict[str, str]:
    """Validate and return environment variables from config."""
    env_vars = {}
    for key, value in env_config.items():
        if ENV_VAR_PATTERN.match(key):
            env_vars[key] = str(value)
        else:
            log.warning("Skipping invalid env var name: %s", key)
    return env_vars


def run_benchmark(
    bench_type: str,
    test_name: str,
    parameters: dict,
    env_config: dict,
    results_dir: Path,
    spyre_devices: str,
    aiu_world_size: str,
) -> bool:
    """Run a single vllm bench command. Returns True on success."""
    cmd = ["vllm", "bench", bench_type]
    cmd.extend(build_command_args(parameters))
    cmd.extend(["--output-json", str(results_dir / f"{test_name}.json")])

    # Build environment
    env = os.environ.copy()
    env["SPYRE_DEVICES"] = spyre_devices
    env["AIU_WORLD_SIZE"] = aiu_world_size
    env.update(build_env_vars(env_config))

    log.info("=== Running %s test: %s ===", bench_type, test_name)
    log.info("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        log.error("Test %s failed with exit code %d", test_name, result.returncode)
        return False
    return True


def run_benchmarks_from_file(
    config_file: Path,
    bench_type: str,
    results_dir: Path,
    spyre_devices: str,
    aiu_world_size: str,
) -> tuple[int, int]:
    """Run all benchmarks from a config file. Returns (passed, failed) counts."""
    if not config_file.exists():
        log.info("No %s config found, skipping", config_file.name)
        return 0, 0

    with open(config_file) as f:
        configs = yaml.safe_load(f)

    if not isinstance(configs, list):
        log.error("%s is not a YAML list", config_file)
        return 0, 1

    passed = 0
    failed = 0
    for config in configs:
        test_name = config.get("test_name", "unknown")
        parameters = config.get("parameters", {})
        env_config = config.get("environment_variables", {})

        success = run_benchmark(
            bench_type=bench_type,
            test_name=test_name,
            parameters=parameters,
            env_config=env_config,
            results_dir=results_dir,
            spyre_devices=spyre_devices,
            aiu_world_size=aiu_world_size,
        )
        if success:
            passed += 1
        else:
            failed += 1

    return passed, failed


def main():
    args = parse_args()
    configs_dir = Path(args.configs_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    total_passed = 0
    total_failed = 0

    # Run latency benchmarks
    passed, failed = run_benchmarks_from_file(
        config_file=configs_dir / "latency-tests.yaml",
        bench_type="latency",
        results_dir=results_dir,
        spyre_devices=args.spyre_devices,
        aiu_world_size=args.aiu_world_size,
    )
    total_passed += passed
    total_failed += failed

    # Run throughput benchmarks
    passed, failed = run_benchmarks_from_file(
        config_file=configs_dir / "throughput-tests.yaml",
        bench_type="throughput",
        results_dir=results_dir,
        spyre_devices=args.spyre_devices,
        aiu_world_size=args.aiu_world_size,
    )
    total_passed += passed
    total_failed += failed

    # Summary
    log.info("=== Benchmark Summary ===")
    log.info("Passed: %d, Failed: %d", total_passed, total_failed)

    result_files = list(results_dir.glob("*.json"))
    log.info("Result files: %d", len(result_files))
    for f in result_files:
        log.info("  %s", f.name)

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
