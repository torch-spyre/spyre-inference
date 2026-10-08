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
`vllm bench latency` / `vllm bench throughput` / `vllm bench serve` commands,
and executes them.
"""

import contextlib
import json
import logging
import os
import platform
import re
import shlex
import signal
import string
import subprocess
import sys
import time
import urllib.request
from argparse import ArgumentParser
from pathlib import Path

import yaml
from resource_sampler import ContainerSampler, write_resource_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Valid environment variable name pattern
ENV_VAR_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")

# Fallbacks for the dataset env vars the configs reference, matching the layout
# on the Spyre benchmark hosts.
DATASET_PATH_DEFAULTS = {
    "SPYRE_AIOPS_DATASET": (
        "/models/online_benchmarking_data_reordered/"
        "aiops_results_2025.11.03_e2ee1b0_correct_order.jsonl"
    ),
    "SPYRE_CICS_DATASET": (
        "/models/online_benchmarking_data_reordered/"
        "cics_results_2025.11.03_e2ee1b0_correct_order.jsonl"
    ),
}


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
        help="fallback SPYRE_DEVICES value for a config that sets no "
        "tensor-parallel size (default: from env or '0'); a config that sets "
        "one derives its own devices and ignores this",
    )
    parser.add_argument(
        "--aiu-world-size",
        type=str,
        default=os.environ.get("AIU_WORLD_SIZE", "1"),
        help="fallback AIU_WORLD_SIZE value, overridden the same way as "
        "--spyre-devices (default: from env or '1')",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="comma-separated model names to run (empty = all); matched "
        "case-insensitively against each config's model",
    )
    parser.add_argument(
        "--tps",
        type=str,
        default="",
        help="comma-separated tensor-parallel sizes to run (empty = all); "
        "matched against each config's tensor-parallel-size",
    )
    parser.add_argument(
        "--bench-types",
        type=str,
        default="",
        help="comma-separated subset of latency,throughput,serve to run (empty = all)",
    )
    return parser.parse_args()


def _config_model(config: dict) -> str | None:
    """Model name for a benchmark config (serve uses server_parameters)."""
    for key in ("parameters", "server_parameters"):
        model = config.get(key, {}).get("model")
        if model:
            return model
    return None


def _config_tp(config: dict) -> int | None:
    """Tensor-parallel size for a benchmark config, or None if it sets none.

    Both spellings are accepted: serve passes `tensor-parallel-size` to the
    server CLI, latency/throughput `tensor_parallel_size`.
    """
    for key in ("parameters", "server_parameters"):
        parameters = config.get(key, {})
        for tp_key in ("tensor-parallel-size", "tensor_parallel_size"):
            if tp_key in parameters:
                return int(parameters[tp_key])
    return None


def _resolve_dataset_path(config: dict) -> None:
    """Expand environment variables in a config's `dataset-path`, in place."""
    parameters = config.get("parameters")
    if not parameters:
        return
    path = parameters.get("dataset-path")
    if not path:
        return
    env = {**DATASET_PATH_DEFAULTS, **os.environ}
    parameters["dataset-path"] = string.Template(str(path)).safe_substitute(env)


def _missing_dataset(config: dict) -> str | None:
    """Dataset path a config needs but which is absent on this host, if any."""
    path = config.get("parameters", {}).get("dataset-path")
    if path and not Path(path).exists():
        return str(path)
    return None


def _select_configs(configs: list, models: set[str], tps: set[int]) -> list:
    """Keep configs whose model and TP are selected (empty selects all).

    A selected config whose dataset is absent is fatal, not skipped: skipping
    would let a serve-only model job report success while measuring nothing.
    """
    selected = []
    missing_datasets = []
    for config in configs:
        model = _config_model(config)
        if models and not (model and model.lower() in models):
            log.info("Skipping %s (model %s not selected)", config.get("test_name"), model)
            continue
        tp = _config_tp(config)
        if tp is None:
            log.error(
                "Config %s sets no tensor-parallel-size; add one to its parameters",
                config.get("test_name"),
            )
            sys.exit(2)
        if tps and tp not in tps:
            log.info("Skipping %s (tp %d not selected)", config.get("test_name"), tp)
            continue
        _resolve_dataset_path(config)
        missing = _missing_dataset(config)
        if missing:
            missing_datasets.append((config.get("test_name"), missing))
            continue
        selected.append(config)

    if missing_datasets:
        for test_name, path in missing_datasets:
            log.error("%s needs dataset %s, which is not present on this host", test_name, path)
        log.error(
            "Point SPYRE_AIOPS_DATASET / SPYRE_CICS_DATASET at this host's copies "
            "of the trace files, or mount them at the paths above."
        )
        sys.exit(2)
    return selected


# Spyre devices are handed out from 0, so TP n takes the first n.
def _spyre_devices_for_tp(tp: int) -> str:
    return ",".join(str(i) for i in range(tp))


def _merge_defaults(defaults: dict, config: dict) -> dict:
    """Merge one test config over the file's `defaults`.

    The parameter sections merge one level deep, so a test overrides individual
    keys instead of replacing a whole section.
    """
    merged = {**defaults, **config}
    for section in ("environment_variables", "server_parameters", "parameters"):
        section_defaults = defaults.get(section) or {}
        section_config = config.get(section) or {}
        if section_defaults or section_config:
            merged[section] = {**section_defaults, **section_config}
    return merged


def _derive_config(config: dict) -> None:
    """Fill in the config values that follow from others, in place.

    A config that sets one of them explicitly keeps its own value.
    `AIU_WORLD_SIZE` must agree with the device list: it is what the platform
    falls back to for its card count in subprocesses that re-import before
    torch_spyre is loaded.
    """
    env_config = config.setdefault("environment_variables", {})

    tp = _config_tp(config)
    if tp is not None:
        env_config.setdefault("SPYRE_DEVICES", _spyre_devices_for_tp(tp))
        env_config.setdefault("AIU_WORLD_SIZE", str(tp))

    health_timeout = config.get("server_health_timeout")
    if health_timeout is not None:
        env_config.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", str(health_timeout))

    server_model = config.get("server_parameters", {}).get("model")
    if server_model and "parameters" in config:
        config["parameters"].setdefault("model", server_model)


def _load_configs(config_file: Path) -> list | None:
    """Read a config file -- a bare list of tests, or a `defaults`/`tests`
    mapping whose tests are merged over `defaults`. None if malformed."""
    with open(config_file) as f:
        raw = yaml.safe_load(f)

    if isinstance(raw, dict):
        tests = raw.get("tests")
        if not isinstance(tests, list):
            log.error("%s has no `tests` list", config_file)
            return None
        defaults = raw.get("defaults") or {}
        configs = [_merge_defaults(defaults, config) for config in tests]
    elif isinstance(raw, list):
        configs = raw
    else:
        log.error("%s is not a YAML list or a defaults/tests mapping", config_file)
        return None

    for config in configs:
        _derive_config(config)
    return configs


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


def format_command(cmd: list[str], env_vars: dict[str, str]) -> str:
    """Render a command as a copy-pasteable shell line, prefixed by its env vars.

    Only the config's own variables are shown, not the inherited environment.
    """
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env_vars.items()))
    line = shlex.join(cmd)
    return f"{prefix} {line}" if prefix else line


def record_command(cmd: list[str], env_vars: dict[str, str], cmd_file: Path) -> str:
    """Write a command line to `cmd_file` and return it."""
    line = format_command(cmd, env_vars)
    try:
        cmd_file.write_text(line + "\n")
    except OSError as e:
        # Losing the record is not a test failure.
        log.warning("Could not write %s: %s", cmd_file.name, e)
    return line


# Invoke the vLLM CLI directly: the dynamo recompile-limit raise the benchmarks
# need is applied by the platform plugin at import (see
# spyre_inference/platform.py::_raise_dynamo_recompile_limits, torch-spyre #444).
# Via sys.executable rather than the `vllm` console script, so the CLI always
# runs in this interpreter's environment.
VLLM_CLI = [sys.executable, "-m", "vllm.entrypoints.cli.main"]


class _Noop:
    """No-op stand-in for ContainerSampler when cgroup metrics are unavailable."""

    def stop(self) -> None:
        pass

    def summary(self, phase: str = "") -> dict:
        return {}


def _make_sampler() -> ContainerSampler:
    """Construct and start a ContainerSampler, degrading to a no-op on OSError."""
    try:
        s = ContainerSampler()
        s.start()
        return s
    except OSError:
        log.warning("cgroup metrics unavailable; host resource sampling disabled")
        return _Noop()  # type: ignore[return-value]


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
    cmd = [*VLLM_CLI, "bench", bench_type]
    cmd.extend(build_command_args(parameters))
    cmd.extend(["--output-json", str(results_dir / f"{test_name}.json")])

    # The CLI device values are only a base: env_config carries the per-test
    # values from _derive_config and is applied last, so it wins.
    env = os.environ.copy()
    env["SPYRE_DEVICES"] = spyre_devices
    env["AIU_WORLD_SIZE"] = aiu_world_size
    config_env = build_env_vars(env_config)
    env.update(config_env)

    cmd_line = record_command(cmd, config_env, results_dir / f"{test_name}.cmd")

    log.info("=== Running %s test: %s ===", bench_type, test_name)
    log.info("Command: %s", cmd_line)

    log_file = results_dir / f"{test_name}.log"
    sampler = _make_sampler()
    with open(log_file, "w") as lf:
        lf.write(f"# {cmd_line}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.PIPE, text=True)
        _, stderr_output = proc.communicate()
    sampler.stop()
    write_resource_metrics(
        test_name, results_dir, sampler.summary(), model=parameters.get("model", "")
    )

    if proc.returncode != 0:
        log.error("Test %s failed with exit code %d", test_name, proc.returncode)
        if stderr_output:
            stderr_lines = stderr_output.strip().splitlines()[-50:]
            log.error("stderr tail:\n%s", "\n".join(stderr_lines))
        return False
    log.info("Test %s passed", test_name)
    return True


def run_benchmarks_from_file(
    config_file: Path,
    bench_type: str,
    results_dir: Path,
    spyre_devices: str,
    aiu_world_size: str,
    models: set[str],
    tps: set[int],
) -> tuple[int, int]:
    """Run all benchmarks from a config file. Returns (passed, failed) counts."""
    if not config_file.exists():
        log.info("No %s config found, skipping", config_file.name)
        return 0, 0

    configs = _load_configs(config_file)
    if configs is None:
        return 0, 1

    configs = _select_configs(configs, models, tps)

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


def run_serve_benchmark(
    test_name: str,
    server_parameters: dict,
    bench_parameters: dict,
    env_config: dict,
    results_dir: Path,
    spyre_devices: str,
    aiu_world_size: str,
    health_timeout: int = 180,
) -> bool:
    """Start vllm serve, wait for health, run bench serve, cleanup."""
    # As in run_benchmark: env_config is applied last and overrides the CLI
    # device values.
    env = os.environ.copy()
    env["SPYRE_DEVICES"] = spyre_devices
    env["AIU_WORLD_SIZE"] = aiu_world_size
    config_env = build_env_vars(env_config)
    env.update(config_env)

    # Build server command
    server_params = dict(server_parameters)
    model = server_params.pop("model")
    host = str(server_params.get("host", "127.0.0.1"))
    port = int(server_params.get("port", 8000))
    server_cmd = [*VLLM_CLI, "serve", model]
    server_cmd.extend(build_command_args(server_params))

    server_cmd_line = record_command(
        server_cmd, config_env, results_dir / f"{test_name}_server.cmd"
    )

    log.info("=== Starting vLLM server for serve test: %s ===", test_name)
    log.info("Server command: %s", server_cmd_line)

    def _kill_server(proc: subprocess.Popen) -> None:
        """Kill the server and its entire process group."""
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    server_log = results_dir / f"{test_name}_server.log"
    bench_proc = None
    bench_stderr = ""
    with open(server_log, "w") as server_lf:
        server_lf.write(f"# {server_cmd_line}\n")
        server_lf.flush()
        server_start_ts = time.monotonic()
        server_proc = subprocess.Popen(
            server_cmd,
            env=env,
            stdout=server_lf,
            stderr=server_lf,
            start_new_session=True,
        )

        compile_sampler = _make_sampler()

        try:
            # Timed from before the spawn: for large models this is mostly the
            # one-time warmup compile, which is worth comparing run to run.
            health_url = f"http://{host}:{port}/health"
            server_startup_sec = None
            deadline = server_start_ts + health_timeout
            while time.monotonic() < deadline:
                if server_proc.poll() is not None:
                    log.error("Server process died with exit code %d", server_proc.returncode)
                    compile_sampler.stop()
                    write_resource_metrics(
                        test_name,
                        results_dir,
                        compile_sampler.summary(phase="compile"),
                        model=model,
                    )
                    if server_log.exists():
                        log.error("Server log:\n%s", server_log.read_text())
                    return False
                try:
                    urllib.request.urlopen(health_url, timeout=2)
                    server_startup_sec = round(time.monotonic() - server_start_ts, 1)
                    log.info("Server ready after %.1fs", server_startup_sec)
                    break
                except Exception:
                    time.sleep(1)

            compile_sampler.stop()

            if server_startup_sec is None:
                log.error("Server did not become healthy within %ds", health_timeout)
                if server_log.exists():
                    log.error("Server log:\n%s", server_log.read_text())
                write_resource_metrics(
                    test_name, results_dir, compile_sampler.summary(phase="compile"), model=model
                )
                return False

            # Run bench serve
            bench_cmd = [*VLLM_CLI, "bench", "serve"]
            bench_cmd.extend(build_command_args(bench_parameters))
            bench_cmd.extend(
                [
                    # The trace prompts already carry their chat template;
                    # re-applying it would change token counts and the
                    # prefix-cache hit rate.
                    "--skip-chat-template",
                    "--save-result",
                    "--result-dir",
                    str(results_dir),
                    "--result-filename",
                    f"{test_name}.json",
                ]
            )

            bench_cmd_line = record_command(
                bench_cmd, config_env, results_dir / f"{test_name}_bench.cmd"
            )

            log.info("=== Running serve benchmark: %s ===", test_name)
            log.info("Bench command: %s", bench_cmd_line)

            bench_log = results_dir / f"{test_name}_bench.log"
            bench_sampler = _make_sampler()
            try:
                with open(bench_log, "w") as blf:
                    blf.write(f"# {bench_cmd_line}\n")
                    blf.flush()
                    bench_proc = subprocess.Popen(
                        bench_cmd, env=env, stdout=blf, stderr=subprocess.PIPE, text=True
                    )
                    _, bench_stderr = bench_proc.communicate()
            finally:
                bench_sampler.stop()

            combined = {
                **compile_sampler.summary(phase="compile"),
                **bench_sampler.summary(),
            }
            write_resource_metrics(test_name, results_dir, combined, model=model)

        finally:
            _kill_server(server_proc)

    if bench_proc.returncode != 0:
        log.error("Serve test %s failed with exit code %d", test_name, bench_proc.returncode)
        if bench_stderr:
            stderr_lines = bench_stderr.strip().splitlines()[-50:]
            log.error("stderr tail:\n%s", "\n".join(stderr_lines))
        return False

    # `vllm bench serve` only measures the request phase. A local artifact:
    # not in ingest_vllm_benchmarks.py's `_SERVE_METRICS`, so it does not
    # reach ClickHouse.
    result_file = results_dir / f"{test_name}.json"
    try:
        data = json.loads(result_file.read_text())
        data["server_startup_sec"] = server_startup_sec
        result_file.write_text(json.dumps(data, indent=2))
    except (OSError, ValueError) as e:
        # The benchmark itself succeeded.
        log.warning("Could not add server_startup_sec to %s: %s", result_file.name, e)

    log.info("Serve test %s passed (server startup %.1fs)", test_name, server_startup_sec)
    return True


def run_serve_benchmarks_from_file(
    config_file: Path,
    results_dir: Path,
    spyre_devices: str,
    aiu_world_size: str,
    models: set[str],
    tps: set[int],
) -> tuple[int, int]:
    """Run all serve benchmarks from a config file. Returns (passed, failed) counts."""
    if not config_file.exists():
        log.info("No %s config found, skipping", config_file.name)
        return 0, 0

    configs = _load_configs(config_file)
    if configs is None:
        return 0, 1

    configs = _select_configs(configs, models, tps)

    passed = 0
    failed = 0
    for config in configs:
        test_name = config.get("test_name", "unknown")
        server_parameters = config.get("server_parameters", {})
        bench_parameters = config.get("parameters", {})
        env_config = config.get("environment_variables", {})
        # Large models pay a one-time warmup compile before /health passes;
        # let a config bump this past the 180s default.
        health_timeout = int(config.get("server_health_timeout", 180))

        success = run_serve_benchmark(
            test_name=test_name,
            server_parameters=server_parameters,
            bench_parameters=bench_parameters,
            env_config=env_config,
            results_dir=results_dir,
            spyre_devices=spyre_devices,
            aiu_world_size=aiu_world_size,
            health_timeout=health_timeout,
        )
        if success:
            passed += 1
        else:
            failed += 1

    return passed, failed


VALID_BENCH_TYPES = ("latency", "throughput", "serve")


def main():
    # s390x: the protobuf upb C runtime SIGSEGVs (upb_DefPool_Free) in vLLM's
    # model-inspection child; force pure-Python protobuf. Set on os.environ so
    # the child, spawned by vLLM not us, inherits it.
    if platform.machine() == "s390x":
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    args = parse_args()
    configs_dir = Path(args.configs_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    models = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    try:
        tps = {int(t.strip()) for t in args.tps.split(",") if t.strip()}
    except ValueError:
        log.error("--tps takes comma-separated integers, got %r", args.tps)
        sys.exit(2)
    bench_types = {b.strip().lower() for b in args.bench_types.split(",") if b.strip()}
    unknown = bench_types - set(VALID_BENCH_TYPES)
    if unknown:
        log.error("Unknown bench types %s; valid: %s", sorted(unknown), VALID_BENCH_TYPES)
        sys.exit(2)
    if not bench_types:
        bench_types = set(VALID_BENCH_TYPES)

    total_passed = 0
    total_failed = 0

    # Run latency benchmarks
    if "latency" in bench_types:
        passed, failed = run_benchmarks_from_file(
            config_file=configs_dir / "latency-tests.yaml",
            bench_type="latency",
            results_dir=results_dir,
            spyre_devices=args.spyre_devices,
            aiu_world_size=args.aiu_world_size,
            models=models,
            tps=tps,
        )
        total_passed += passed
        total_failed += failed

    # Run throughput benchmarks
    if "throughput" in bench_types:
        passed, failed = run_benchmarks_from_file(
            config_file=configs_dir / "throughput-tests.yaml",
            bench_type="throughput",
            results_dir=results_dir,
            spyre_devices=args.spyre_devices,
            aiu_world_size=args.aiu_world_size,
            models=models,
            tps=tps,
        )
        total_passed += passed
        total_failed += failed

    # Run serve benchmarks (after latency/throughput to avoid port conflicts)
    if "serve" in bench_types:
        passed, failed = run_serve_benchmarks_from_file(
            config_file=configs_dir / "serve-tests.yaml",
            results_dir=results_dir,
            spyre_devices=args.spyre_devices,
            aiu_world_size=args.aiu_world_size,
            models=models,
            tps=tps,
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
