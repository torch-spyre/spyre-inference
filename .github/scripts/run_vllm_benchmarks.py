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

"""Run vLLM benchmarks from YAML config files.

Reads the benchmark configs of one suite, builds the appropriate
`vllm bench latency` / `vllm bench throughput` / `vllm bench serve` commands,
and executes them.

Serve entries run one server config at a time. Each server config first gets a
startup record: the server starts on an empty private compile cache, then again on
the cache it filled, and the record holds both startup times and the number of
files each start added to the cache. The serve benchmarks then run on that warm
cache. `--startup-only` and `--serve-only` run one phase alone.
"""

import contextlib
import hashlib
import importlib.util
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
import tempfile
import time
import urllib.request
from argparse import ArgumentParser
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Valid environment variable name pattern
ENV_VAR_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")

# The scripts are not a package, so load the fetcher by path.
_FETCH_SCRIPT = Path(__file__).resolve().parent / "fetch_bench_datasets.py"
_fetch_spec = importlib.util.spec_from_file_location("fetch_bench_datasets", _FETCH_SCRIPT)
assert _fetch_spec and _fetch_spec.loader
fetch_bench_datasets = importlib.util.module_from_spec(_fetch_spec)
_fetch_spec.loader.exec_module(fetch_bench_datasets)


def _dataset_path_defaults() -> dict[str, str]:
    cache = fetch_bench_datasets.cache_dir()
    return {var: str(cache / spec["path"]) for var, spec in fetch_bench_datasets.ARTIFACTS.items()}


VALID_BENCH_TYPES = ("latency", "throughput", "serve")
# The two phases of a serve run: the startup records, and the serve benchmarks themselves.
PHASES = ("startup", "serve")
SUITES = ("minimal", "base", "reference", "power-applications")
DEFAULT_SUITE = "base"


def parse_args():
    parser = ArgumentParser(description="Run vLLM benchmarks from YAML configs")
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="vllm-benchmarks/benchmarks",
        help="directory searched for *-runs.yaml config files, or a single config "
        "file, which runs unfiltered by suite",
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
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument(
        "--startup-only",
        action="store_true",
        help="for serve entries, record only the cold and warm server startup of each "
        "server config, without running the serve benchmarks",
    )
    phase.add_argument(
        "--serve-only",
        action="store_true",
        help="for serve entries, run only the serve benchmarks, on the compile cache the "
        "environment sets, without the startup records",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=DEFAULT_SUITE,
        help=f"suite to run: one of {', '.join(SUITES)} (default: {DEFAULT_SUITE}); "
        "it runs the entries whose `labels` name it",
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
    env = {**_dataset_path_defaults(), **os.environ}
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
        raw_path = str(config.get("parameters", {}).get("dataset-path", ""))
        _resolve_dataset_path(config)
        missing = _missing_dataset(config)
        if missing:
            missing_datasets.append((config.get("test_name"), missing, raw_path))
            continue
        selected.append(config)

    if missing_datasets:
        dataset_vars = set()
        for test_name, path, raw_path in missing_datasets:
            log.error("%s needs dataset %s, which is not present on this host", test_name, path)
            dataset_vars.update(string.Template(raw_path).get_identifiers())
        log.error(
            "Export %s to existing copies of the trace files, or run `make perf-tests` "
            "with FETCH_BENCH_DATA=1 to fetch them into SPYRE_BENCH_DATA_DIR.",
            " / ".join(sorted(dataset_vars)) or "the SPYRE_*_DATASET variables",
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


def bench_env(env_config: dict, spyre_devices: str, aiu_world_size: str) -> tuple[dict, dict]:
    """The process env, and the config's own part of it.

    The CLI device values are only a base: `env_config` carries the per-test
    values from _derive_config and is applied last, so it wins.
    """
    env = os.environ.copy()
    env["SPYRE_DEVICES"] = spyre_devices
    env["AIU_WORLD_SIZE"] = aiu_world_size
    config_env = build_env_vars(env_config)
    env.update(config_env)
    return env, config_env


# Invoke the vLLM CLI directly: the dynamo recompile-limit raise the benchmarks
# need is applied by the platform plugin at import (see
# spyre_inference/platform.py::_raise_dynamo_recompile_limits, torch-spyre #444).
# Via sys.executable rather than the `vllm` console script, so the CLI always
# runs in this interpreter's environment.
VLLM_CLI = [sys.executable, "-m", "vllm.entrypoints.cli.main"]


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

    env, config_env = bench_env(env_config, spyre_devices, aiu_world_size)

    cmd_line = record_command(cmd, config_env, results_dir / f"{test_name}.cmd")

    log.info("=== Running %s test: %s ===", bench_type, test_name)
    log.info("Command: %s", cmd_line)

    log_file = results_dir / f"{test_name}.log"
    with open(log_file, "w") as lf:
        lf.write(f"# {cmd_line}\n")
        lf.flush()
        result = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        log.error("Test %s failed with exit code %d", test_name, result.returncode)
        if result.stderr:
            stderr_lines = result.stderr.strip().splitlines()[-50:]
            log.error("stderr tail:\n%s", "\n".join(stderr_lines))
        return False
    log.info("Test %s passed", test_name)
    return True


def run_benchmarks(
    configs: list,
    bench_type: str,
    results_dir: Path,
    spyre_devices: str,
    aiu_world_size: str,
) -> tuple[int, int]:
    """Run latency or throughput benchmarks. Returns (passed, failed) counts."""
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


@contextlib.contextmanager
def running_server(
    server_parameters: dict,
    env: dict,
    config_env: dict,
    results_dir: Path,
    log_name: str,
    health_timeout: int,
):
    """Start `vllm serve`; yield its seconds to a healthy /health (None on failure), then kill it.

    Timed from before the spawn: for large models this is mostly the warmup compile.
    """
    server_params = dict(server_parameters)
    model = server_params.pop("model")
    host = str(server_params.get("host", "127.0.0.1"))
    port = int(server_params.get("port", 8000))
    server_cmd = [*VLLM_CLI, "serve", model]
    server_cmd.extend(build_command_args(server_params))

    server_cmd_line = record_command(server_cmd, config_env, results_dir / f"{log_name}.cmd")
    log.info("Server command: %s", server_cmd_line)

    server_log = results_dir / f"{log_name}.log"
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
        try:
            health_url = f"http://{host}:{port}/health"
            startup_sec = None
            deadline = server_start_ts + health_timeout
            while time.monotonic() < deadline:
                if server_proc.poll() is not None:
                    log.error("Server process died with exit code %d", server_proc.returncode)
                    log.error("Server log:\n%s", server_log.read_text())
                    break
                try:
                    urllib.request.urlopen(health_url, timeout=2)
                    startup_sec = round(time.monotonic() - server_start_ts, 1)
                    log.info("Server ready after %.1fs", startup_sec)
                    break
                except Exception:
                    time.sleep(1)
            else:
                log.error("Server did not become healthy within %ds", health_timeout)
                log.error("Server log:\n%s", server_log.read_text())
            yield startup_sec
        finally:
            _kill_server(server_proc)


def run_serve_benchmark(
    test_name: str,
    server_parameters: dict,
    bench_parameters: dict,
    env: dict,
    config_env: dict,
    results_dir: Path,
    health_timeout: int = 180,
) -> bool:
    """Start vllm serve, wait for health, run bench serve, cleanup."""
    log.info("=== Starting vLLM server for serve test: %s ===", test_name)
    with running_server(
        server_parameters, env, config_env, results_dir, f"{test_name}_server", health_timeout
    ) as server_startup_sec:
        if server_startup_sec is None:
            return False
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
        with open(bench_log, "w") as blf:
            blf.write(f"# {bench_cmd_line}\n")
            blf.flush()
            result = subprocess.run(
                bench_cmd, env=env, stdout=blf, stderr=subprocess.PIPE, text=True
            )

    if result.returncode != 0:
        log.error("Serve test %s failed with exit code %d", test_name, result.returncode)
        if result.stderr:
            stderr_lines = result.stderr.strip().splitlines()[-50:]
            log.error("stderr tail:\n%s", "\n".join(stderr_lines))
        return False

    # `vllm bench serve` only measures the request phase. A local artifact:
    # not in spyre_clickhouse_ingest.vllm's `_SERVE_METRICS`, so it does not
    # reach ClickHouse; the startup records do.
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


# ── startup records ──

# Set by a caller to keep a run off a compile cache; a startup record needs the caches on.
CACHE_DISABLING_ENV_VARS = (
    "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
    "TORCHINDUCTOR_FX_GRAPH_CACHE",
    "TORCHINDUCTOR_AUTOGRAD_CACHE",
    "VLLM_DISABLE_COMPILE_CACHE",
)
STARTUP_STATES = ("cold", "warm")


# Derived from the TP and from server_health_timeout: they do not change what the server
# compiles, so they stay out of the server config's identity.
_DERIVED_ENV_VARS = ("SPYRE_DEVICES", "AIU_WORLD_SIZE", "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS")


def server_key(config: dict) -> str:
    """What a server start depends on: its parameters and its own env."""
    env = {
        k: v
        for k, v in config.get("environment_variables", {}).items()
        if k not in _DERIVED_ENV_VARS
    }
    return json.dumps([config.get("server_parameters", {}), env], sort_keys=True)


def startup_name(config: dict) -> str:
    """`startup_<model>_tp<N>_len<L>_seqs<S>_h<hash>` for the entry's server config.

    <model> is the entry's test_name slug. Two server configs can share everything up
    to the seqs part (e.g. a decoder file and a power-use-case file for the same
    model), so a hash of the server config keeps their records apart, whichever suite
    selects them. The `h` prefix keeps the hash out of the ingest's shape tokens.
    """
    server = config.get("server_parameters", {})
    tp = _config_tp(config)
    model = config["test_name"].removeprefix("serve_").split(f"_tp{tp}_", 1)[0]
    digest = hashlib.sha256(server_key(config).encode()).hexdigest()[:8]
    return (
        f"startup_{model}_tp{tp}_len{server.get('max-model-len')}"
        f"_seqs{server.get('max-num-seqs')}_h{digest}"
    )


def server_groups(configs: list) -> list[tuple[str, list]]:
    """The serve entries grouped by server config, in first-seen order, each with its name."""
    groups: dict[str, list] = {}
    for config in configs:
        groups.setdefault(server_key(config), []).append(config)
    return [(startup_name(entries[0]), entries) for entries in groups.values()]


def count_cache_files(cache_root: Path) -> int:
    return sum(len(files) for _, _, files in os.walk(cache_root))


def cache_env(env_config: dict, cache_root: Path) -> dict:
    """`env_config` pointed at one private compile cache, with every cache layer on."""
    return {
        **env_config,
        "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "inductor"),
        "VLLM_CACHE_ROOT": str(cache_root / "vllm"),
        "SPYRE_KERNEL_CACHE": "1",
    }


def _pytorch_records(model: str, server_parameters: dict, metrics: dict) -> list:
    """`metrics` in the `*.pytorch.json` shape vLLM writes, which the ingest reads first."""
    return [
        {
            "benchmark": {"name": "vLLM benchmark", "extra_info": {"args": server_parameters}},
            "model": {"name": model},
            "metric": {"name": metric, "benchmark_values": [value], "extra_info": {}},
        }
        for metric, value in metrics.items()
    ]


def run_startup_benchmark(
    name: str,
    config: dict,
    env: dict,
    config_env: dict,
    cache_root: Path,
    results_dir: Path,
) -> bool:
    """Start the server on an empty cache, then on the cache it filled.

    Records each start's seconds to a healthy /health and the files it added to the
    cache. A warm start that adds files means some compiled artifact got a key that is
    not stable across processes.
    """
    server_parameters = config.get("server_parameters", {})
    health_timeout = int(config.get("server_health_timeout", 180))
    metrics: dict[str, float] = {}
    files_before = count_cache_files(cache_root)
    for state in STARTUP_STATES:
        log.info("=== Starting vLLM server for startup record: %s (%s) ===", name, state)
        with running_server(
            server_parameters,
            env,
            config_env,
            results_dir,
            f"{name}_{state}_server",
            health_timeout,
        ) as sec:
            pass
        if sec is None:
            log.error("Startup record %s failed on its %s start", name, state)
            return False
        files_after = count_cache_files(cache_root)
        metrics[f"{state}_startup_s"] = sec
        metrics[f"{state}_new_cache_files"] = files_after - files_before
        files_before = files_after

    model = server_parameters["model"]
    (results_dir / f"{name}.json").write_text(
        json.dumps({"model": model, "record_type": "startup", **metrics}, indent=2)
    )
    if os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT"):
        (results_dir / f"{name}.pytorch.json").write_text(
            json.dumps(_pytorch_records(model, server_parameters, metrics), indent=2)
        )
    if metrics["warm_new_cache_files"]:
        log.warning(
            "Startup record %s: the warm start added %d cache files; some compiled "
            "artifact is not reused across processes",
            name,
            metrics["warm_new_cache_files"],
        )
    log.info(
        "Startup record %s passed (cold %.1fs, %d files; warm %.1fs, %d files)",
        name,
        metrics["cold_startup_s"],
        metrics["cold_new_cache_files"],
        metrics["warm_startup_s"],
        metrics["warm_new_cache_files"],
    )
    return True


def _phase_env(
    env_config: dict, cache_root: Path | None, spyre_devices: str, aiu_world_size: str
) -> tuple[dict, dict]:
    """bench_env, pointed at `cache_root` with every cache layer on when it is given."""
    if cache_root is None:
        return bench_env(env_config, spyre_devices, aiu_world_size)
    env, config_env = bench_env(cache_env(env_config, cache_root), spyre_devices, aiu_world_size)
    for var in CACHE_DISABLING_ENV_VARS:
        if var not in config_env:
            env.pop(var, None)
    return env, config_env


def run_serve_benchmarks(
    configs: list,
    results_dir: Path,
    spyre_devices: str,
    aiu_world_size: str,
    phases: frozenset[str] = frozenset(PHASES),
) -> tuple[int, int]:
    """Run serve benchmarks, one server config at a time. Returns (passed, failed) counts.

    With the `startup` phase, each server config gets a startup record on a private
    compile cache, under TMPDIR, and its serve entries then run on that warm cache; the
    cache is deleted once they are done. With `serve` alone, the entries run on whatever
    cache the environment sets.
    """
    passed = 0
    failed = 0
    for name, entries in server_groups(configs):
        with contextlib.ExitStack() as stack:
            cache_root = None
            if "startup" in phases:
                cache_root = Path(
                    stack.enter_context(tempfile.TemporaryDirectory(prefix=f"{name}-cache-"))
                )
                env, config_env = _phase_env(
                    entries[0].get("environment_variables", {}),
                    cache_root,
                    spyre_devices,
                    aiu_world_size,
                )
                if run_startup_benchmark(
                    name, entries[0], env, config_env, cache_root, results_dir
                ):
                    passed += 1
                else:
                    failed += 1
                    # A server that cannot start fails its serve entries too.
                    if "serve" in phases:
                        log.error("Skipping the %d serve entries of %s", len(entries), name)
                        failed += len(entries)
                    continue

            if "serve" not in phases:
                continue
            for config in entries:
                env, config_env = _phase_env(
                    config.get("environment_variables", {}),
                    cache_root,
                    spyre_devices,
                    aiu_world_size,
                )
                success = run_serve_benchmark(
                    test_name=config.get("test_name", "unknown"),
                    server_parameters=config.get("server_parameters", {}),
                    bench_parameters=config.get("parameters", {}),
                    env=env,
                    config_env=config_env,
                    results_dir=results_dir,
                    # Large models pay a one-time warmup compile before /health
                    # passes; let a config bump this past the 180s default.
                    health_timeout=int(config.get("server_health_timeout", 180)),
                )
                if success:
                    passed += 1
                else:
                    failed += 1

    return passed, failed


def _bench_type(config_file: Path) -> str | None:
    """Bench type named by a config file's prefix, e.g. `serve-x-runs.yaml` -> serve."""
    prefix = config_file.name.split("-", 1)[0]
    return prefix if prefix in VALID_BENCH_TYPES else None


def config_files(configs_path: Path) -> list[Path]:
    """Config files under `configs_path`, or `configs_path` itself if it is a file.

    Exits when there is none, or when a file's prefix names no bench type.
    """
    files = [configs_path] if configs_path.is_file() else sorted(configs_path.rglob("*-runs.yaml"))
    if not files:
        log.error("No *-runs.yaml config file in %s", configs_path)
        sys.exit(2)
    for config_file in files:
        if _bench_type(config_file) is None:
            log.error("%s: name must start with one of %s", config_file, VALID_BENCH_TYPES)
            sys.exit(2)
    return files


def test_name(config: dict, config_file: Path) -> str:
    """The entry's test_name; exits when it is missing or not a string."""
    name = config.get("test_name")
    if not isinstance(name, str) or not name:
        log.error("%s: an entry has no test_name", config_file)
        sys.exit(2)
    return name


def _labels(config: dict, required: bool = True) -> list[str]:
    """Suite labels of an entry; exits on unknown labels, or missing ones if `required`."""
    labels = config.get("labels")
    if labels is None and not required:
        return []
    if (
        not isinstance(labels, list)
        or not all(isinstance(label, str) and label in SUITES for label in labels)
        or (required and not labels)
    ):
        log.error(
            "%s has labels %r; it needs a non-empty list of %s",
            config.get("test_name"),
            labels,
            SUITES,
        )
        sys.exit(2)
    return labels


def collect_configs(configs_path: Path, suite: str, bench_types: set[str]) -> dict[str, list]:
    """Map each selected bench type to the configs of `suite`; a single file runs unfiltered."""
    unfiltered = configs_path.is_file()
    selected: dict[str, list] = {b: [] for b in VALID_BENCH_TYPES if b in bench_types}
    seen: dict[str, Path] = {}
    for config_file in config_files(configs_path):
        bench_type = _bench_type(config_file)
        configs = _load_configs(config_file)
        if configs is None:
            sys.exit(2)
        for config in configs:
            name = test_name(config, config_file)
            if name in seen:
                log.error("test_name %s is in both %s and %s", name, seen[name], config_file)
                sys.exit(2)
            seen[name] = config_file
            labels = _labels(config, required=not unfiltered)
            if bench_type not in selected:
                continue
            if unfiltered or suite in labels:
                selected[bench_type].append(config)
    return selected


def main():
    # Turn SIGTERM into SystemExit so the running server's process group, started in its
    # own session, is still killed.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))

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

    suite = args.suite.strip() or DEFAULT_SUITE
    if suite not in SUITES:
        log.error("Unknown suite %r; valid: %s", suite, SUITES)
        sys.exit(2)
    if not configs_dir.exists():
        log.error("Config path %s does not exist", configs_dir)
        sys.exit(2)
    selected = collect_configs(configs_dir, suite, bench_types)
    selected = {b: _select_configs(c, models, tps) for b, c in selected.items()}

    total_passed = 0
    total_failed = 0

    for bench_type in ("latency", "throughput"):
        if bench_type in selected:
            passed, failed = run_benchmarks(
                configs=selected[bench_type],
                bench_type=bench_type,
                results_dir=results_dir,
                spyre_devices=args.spyre_devices,
                aiu_world_size=args.aiu_world_size,
            )
            total_passed += passed
            total_failed += failed

    # Run serve benchmarks (after latency/throughput to avoid port conflicts)
    if "serve" in selected:
        phases = frozenset(
            ["startup"] if args.startup_only else ["serve"] if args.serve_only else PHASES
        )
        passed, failed = run_serve_benchmarks(
            configs=selected["serve"],
            results_dir=results_dir,
            spyre_devices=args.spyre_devices,
            aiu_world_size=args.aiu_world_size,
            phases=phases,
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
