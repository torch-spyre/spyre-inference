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
import datetime
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
import time
import urllib.request
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from dataclasses import dataclass
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
    parser.add_argument(
        "--suite",
        type=str,
        default="",
        help="config suite to run: empty for the CI configs (<type>-tests.yaml), "
        "or e.g. 'dev' for <type>-tests-dev.yaml; a bench type with no file "
        "for the suite is skipped, and a suite with no file at all is an error",
    )
    parser.add_argument(
        "--junit-xml",
        type=str,
        default="",
        help="write one JUnit testcase per benchmark run to this file (empty = none)",
    )
    parser.add_argument(
        "--list-legs",
        action="store_true",
        help="print the CI legs, one per selected (model, tp), as JSON and run nothing",
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


def _resolve_dataset_path(config: dict) -> list[str]:
    """Expand environment variables in a config's `dataset-path`, in place.

    Returns the identifiers that neither the environment nor the fetch cache
    provided, which safe_substitute leaves as literal `${VAR}` or `$VAR` text.
    """
    parameters = config.get("parameters")
    if not parameters:
        return []
    path = parameters.get("dataset-path")
    if not path:
        return []
    env = {**_dataset_path_defaults(), **os.environ}
    template = string.Template(str(path))
    parameters["dataset-path"] = template.safe_substitute(env)
    return [var for var in template.get_identifiers() if var not in env]


def _missing_dataset(config: dict) -> str | None:
    """Dataset path a config needs but which is absent on this host, if any."""
    path = config.get("parameters", {}).get("dataset-path")
    if path and not Path(path).exists():
        return str(path)
    return None


# Dataset variables that are optional because nothing provides them yet (#1538).
# Anything else unresolved is a typo or an unfetched dataset and stays fatal.
_OPTIONAL_DATASET_VARS = {"SPYRE_VISION_DATASET"}


def _select_configs(configs: list, models: set[str], tps: set[int]) -> list:
    """Keep configs whose model and TP are selected (empty selects all).

    A selected config whose dataset is absent is fatal, not skipped: skipping
    would let a serve-only model job report success while measuring nothing.
    The exception is a config whose dataset-path names only variables in
    _OPTIONAL_DATASET_VARS, which nothing provides yet and cannot resolve on
    any host: that one is skipped with a WARNING.
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
        unresolved = _resolve_dataset_path(config)
        missing = _missing_dataset(config)
        if missing:
            if unresolved and set(unresolved) <= _OPTIONAL_DATASET_VARS:
                log.warning(
                    "Skipping %s (dataset variable %s is unset and has no "
                    "default; export the SPYRE_*_DATASET variables to run it)",
                    config.get("test_name"),
                    " / ".join(sorted(unresolved)),
                )
                continue
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


_CACHE_CONFIG = Path(__file__).resolve().parents[1] / "cache_config" / "hf_models_and_datasets.yaml"


def _pinned_revisions() -> dict[str, str]:
    """Model -> revision for the models the CI cache holds only at a pinned revision."""
    try:
        raw = yaml.safe_load(_CACHE_CONFIG.read_text()) or {}
    except OSError:
        return {}
    entries = (raw.get("public_models") or []) + (raw.get("gated_models") or [])
    return {
        e["repo"].lower(): e["revision"]
        for e in entries
        if isinstance(e, dict) and e.get("revision")
    }


def _local_snapshot(model: str, revision: str) -> str | None:
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model, revision=revision, local_files_only=True)
    except Exception:  # noqa: BLE001
        return None


def _pin_revision(config: dict, pins: dict[str, str]) -> None:
    """Run a pinned model at its cached revision: offline, `main` does not resolve for it."""
    model = _config_model(config)
    revision = pins.get((model or "").lower())
    if not model or not revision:
        return
    engine = config["server_parameters"] if "server_parameters" in config else config["parameters"]
    revision = engine.setdefault("revision", revision)
    # bench serve and throughput load the tokenizer without a revision, so give them the snapshot.
    snapshot = _local_snapshot(model, revision)
    if snapshot:
        config.setdefault("parameters", {}).setdefault("tokenizer", snapshot)


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


@dataclass
class BenchResult:
    bench_type: str
    test_name: str
    model: str
    seconds: float
    error: str | None = None


def _tail(text: str | None, lines: int = 50) -> str:
    return "\n".join((text or "").strip().splitlines()[-lines:])


def _bench_failure(test_name: str, result: subprocess.CompletedProcess) -> str:
    """Log a failed `vllm bench` run and return its failure text."""
    log.error("Test %s failed with exit code %d", test_name, result.returncode)
    tail = _tail(result.stderr)
    if tail:
        log.error("stderr tail:\n%s", tail)
    return f"exit code {result.returncode}\n{tail}".strip()


def run_benchmark(
    bench_type: str,
    test_name: str,
    parameters: dict,
    env_config: dict,
    results_dir: Path,
    spyre_devices: str,
    aiu_world_size: str,
) -> str | None:
    """Run a single vllm bench command. Returns None on success, else the failure text."""
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
    with open(log_file, "w") as lf:
        lf.write(f"# {cmd_line}\n")
        lf.flush()
        result = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return _bench_failure(test_name, result)
    log.info("Test %s passed", test_name)
    return None


def run_serve_benchmark(
    test_name: str,
    server_parameters: dict,
    bench_parameters: dict,
    env_config: dict,
    results_dir: Path,
    spyre_devices: str,
    aiu_world_size: str,
    health_timeout: int = 180,
) -> str | None:
    """Serve, wait for health, run bench serve, clean up; None on success, else the failure."""
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

    def _server_failure(summary: str) -> str:
        log.error("%s", summary)
        text = server_log.read_text() if server_log.exists() else ""
        if text:
            log.error("Server log:\n%s", text)
        return f"{summary}\n{_tail(text)}".strip()

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
            # Timed from before the spawn: for large models this is mostly the
            # one-time warmup compile, which is worth comparing run to run.
            health_url = f"http://{host}:{port}/health"
            server_startup_sec = None
            deadline = server_start_ts + health_timeout
            while time.monotonic() < deadline:
                if server_proc.poll() is not None:
                    return _server_failure(
                        f"Server process died with exit code {server_proc.returncode}"
                    )
                try:
                    urllib.request.urlopen(health_url, timeout=2)
                    server_startup_sec = round(time.monotonic() - server_start_ts, 1)
                    log.info("Server ready after %.1fs", server_startup_sec)
                    break
                except Exception:
                    time.sleep(1)

            if server_startup_sec is None:
                return _server_failure(f"Server did not become healthy within {health_timeout}s")

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
            with open(bench_log, "w") as blf:
                blf.write(f"# {bench_cmd_line}\n")
                blf.flush()
                result = subprocess.run(
                    bench_cmd, env=env, stdout=blf, stderr=subprocess.PIPE, text=True
                )
        finally:
            _kill_server(server_proc)

    if result.returncode != 0:
        return _bench_failure(test_name, result)

    # `vllm bench serve` only measures the request phase. A local artifact:
    # not in spyre_clickhouse_ingest.vllm's `_SERVE_METRICS`, so it does not
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
    return None


def run_benchmarks_from_file(
    config_file: Path,
    bench_type: str,
    results_dir: Path,
    spyre_devices: str,
    aiu_world_size: str,
    models: set[str],
    tps: set[int],
) -> list[BenchResult]:
    """Run every selected benchmark in a config file."""
    if not config_file.exists():
        log.info("No %s config found, skipping", config_file.name)
        return []

    configs = _load_configs(config_file)
    if configs is None:
        return [BenchResult(bench_type, config_file.name, "", 0.0, "malformed config file")]

    pins = _pinned_revisions()
    results = []
    for config in _select_configs(configs, models, tps):
        test_name = config.get("test_name", "unknown")
        _pin_revision(config, pins)
        env_config = config.get("environment_variables", {})
        start = time.monotonic()
        if bench_type == "serve":
            error = run_serve_benchmark(
                test_name=test_name,
                server_parameters=config.get("server_parameters", {}),
                bench_parameters=config.get("parameters", {}),
                env_config=env_config,
                results_dir=results_dir,
                spyre_devices=spyre_devices,
                aiu_world_size=aiu_world_size,
                # Large models pay a one-time warmup compile before /health passes;
                # let a config bump this past the 180s default.
                health_timeout=int(config.get("server_health_timeout", 180)),
            )
        else:
            error = run_benchmark(
                bench_type=bench_type,
                test_name=test_name,
                parameters=config.get("parameters", {}),
                env_config=env_config,
                results_dir=results_dir,
                spyre_devices=spyre_devices,
                aiu_world_size=aiu_world_size,
            )
        model = _config_model(config) or ""
        results.append(BenchResult(bench_type, test_name, model, time.monotonic() - start, error))
    return results


# Spyre runner pools by card count; a leg takes the smallest one its TP fits.
CARD_POOLS = (1, 2, 4)


def list_legs(config_files: list[tuple[str, Path]], models: set[str], tps: set[int]) -> list[dict]:
    """One CI leg per selected (model, tp), each run on a runner with enough cards."""
    legs: dict[tuple[str, int], str] = {}
    for _, config_file in config_files:
        configs = _load_configs(config_file) if config_file.exists() else []
        if configs is None:
            sys.exit(2)
        for config in configs:
            model, tp = _config_model(config), _config_tp(config)
            if not model or (models and model.lower() not in models):
                continue
            if tp is None or tp > CARD_POOLS[-1]:
                log.error(
                    "%s needs a tensor-parallel-size of 1 to %d",
                    config.get("test_name"),
                    CARD_POOLS[-1],
                )
                sys.exit(2)
            if not tps or tp in tps:
                legs.setdefault((model.lower(), tp), model)
    return [
        {
            "model": model,
            "tp": tp,
            "cards": next(n for n in CARD_POOLS if n >= tp),
            "key": f"{re.sub(r'[^a-z0-9.]+', '-', key).strip('-')}-tp{tp}",
        }
        for (key, tp), model in legs.items()
    ]


# Characters XML 1.0 cannot carry, e.g. the ANSI escapes in vLLM's stderr.
_XML_INVALID = re.compile("[^\t\n\r\x20-퟿-�\U00010000-\U0010ffff]")


def write_junit(path: Path, results: list[BenchResult]) -> None:
    """One testcase per benchmark, read by the same summary and ingest as the pytest legs."""
    suite = ET.Element(
        "testsuite",
        name="vllm-benchmarks",
        tests=str(len(results)),
        failures=str(sum(r.error is not None for r in results)),
        errors="0",
        skipped="0",
        time=f"{sum(r.seconds for r in results):.3f}",
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    )
    arch = re.sub(r"[^a-zA-Z0-9_]", "_", platform.machine() or "").strip("_") or "unknown"
    for r in results:
        case = ET.SubElement(
            suite,
            "testcase",
            classname=f"vllm_bench.{r.bench_type}",
            name=r.test_name,
            time=f"{r.seconds:.3f}",
        )
        # The tags the pytest legs carry (tests/plugin/spyre_testing_plugin/tags.py).
        tags = [f"platform__{arch}", "testtype__perf"] + ([f"model__{r.model}"] if r.model else [])
        props = ET.SubElement(case, "properties")
        for tag in tags:
            ET.SubElement(props, "property", name="tag", value=tag)
        if r.error is not None:
            text = _XML_INVALID.sub("", r.error)
            ET.SubElement(case, "failure", message=text.splitlines()[0] if text else "").text = text
    root = ET.Element("testsuites")
    root.append(suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


VALID_BENCH_TYPES = ("latency", "throughput", "serve")


def _suite_config_files(configs_dir: Path, suite: str, bench_types: set[str]) -> dict[str, Path]:
    """Map each selected bench type to its `<type>-tests[-<suite>].yaml` config."""
    suffix = f"-{suite}" if suite else ""
    return {
        b: configs_dir / f"{b}-tests{suffix}.yaml" for b in VALID_BENCH_TYPES if b in bench_types
    }


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

    suite = args.suite.strip()
    config_files = _suite_config_files(configs_dir, suite, bench_types)
    # A mistyped suite would otherwise run nothing and still exit 0.
    if suite and not any(f.exists() for f in config_files.values()):
        log.error(
            "Suite %r has no config in %s: expected one of %s",
            suite,
            configs_dir,
            [f.name for f in config_files.values()],
        )
        sys.exit(2)

    # Serve runs after latency/throughput to avoid port conflicts.
    files = list(config_files.items())
    if "serve" in bench_types:
        files.append(("serve", configs_dir / "multimodal-serve-tests.yaml"))

    if args.list_legs:
        print(json.dumps(list_legs(files, models, tps)))
        return

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for bench_type, config_file in files:
        results += run_benchmarks_from_file(
            config_file=config_file,
            bench_type=bench_type,
            results_dir=results_dir,
            spyre_devices=args.spyre_devices,
            aiu_world_size=args.aiu_world_size,
            models=models,
            tps=tps,
        )
    failed = sum(r.error is not None for r in results)

    # Summary
    log.info("=== Benchmark Summary ===")
    log.info("Passed: %d, Failed: %d", len(results) - failed, failed)

    result_files = list(results_dir.glob("*.json"))
    log.info("Result files: %d", len(result_files))
    for f in result_files:
        log.info("  %s", f.name)

    if args.junit_xml:
        write_junit(Path(args.junit_xml), results)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
