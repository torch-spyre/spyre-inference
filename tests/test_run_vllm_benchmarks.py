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

"""CPU-only tests for the benchmark runner's config layer (no hardware needed).

What the benchmarks depend on and a typo would silently break: tests merge over
`defaults` per section rather than replacing it, TP drives
SPYRE_DEVICES/AIU_WORLD_SIZE and server_health_timeout drives
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS, the bench side inherits the server's model,
`dataset-path` expands from the environment, a selected test whose dataset is
absent aborts the run instead of being skipped, and `labels` pick the suite.

All checked-in configs are loaded as-is at the end, so a config edit that
breaks these invariants fails here.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / ".github" / "scripts" / "run_vllm_benchmarks.py"
_spec = importlib.util.spec_from_file_location("run_vllm_benchmarks", _SCRIPT)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

_MATRIX_SCRIPT = _REPO / ".github" / "scripts" / "generate_vllm_benchmark_matrix.py"
_matrix_spec = importlib.util.spec_from_file_location(
    "generate_vllm_benchmark_matrix", _MATRIX_SCRIPT
)
matrix = importlib.util.module_from_spec(_matrix_spec)
_matrix_spec.loader.exec_module(matrix)

_TABLES_SCRIPT = _REPO / ".github" / "scripts" / "generate_bench_config_tables.py"
_tables_spec = importlib.util.spec_from_file_location("bench_config_tables", _TABLES_SCRIPT)
tables = importlib.util.module_from_spec(_tables_spec)
_tables_spec.loader.exec_module(tables)

_CONFIGS = _REPO / "vllm-benchmarks" / "benchmarks"
CONFIG_FILES = runner.config_files(_CONFIGS)
SERVE_FILES = [f for f in CONFIG_FILES if f.name.startswith("serve-")]
OFFLINE_FILES = [f for f in CONFIG_FILES if not f.name.startswith("serve-")]


def _write(tmp_path, raw) -> Path:
    path = tmp_path / "serve-tests.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _serve_config(dataset: str, **overrides) -> dict:
    config = {
        "test_name": "t",
        "server_parameters": {"model": "m", "tensor-parallel-size": 1},
        "parameters": {"dataset-path": dataset},
    }
    config.update(overrides)
    return config


# ── defaults merging ──


def test_defaults_merge_per_section(tmp_path):
    """A test overrides individual keys; the other defaults in that section stay."""
    path = _write(
        tmp_path,
        {
            "defaults": {
                "server_parameters": {"max-num-seqs": 4, "enable-prefix-caching": True},
                "parameters": {"num-prompts": 100, "dataset-name": "custom"},
                "environment_variables": {"DTLOG_LEVEL": "error"},
            },
            "tests": [
                {
                    "test_name": "t",
                    "server_parameters": {
                        "model": "m",
                        "tensor-parallel-size": 1,
                        "max-num-seqs": 8,
                    },
                    "parameters": {"num-prompts": 50},
                }
            ],
        },
    )
    (config,) = runner._load_configs(path)

    assert config["server_parameters"]["max-num-seqs"] == 8
    assert config["server_parameters"]["enable-prefix-caching"] is True
    assert config["parameters"]["num-prompts"] == 50
    assert config["parameters"]["dataset-name"] == "custom"
    assert config["environment_variables"]["DTLOG_LEVEL"] == "error"


def test_defaults_are_not_shared_between_tests(tmp_path):
    """Each test gets its own sections, so deriving into one does not leak."""
    path = _write(
        tmp_path,
        {
            "defaults": {"parameters": {"num-prompts": 100}},
            "tests": [
                {
                    "test_name": "a",
                    "server_parameters": {"model": "m", "tensor-parallel-size": 1},
                },
                {
                    "test_name": "b",
                    "server_parameters": {"model": "m", "tensor-parallel-size": 4},
                },
            ],
        },
    )
    a, b = runner._load_configs(path)

    assert a["environment_variables"]["AIU_WORLD_SIZE"] == "1"
    assert b["environment_variables"]["AIU_WORLD_SIZE"] == "4"
    assert a["parameters"] is not b["parameters"]


def test_bare_list_still_loads(tmp_path):
    """A bare list with no `defaults` mapping still loads."""
    path = _write(
        tmp_path, [{"test_name": "t", "parameters": {"model": "m", "tensor_parallel_size": 2}}]
    )
    (config,) = runner._load_configs(path)

    assert config["environment_variables"]["SPYRE_DEVICES"] == "0,1"


@pytest.mark.parametrize("raw", [{"defaults": {}}, {"tests": {}}, "scalar"])
def test_malformed_file_returns_none(tmp_path, raw):
    assert runner._load_configs(_write(tmp_path, raw)) is None


# ── derived values ──


@pytest.mark.parametrize("tp,devices", [(1, "0"), (2, "0,1"), (4, "0,1,2,3")])
def test_tp_drives_devices_and_world_size(tmp_path, tp, devices):
    path = _write(
        tmp_path,
        {
            "tests": [
                {"test_name": "t", "server_parameters": {"model": "m", "tensor-parallel-size": tp}}
            ]
        },
    )
    (config,) = runner._load_configs(path)

    assert config["environment_variables"]["SPYRE_DEVICES"] == devices
    assert config["environment_variables"]["AIU_WORLD_SIZE"] == str(tp)


def test_explicit_env_wins_over_derived(tmp_path):
    path = _write(
        tmp_path,
        {
            "tests": [
                {
                    "test_name": "t",
                    "server_parameters": {"model": "m", "tensor-parallel-size": 2},
                    "environment_variables": {"SPYRE_DEVICES": "2,3"},
                }
            ]
        },
    )
    (config,) = runner._load_configs(path)

    assert config["environment_variables"]["SPYRE_DEVICES"] == "2,3"
    assert config["environment_variables"]["AIU_WORLD_SIZE"] == "2"


def test_health_timeout_drives_execute_model_timeout(tmp_path):
    path = _write(
        tmp_path,
        {
            "tests": [
                {
                    "test_name": "t",
                    "server_health_timeout": 3600,
                    "server_parameters": {"model": "m", "tensor-parallel-size": 1},
                }
            ]
        },
    )
    (config,) = runner._load_configs(path)

    assert config["environment_variables"]["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] == "3600"


def test_bench_model_inherits_from_server(tmp_path):
    path = _write(
        tmp_path,
        {
            "tests": [
                {
                    "test_name": "t",
                    "server_parameters": {"model": "ibm-granite/x", "tensor-parallel-size": 1},
                    "parameters": {"num-prompts": 10},
                }
            ]
        },
    )
    (config,) = runner._load_configs(path)

    assert config["parameters"]["model"] == "ibm-granite/x"


# ── dataset resolution ──


def test_dataset_path_expands_from_env(monkeypatch, tmp_path):
    trace = tmp_path / "cics.jsonl"
    trace.touch()
    monkeypatch.setenv("SPYRE_CICS_DATASET", str(trace))
    config = _serve_config("${SPYRE_CICS_DATASET}")
    runner._derive_config(config)

    (selected,) = runner._select_configs([config], set(), set())

    assert selected["parameters"]["dataset-path"] == str(trace)


def test_unset_dataset_var_falls_back_to_fetch_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("SPYRE_AIOPS_DATASET", raising=False)
    monkeypatch.setenv("SPYRE_BENCH_DATA_DIR", str(tmp_path))
    config = _serve_config("${SPYRE_AIOPS_DATASET}")
    runner._resolve_dataset_path(config)

    expected = tmp_path / runner.fetch_bench_datasets.ARTIFACTS["SPYRE_AIOPS_DATASET"]["path"]
    assert config["parameters"]["dataset-path"] == str(expected)


def test_missing_dataset_aborts_the_run(monkeypatch, tmp_path):
    """A skip here would let a serve-only model job go green measuring nothing."""
    monkeypatch.setenv("SPYRE_CICS_DATASET", str(tmp_path / "absent.jsonl"))
    config = _serve_config("${SPYRE_CICS_DATASET}")
    runner._derive_config(config)

    with pytest.raises(SystemExit) as excinfo:
        runner._select_configs([config], set(), set())

    assert excinfo.value.code == 2


def test_missing_dataset_of_a_deselected_test_is_not_fatal(monkeypatch, tmp_path):
    """Filtering runs first, so an absent trace only matters for a selected test."""
    monkeypatch.setenv("SPYRE_CICS_DATASET", str(tmp_path / "absent.jsonl"))
    config = _serve_config("${SPYRE_CICS_DATASET}")
    runner._derive_config(config)

    assert runner._select_configs([config], {"other-model"}, set()) == []
    assert runner._select_configs([config], set(), {4}) == []


def test_config_without_tp_aborts_the_run():
    with pytest.raises(SystemExit) as excinfo:
        runner._select_configs([{"test_name": "t", "parameters": {"model": "m"}}], set(), set())

    assert excinfo.value.code == 2


# ── filters ──


def test_model_filter_is_case_insensitive(tmp_path):
    trace = tmp_path / "t.jsonl"
    trace.touch()
    configs = [_serve_config(str(trace))]
    configs[0]["server_parameters"]["model"] = "IBM-Granite/X"
    runner._derive_config(configs[0])

    assert len(runner._select_configs(configs, {"ibm-granite/x"}, set())) == 1


# ── suites ──


def _write_runs(directory: Path, name: str, labels_per_test: dict[str, list | None]) -> Path:
    tests = []
    for test_name, labels in labels_per_test.items():
        config = {
            "test_name": test_name,
            "parameters": {"model": f"m-{test_name}", "tensor_parallel_size": 1},
        }
        if labels is not None:
            config["labels"] = labels
        tests.append(config)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(yaml.safe_dump({"tests": tests}))
    return path


def _names(selected: dict[str, list]) -> dict[str, list[str]]:
    return {b: [c["test_name"] for c in configs] for b, configs in selected.items()}


@pytest.mark.parametrize(
    ("suite", "expected"),
    [
        ("minimal", ["a"]),
        ("base", ["a", "b"]),
        ("reference", ["c"]),
        ("exhaustive", ["a", "b", "c", "d"]),
    ],
)
def test_labels_select_the_suite(tmp_path, suite, expected):
    """An entry runs in the suites it names, and every entry runs in exhaustive."""
    _write_runs(
        tmp_path / "decoder",
        "latency-x-runs.yaml",
        {"a": ["minimal", "base"], "b": ["base"], "c": ["reference"], "d": None},
    )

    selected = runner.collect_configs(tmp_path, suite, {"latency"})

    assert _names(selected) == {"latency": expected}


def test_bench_type_comes_from_the_file_prefix(tmp_path):
    _write_runs(tmp_path / "a", "latency-x-runs.yaml", {"lat": ["base"]})
    _write_runs(tmp_path / "b", "serve-x-runs.yaml", {"srv": ["base"]})
    _write_runs(tmp_path, "notes.yaml", {"ignored": ["base"]})

    selected = runner.collect_configs(tmp_path, "base", {"latency", "serve"})

    assert _names(selected) == {"latency": ["lat"], "serve": ["srv"]}


def test_single_config_file_runs_unfiltered(tmp_path):
    path = _write_runs(tmp_path, "serve-x-runs.yaml", {"a": ["base"], "b": None})

    selected = runner.collect_configs(path, "minimal", {"serve"})

    assert _names(selected) == {"serve": ["a", "b"]}


@pytest.mark.parametrize("labels", [["smoke"], ["exhaustive"], "base"])
def test_unknown_label_aborts_the_run(tmp_path, labels):
    """A typo would otherwise drop the entry from its suite without a word."""
    _write_runs(tmp_path, "serve-x-runs.yaml", {"a": labels})

    with pytest.raises(SystemExit) as exc:
        runner.collect_configs(tmp_path, "base", {"serve"})
    assert exc.value.code == 2


def test_duplicate_test_name_aborts_the_run(tmp_path):
    """A test_name names the result file, so a duplicate would overwrite it."""
    _write_runs(tmp_path / "a", "serve-x-runs.yaml", {"t": ["base"]})
    _write_runs(tmp_path / "b", "serve-y-runs.yaml", {"t": None})

    with pytest.raises(SystemExit) as exc:
        runner.collect_configs(tmp_path, "exhaustive", {"serve"})
    assert exc.value.code == 2


def test_unknown_suite_aborts_the_run(monkeypatch, tmp_path):
    _write_runs(tmp_path, "latency-x-runs.yaml", {"a": ["base"]})
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vllm_benchmarks.py",
            "--configs-dir",
            str(tmp_path),
            "--results-dir",
            str(tmp_path / "results"),
            "--suite",
            "dev",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def test_matrix_follows_the_suite(tmp_path):
    """The matrix holds exactly the models the run would bench for that suite."""
    _write_runs(tmp_path, "latency-x-runs.yaml", {"small": ["minimal"], "big": None})

    minimal = matrix.generate_benchmark_matrix(str(tmp_path), [], "minimal")
    exhaustive = matrix.generate_benchmark_matrix(str(tmp_path), [], "exhaustive")

    assert minimal["include"] == [{"models": "m-small"}]
    assert exhaustive["include"] == [{"models": "m-small"}, {"models": "m-big"}]


# ── the checked-in config ──

# Each trace is reached through its own env var. `all` at TP1 replays the trace
# truncated to the TP1 context.
DATASET_PATH_VARS = {
    "aiops": "${SPYRE_AIOPS_DATASET}",
    "cics": "${SPYRE_CICS_DATASET}",
    "all": "${SPYRE_ALL_SEQUENCES_DATASET}",
    "vision": "${SPYRE_VISION_DATASET}",
}
TP1_DATASET_PATH_VARS = {**DATASET_PATH_VARS, "all": "${SPYRE_ALL_SEQUENCES_TRUNCATED_8K_DATASET}"}
# One decoder setup per TP: (max-model-len, max-num-seqs).
DECODER_SETUP = {4: (32768, 32), 1: (8192, 8)}


def test_checked_in_configs_are_found():
    assert SERVE_FILES and OFFLINE_FILES


def test_checked_in_test_names_and_labels_are_valid():
    """collect_configs exits on a duplicate test_name or an unknown label."""
    selected = runner.collect_configs(_CONFIGS, "exhaustive", set(runner.VALID_BENCH_TYPES))

    assert all(selected.values())
    for suite in runner.SUITE_LABELS:
        assert any(runner.collect_configs(_CONFIGS, suite, {"serve"}).values()), suite


def test_readme_config_tables_are_up_to_date():
    current = tables.README.read_text()

    assert tables.updated_readme(current, tables.render(tables.load_entries())) == current


def test_tables_reject_a_duplicate_test_name(monkeypatch, tmp_path):
    _write_runs(tmp_path / "a", "serve-x-runs.yaml", {"t": ["base"]})
    _write_runs(tmp_path / "b", "latency-y-runs.yaml", {"t": None})
    monkeypatch.setattr(tables, "CONFIGS_DIR", tmp_path)
    monkeypatch.setattr(tables, "README", tmp_path / "README.md")

    with pytest.raises(SystemExit) as exc:
        tables.load_entries()
    assert exc.value.code == 2


@pytest.mark.parametrize("config_file", SERVE_FILES, ids=lambda f: f.name)
def test_serve_runs_yaml_derives_consistently(config_file):
    configs = runner._load_configs(config_file)
    assert configs

    for config in configs:
        env = config["environment_variables"]
        tp = runner._config_tp(config)
        assert tp is not None, config["test_name"]
        assert env["AIU_WORLD_SIZE"] == str(tp)
        assert env["SPYRE_DEVICES"].split(",") == [str(i) for i in range(tp)]
        # The bench client must target the server's model, not a stale copy.
        assert config["parameters"]["model"] == config["server_parameters"]["model"]

        name = config["test_name"]
        server = config["server_parameters"]
        parameters = config["parameters"]
        # A name maps to one config: it spells out the TP and, for decoders, the
        # concurrency the entry runs at.
        assert f"_tp{tp}_" in name, name
        if not parameters["dataset-name"].startswith("random"):
            assert f"_conc{parameters['max-concurrency']}_" in name, name
            # The name's suffix has to say which trace the entry replays.
            trace = name.rsplit("_", 1)[1]
            path_vars = TP1_DATASET_PATH_VARS if tp == 1 else DATASET_PATH_VARS
            assert trace in path_vars, name
            # _select_configs expands the env var, so at load time it is still
            # the literal the entry names.
            assert parameters["dataset-path"] == path_vars[trace], name
            assert (server["max-model-len"], server["max-num-seqs"]) == DECODER_SETUP[tp], name
        elif parameters["dataset-name"] == "random-mm":
            assert "dataset-path" not in parameters, name
            assert parameters["input-len"] <= server["max-model-len"], name
        else:
            # Generated prompts need no staged file, and a request over the
            # server's context is rejected.
            assert "dataset-path" not in parameters, config["test_name"]
            assert parameters["random-input-len"] <= config["server_parameters"]["max-model-len"], (
                config["test_name"]
            )


# The offline files spell input/output length under their own flag names.
OFFLINE_LEN_KEYS = {
    "latency": ("input-len", "output-len"),
    "throughput": ("random-input-len", "random-output-len"),
}


@pytest.mark.parametrize("config_file", OFFLINE_FILES, ids=lambda f: f.name)
def test_offline_tests_yaml_derives_consistently(config_file):
    """Every offline entry derives its devices, and names the shape it runs."""
    input_key, output_key = OFFLINE_LEN_KEYS[config_file.name.split("-", 1)[0]]
    configs = runner._load_configs(config_file)
    assert configs

    for config in configs:
        name = config["test_name"]
        env = config["environment_variables"]
        tp = runner._config_tp(config)
        assert tp is not None, name
        assert env["AIU_WORLD_SIZE"] == str(tp)
        assert env["SPYRE_DEVICES"].split(",") == [str(i) for i in range(tp)]

        parameters = config["parameters"]
        # in<N>_out<N>/tp<N> in the name are the benchmark's identity downstream
        # (ingest_vllm_benchmarks.py::_parse_input_shapes), so a name that
        # disagrees with the flags silently mislabels a trend line.
        assert f"_tp{tp}_" in name, name
        shape = f"_in{parameters[input_key]}_out{parameters[output_key]}"
        assert name.endswith(shape), name

        # The in64_out64 entries are a short shape at max-model-len 128; every
        # other splits max-model-len evenly between prompt and output.
        if shape == "_in64_out64":
            continue
        half = parameters["max-model-len"] // 2
        assert parameters[input_key] == parameters[output_key] == half, name
