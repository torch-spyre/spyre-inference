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
`dataset-path` expands from the environment, and a selected test whose dataset is
absent aborts the run instead of being skipped.

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

_CONFIGS = _REPO / "vllm-benchmarks" / "benchmarks" / "spyre"
SERVE_TESTS = _CONFIGS / "serve-tests.yaml"
SERVE_DEV_TESTS = _CONFIGS / "serve-tests-dev.yaml"
LATENCY_TESTS = _CONFIGS / "latency-tests.yaml"
THROUGHPUT_TESTS = _CONFIGS / "throughput-tests.yaml"
LATENCY_DEV_TESTS = _CONFIGS / "latency-tests-dev.yaml"
THROUGHPUT_DEV_TESTS = _CONFIGS / "throughput-tests-dev.yaml"


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


@pytest.mark.parametrize("template", ["${SPYRE_VISION_DATASET}", "$SPYRE_VISION_DATASET"])
def test_dataset_var_without_default_skips_the_entry(monkeypatch, tmp_path, caplog, template):
    """A var nothing defines (SPYRE_VISION_DATASET, #1538) skips, not aborts, the run."""
    monkeypatch.delenv("SPYRE_VISION_DATASET", raising=False)
    monkeypatch.setenv("SPYRE_BENCH_DATA_DIR", str(tmp_path))
    config = _serve_config(template)
    runner._derive_config(config)

    with caplog.at_level("WARNING", logger=runner.log.name):
        assert runner._select_configs([config], set(), set()) == []

    assert "SPYRE_VISION_DATASET" in caplog.text


def test_unrecognised_dataset_var_stays_fatal(monkeypatch, tmp_path):
    """A typo like ${SPYRE_CICS_DATASET} is not an optional var: it aborts."""
    monkeypatch.setenv("SPYRE_BENCH_DATA_DIR", str(tmp_path))
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


@pytest.mark.parametrize(
    ("suite", "expected"),
    [("", "latency-tests.yaml"), ("dev", "latency-tests-dev.yaml")],
)
def test_suite_selects_config_file(tmp_path, suite, expected):
    files = runner._suite_config_files(tmp_path, suite, {"latency"})

    assert files == {"latency": tmp_path / expected}


def test_unknown_suite_aborts_the_run(monkeypatch, tmp_path):
    (tmp_path / "latency-tests-dev.yaml").write_text("[]")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vllm_benchmarks.py",
            "--configs-dir",
            str(tmp_path),
            "--results-dir",
            str(tmp_path / "results"),
            "--suite",
            "devs",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def test_matrix_skips_dev_configs(tmp_path):
    spyre = tmp_path / "spyre"
    spyre.mkdir()
    for name, model in [
        ("latency-tests.yaml", "ci-model"),
        ("latency-tests-dev.yaml", "dev-model"),
    ]:
        (spyre / name).write_text(
            yaml.safe_dump([{"test_name": "t", "parameters": {"model": model}}])
        )

    result = matrix.generate_benchmark_matrix(str(tmp_path), [])

    assert result["include"] == [{"models": "ci-model"}]


# ── CI legs ──


def _offline(model: str, tp: int) -> dict:
    return {"test_name": f"t_{tp}", "parameters": {"model": model, "tensor_parallel_size": tp}}


def _legs_files(tmp_path) -> list:
    latency = tmp_path / "latency-tests.yaml"
    latency.write_text(yaml.safe_dump([_offline("Org/A", 1), _offline("org/a", 4)]))
    serve = _write(
        tmp_path,
        {
            "tests": [
                _serve_config("x", server_parameters={"model": "Org/A", "tensor-parallel-size": 1}),
                _serve_config("x", server_parameters={"model": "org/b", "tensor-parallel-size": 2}),
            ]
        },
    )
    return [("latency", latency), ("serve", serve), ("serve", tmp_path / "absent.yaml")]


def test_legs_split_by_model_and_tp_on_enough_cards(tmp_path):
    legs = runner.list_legs(_legs_files(tmp_path), set(), set())

    assert legs == [
        {"model": "Org/A", "tp": 1, "cards": 1, "key": "org-a-tp1"},
        {"model": "org/a", "tp": 4, "cards": 4, "key": "org-a-tp4"},
        {"model": "org/b", "tp": 2, "cards": 2, "key": "org-b-tp2"},
    ]


def test_legs_follow_the_model_and_tp_filters(tmp_path):
    legs = runner.list_legs(_legs_files(tmp_path), {"org/a"}, {4})

    assert [(leg["model"], leg["tp"]) for leg in legs] == [("org/a", 4)]


def test_tp_beyond_the_largest_runner_aborts_the_listing(tmp_path):
    latency = tmp_path / "latency-tests.yaml"
    latency.write_text(yaml.safe_dump([_offline("m", 8)]))

    with pytest.raises(SystemExit) as excinfo:
        runner.list_legs([("latency", latency)], set(), set())
    assert excinfo.value.code == 2


def test_every_ci_config_lands_on_a_leg_with_enough_cards():
    legs = runner.list_legs(
        [("latency", LATENCY_TESTS), ("throughput", THROUGHPUT_TESTS), ("serve", SERVE_TESTS)],
        set(),
        set(),
    )
    cards = {(leg["model"].lower(), leg["tp"]): leg["cards"] for leg in legs}
    assert len({leg["key"] for leg in legs}) == len(legs)

    for config_file in (LATENCY_TESTS, THROUGHPUT_TESTS, SERVE_TESTS):
        for config in runner._load_configs(config_file):
            tp = runner._config_tp(config)
            assert cards[(runner._config_model(config).lower(), tp)] >= tp, config["test_name"]


# ── pinned revisions ──


def test_cache_config_pins_are_read():
    pins = runner._pinned_revisions()

    assert pins["google/gemma-4-26b-a4b"] == "24548b62aa021d562695c04aaf7758a1ea47990b"
    assert "ibm-ai-platform/micro-g3.3-8b-instruct-1b" not in pins


def test_pinned_serve_model_runs_at_its_cached_revision(monkeypatch):
    monkeypatch.setattr(runner, "_local_snapshot", lambda model, rev: f"/snap/{model}/{rev}")
    config = _serve_config("x", server_parameters={"model": "Org/M", "tensor-parallel-size": 1})
    runner._derive_config(config)

    runner._pin_revision(config, {"org/m": "abc"})

    assert config["server_parameters"]["revision"] == "abc"
    # bench serve has no revision flag, so its tokenizer is the local snapshot.
    assert config["parameters"]["tokenizer"] == "/snap/Org/M/abc"


def test_pinned_offline_model_keeps_an_explicit_revision(monkeypatch):
    monkeypatch.setattr(runner, "_local_snapshot", lambda model, rev: f"/snap/{rev}")
    config = _offline("org/m", 1)
    config["parameters"]["revision"] = "mine"

    runner._pin_revision(config, {"org/m": "abc"})

    assert config["parameters"]["revision"] == "mine"
    assert config["parameters"]["tokenizer"] == "/snap/mine"


def test_unpinned_or_uncached_model_is_left_alone(monkeypatch):
    monkeypatch.setattr(runner, "_local_snapshot", lambda model, rev: None)
    unpinned, uncached = _offline("org/x", 1), _offline("org/m", 1)

    runner._pin_revision(unpinned, {"org/m": "abc"})
    runner._pin_revision(uncached, {"org/m": "abc"})

    assert unpinned["parameters"] == _offline("org/x", 1)["parameters"]
    assert uncached["parameters"]["revision"] == "abc"
    assert "tokenizer" not in uncached["parameters"]


# ── JUnit ──


def test_junit_holds_one_case_per_benchmark(tmp_path):
    import xml.etree.ElementTree as ET

    path = tmp_path / "out" / "junit.xml"
    runner.write_junit(
        path,
        [
            runner.BenchResult("latency", "lat_a", "org/a", 1.5),
            runner.BenchResult("serve", "srv_b", "", 2.0, "exit code 1\n\x1b[31mboom\x1b[0m"),
        ],
    )

    suite = ET.parse(path).getroot().find("testsuite")
    assert (suite.get("tests"), suite.get("failures")) == ("2", "1")
    passed, failed = suite.findall("testcase")
    assert (passed.get("classname"), passed.get("name")) == ("vllm_bench.latency", "lat_a")
    assert passed.find("failure") is None
    tags = [p.get("value") for p in passed.iter("property")]
    assert "testtype__perf" in tags and "model__org/a" in tags
    failure = failed.find("failure")
    assert failure.get("message") == "exit code 1"
    assert failure.text == "exit code 1\n[31mboom[0m"


def test_main_writes_junit_and_fails_on_a_failed_benchmark(monkeypatch, tmp_path):
    import xml.etree.ElementTree as ET

    (tmp_path / "latency-tests.yaml").write_text(
        yaml.safe_dump(
            [{**_offline("m", 1), "test_name": "ok"}, {**_offline("m", 1), "test_name": "bad"}]
        )
    )
    monkeypatch.setattr(
        runner, "run_benchmark", lambda **kw: "exit code 1" if kw["test_name"] == "bad" else None
    )
    junit = tmp_path / "junit.xml"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vllm_benchmarks.py",
            "--configs-dir",
            str(tmp_path),
            "--results-dir",
            str(tmp_path / "results"),
            "--bench-types",
            "latency",
            "--junit-xml",
            str(junit),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        runner.main()

    assert exc.value.code == 1
    cases = ET.parse(junit).getroot().iter("testcase")
    assert {c.get("name"): c.find("failure") is not None for c in cases} == {
        "ok": False,
        "bad": True,
    }


# ── the checked-in config ──

# Each trace is replayed at the one max-model-len that fits its requests, and is
# reached through its own env var.
DATASET_CONTEXT_LEN = {"aiops": 4096, "cics": 8192, "all": 32768}
DATASET_PATH_VARS = {
    "aiops": "${SPYRE_AIOPS_DATASET}",
    "cics": "${SPYRE_CICS_DATASET}",
    "all": "${SPYRE_ALL_SEQUENCES_DATASET}",
}


@pytest.mark.parametrize("config_file", [SERVE_TESTS, SERVE_DEV_TESTS])
def test_serve_tests_yaml_derives_consistently(config_file):
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

        parameters = config["parameters"]
        if not parameters["dataset-name"].startswith("random"):
            # The name's suffix has to say which trace the entry replays, and
            # each trace needs a max-model-len that fits its requests: longer
            # ones are rejected.
            trace = config["test_name"].rsplit("_", 1)[1]
            assert trace in DATASET_CONTEXT_LEN, config["test_name"]
            # _select_configs expands the env var, so at load time it is still
            # the literal the entry names.
            assert parameters["dataset-path"] == DATASET_PATH_VARS[trace]
            assert config["server_parameters"]["max-model-len"] == DATASET_CONTEXT_LEN[trace]
        else:
            # Generated prompts need no staged file, and a request over the
            # server's context is rejected.
            assert "dataset-path" not in parameters, config["test_name"]
            assert parameters["random-input-len"] <= config["server_parameters"]["max-model-len"], (
                config["test_name"]
            )


def test_serve_dev_tests_yaml_extends_serve_tests_yaml():
    """The dev serve file holds every CI serve entry unchanged, plus only `_all` ones."""
    ci = {c["test_name"]: c for c in runner._load_configs(SERVE_TESTS)}
    dev = {c["test_name"]: c for c in runner._load_configs(SERVE_DEV_TESTS)}

    for name, config in ci.items():
        assert dev.get(name) == config, name
    extra = dev.keys() - ci.keys()
    assert extra
    assert all(name.endswith("_all") for name in extra), sorted(extra)


# The offline files spell input/output length under their own flag names.
OFFLINE_LEN_KEYS = {
    LATENCY_TESTS: ("input-len", "output-len"),
    THROUGHPUT_TESTS: ("random-input-len", "random-output-len"),
    LATENCY_DEV_TESTS: ("input-len", "output-len"),
    THROUGHPUT_DEV_TESTS: ("random-input-len", "random-output-len"),
}


@pytest.mark.parametrize("config_file", list(OFFLINE_LEN_KEYS))
def test_offline_tests_yaml_derives_consistently(config_file):
    """Every offline entry derives its devices, and names the shape it runs."""
    input_key, output_key = OFFLINE_LEN_KEYS[config_file]
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
        assert name.endswith((shape, f"{shape}_smoke")), name

        # A `_smoke` entry is its own short shape; every other splits
        # max-model-len evenly between prompt and output.
        if name.endswith("_smoke"):
            continue
        half = parameters["max-model-len"] // 2
        assert parameters[input_key] == parameters[output_key] == half, name
