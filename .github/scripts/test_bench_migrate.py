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

"""Every `vllm bench` entry the runner can reach has a Bench definition that keeps its benchmark id.

The legacy configs under `vllm-benchmarks/benchmarks/spyre/` are the source of truth for what
runs; the definitions under `benchmarks/` must name each entry, hash to the id the existing ingest
writes for its test name, and run the same entry through the same runner. Id comparisons need the
ingest package; the engine-backed ones need the engine as well.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks"
CONFIGS = REPO / "vllm-benchmarks" / "benchmarks" / "spyre"
SCRIPTS = Path(__file__).resolve().parent
KINDS = ("latency", "throughput", "serve")
WRAPPER = ["python3", ".github/scripts/bench_perf_vllm.py", "run"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load("run_vllm_benchmarks", SCRIPTS / "run_vllm_benchmarks.py")
perf = _load("bench_perf_vllm", SCRIPTS / "bench_perf_vllm.py")


def _need(module):
    if os.environ.get("BENCH_NIGHTLY_REQUIRE_ENGINE"):
        return importlib.import_module(module)
    return pytest.importorskip(module)


def _docs():
    out = []
    for path in sorted(BENCH.rglob("*.yaml")):
        out += [d for d in yaml.safe_load_all(path.read_text()) if d]
    return out


def _vllm_benchmarks():
    return [
        d
        for d in _docs()
        if d.get("kind") == "Benchmark"
        and (d["spec"].get("identity") or {}).get("profile") in ("vllm", "vllm-sweep")
    ]


def _reachable_entries():
    """(file name, config) of every entry the runner can select: the top-level suite files."""
    out = []
    for path in sorted(CONFIGS.glob("*-tests*.yaml")):
        out += [(path.name, c) for c in runner._load_configs(path) or []]
    return out


def _producer_entry(test_name):
    vllm = _need("spyre_clickhouse_ingest.vllm")
    row = {
        "extra": json.dumps({"test_name": test_name, "device": "spyre"}),
        "metric": "avg_latency",
        "actual": 1.0,
    }
    return vllm.bench_entries([row])[0]


def _by_identity():
    return {b["spec"]["identity"]["name"]: b for b in _vllm_benchmarks()}


def test_every_reachable_entry_has_one_identity_benchmark():
    ours = _by_identity()
    assert len(ours) == len(_vllm_benchmarks()), "two Benchmarks share an ingest name"
    names = {c["test_name"].lower() for _, c in _reachable_entries()}
    assert sorted(names - set(ours)) == []
    assert sorted(set(ours) - names) == [], "a definition names no runnable entry"


def test_the_variant_is_the_ingest_name_and_its_environment_matches_tp():
    envs = {d["metadata"]["name"]: d for d in _docs() if d.get("kind") == "Environment"}
    for b in _vllm_benchmarks():
        (variant,) = b["spec"]["variants"]
        assert variant["name"] == b["spec"]["identity"]["name"]
        cards = envs[variant["environment"]]["spec"]["devices"]["requests"][0]["count"]
        assert cards == variant["system"]["tp"], variant["name"]


def test_identity_blocks_match_the_ingest_producer():
    checked = 0
    for b in _vllm_benchmarks():
        blk = b["spec"]["identity"]
        entry = _producer_entry(blk["name"])
        profile = "vllm-sweep" if len(entry["disc_keys"]) > 5 else "vllm"
        assert blk["profile"] == profile, blk["name"]
        assert [d["key"] for d in blk["discriminators"]] == list(entry["disc_keys"]), blk["name"]
        got = {d["key"]: str(d["value"]) for d in blk["discriminators"]}
        assert got == {k: str(entry["disc"].get(k, "")) for k in entry["disc_keys"]}, blk["name"]
        assert entry["tags"] == [] and not blk.get("tags")
        checked += 1
    assert checked >= 40


def test_stored_ids_equal_the_ingest_producer_ids():
    engine = _need("spyre_bench.engine")
    identity = _need("spyre_clickhouse_ingest.identity")
    cat = engine.Catalog([str(BENCH)], [], [])
    assert not cat.errors, cat.errors
    ours = {c["name"]: c for c in cat.cases}
    for _, config in _reachable_entries():
        entry = _producer_entry(config["test_name"])
        theirs = identity.BenchmarkId.derive(
            "spyre-inference", entry["name"], entry["tags"], entry["disc"], entry["disc_keys"]
        )
        stored = cat.ids(ours[config["test_name"].lower()])["benchmark_id"]
        assert str(theirs) == stored, config["test_name"]


def _dataset_for(var):
    docs = _docs()
    asset = next(
        d["metadata"]["name"]
        for d in docs
        if d.get("kind") == "Asset"
        and (d["metadata"].get("annotations") or {}).get("consumer.env") == var
    )
    return next(
        d["metadata"]["name"]
        for d in docs
        if d.get("kind") == "Dataset" and d["spec"]["asset"] == asset
    )


def test_each_definition_keeps_the_entry_parameters():
    ours = _by_identity()
    for fname, c in _reachable_entries():
        name = c["test_name"].lower()
        (variant,) = ours[name]["spec"]["variants"]
        sysd, wl = variant["system"], variant["workload"]
        if fname.startswith(("latency", "throughput")):
            p = c["parameters"]
            assert sysd["tp"] == int(p["tensor_parallel_size"]), name
            assert sysd["model"] == p["model"].split("/")[-1].lower(), name
            assert sysd["flags"]["max-model-len"] == p["max-model-len"], name
            if fname.startswith("latency"):
                assert wl["shape"]["input_len"] == p["input-len"], name
                assert wl["shape"]["output_len"] == p["output-len"], name
                assert wl["load"]["iterations"] == p["num-iters"], name
            else:
                assert wl["input_len"] == p["random-input-len"], name
                assert wl["prompts"] == p["num-prompts"], name
            continue
        sp, p = c["server_parameters"], c["parameters"]
        assert sysd["tp"] == sp["tensor-parallel-size"], name
        assert sysd["model"] == sp["model"].split("/")[-1].lower(), name
        assert sysd["flags"]["max-model-len"] == sp["max-model-len"], name
        assert wl["load"]["concurrency"] == p["max-concurrency"], name
        assert wl["load"]["requests"] == p["num-prompts"], name
        path = p.get("dataset-path", "")
        if path.startswith("${"):
            assert [i["dataset"] for i in wl["inputs"]] == [_dataset_for(path[2:-1])], name
        else:
            assert "inputs" not in wl, name


def _opt_of(argv):
    return lambda flag: argv[argv.index(flag) + 1] if flag in argv else ""


def test_each_command_selects_its_own_entry():
    """The suite and bench type in the argv reach a config file that holds the test name."""
    checked = 0
    for b in _vllm_benchmarks():
        argv = b["spec"]["run"]["command"]
        assert argv[:3] == WRAPPER and "--" in argv, b["metadata"]["name"]
        inner = argv[argv.index("--") + 1 :]
        opt = _opt_of(inner)
        assert opt("--tests") == "{case}" and opt("--results-dir") == "{raw_dir}"
        kind, suite = opt("--bench-types"), opt("--suite")
        assert kind in KINDS and opt("--configs-dir") == "vllm-benchmarks/benchmarks/spyre"
        files = runner._suite_config_files(CONFIGS, suite, {kind})
        if kind == "serve":
            files["mm"] = CONFIGS / "multimodal-serve-tests.yaml"
        known = {
            c["test_name"].lower() for f in files.values() for c in runner._load_configs(f) or []
        }
        assert b["spec"]["identity"]["name"] in known, b["metadata"]["name"]
        assert b["spec"]["run"]["repeats"] == 1
        checked += 1
    assert checked >= 40


def test_entries_no_runner_option_reaches_are_not_defined():
    """Config folders no runner option reaches (power use case, per-model serve) stay undefined."""
    reachable = {p.name for p in CONFIGS.glob("*-tests*.yaml")}
    for sub in ("power_use_case", "serve-tests", "serve_tests"):
        for path in (CONFIGS / sub).rglob("*.yaml"):
            assert path.name not in reachable, path
            tests = (yaml.safe_load(path.read_text()) or {}).get("tests") or []
            assert tests, path


def test_perf_extended_selects_every_perf_case_but_the_quality_ones():
    docs = _docs()
    sel = next(
        d for d in docs if d.get("kind") == "Selection" and d["metadata"]["name"] == "perf-extended"
    )
    assert sel["spec"]["selector"]["groupings"]["tier"] == {"at_most": "extended"}
    tiers = {b["spec"]["tier"] for b in _vllm_benchmarks()}
    assert tiers <= {"sanity", "core", "extended"}


# ---------------------------------------------------------------- recorder


def test_latency_record_becomes_the_batch_latency_sample():
    assert perf.samples_of({"avg_latency": 0.5, "percentiles": {"99": 1.0}}) == {
        "llm.batch_latency.mean": [0.5]
    }


def test_throughput_record_keeps_only_the_request_rate():
    got = perf.samples_of(
        {"requests_per_second": 2.0, "tokens_per_second": 99.0, "elapsed_time": 8}
    )
    assert got == {"llm.request_throughput": [2.0]}


def test_serve_record_converts_milliseconds_to_seconds():
    got = perf.samples_of(
        {
            "request_throughput": 3.0,
            "output_throughput": 40.0,
            "mean_ttft_ms": 250.0,
            "p99_ttft_ms": 900.0,
            "median_itl_ms": 20.0,
            "p99_e2el_ms": 5000.0,
            "mean_tpot_ms": 21.0,
        }
    )
    assert got["llm.ttft.mean"] == [0.25] and got["llm.e2el.p99"] == [5.0]
    assert got["llm.output_throughput"] == [40.0] and "llm.tpot.mean" not in got


def test_nonfinite_and_boolean_values_are_dropped():
    assert perf.samples_of({"avg_latency": float("nan")}) == {}
    assert perf.samples_of({"request_throughput": True}) == {}


def _run_wrapper(tmp_path, body, case="latency_demo_tp1_in8_out8"):
    script = tmp_path / "fake_runner.py"
    script.write_text(body)
    raw, rep = tmp_path / "raw", tmp_path / "rep"
    raw.mkdir()
    cmd = [
        sys.executable, str(SCRIPTS / "bench_perf_vllm.py"), "run", "--case", case.lower(),
        "--repeat-dir", str(rep), "--raw-dir", str(raw),
        "--", sys.executable, str(script), str(raw), case,
    ]  # fmt: skip
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env={**os.environ, "BENCH_QUIET": "1"}
    )
    return proc, rep / "metrics.json"


def test_wrapper_runs_the_runner_then_writes_the_sidecar(tmp_path):
    body = (
        "import json, sys\n"
        "doc = {'avg_latency': 0.25, 'latencies': [0.25]}\n"
        "json.dump(doc, open(sys.argv[1] + '/' + sys.argv[2] + '.json', 'w'))\n"
    )
    proc, sidecar = _run_wrapper(tmp_path, body, "latency_Demo_tp1_in8_out8")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(sidecar.read_text()) == {"samples": {"llm.batch_latency.mean": [0.25]}}


def test_wrapper_passes_the_runner_failure_through_and_writes_nothing(tmp_path):
    proc, sidecar = _run_wrapper(tmp_path, "import sys\nsys.exit(7)\n")
    assert proc.returncode == 7 and not sidecar.exists()


def test_wrapper_fails_when_the_runner_left_no_result(tmp_path):
    proc, sidecar = _run_wrapper(tmp_path, "pass\n")
    assert proc.returncode == 2 and "left no" in proc.stderr and not sidecar.exists()


def test_wrapper_fails_when_the_result_has_no_registered_metric(tmp_path):
    body = "import sys\nopen(sys.argv[1] + '/' + sys.argv[2] + '.json', 'w').write('{\"x\": 1}')\n"
    proc, _ = _run_wrapper(tmp_path, body)
    assert proc.returncode == 2 and "no registered metric" in proc.stderr


# ---------------------------------------------------------------- engine


def test_engine_run_records_samples_for_a_latency_a_throughput_and_a_serve_case(tmp_path):
    """Real engine: each kind's command, with the runner swapped for one writing a native result."""
    exe = _engine_cli()
    fake = tmp_path / "fake_runner.py"
    fake.write_text(
        "import json, sys\n"
        "raw, case = sys.argv[1], sys.argv[2]\n"
        "doc = {'latency': {'avg_latency': 0.5},\n"
        "       'throughput': {'requests_per_second': 2.0, 'tokens_per_second': 9.0},\n"
        "       'serve': {'request_throughput': 3.0, 'output_throughput': 40.0,\n"
        "                 'mean_ttft_ms': 250.0,\n"
        "                 'p99_ttft_ms': 900.0, 'median_itl_ms': 20.0}}[case.split('_')[0]]\n"
        "json.dump(doc, open(f'{raw}/{case}.json', 'w'))\n"
    )
    cat = tmp_path / "cat"
    shutil_copy = __import__("shutil").copytree
    shutil_copy(BENCH, cat)
    cases = [
        "latency_granite33-8b_tp1_in1024_out1024",
        "throughput_granite33-8b_tp1_in1024_out1024",
    ]
    cases += ["serve_granite33-8b_tp1_cics"]
    for path in cat.rglob("*.yaml"):
        text = path.read_text()
        if "bench_perf_vllm.py" not in text:
            continue
        doc = yaml.safe_load(text)
        if doc.get("kind") != "Benchmark" or doc["spec"]["identity"]["name"] not in cases:
            continue
        run = doc["spec"]["run"]
        i = run["command"].index("--")
        run["command"] = [
            *run["command"][: i + 1],
            sys.executable,
            str(fake),
            "{raw_dir}",
            "{case}",
        ]
        path.write_text(yaml.safe_dump(doc))
    env = {**os.environ, "BENCH_PURPOSE": "on-demand", "BENCH_REUSE": "never"}
    plan = tmp_path / "plan.json"
    made = subprocess.run(
        [
            exe, "plan", "--catalog", str(cat), "--cases", ",".join(cases), "--out", str(plan),
            "--arch", "x86_64", "--subject-id", "9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2",
        ],
        capture_output=True, text=True, env=env, cwd=REPO,
    )  # fmt: skip
    assert made.returncode == 0, made.stderr
    out = tmp_path / "out"
    ran = subprocess.run(
        [exe, "run", "--plan", str(plan), "--catalog", str(cat), "--out", str(out)],
        capture_output=True, text=True, env=env, cwd=REPO,
    )  # fmt: skip
    assert ran.returncode == 0, ran.stdout + ran.stderr
    results = {
        json.loads(p.read_text())["status"]["case"]["name"]: json.loads(p.read_text())
        for p in out.rglob("result.json")
    }
    assert sorted(results) == sorted(cases)
    lat = results[cases[0]]["status"]["repeats"][0]["samples"]
    assert lat == {"llm.batch_latency.mean": [0.5]}
    the = results[cases[1]]["status"]["repeats"][0]["samples"]
    assert the == {"llm.request_throughput": [2.0]}
    srv = results[cases[2]]["status"]["repeats"][0]["samples"]
    assert srv["llm.ttft.p99"] == [0.9] and srv["llm.output_throughput"] == [40.0]


def _engine_cli():
    import shutil

    exe = shutil.which("spyre-bench")
    if exe is None:
        if os.environ.get("BENCH_NIGHTLY_REQUIRE_ENGINE"):
            pytest.fail("spyre-bench is not installed")
        pytest.skip("spyre-bench is not installed")
    helps = subprocess.run([exe, "--help"], capture_output=True, text=True).stdout
    if "plan " not in helps or "run " not in helps:
        pytest.skip("the engine has no plan and run verbs")
    return exe


# ---------------------------------------------------------------- bench bundle (side by side)


def _workflow():
    return yaml.safe_load((REPO / ".github/workflows/bench-on-demand.yml").read_text())


def test_workflow_bundle_steps_are_advisory_and_off_by_default():
    wf = _workflow()
    inputs = wf[True]["workflow_dispatch"]["inputs"]
    assert inputs["bench_mode"]["default"] == "off"
    assert inputs["bench_mode"]["options"] == ["off", "side-by-side"]
    assert wf["jobs"]["leg"]["env"]["BENCH_MODE"] == "${{ inputs.bench_mode || 'off' }}"
    steps = {s["name"]: s for s in wf["jobs"]["leg"]["steps"] if "name" in s}
    dev, prod = steps["Record the bench results (dev)"], steps["Record the bench results (prod)"]
    for step, secret in ((dev, "_DEV"), (prod, "")):
        assert step["continue-on-error"] is True
        assert "env.BENCH_MODE == 'side-by-side'" in step["if"]
        assert step["uses"].startswith(
            "torch-spyre/torch-spyre/.github/actions/ingest-xml-to-clickhouse@"
        )
        assert step["with"]["clickhouse-host"] == "${{ secrets.CLICKHOUSE_HOST%s }}" % secret
    assert "!= 'prod'" in dev["if"] and "== 'prod'" in prod["if"]


def test_the_bench_bundle_step_names_no_workflow_and_trusts_ci_only_from_main():
    wf = _workflow()
    for step in wf["jobs"]["leg"]["steps"]:
        if not step.get("name", "").startswith("Record the bench results"):
            continue
        with_ = step["with"]
        assert with_["bench-dir"] == "results/bench" and with_["trigger-type"] == "perf"
        assert "workflow" not in with_, (
            "the legacy v2 rows have no source job; the series is the id"
        )
        assert with_["channel-trust"].startswith("${{ github.ref == 'refs/heads/main' && ")
        assert "inputs.engine_ref == 'main'" in with_["channel-trust"]
        assert "xml-dir" not in with_ and "vllm-results-dir" not in with_


def test_the_legacy_ingest_steps_are_unchanged_by_the_bundle():
    wf = _workflow()
    steps = {s["name"]: s for s in wf["jobs"]["leg"]["steps"] if "name" in s}
    for name in ("Ingest the results (dev)", "Ingest the results (prod)"):
        assert steps[name]["uses"] == "./.github/actions/ingest-vllm-benchmarks"
        assert steps[name]["with"]["results-dir"] == "results/bench/native"
        assert "continue-on-error" not in steps[name]


def test_workflow_rejects_side_by_side_without_the_engine_runner_and_a_subject():
    checks = next(s for s in _workflow()["jobs"]["plan"]["steps"] if s.get("id") == "mode")["run"]
    assert "side-by-side" in checks and "needs runner: engine and a subject_id" in checks
    assert "subject_id is a lower-case UUID" in checks


def test_pooling_zero_output_throughput_is_not_recorded():
    got = perf.samples_of({"request_throughput": 3.0, "output_throughput": 0.0})
    assert got == {"llm.request_throughput": [3.0]}
