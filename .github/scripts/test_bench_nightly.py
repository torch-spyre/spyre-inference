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

"""The nightly bench route's definitions against the real ingest producer.

`benchmarks/decoder/` declares `identity` blocks that must hash to the ids that
`spyre_clickhouse_ingest.vllm.bench_entries` writes for the same test names. The id comparison
needs the engine; the rest needs only the ingest package, which the ingest-script-tests job
already installs.
"""

import importlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks"
CONFIGS = REPO / "vllm-benchmarks" / "benchmarks" / "spyre"
COMPONENT = "spyre-inference"

_spec = importlib.util.spec_from_file_location(
    "bench_nightly_cases", Path(__file__).with_name("bench_nightly_cases.py")
)
cases_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cases_mod)


def _docs(*parts):
    out = []
    for path in (
        sorted(BENCH.joinpath(*parts).rglob("*.yaml")) if parts else sorted(BENCH.rglob("*.yaml"))
    ):
        out += [d for d in yaml.safe_load_all(path.read_text()) if d]
    return out


def _by_kind(kind):
    return [d for d in _docs() if d.get("kind") == kind]


def _identity_benchmarks():
    """The nightly Benchmarks; test_bench_migrate.py covers the rest of the vllm-profile ones."""
    nightly = set(_nightly_cases())
    return [
        b
        for b in _by_kind("Benchmark")
        if (b["spec"].get("identity") or {}).get("mode") == "ingest"
        and b["spec"]["identity"].get("profile") == "vllm"
        and b["spec"]["identity"]["name"] in nightly
    ]


def _nightly_cases():
    sel = next(s for s in _by_kind("Selection") if s["metadata"]["name"] == "nightly")
    return sel["spec"]["cases"]


def _need(module):
    """importorskip, except that BENCH_NIGHTLY_REQUIRE_ENGINE=1 makes a missing module a failure."""
    if os.environ.get("BENCH_NIGHTLY_REQUIRE_ENGINE"):
        return importlib.import_module(module)
    return pytest.importorskip(module)


def _unavailable(reason):
    if os.environ.get("BENCH_NIGHTLY_REQUIRE_ENGINE"):
        pytest.fail(reason)
    pytest.skip(reason)


def _producer_entry(test_name):
    vllm = _need("spyre_clickhouse_ingest.vllm")
    row = {
        "extra": json.dumps({"test_name": test_name, "device": "spyre"}),
        "metric": "avg_latency",
        "actual": 1.0,
    }
    return vllm.bench_entries([row])[0]


@pytest.fixture(scope="module")
def nightly_cases():
    return cases_mod.load_cases(CONFIGS)


def test_selection_is_the_nightly_config_names(nightly_cases):
    assert sorted(_nightly_cases()) == sorted(c["case"] for c in nightly_cases)


def test_every_selected_case_has_one_identity_benchmark():
    names = [b["spec"]["identity"]["name"] for b in _identity_benchmarks()]
    assert sorted(set(_nightly_cases()) - set(names)) == []
    assert len(names) == len(set(names))


@pytest.mark.parametrize("kind", cases_mod.KINDS)
def test_shared_entries_keep_the_ci_parameters(kind):
    ci = {e["test_name"]: e for e in yaml.safe_load((CONFIGS / f"{kind}-tests.yaml").read_text())}
    nightly = {
        e["test_name"]: e
        for e in yaml.safe_load((CONFIGS / f"{kind}-tests-nightly.yaml").read_text())
    }
    shared = set(ci) & set(nightly)
    assert shared, "the nightly suite must reuse at least one CI entry"
    for name in shared:
        assert nightly[name] == ci[name], name


def test_companion_differs_from_its_pair_only_in_output_len():
    entries = {
        e["test_name"]: e
        for e in yaml.safe_load((CONFIGS / "latency-tests-nightly.yaml").read_text())
    }
    for name, e in entries.items():
        if not name.endswith("_out1"):
            continue
        pair = entries[name[: -len("out1")] + "out64"]
        assert e["parameters"] == {**pair["parameters"], "output-len": 1}
        assert e["environment_variables"] == pair["environment_variables"]


def test_identity_blocks_match_the_ingest_producer():
    """Discriminator keys, order and values equal what the real producer hashes for the name."""
    checked = 0
    for b in _identity_benchmarks():
        blk = b["spec"]["identity"]
        entry = _producer_entry(blk["name"])
        assert blk["profile"] == "vllm"
        assert [d["key"] for d in blk["discriminators"]] == list(entry["disc_keys"]), blk["name"]
        got = {d["key"]: str(d["value"]) for d in blk["discriminators"]}
        assert got == {k: str(entry["disc"].get(k, "")) for k in entry["disc_keys"]}, blk["name"]
        assert entry["tags"] == [] and not blk.get("tags")
        checked += 1
    assert checked >= len(_nightly_cases())


def test_variant_name_is_the_ingest_name_and_environment_matches_tp():
    envs = {e["metadata"]["name"]: e for e in _by_kind("Environment")}
    for b in _identity_benchmarks():
        (variant,) = b["spec"]["variants"]
        assert variant["name"] == b["spec"]["identity"]["name"]
        cards = envs[variant["environment"]]["spec"]["devices"]["requests"][0]["count"]
        assert cards == variant["system"]["tp"]


def test_every_nightly_benchmark_has_a_cost_hint():
    for b in _identity_benchmarks():
        assert b["spec"].get("est_hours", 0) > 0, b["metadata"]["name"]


def test_pinning_defaults_to_none_and_is_a_series_label():
    envs = _by_kind("Environment")
    assert envs
    for e in envs:
        policy = e["spec"]["host"]["cpu_pinning"]["policy"]
        assert policy == (
            "device_local" if e["metadata"]["name"].endswith("-device-local") else "none"
        )
    project = next(d for d in _by_kind("Project"))
    assert "env.label.cpu_pinning" in project["spec"]["series_labels"]
    assert project["spec"]["default_environment"] in {
        e["metadata"]["name"] for e in envs if "device-local" not in e["metadata"]["name"]
    }


def test_decode_per_token_derive_uses_the_declared_pair():
    analysis = next(a for a in _by_kind("Analysis") if a["metadata"]["name"] == "decode-per-token")
    derive = analysis["spec"]["derive"]
    cases = set(_nightly_cases())
    assert derive["op"] == "slope"
    assert {derive["cases"]["first"], derive["cases"]["second"]} <= cases
    first, second = derive["cases"]["first"], derive["cases"]["second"]
    assert derive["x"] == {
        "first": int(first.rsplit("out", 1)[1]),
        "second": int(second.rsplit("out", 1)[1]),
    }


def test_leg_cases_group_by_model_and_tp(nightly_cases):
    legs = {(c["model"], c["tp"]) for c in nightly_cases}
    assert len(legs) == 3
    grouped = [
        n for model, tp in sorted(legs) for n in cases_mod.leg_cases(nightly_cases, model, tp)
    ]
    assert sorted(grouped) == sorted(_nightly_cases())
    assert cases_mod.leg_cases(nightly_cases, "no/such-model", 1) == []


def test_leg_cases_cli_fails_on_an_unknown_leg(capsys):
    assert (
        cases_mod.main(["--configs-dir", str(CONFIGS), "--model", "no/such-model", "--tp", "1"])
        == 2
    )
    assert (
        cases_mod.main(
            ["--configs-dir", str(CONFIGS), "--model", "google/gemma-4-26B-A4B", "--tp", "1"]
        )
        == 2
    )
    assert "no nightly case" in capsys.readouterr().err


def test_stored_ids_equal_the_ingest_producer_ids():
    """Needs the engine: its adapter calls BenchmarkId.derive on the same producer entry."""
    engine = _need("spyre_bench.engine")
    identity = _need("spyre_clickhouse_ingest.identity")
    cat = engine.Catalog([str(BENCH)], [], [])
    if cat.errors and all("unknown key 'command'" in e for e in cat.errors):
        _unavailable("the engine predates run.command (the plan and run verbs)")
    assert not cat.errors, cat.errors
    ours = {c["name"]: c for c in cat.cases}
    for test_name in sorted({c["test_name"] for c in cases_mod.load_cases(CONFIGS)}):
        entry = _producer_entry(test_name)
        theirs = identity.BenchmarkId.derive(
            COMPONENT, entry["name"], entry["tags"], entry["disc"], entry["disc_keys"]
        )
        assert str(theirs) == cat.ids(ours[test_name.lower()])["benchmark_id"], test_name


def test_every_nightly_benchmark_runs_one_case_through_the_nightly_suite():
    for b in _identity_benchmarks():
        run = b["spec"]["run"]
        cmd = run["command"]
        assert cmd[cmd.index("--suite") + 1] == "nightly"
        assert cmd[cmd.index("--tests") + 1] == "{case}"
        assert cmd[:3] == ["python3", ".github/scripts/bench_perf_vllm.py", "run"]
        assert cmd[cmd.index("--results-dir") + 1] == "{raw_dir}"
        assert run["repeats"] == 1


FAKE_ENGINE = """#!/bin/sh
if [ "$1" = "--help" ]; then
  printf '  plan   x\\n  run   x\\n  verdict   x\\n  validate   x\\n'
  exit 0
fi
printf '%s\\n' "$*" >> "$FAKE_LOG"
while [ $# -gt 0 ]; do
  [ "$1" = "--out" ] && [ "${2##*.}" = "json" ] && echo '{}' > "$2"
  shift
done
"""


def _make(tmp_path, *args, **env):
    import os
    import shutil
    import subprocess

    if shutil.which("make") is None:
        pytest.skip("make is not installed")
    fake = tmp_path / "spyre-bench"
    fake.write_text(FAKE_ENGINE)
    fake.chmod(0o755)
    log = tmp_path / "calls.log"
    proc = subprocess.run(
        [
            "make",
            "--no-print-directory",
            f"BENCH={fake}",
            "BENCH_PREFIX=",
            f"BENCH_RESULTS_DIR={tmp_path / 'res'}",
            *args,
        ],
        cwd=REPO,
        env={**os.environ, "FAKE_LOG": str(log), **env},
        capture_output=True,
        text=True,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls


def test_bench_run_plans_then_runs_the_plan(tmp_path):
    proc, calls = _make(tmp_path, "bench-run", "BENCH_SELECTION=nightly")
    assert proc.returncode == 0, proc.stderr
    assert [c.split()[0] for c in calls] == ["plan", "run"]
    plan_file = str(tmp_path / "res" / "plan.json")
    assert f"--out {plan_file}" in calls[0]
    assert f"--plan {plan_file}" in calls[1] and "--cases" not in calls[1]


def test_bench_run_uses_a_given_plan_without_planning(tmp_path):
    proc, calls = _make(tmp_path, "bench-run", "BENCH_SELECTION=nightly", "BENCH_PLAN=given.json")
    assert proc.returncode == 0, proc.stderr
    assert [c.split()[0] for c in calls] == ["run"] and "--plan given.json" in calls[0]


def test_engine_leg_runs_only_the_legs_cases(tmp_path):
    proc, calls = _make(
        tmp_path,
        "bench-leg",
        "BENCH_RUNNER=engine",
        "BENCH_LEG_MODEL=ibm-granite/granite-3.3-8b-instruct",
        "BENCH_LEG_TP=4",
    )
    # The fake engine leaves no JSON: the collect step must fail the leg, not ingest nothing.
    assert proc.returncode != 0
    assert (
        "--cases latency_granite33-8b_tp4_in64_out64,latency_granite33-8b_tp4_in64_out1" in calls[0]
    )


def test_native_runner_refuses_a_pinning_policy(tmp_path):
    proc, calls = _make(
        tmp_path,
        "bench-native",
        "BENCH_LEG_MODEL=m",
        "BENCH_LEG_TP=1",
        "BENCH_CPU_PINNING=device_local",
    )
    assert proc.returncode != 0 and calls == []
    assert "cannot pin" in proc.stderr


def test_verdict_needs_a_run_id_and_passes_it_quietly(tmp_path):
    proc, calls = _make(tmp_path, "bench-verdict")
    assert proc.returncode != 0 and calls == []
    proc, calls = _make(tmp_path, "bench-verdict", "BENCH_VERDICT_RUN_ID=abc")
    assert proc.returncode == 0, proc.stderr
    assert calls == ["verdict --run-id abc --store clickhouse --quiet"]


def test_missing_engine_fails_with_the_install_command(tmp_path):
    proc, _ = _make(
        tmp_path, "bench-verdict", "BENCH_VERDICT_RUN_ID=abc", "BENCH=/nonexistent/spyre-bench"
    )
    assert proc.returncode == 2
    assert "Install spyre-bench" in proc.stderr and "uv pip install" in proc.stderr


def test_missing_engine_is_a_notice_only_when_optional_and_not_required(tmp_path):
    args = ("bench-verdict", "BENCH_VERDICT_RUN_ID=abc", "BENCH=/nonexistent/spyre-bench")
    proc, _ = _make(tmp_path, *args, "BENCH_OPTIONAL=1")
    assert proc.returncode == 0 and "skipped 'verdict' (BENCH_OPTIONAL=1)" in proc.stderr
    proc, _ = _make(tmp_path, *args, "BENCH_OPTIONAL=1", "BENCH_REQUIRED=1")
    assert proc.returncode == 2


def test_the_validate_hook_skips_without_the_engine():
    import yaml

    hooks = yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text())["repos"]
    entry = next(h["entry"] for r in hooks for h in r["hooks"] if h["id"] == "bench-validate")
    assert "BENCH_OPTIONAL=1" in entry


def test_stub_leg_lists_cases_and_fails_on_an_unknown_leg(tmp_path):
    stub = ("bench-leg", "BENCH_RUNNER=stub", "BENCH_LEG_TP=4")
    ok, _ = _make(tmp_path, *stub, "BENCH_LEG_MODEL=ibm-granite/granite-3.3-8b-instruct")
    assert ok.returncode == 0 and "latency_granite33-8b_tp4_in64_out1" in ok.stdout
    bad, _ = _make(tmp_path, *stub, "BENCH_LEG_MODEL=no/such-model")
    assert bad.returncode != 0


def _engine_cli():
    import shutil
    import subprocess

    exe = shutil.which("spyre-bench")
    if exe is None:
        _unavailable("spyre-bench is not installed")
    helps = subprocess.run([exe, "--help"], capture_output=True, text=True).stdout
    if "plan " not in helps or "run " not in helps:
        _unavailable("the engine has no plan and run verbs")
    return exe


def test_engine_plan_renders_one_runner_command_per_case(tmp_path):
    """Real engine: plan the Selection and read plan.json; the dry run must succeed."""
    import subprocess

    exe = _engine_cli()
    plan = tmp_path / "plan.json"
    env = {**os.environ, "BENCH_PURPOSE": "nightly", "BENCH_REUSE": "never"}
    plan_cmd = [exe, "plan", "--catalog", str(BENCH), "--select", "nightly", "--out", str(plan)]
    made = subprocess.run(plan_cmd, capture_output=True, text=True, env=env, cwd=REPO)
    assert made.returncode == 0, made.stderr
    cases = json.loads(plan.read_text())["cases"]
    assert sorted(c["name"] for c in cases) == sorted(_nightly_cases())
    for case in cases:
        argv = case["command"]
        assert argv[argv.index("--tests") + 1] == case["name"]
        assert argv[argv.index("--suite") + 1] == "nightly"
        assert argv[argv.index("--results-dir") + 1].endswith("/raw") or "{raw_dir}" in argv[-1]
    run_cmd = [exe, "run", "--plan", str(plan), "--out", str(tmp_path / "out"), "--dry-run"]
    dry = subprocess.run(run_cmd, capture_output=True, text=True, env=env, cwd=REPO)
    assert dry.returncode == 0, dry.stderr
    assert len([x for x in dry.stdout.splitlines() if "--tests" in x]) == len(cases)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stored_run_ids_prefer_run_json_then_derive(tmp_path):
    ids = _load_script("bench_nightly_run_ids")
    (tmp_path / "run.json").write_text(json.dumps({"stored_run_ids": ["a", "b", "a"]}))
    assert ids.stored_run_ids(tmp_path, "123") == ["a", "b"]
    (tmp_path / "run.json").write_text(json.dumps({"stored_run_ids": []}))
    derived = ids.stored_run_ids(tmp_path, "123")
    assert len(derived) == 1 and derived == ids.stored_run_ids(tmp_path / "none", "123")


def test_derived_run_id_equals_the_ingest_scripts():
    """The verdict reads the id the ingest action stores: run_id_of('gha', run id, arch, 'perf')."""
    ingest = pytest.importorskip("spyre_clickhouse_ingest")
    ids = _load_script("bench_nightly_run_ids")
    assert ids.stored_run_ids(Path("/nonexistent"), "4242") == [
        str(ingest.run_id_of("gha", "4242", "x86_64", "perf"))
    ]


def test_make_collect_copies_the_engine_cases(tmp_path):
    raw = tmp_path / "res" / "cases" / "c1" / "raw"
    raw.mkdir(parents=True)
    for n in ("latency_demo.json", "latency_demo.pytorch.json"):
        (raw / n).write_text("{}")
    (tmp_path / "res" / "run.json").write_text(
        json.dumps({"cases": [{"name": "latency_demo", "status": "ok", "dir": "cases/c1"}]})
    )
    proc, _ = _make(tmp_path, "bench-collect")
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "res" / "native" / "latency_demo.pytorch.json").exists()


def test_vllm_runner_env_is_passed_to_the_engine():
    text = (REPO / "mk" / "bench-nightly.mk").read_text()
    line = next(x for x in text.splitlines() if x.startswith("BENCH_PASS_ENV"))
    for name in ("HF_HOME", "HF_HUB_OFFLINE", "SAVE_TO_PYTORCH_BENCHMARK_FORMAT", "STORAGE_1_DIR"):
        assert name in line
    names = line.split("?=")[1].strip().split(",")
    assert not [n for n in names if any(w in n for w in ("PASS", "SECRET", "CRED", "KEY"))]
    assert "HF_TOKEN" not in names


def _run_json(tmp_path, ok_cases, files):
    run = {"cases": [{"name": n, "status": "ok", "dir": f"cases/{n}"} for n in ok_cases]}
    (tmp_path / "run.json").write_text(json.dumps(run))
    for case, names in files.items():
        raw = tmp_path / "cases" / case / "raw"
        raw.mkdir(parents=True)
        for n in names:
            (raw / n).write_text("{}")


def test_collect_copies_results_and_flags_missing_ones(tmp_path):
    collect = _load_script("bench_nightly_collect")
    _run_json(
        tmp_path,
        ["latency_demo_tp1", "latency_other_tp1", "latency_nosib_tp1"],
        {
            "latency_demo_tp1": ["latency_Demo_tp1.json", "latency_Demo_tp1.pytorch.json", "x.log"],
            "latency_other_tp1": ["unrelated.log"],
            "latency_nosib_tp1": ["latency_nosib_tp1.json"],
        },
    )
    problems = collect.collect(tmp_path, tmp_path / "native")
    assert sorted(p.split(":")[0] for p in problems) == ["latency_nosib_tp1", "latency_other_tp1"]
    assert sorted(f.name for f in (tmp_path / "native").iterdir()) == [
        "latency_Demo_tp1.json",
        "latency_Demo_tp1.pytorch.json",
        "latency_nosib_tp1.json",
    ]


def test_collect_reports_a_name_collision(tmp_path):
    collect = _load_script("bench_nightly_collect")
    _run_json(
        tmp_path,
        ["a_tp1", "b_tp1"],
        {
            "a_tp1": ["a_tp1.json", "a_tp1.pytorch.json", "shared.json"],
            "b_tp1": ["b_tp1.json", "b_tp1.pytorch.json", "shared.json"],
        },
    )
    assert any("collides" in p for p in collect.collect(tmp_path, tmp_path / "native"))


def test_pass_env_is_completed_inside_the_prefix_and_never_names_secrets():
    text = (REPO / "mk" / "bench-nightly.mk").read_text()
    assert "BENCH_PREFIX +=" in text and "grep -Ev 'TOKEN|PASS|SECRET|KEY|CRED'" in text


def test_helper_run_id_is_the_id_the_ingest_action_stores(tmp_path):
    """Recompute with the real ingest the way the action calls it (numeric --run-id, arch, perf)."""
    pytest.importorskip("clickhouse_connect", reason="the ingest script imports clickhouse_connect")
    identity = _need("spyre_clickhouse_ingest.identity")
    script = _load_script("ingest_vllm_benchmarks")
    ids = _load_script("bench_nightly_run_ids")
    import argparse

    gha = "9876543210"
    action_args = argparse.Namespace(
        run_id=gha, v2_run_id="", gha_run_id=gha, arch="x86_64", test_type="perf"
    )
    stored = script.resolve_v2_run_id(action_args)
    assert stored == identity.RunId.derive("gha", gha, "x86_64", "perf") != ""
    assert ids.stored_run_ids(tmp_path, gha) == [stored]
    other = argparse.Namespace(
        run_id=gha, v2_run_id="", gha_run_id=gha, arch="ppc64le", test_type="perf"
    )
    assert script.resolve_v2_run_id(other) != stored


def test_action_passes_the_inputs_the_helper_assumes():
    text = (REPO / ".github" / "workflows" / "bench-on-demand.yml").read_text()
    assert "run-id: ${{ inputs.run_id == '' && github.run_id || '' }}" in text
    assert "v2-run-id: ${{ inputs.run_id }}" in text
    action = (REPO / ".github" / "actions" / "ingest-vllm-benchmarks" / "action.yml").read_text()
    assert (
        '--gha-run-id "${{ inputs.run-id }}"' in action
        and '--test-type "${{ inputs.test-type }}"' in action
    )
    assert "arch:" in action and "default: x86_64" in action and "default: perf" in action


FAKE_VERDICT = """#!/bin/sh
case "$FAKE_MODE" in
  fail) exit 3 ;;
  junk) echo not-json ;;
  empty) echo '{"summary": {"total": 0, "pass": 0, "worst": "pass"}}' ;;
  *) echo '{"summary": {"total": 2, "pass": 1, "regressed": 1, "worst": "regressed"}}' ;;
esac
"""


def _verdict(tmp_path, mode, monkeypatch, run_json=None):
    verdict = _load_script("bench_nightly_verdict")
    fake = tmp_path / "bench"
    fake.write_text(FAKE_VERDICT)
    fake.chmod(0o755)
    res = tmp_path / "res"
    res.mkdir(exist_ok=True)
    if run_json is not None:
        (res / "run.json").write_text(json.dumps(run_json))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("FAKE_MODE", mode)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    rc = verdict.verdict(str(fake), res, "123")
    return rc, (summary.read_text() if summary.exists() else "")


def test_verdict_passes_only_with_rows(tmp_path, monkeypatch, capsys):
    rc, summary = _verdict(tmp_path, "ok", monkeypatch, {"stored_run_ids": ["r1"]})
    assert rc == 0 and summary == ""
    assert "regressed 1" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("empty", "zero verdict rows"),
        ("fail", "verdict failed"),
        ("junk", "unreadable verdict output"),
    ],
)
def test_verdict_fails_with_a_reason(tmp_path, monkeypatch, mode, reason):
    rc, summary = _verdict(tmp_path, mode, monkeypatch, {"stored_run_ids": ["r1"]})
    assert rc == 1 and summary.strip().startswith("no verdict: ") and reason in summary


def test_verdict_fails_without_a_stored_run_id(tmp_path, monkeypatch):
    verdict = _load_script("bench_nightly_verdict")
    monkeypatch.setattr(
        verdict,
        "_run_ids",
        lambda: type("M", (), {"stored_run_ids": staticmethod(lambda *a, **k: [])}),
    )
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "s.md"))
    assert verdict.verdict("unused", tmp_path, "1") == 1
    assert "no verdict: no stored run id" in (tmp_path / "s.md").read_text()


def test_workflow_skips_the_verdict_when_ingest_did_not_run():
    text = (REPO / ".github" / "workflows" / "bench-on-demand.yml").read_text()
    assert "ingest skipped" in text
    for tag in ("dev", "prod"):
        block = text.split(f"- name: Verdict ({tag})")[1].split("- name:")[0]
        assert (
            f"steps.ingest_{tag}.outcome == 'success'" in block and "continue-on-error" not in block
        )


def test_workflow_is_manual_only():
    text = (REPO / ".github" / "workflows" / "bench-on-demand.yml").read_text()
    head = text.split("jobs:")[0]
    assert "workflow_dispatch:" in head
    assert "schedule" not in text and "cron" not in text and "BENCH_NIGHTLY_ENABLED" not in text
    assert "pull_request" not in head
    assert not (REPO / ".github" / "workflows" / "bench-nightly.yml").exists()


def test_make_targets_have_no_github_only_assumptions():
    for name in ("bench.mk", "bench-nightly.mk"):
        text = (REPO / "mk" / name).read_text()
        assert "GITHUB_" not in text and "RUNNER_TEMP" not in text and "github.com" not in text


def test_jenkins_stored_run_id_is_the_orchestrator_uuid_used_verbatim(tmp_path):
    """Jenkins threads its run uuid as --v2-run-id; the real ingest script stores exactly that."""
    pytest.importorskip("clickhouse_connect", reason="the ingest script imports clickhouse_connect")
    identity = _need("spyre_clickhouse_ingest.identity")
    script = _load_script("ingest_vllm_benchmarks")
    ids = _load_script("bench_nightly_run_ids")
    import argparse

    orch = "6f0a1c52-3b7e-5d44-9a10-2c8b7d6e4f31"
    args = argparse.Namespace(
        run_id=orch, v2_run_id="", gha_run_id="", arch="x86_64", test_type="perf"
    )
    assert script.resolve_v2_run_id(args) == orch
    assert ids.stored_run_ids(tmp_path, orchestrator_run_id=orch.upper()) == [orch]
    # A bare build number would be read as a GitHub run id (source gha): Jenkins must pass the uuid.
    numeric = argparse.Namespace(
        run_id="57", v2_run_id="", gha_run_id="", arch="x86_64", test_type="perf"
    )
    assert (
        script.resolve_v2_run_id(numeric)
        == identity.RunId.derive("gha", "57", "x86_64", "perf")
        != orch
    )
    assert ids.stored_run_ids(tmp_path) == []


def test_bench_ingest_passes_the_action_arguments(tmp_path):
    """bench-ingest calls the ingest script with the arguments the composite action passes."""
    fake = tmp_path / "py"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$FAKE_LOG"\n')
    fake.chmod(0o755)
    proc, _ = _make(
        tmp_path,
        "bench-ingest",
        f"BENCH_HELPER_PY={fake}",
        "BENCH_STORED_RUN_ID=6f0a1c52-3b7e-5d44-9a10-2c8b7d6e4f31",
        "BENCH_BRANCH=main",
        "BENCH_SHA=abc123",
        "BENCH_LEG_KEY=leg-a",
        "BENCH_REPOSITORY=org/repo",
    )
    assert proc.returncode == 0, proc.stderr
    call = (tmp_path / "calls.log").read_text().strip()
    action = (REPO / ".github" / "actions" / "ingest-vllm-benchmarks" / "action.yml").read_text()
    for flag in call.split():
        if flag.startswith("--"):
            assert f'{flag} "' in action or flag in ("--results-dir",), flag
    for flag in (
        "--workflow",
        "--branch",
        "--sha",
        "--v2-run-id",
        "--gha-run-id",
        "--job-id",
        "--pr-number",
        "--arch",
    ):
        assert flag in call
    for flag in ("--test-type perf", "--state passed", "--leg-key leg-a", "--repository org/repo"):
        assert flag in call
    proc, _ = _make(tmp_path, "bench-ingest")
    assert proc.returncode == 2


# --- the dispatch contract and the workflow gates, structurally ---------------------------------


def _workflow():
    import yaml as _yaml

    doc = _yaml.safe_load((REPO / ".github" / "workflows" / "bench-on-demand.yml").read_text())
    doc["on"] = doc.pop(True)  # YAML reads the key `on` as a boolean
    return doc


def _step(job, name):
    return next(x for x in job["steps"] if x.get("name") == name)


def test_dispatch_inputs_are_the_documented_contract():
    inputs = _workflow()["on"]["workflow_dispatch"]["inputs"]
    assert set(_workflow()["on"]) == {"workflow_dispatch"}
    assert inputs["prebaked_image"]["type"] == "string" and not inputs["prebaked_image"]["required"]
    assert inputs["run_id"]["type"] == "string" and not inputs["run_id"]["required"]
    assert inputs["purpose"]["type"] == "choice"
    assert inputs["purpose"]["options"] == ["on-demand", "nightly", "weekly"]
    assert inputs["purpose"]["default"] == "on-demand"
    assert inputs["target"]["default"] == "dev" and inputs["target"]["options"] == ["dev", "prod"]
    assert inputs["mode"]["options"] == ["run", "smoke"]


def test_inputs_are_validated_by_regex_before_anything_runs():
    import re

    script = _step(_workflow()["jobs"]["plan"], "Resolve mode and check the inputs")["run"]
    image = re.search(r'IMAGE_IN\}" =~ (\^\(image_[^ ]+\$)', script).group(1)
    uuid_re = re.search(r'RUN_ID_IN\}" =~ (\^\(\[0-9a-f\][^ ]+\$)', script).group(1)
    assert re.match(image.replace("(", "(?:", 1), "image_torch_spyre") and not re.match(
        image, "image_x; rm"
    )
    assert re.match(uuid_re, "6f0a1c52-3b7e-5d44-9a10-2c8b7d6e4f31") and not re.match(
        uuid_re, "6F0A1C52-x"
    )
    assert re.match(uuid_re, "")
    assert "refs/heads/main" in script and "target prod is dispatched from main only" in script
    assert "the native runner cannot pin CPUs" in script


def test_the_plan_job_is_gated_to_this_repository_and_guards_empty_legs():
    wf = _workflow()
    assert (
        wf["jobs"]["plan"]["if"].strip()
        == "${{ github.repository == 'torch-spyre/spyre-inference' }}"
    )
    legs = _step(wf["jobs"]["plan"], "List legs")["run"]
    assert '== "[]"' in legs and "exit 1" in legs
    assert wf["jobs"]["leg"]["needs"] == "plan"


def test_legs_run_in_the_requested_image_set_and_skip_the_rebuild_when_prebaked():
    leg = _workflow()["jobs"]["leg"]
    assert (
        "needs.plan.outputs.image" in leg["runs-on"] and "image_torch_spyre" not in leg["runs-on"]
    )
    assert leg["env"]["PREBAKED"] == "${{ inputs.prebaked_image != '' }}"
    for name in ("Install Spyre RPMs", "uv cache (NFS)", "Build spyre-inference"):
        assert "env.PREBAKED != 'true'" in _step(leg, name)["if"]
    plan = _workflow()["jobs"]["plan"]
    assert plan["outputs"]["image"] == "${{ steps.mode.outputs.image }}"


def test_ingest_steps_store_the_orchestrator_run_id_and_purpose():
    leg = _workflow()["jobs"]["leg"]
    for tag, cond in (("dev", "env.TARGET != 'prod'"), ("prod", "env.TARGET == 'prod'")):
        step = _step(leg, f"Ingest the results ({tag})")
        assert (
            cond in step["if"]
            and "env.MODE == 'run'" in step["if"]
            and "!cancelled()" in step["if"]
        )
        w = step["with"]
        assert w["v2-run-id"] == "${{ inputs.run_id }}"
        assert w["run-id"] == "${{ inputs.run_id == '' && github.run_id || '' }}"
        assert w["purpose"] == "${{ inputs.purpose || 'on-demand' }}"
        assert w["rpm-lock"] == "${{ env.PREBAKED != 'true' && 'spyre-rpms.lock' || '' }}"
        assert w["leg-key"] == "${{ matrix.leg.key }}"
    secrets_in = {
        s.get("name"): [k for k in s.get("env", {}) if k.startswith("CLICKHOUSE")]
        for s in leg["steps"]
    }
    assert not secrets_in["Run the leg"]


def test_the_verdict_steps_use_the_orchestrator_run_id_and_need_a_landed_ingest():
    leg = _workflow()["jobs"]["leg"]
    for tag in ("dev", "prod"):
        step = _step(leg, f"Verdict ({tag})")
        assert f"steps.ingest_{tag}.outcome == 'success'" in step["if"]
        assert step["env"]["RUN_ID_IN"] == "${{ inputs.run_id }}"
        assert '--run-id "${RUN_ID_IN}"' in step["run"] and "continue-on-error" not in step


def test_ingest_script_records_the_purpose_tag(monkeypatch):
    pytest.importorskip("clickhouse_connect", reason="the ingest script imports clickhouse_connect")
    script = _load_script("ingest_vllm_benchmarks")
    seen = []
    monkeypatch.setattr(script, "write_benchmarks", lambda *a, **k: seen.append(k))
    script._write_v2_benchmarks(None, "db", [], "rid", "leg-a", "nightly")
    script._write_v2_benchmarks(None, "db", [], "rid", "", "weekly")
    script._write_v2_benchmarks(None, "db", [], "rid", "", "")
    assert seen[0]["row_props"] == {"upload_id": "leg-a", "run.purpose": "nightly"}
    assert seen[1]["row_props"] == {"run.purpose": "weekly"}
    assert "row_props" not in seen[2]
    with pytest.raises(SystemExit):
        monkeypatch.setattr(
            script.sys,
            "argv",
            ["x", "--results-dir", "d", "--branch", "b", "--sha", "s", "--purpose", "bogus"],
        )
        script.parse_args()


def test_prebaked_runs_use_the_images_tree_and_absolute_results():
    leg = _workflow()["jobs"]["leg"]
    assert (
        leg["env"]["SRC_DIR"]
        == "${{ inputs.prebaked_image != '' && '/home/senuser/spyre-inference' || '.' }}"
    )
    run = _step(leg, "Run the leg")["run"]
    assert (
        'make -C "${SRC_DIR}" bench-leg' in run
        and 'BENCH_RESULTS_DIR="${GITHUB_WORKSPACE}/results/bench"' in run
    )


def test_engine_ref_is_validated_and_prod_pins_it():
    import re

    script = _step(_workflow()["jobs"]["plan"], "Resolve mode and check the inputs")["run"]
    assert "engine_ref is a branch, tag or commit name" in script
    assert "target prod takes engine_ref main or a full commit sha" in script
    assert re.search(r"\^\(main\|\[0-9a-f\]\{40\}\)\$", script)
    group = _workflow()["concurrency"]["group"]
    assert (
        "inputs.run_id || github.run_id" in group
        and _workflow()["concurrency"]["cancel-in-progress"] is False
    )
