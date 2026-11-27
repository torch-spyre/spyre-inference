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

"""The per-test env/phases/compile-summary records written beside a benchmark.

No card and no vLLM: the records are host facts plus bookkeeping, and the one
end-to-end case fakes the bench subprocess.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / ".github" / "scripts" / "run_vllm_benchmarks.py"
sys.path.insert(0, str(_SCRIPT.parent))
bench_support = importlib.import_module("bench_support")
_spec = importlib.util.spec_from_file_location("run_vllm_benchmarks", _SCRIPT)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

Recorder = bench_support.Recorder


def _rec(tmp_path, name="t", config=None):
    return Recorder(tmp_path, name, config if config is not None else {"bench_type": "latency"})


def _summary(total_s, passes, cache):
    return json.dumps({"v": 1, "total_s": total_s, "passes": passes, "cache": cache})


# ── where the records land ──


def test_records_land_in_a_subdirectory_not_the_results_root(tmp_path):
    """The ingest globs `<results-dir>/*.json`; these must never appear in it."""
    _rec(tmp_path).write()
    assert sorted(p.name for p in (tmp_path / "t").iterdir()) == ["env.json", "phases.json"]
    assert list(tmp_path.glob("*.json")) == []


def test_two_tests_do_not_share_a_directory(tmp_path):
    _rec(tmp_path, "a").write()
    _rec(tmp_path, "b").write()
    assert (tmp_path / "a" / "env.json").is_file()
    assert (tmp_path / "b" / "env.json").is_file()


# ── phases ──


def test_phases_are_closed_in_order(tmp_path):
    rec = _rec(tmp_path)
    rec.mark("load")
    rec.mark("serve")
    rec.write()
    phases = json.loads((tmp_path / "t" / "phases.json").read_text())["phases"]
    assert [p["name"] for p in phases] == ["load", "serve"]
    assert all(p["t_end"] is not None for p in phases), "write() must close the open phase"
    assert all(p["t_end"] >= p["t_start"] for p in phases)
    assert phases[1]["t_start"] >= phases[0]["t_end"]


def test_phases_file_is_written_even_with_no_marks(tmp_path):
    _rec(tmp_path).write()
    assert json.loads((tmp_path / "t" / "phases.json").read_text())["phases"] == []


def test_server_startup_is_recorded(tmp_path):
    rec = _rec(tmp_path)
    rec.set_server_startup(12.5)
    rec.write()
    d = json.loads((tmp_path / "t" / "phases.json").read_text())
    assert d["server_startup_sec"] == 12.5
    assert d["clock"] == "monotonic"


# ── the child's environment ──


def test_child_env_points_at_this_test_and_turns_timing_on(tmp_path):
    rec = _rec(tmp_path)
    env = {}
    rec.child_env(env)
    assert env[bench_support.COMPILE_SUMMARY_VAR] == str(tmp_path / "t" / "compile")
    assert env[bench_support.TIMING_VAR] == "1"


def test_child_env_does_not_overrule_an_operator(tmp_path):
    env = {bench_support.COMPILE_SUMMARY_VAR: "/their/dir", bench_support.TIMING_VAR: "0"}
    _rec(tmp_path).child_env(env)
    assert env[bench_support.COMPILE_SUMMARY_VAR] == "/their/dir"
    assert env[bench_support.TIMING_VAR] == "0"


# ── merging torch-spyre's per-process summaries ──


def test_compile_summaries_merge_across_processes(tmp_path):
    rec = _rec(tmp_path)
    rec.compile_dir.mkdir(parents=True)
    (rec.compile_dir / "compile_summary.1.json").write_text(
        _summary(2.0, {"P:a": 0.5}, {"entries": 2, "hits": 1, "misses": 1})
    )
    (rec.compile_dir / "compile_summary.2.json").write_text(
        _summary(3.0, {"P:a": 0.25, "P:b": 1.0}, {"entries": 1, "hits": 0, "misses": 1})
    )
    rec.write()
    merged = json.loads((tmp_path / "t" / "compile_summary.json").read_text())
    assert merged["total_s"] == 5.0
    assert merged["passes"] == {"P:a": 0.75, "P:b": 1.0}
    assert merged["cache"] == {"entries": 3, "hits": 1, "misses": 2}
    assert merged["n_processes"] == 2


def test_no_summary_file_when_the_image_wrote_none(tmp_path):
    """An older image writes nothing; that is absence, not an empty measurement."""
    rec = _rec(tmp_path)
    rec.write()
    assert not (tmp_path / "t" / "compile_summary.json").exists()


def test_a_malformed_summary_is_skipped_not_fatal(tmp_path):
    rec = _rec(tmp_path)
    rec.compile_dir.mkdir(parents=True)
    (rec.compile_dir / "compile_summary.1.json").write_text("{not json")
    (rec.compile_dir / "compile_summary.2.json").write_text(
        _summary(4.0, {}, {"entries": 1, "hits": 0, "misses": 1})
    )
    rec.write()
    merged = json.loads((tmp_path / "t" / "compile_summary.json").read_text())
    assert merged["total_s"] == 4.0
    assert merged["n_processes"] == 2, "both files were seen; one carried nothing usable"


def test_a_summary_with_null_fields_does_not_break_the_merge(tmp_path):
    rec = _rec(tmp_path)
    rec.compile_dir.mkdir(parents=True)
    (rec.compile_dir / "compile_summary.1.json").write_text(
        json.dumps({"total_s": None, "passes": None, "cache": None})
    )
    rec.write()
    merged = json.loads((tmp_path / "t" / "compile_summary.json").read_text())
    assert merged["total_s"] == 0.0 and merged["passes"] == {}


# ── the environment fingerprint ──


def test_env_fingerprint_never_raises_and_says_what_it_could_not_read():
    """Runs on any host: off Linux the /proc and /sys reads fail and must be reported."""
    fp = bench_support.env_fingerprint()
    assert isinstance(fp, dict)
    assert isinstance(fp["_missing"], dict)


def test_env_fingerprint_is_cached():
    assert bench_support.env_fingerprint() is bench_support.env_fingerprint()


def test_env_json_carries_the_keys_the_ingest_reads(tmp_path):
    """torch-spyre's benchmeta reader keys off these names; renaming one drops a prop."""
    _rec(tmp_path).write()
    env = json.loads((tmp_path / "t" / "env.json").read_text())
    for key in ("host", "cpu", "cgroup", "mem_total_b", "kernel", "os", "cards", "card_kind"):
        assert key in env, key
    assert {"model", "threads"} <= set(env["cpu"])
    assert {"cpu_max", "memory_max"} <= set(env["cgroup"])


def test_runtime_config_is_kept_beside_the_host_facts(tmp_path):
    _rec(tmp_path, config={"bench_type": "serve", "parameters": {"num-prompts": 8}}).write()
    env = json.loads((tmp_path / "t" / "env.json").read_text())
    assert env["runtime_config"]["parameters"]["num-prompts"] == 8


def test_an_unwritable_directory_does_not_fail_the_benchmark(tmp_path):
    """A record is best-effort: losing it must never turn a passing run into a failure."""
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    Recorder(blocked, "t", {}).write()


# ── end to end through the runner, with the bench subprocess faked ──


def test_run_benchmark_writes_the_records_and_no_extra_result_json(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, env=None, **kwargs):
        seen["env"] = env
        # vllm bench writes its own result JSON; stand in for it.
        (tmp_path / "t.json").write_text("{}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "_make_sampler", lambda: runner._Noop())
    monkeypatch.setattr(runner, "write_resource_metrics", lambda *a, **k: None)

    failure = runner.run_benchmark(
        bench_type="latency",
        test_name="t",
        parameters={"model": "m"},
        env_config={},
        results_dir=tmp_path,
        spyre_devices="0",
        aiu_world_size="1",
    )
    assert failure is None
    assert (tmp_path / "t" / "env.json").is_file()
    assert (tmp_path / "t" / "phases.json").is_file()
    # The ingest reads this glob: only the bench's own result may appear in it.
    assert [p.name for p in tmp_path.glob("*.json")] == ["t.json"]
    assert seen["env"][bench_support.COMPILE_SUMMARY_VAR] == str(tmp_path / "t" / "compile")
    phases = json.loads((tmp_path / "t" / "phases.json").read_text())["phases"]
    assert [p["name"] for p in phases] == ["run"]


def test_records_are_written_when_the_benchmark_fails(tmp_path, monkeypatch):
    def fake_run(cmd, env=None, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "_make_sampler", lambda: runner._Noop())
    monkeypatch.setattr(runner, "write_resource_metrics", lambda *a, **k: None)

    failure = runner.run_benchmark(
        bench_type="latency",
        test_name="t",
        parameters={"model": "m"},
        env_config={},
        results_dir=tmp_path,
        spyre_devices="0",
        aiu_world_size="1",
    )
    assert failure is not None, "the failure must still be reported"
    assert (tmp_path / "t" / "env.json").is_file(), "and the records must say where it ran"
