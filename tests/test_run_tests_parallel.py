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

"""CPU-only tests for `make tests PARALLEL=...` (scripts/run_tests_parallel.py).

The invariants: the parallel run walks the same targets as the serial run and CI, two
shards never share a card, and a shard that drops or duplicates tests fails the run.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "run_tests_parallel.py"
_MATRIX = _REPO / ".github" / "workflows" / "_test_matrix.yaml"
_spec = importlib.util.spec_from_file_location("run_tests_parallel", _SCRIPT)
rtp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rtp  # @dataclass looks its module up here
_spec.loader.exec_module(rtp)


def _make_targets() -> list[str]:
    out = subprocess.run(
        ["make", "-s", "--no-print-directory", "print-test-targets"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return out.split()


# ---------------------------------------------------------------------------
# Same targets as CI and the serial run
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
@pytest.mark.parametrize("tier", ["regression", "trunk"])
def test_targets_match_the_ci_matrix(tier):
    """The Makefile's list is exactly CI's x86_64 legs for the tier, minus the legs that
    need an extra dependency group (triton-cpu), which the serial run also leaves out."""
    doc = yaml.safe_load(_MATRIX.read_text())
    ci = {
        e["test_target"]
        for e in doc["jobs"]["test"]["strategy"]["matrix"]["include"]
        if tier in e["test_types"].split()
        and "x86_64" in e["runs_on"]
        and not e.get("extra_groups")
        and e["test_target"] != "parallel-ab"  # TEMPORARY A/B leg
    }
    targets = _make_targets()
    assert len(targets) == len(set(targets))
    assert set(targets) == ci


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_every_target_has_a_card_count():
    needs = rtp.card_needs_from_matrix(_MATRIX)
    for t in _make_targets():
        assert needs[t] in (1, 2, 4), t
    assert needs["test-smoke-shard-0"] == 1
    assert needs["test-distributed-shard-0"] == 2
    assert needs["test-distributed-tp4"] == 4


# ---------------------------------------------------------------------------
# Card detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "requested", "want"),
    [
        ({"AIU_WORLD_SIZE": "4"}, "auto", ["0", "1", "2", "3"]),
        ({"AIU_WORLD_SIZE": "4"}, "2", ["0", "1"]),
        (
            {"PCIDEVICE_IBM_COM_AIU_PF": "0000:1a:00.0,0000:1b:00.0", "AIU_WORLD_SIZE": "8"},
            "auto",
            ["0", "1"],
        ),
        ({"SPYRE_DEVICES": "1,5,7", "AIU_WORLD_SIZE": "8"}, "auto", ["1", "5", "7"]),
        ({"SPYRE_DEVICES": "1,5,7"}, "2", ["1", "5"]),
    ],
)
def test_detect_devices(env, requested, want):
    assert rtp.detect_devices(env, requested) == want


@pytest.mark.parametrize(
    ("env", "requested", "msg"),
    [
        ({}, "auto", "no Spyre cards"),
        ({"AIU_WORLD_SIZE": "2"}, "4", "only 2 card"),
        ({"AIU_WORLD_SIZE": "2"}, "0", "positive integer"),
        ({"AIU_WORLD_SIZE": "2"}, "two", "positive integer"),
    ],
)
def test_detect_devices_errors(env, requested, msg):
    with pytest.raises(ValueError, match=msg):
        rtp.detect_devices(env, requested)


# ---------------------------------------------------------------------------
# Scheduling: never two shards on one card
# ---------------------------------------------------------------------------


def test_card_pool_never_shares_a_card():
    pool = rtp.CardPool(["0", "1", "2", "3"])
    in_use: set[str] = set()
    lock = threading.Lock()
    errors = []

    def job(need):
        cards = pool.acquire(need)
        with lock:
            if in_use & set(cards):
                errors.append(f"overlap {cards} with {in_use}")
            in_use.update(cards)
        time.sleep(0.002)
        with lock:
            in_use.difference_update(cards)
        pool.release(cards)

    threads = [threading.Thread(target=job, args=(n,)) for n in [4, 2, 2, 1, 1, 1, 1] * 10]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert sorted(pool._free) == ["0", "1", "2", "3"]


def test_job_wider_than_host_gets_every_card():
    # As in a serial run on a 2-card host, the TP=4 job runs with both cards (and skips).
    assert rtp.CardPool(["0", "1"]).acquire(4) == ["0", "1"]


def test_job_env_pins_cards_and_scopes_the_reaper():
    env = rtp.job_env({"PATH": "/bin"}, ["2", "3"])
    assert env["SPYRE_DEVICES"] == "2,3"
    assert env["AIU_WORLD_SIZE"] == "2"
    assert env["SPYRE_PARALLEL_WORKER"] == "1"
    assert env["SPYRE_TEST_CARD_SCOPE"].startswith("cards-2-3-")
    assert env["TORCHINDUCTOR_CACHE_DIR"].endswith("__card_2")


def test_run_order_widest_first():
    needs = {"a": 1, "b": 4, "c": 2, "d": 1}
    assert rtp.run_order(list(needs), needs) == ["b", "c", "a", "d"]


# ---------------------------------------------------------------------------
# Parity guard
# ---------------------------------------------------------------------------


def test_parse_collected_skips_annotations_and_summary():
    out = textwrap.dedent(
        """\
        Running tests for TEST_TYPE=regression MARK_OVERRIDE=...
        ::error::not a test id
        tests/a.py::test_x[1]
        tests/a.py::TestK::test_y
        ../../.cache/vllm-upstream-tests/worktree-abc/tests/v1/t.py::test_z

        3 tests collected in 1.2s
        """
    )
    assert rtp.parse_collected(out) == [
        "tests/a.py::test_x[1]",
        "tests/a.py::TestK::test_y",
        "../../.cache/vllm-upstream-tests/worktree-abc/tests/v1/t.py::test_z",
    ]


def test_shard_family():
    assert rtp.shard_family("test-smoke-shard-10") == "test-smoke"
    assert rtp.shard_family("test-distributed-shard-0") == "test-distributed"
    assert rtp.shard_family("test-encoder-attention") is None
    # No unsharded twin with the same marker expression: never checked as a family.
    assert rtp.shard_family("test-e2e-smoke-shard-0") is None


def _leaf_pytest(target: str) -> str:
    out = subprocess.run(
        ["make", "-n", "--no-print-directory", target, "JUNIT_XML="],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (line,) = [ln for ln in out.splitlines() if " pytest " in ln]
    return line[line.index(" pytest ") :]


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
@pytest.mark.parametrize("prefix", sorted(rtp.FAMILIES))
def test_family_unsharded_target_selects_the_same_tests(prefix):
    """The parity guard compares a family's shards to its unsharded target, so the two
    pytest command lines must differ only by the shard flags."""
    sharded = re.sub(r"--[a-z-]+-shards?(-id)?=\d+ ?", "", _leaf_pytest(f"{prefix}-0"))
    assert " ".join(sharded.split()) == " ".join(_leaf_pytest(rtp.FAMILIES[prefix]).split())


_FAMILY = {"test-smoke": ["test-smoke-shard-0", "test-smoke-shard-1"]}


def test_check_collection_ok():
    collected = {
        "test-smoke": ["a", "b", "c"],
        "test-smoke-shard-0": ["a", "c"],
        "test-smoke-shard-1": ["b"],
    }
    assert rtp.check_collection(collected, _FAMILY) == []


def test_check_collection_flags_a_dropped_test():
    collected = {
        "test-smoke": ["a", "b", "c"],
        "test-smoke-shard-0": ["a"],
        "test-smoke-shard-1": ["b"],
    }
    assert rtp.check_collection(collected, _FAMILY) == ["test-smoke: 1 test(s) in no shard, e.g. c"]


def test_check_collection_flags_an_empty_shard():
    collected = {"test-smoke": ["a"], "test-smoke-shard-0": ["a"], "test-smoke-shard-1": []}
    assert rtp.check_collection(collected, _FAMILY) == ["test-smoke-shard-1: collected 0 test IDs"]


def test_check_collection_flags_a_test_in_two_shards():
    collected = {
        "test-smoke": ["a", "b"],
        "test-smoke-shard-0": ["a", "b"],
        "test-smoke-shard-1": ["b"],
    }
    assert rtp.check_collection(collected, _FAMILY) == [
        "b: in both test-smoke-shard-0 and test-smoke-shard-1"
    ]


def _junit(path: Path, n: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = "".join(f'<testcase classname="c" name="t{i}"/>' for i in range(n))
    path.write_text(f'<testsuites><testsuite name="pytest">{cases}</testsuite></testsuites>')
    return path


def test_check_results(tmp_path):
    paths = {
        "a": _junit(tmp_path / "a.xml", 3),
        "b": _junit(tmp_path / "b.xml", 2),
        "c": tmp_path / "missing.xml",
    }
    assert rtp.check_results({"a": 3, "b": 3, "c": 1}, paths) == [
        "b: collected 3 test(s) but JUnit has 2",
        f"c: no JUnit file at {tmp_path / 'missing.xml'}",
    ]


# ---------------------------------------------------------------------------
# End to end, with a fake `make` that stands in for pytest
# ---------------------------------------------------------------------------

_FAKE_MAKE = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    # argv: target, then VAR=value pairs. Emits collection output or writes JUnit.
    import json, os, sys, time
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target, kv = args[0], dict(a.split("=", 1) for a in args[1:] if "=" in a)
    plan = json.load(open(os.environ["FAKE_PLAN"]))
    ids = plan["run" if "JUNIT_XML" in kv and kv["JUNIT_XML"] else "collect"].get(
        target, plan["collect"][target])
    if "--collect-only" in kv.get("PYTEST_ARGS", ""):
        print("\\n".join(ids))
        sys.exit(0 if ids else 5)
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps({"t": target, "cards": os.environ["SPYRE_DEVICES"],
                            "start": time.time()}) + "\\n")
    time.sleep(0.05)
    cases = "".join(f'<testcase classname="c" name="{i}"/>' for i in ids)
    open(kv["JUNIT_XML"], "w").write(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps({"t": target, "end": time.time()}) + "\\n")
    """
)


def _fake_run(tmp_path, collect, run=None, cards="auto", world="4"):
    make = tmp_path / "fake-make"
    make.write_text(_FAKE_MAKE)
    make.chmod(0o755)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"collect": collect, "run": run or {}}))
    matrix = tmp_path / "matrix.yaml"
    include = [
        {"test_target": t, "runs_on": ["x86_64", f"spyre_pf_x{2 if 'dist' in t else 1}"]}
        for t in {**_COLLECT, **collect}
    ]
    matrix.write_text(
        yaml.safe_dump({"jobs": {"test": {"strategy": {"matrix": {"include": include}}}}})
    )
    log = tmp_path / "log.jsonl"
    env = {**os.environ, "FAKE_PLAN": str(plan), "FAKE_LOG": str(log), "AIU_WORLD_SIZE": world}
    env.pop("SPYRE_DEVICES", None)
    env.pop("PCIDEVICE_IBM_COM_AIU_PF", None)
    targets = [t for t in collect if "shard" in t]
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--make",
            str(make),
            "--cards",
            cards,
            "--matrix",
            str(matrix),
            "--results-dir",
            str(tmp_path / "out"),
            *targets,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    return proc, log


def _ids(*names):
    return [f"tests/t.py::{n}" for n in names]


_COLLECT = {
    "test-smoke": _ids("s1", "s2", "s3", "s4"),
    "test-smoke-shard-0": _ids("s1", "s3"),
    "test-smoke-shard-1": _ids("s2"),
    "test-smoke-shard-2": _ids("s4"),
    "test-distributed": _ids("d1", "d2"),
    "test-distributed-shard-0": _ids("d1"),
    "test-distributed-shard-1": _ids("d2"),
}


def test_end_to_end_runs_every_shard_on_its_own_cards(tmp_path):
    proc, log = _fake_run(tmp_path, _COLLECT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    events = [json.loads(line) for line in log.read_text().splitlines()]
    starts = {e["t"]: e for e in events if "start" in e}
    ends = {e["t"]: e["end"] for e in events if "end" in e}
    assert set(starts) == {t for t in _COLLECT if "shard" in t}
    assert len(starts["test-distributed-shard-0"]["cards"].split(",")) == 2
    # Jobs that overlapped in time never shared a card.
    for a in starts:
        for b in starts:
            if a < b and starts[a]["start"] < ends[b] and starts[b]["start"] < ends[a]:
                assert not set(starts[a]["cards"].split(",")) & set(
                    starts[b]["cards"].split(",")
                ), (a, b)
    assert "collected 6 test(s)" in proc.stdout


def test_end_to_end_stops_when_a_shard_drops_a_test(tmp_path):
    collect = {**_COLLECT, "test-smoke-shard-2": []}
    proc, log = _fake_run(tmp_path, collect)
    assert proc.returncode == 3
    assert "test-smoke-shard-2: collected 0 test IDs" in proc.stdout
    assert "test-smoke: 1 test(s) in no shard, e.g. tests/t.py::s4" in proc.stdout
    assert not log.exists()  # nothing ran


def test_end_to_end_subset_of_a_family_skips_the_family_check(tmp_path):
    collect = {k: v for k, v in _COLLECT.items() if k != "test-smoke-shard-2"}
    proc, _ = _fake_run(tmp_path, collect)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_end_to_end_fails_when_a_shard_runs_fewer_tests(tmp_path):
    proc, _ = _fake_run(tmp_path, _COLLECT, run={"test-smoke-shard-0": _ids("s1")})
    assert proc.returncode == 1
    assert "test-smoke-shard-0: collected 2 test(s) but JUnit has 1" in proc.stdout


def test_end_to_end_refuses_more_cards_than_the_host_has(tmp_path):
    proc, _ = _fake_run(tmp_path, _COLLECT, cards="8", world="4")
    assert proc.returncode == 2
    assert "PARALLEL=8 but only 4 card(s) detected" in proc.stdout


# ---------------------------------------------------------------------------
# The VFIO reaper only sees its own shard's processes
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="needs /proc")
def test_reaper_scope_matches_only_its_own_shard():
    from spyre_testing_plugin import vfio_reaper

    env = {**os.environ, vfio_reaper.CARD_SCOPE_ENV: "cards-1-42"}
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], env=env)
    try:
        time.sleep(0.2)  # let the child exec, so /proc/<pid>/environ is its own
        assert vfio_reaper._in_card_scope(child.pid, "cards-1-42")
        assert not vfio_reaper._in_card_scope(child.pid, "cards-2-42")
        assert not vfio_reaper._in_card_scope(child.pid, "cards-1")
        assert vfio_reaper._in_card_scope(child.pid, None)  # serial run: every holder
    finally:
        child.kill()
        child.wait()
