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

from __future__ import annotations

import io
import json
import pathlib
import subprocess
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ci_time_report as ctr  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[2]
HOUR = 3600.0


def _baseline(**extra):
    # A 1 h tier, so pct == net minutes / 0.6.
    return {"tests/a.py::test_big": 3000.0, "tests/b.py::test_small": 600.0, **extra}


def _pr(d, suite="S"):
    return {n: (s, suite) for n, s in d.items()}


def test_new_tests_count_against_the_tier():
    pr = _pr({**_baseline(), "tests/c.py::test_new": 180.0})
    r = ctr.compute(pr, [_baseline()] * 5)
    assert [t.nodeid for t in r.new] == ["tests/c.py::test_new"]
    assert r.baseline_s == HOUR
    assert r.pct == 5.0 and r.over


def _collected(nodeids):
    return {ctr.junit_key(n) for n in nodeids}


def test_a_rename_nets_out():
    base = _baseline(**{"tests/c.py::test_x[old]": 300.0})
    pr = _pr({**_baseline(), "tests/c.py::test_x[new]": 300.0})
    r = ctr.compute(pr, [base] * 5, collected=_collected(pr))
    assert len(r.new) == 1 and list(r.removed) == ["tests/c.py::test_x[old]"]
    assert r.net_s == 0 and not r.over


def test_a_skipped_test_is_not_removed():
    # Durations omit skipped tests, e.g. trunk-only ones in a regression-tier PR run.
    trunk_only = "tests/e2e/test_xpress_batch.py::test_qwen3_xpress_batch_four[defaults]"
    base = _baseline(**{trunk_only: 320.0})
    pr = _pr({**_baseline(), "tests/e2e/test_xpress_batch.py::test_new": 600.0})
    r = ctr.compute(pr, [base] * 5, collected=_collected([*pr, trunk_only]))
    assert r.removed == {} and r.net_s == 600.0


def test_removed_needs_the_newest_baseline_and_most_runs():
    gone_on_base = _baseline(**{"tests/c.py::test_old": 300.0})
    r = ctr.compute(_pr(_baseline()), [_baseline()] + [gone_on_base] * 4, collected=set())
    assert r.removed == {}
    r = ctr.compute(_pr(_baseline()), [gone_on_base] * 2 + [_baseline()] * 3, collected=set())
    assert r.removed == {}


def test_known_in_any_baseline_run_is_not_new():
    pr = _pr({**_baseline(), "tests/c.py::test_sometimes": 120.0})
    r = ctr.compute(pr, [_baseline()] * 4 + [_baseline(**{"tests/c.py::test_sometimes": 1.0})])
    assert r.new == []


def test_tests_new_on_the_base_branch_are_not_the_prs():
    pr = _pr({**_baseline(), "tests/main_only.py::test_merged_meanwhile": 600.0})
    r = ctr.compute(pr, [_baseline()] * 5, base_only_files={"tests/main_only.py"})
    assert r.new == [] and len(r.on_base) == 1 and r.net_s == 0


def test_a_new_leg_counts_its_setup():
    r = ctr.compute(_pr(_baseline()), [_baseline()] * 5, legs={"Spyre probe tests": 1})
    assert r.net_s == ctr.LEG_OVERHEAD_S
    assert r.over  # 7.2 min of a 1 h tier


def test_baseline_total_is_the_median():
    runs = [_baseline(), _baseline(), {k: 3 * v for k, v in _baseline().items()}]
    assert ctr.compute(_pr(_baseline()), runs).baseline_s == HOUR


def test_only_baseline_runs_cap_is_used():
    extra = [_baseline(**{"tests/z.py::test_ancient": 9.0})]
    pr = _pr({**_baseline(), "tests/z.py::test_ancient": 9.0})
    r = ctr.compute(pr, [_baseline()] * ctr.BASELINE_RUNS + extra)
    assert [t.nodeid for t in r.new] == ["tests/z.py::test_ancient"]


def test_clean_durations_drops_garbage():
    raw = {
        "ok": 1.5,
        "int": 2,
        "neg": -1,
        "nan": float("nan"),
        "inf": 1e9,
        "s": "x",
        "b": True,
        3: 1,
    }
    assert ctr.clean_durations(raw) == {"ok": 1.5, "int": 2.0}
    assert ctr.clean_durations(["not", "a", "dict"]) == {}
    assert len(next(iter(ctr.clean_durations({"x" * 5000: 1.0})))) == ctr._MAX_NODEID_CHARS


def test_suite_name_strips_shard_and_markup():
    assert ctr.suite_name("Spyre e2e smoke TP=2 (shard 1/2)") == "Spyre e2e smoke TP=2"
    assert "@" not in ctr.suite_name("x @everyone | [l](http://e)")
    assert ctr.suite_name("@@@") == "(unnamed leg)"


def test_matrix_legs_of_the_real_matrix():
    legs = ctr.matrix_legs((_REPO / ctr.MATRIX_PATH).read_text())
    assert legs and all("cfg" in leg for leg in legs)
    counts = ctr.tier_leg_counts(legs)
    assert "Spyre attention smoke" not in counts
    assert counts["Non-distributed spyre tests"] >= 1


def test_leg_delta_by_suite_survives_a_reshard():
    head = {"Smoke": 12, "New suite": 1}
    base = {"Smoke": 11, "Old suite": 2}
    assert ctr.leg_delta(head, base) == {"New suite": 1, "Old suite": -2, "Smoke": 1}
    assert ctr.leg_delta({"A": 2}, {"A": 2}) == {}


def test_slot_suites_recovers_retry_legs_in_sorted_descriptor_order():
    legs = [{"cfg": f"Suite {i} (shard 1/1)"} for i in range(12)]
    # Sorted as strings: failed-suite-1-10 precedes failed-suite-1-2.
    suites = ctr.slot_suites(legs, ["failed-suite-1-2", "failed-suite-1-10"])
    assert suites["retry-0"] == "Suite 10" and suites["retry-1"] == "Suite 2"
    assert suites["3"] == "Suite 3"


def test_merge_pr_legs_keeps_the_original_suite_and_larger_time():
    slices = {"retry-0": {"t::a": 50.0, "t::b": 5.0}, "0": {"t::a": 10.0}}
    merged = ctr.merge_pr_legs(slices, {"0": "Smoke"})
    assert merged["t::a"] == (50.0, "Smoke")
    assert merged["t::b"] == (5.0, ctr.RETRY_SUITE)


def test_comment_escapes_pr_controlled_node_ids():
    evil = "tests/x.py::test_y[`@everyone`|\n<!-- ci-time-report -->]"
    r = ctr.compute(_pr({**_baseline(), evil: 600.0}), [_baseline()] * 5)
    body = ctr.render_comment(r)
    assert body.startswith(ctr.MARKER + "\n")
    line = next(ln for ln in body.splitlines() if "test_y" in ln)
    assert line.startswith("- `") and "`@everyone`" not in line and "\\|" in line
    assert body.count(ctr.MARKER) == 2  # the escaped copy sits inside a code span


def test_comment_wording_above_and_below():
    over = ctr.compute(_pr({**_baseline(), "tests/c.py::test_new": 600.0}), [_baseline()] * 5)
    assert "Please confirm" in ctr.render_comment(over)
    under = ctr.compute(_pr(_baseline()), [_baseline()] * 5)
    assert "below" in ctr.render_comment(under) and "Please confirm" not in ctr.render_comment(
        under
    )


def test_summary_lists_new_removed_and_base_only():
    base = _baseline(**{"tests/c.py::test_old": 60.0})
    pr = _pr({**_baseline(), "tests/c.py::test_new": 90.0, "tests/m.py::test_main": 5.0})
    r = ctr.compute(pr, [base] * 5, collected=_collected(pr), base_only_files={"tests/m.py"})
    s = ctr.render_summary(r)
    assert "test_new" in s and "test_old" in s and "new on the base branch" in s


def _fake_gh(monkeypatch, comments):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        out = json.dumps([comments]) if "comments?per_page" in cmd[2] else "{}"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(ctr.subprocess, "run", run)
    return calls


def _report(minutes):
    return ctr.compute(_pr({**_baseline(), "tests/c.py::t": minutes * 60}), [_baseline()] * 5)


def test_no_comment_below_threshold_without_a_previous_one(monkeypatch):
    calls = _fake_gh(monkeypatch, [])
    r = _report(0.5)
    ctr.upsert_comment("o/r", 7, ctr.render_comment(r), r)
    assert all("POST" not in c and "PATCH" not in c for c in calls)


def test_posts_once_then_edits_in_place(monkeypatch):
    calls = _fake_gh(monkeypatch, [])
    r = _report(10)
    ctr.upsert_comment("o/r", 7, ctr.render_comment(r), r)
    assert any("POST" in c and "/repos/o/r/issues/7/comments" in c for c in calls)

    mine = {"id": 99, "user": {"login": ctr.BOT_LOGIN}, "body": ctr.MARKER + "\nold"}
    spoof = {"id": 5, "user": {"login": "someone"}, "body": ctr.MARKER}
    calls = _fake_gh(monkeypatch, [spoof, mine])
    r = _report(0.5)  # back under: still edits the existing comment
    ctr.upsert_comment("o/r", 7, ctr.render_comment(r), r)
    assert any("PATCH" in c and "/repos/o/r/issues/comments/99" in c for c in calls)


def test_read_json_artifact_reads_in_memory_and_ignores_paths(monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../escape.json", json.dumps({"t::a": 1.0, "t::b": "x"}))
    monkeypatch.setattr(ctr, "gh_bytes", lambda path: buf.getvalue())
    assert ctr.read_json_artifact("o/r", {"id": 1, "name": "durations-data-0"}) == {"t::a": 1.0}
    assert ctr.read_json_artifact("o/r", {"id": 1, "name": "x", "size_in_bytes": 1 << 40}) == {}


def test_newest_by_name_keeps_the_latest_attempt():
    arts = [
        {"name": "durations-data-0", "id": 2, "created_at": "2026-11-10T02:00:00Z"},
        {"name": "durations-data-0", "id": 1, "created_at": "2026-11-10T01:00:00Z"},
        {"name": "junit-x", "id": 3, "created_at": "2026-11-10T03:00:00Z"},
    ]
    assert {k: v["id"] for k, v in ctr.newest_by_name(arts, "durations-data-").items()} == {
        "durations-data-0": 2
    }


def test_comment_lists_only_slow_new_tests_and_links():
    pr = _pr({**_baseline(), "tests/c.py::test_slow": 600.0, "tests/c.py::test_fast": 2.0})
    r = ctr.compute(pr, [_baseline()] * 5)
    body = ctr.render_comment(r, "https://x/runs/1", "https://x/runs/2")
    assert "test_slow" in body and "test_fast" not in body
    assert "[PR run](https://x/runs/1)" in body and "(https://x/runs/2)" in body


def _matrix(*cfgs):
    include = "".join(f"          - cfg: {c}\n            test_types: {t}\n" for c, t in cfgs)
    return f"jobs:\n  test:\n    strategy:\n      matrix:\n        include:\n{include}"


def _fake_build(monkeypatch, pr_files, matrices):
    run = {"id": 1, "head_sha": "head", "created_at": "2026-11-10T12:00:00Z", "run_attempt": 1}
    pr = {"number": 7, "base": {"ref": "main"}}
    arts = [{"name": "durations-data-0", "id": 1}, {"name": "junit-test-smoke-shard-0", "id": 2}]
    monkeypatch.setattr(ctr, "run_artifacts", lambda repo, rid: arts)
    monkeypatch.setattr(ctr, "read_json_artifact", lambda repo, a: _baseline())
    monkeypatch.setattr(ctr, "read_artifact", lambda repo, a, sfx, parse: _collected(_baseline()))
    monkeypatch.setattr(ctr, "baseline_runs", lambda *a: [({"head_sha": "b"}, _baseline())] * 5)
    monkeypatch.setattr(ctr, "pr_changed_files", lambda repo, n: pr_files)
    monkeypatch.setattr(ctr, "base_only_test_files", lambda *a: set())
    monkeypatch.setattr(ctr, "file_at", lambda repo, path, ref: matrices[ref])
    monkeypatch.setattr(ctr, "gh_json", lambda path, **kw: {"merge_base_commit": {"sha": "fork"}})
    return ctr.build_report("o/r", run, pr)


def test_new_legs_are_diffed_against_the_merge_base(monkeypatch):
    fork = _matrix(("Smoke (shard 1/1)", "regression trunk"))
    head = _matrix(
        ("Smoke (shard 1/1)", "regression trunk"),
        ("Probes (shard 1/1)", "regression trunk"),
        ("Smoke e2e (shard 1/1)", "integration"),
    )
    # The base tip gained a shard since the PR forked.
    tip = _matrix(("Smoke (shard 1/2)", "regression trunk"), ("Smoke (shard 2/2)", "regression"))
    matrices = {"fork": fork, "head": head, "main": tip}
    r = _fake_build(monkeypatch, {ctr.MATRIX_PATH}, matrices)
    assert r.leg_delta == {"Probes": 1}
    assert r.net_s == ctr.LEG_OVERHEAD_S


def test_a_pr_that_leaves_the_matrix_alone_adds_no_legs(monkeypatch):
    tip = _matrix(("Smoke (shard 1/2)", "regression trunk"), ("Smoke (shard 2/2)", "regression"))
    r = _fake_build(monkeypatch, {"tests/x.py"}, {"main": tip, "head": _matrix()})
    assert r.leg_delta == {} and r.net_s == 0
    assert "unchanged" in ctr.render_summary(r)


def test_junit_keys_match_pytest_classnames():
    xml = b"""<testsuites><testsuite>
      <testcase classname="tests.e2e.test_x" name="test_a[p]"/>
      <testcase classname="tests.test_y.TestK" name="test_b"><skipped/></testcase>
      <testcase classname="" name="test_upstream[m]"/>
    </testsuite></testsuites>"""
    cases = ctr.junit_cases(xml)
    for nodeid in [
        "tests/e2e/test_x.py::test_a[p]",
        "tests/test_y.py::TestK::test_b",
        "::test_upstream[m]",
    ]:
        assert ctr.junit_key(nodeid) in cases


def test_junit_with_a_dtd_is_refused():
    xml = b'<!DOCTYPE x [<!ENTITY a "aaaa">]><testsuites>&a;</testsuites>'
    try:
        ctr.junit_cases(xml)
    except ValueError:
        return
    raise AssertionError("DTD accepted")


def test_renamed_file_counts_its_old_path(monkeypatch):
    page = [{"filename": "tests/b.py", "previous_filename": "tests/a.py"}, {"filename": "x"}]
    monkeypatch.setattr(ctr, "gh_json", lambda path, **kw: [page])
    assert ctr.pr_changed_files("o/r", 1) == {"tests/a.py", "tests/b.py", "x"}


def test_find_comment_ignores_quotes_of_the_marker(monkeypatch):
    quoting = {"id": 1, "user": {"login": ctr.BOT_LOGIN}, "body": "echo: " + ctr.MARKER}
    mine = {"id": 2, "user": {"login": ctr.BOT_LOGIN}, "body": ctr.MARKER + "\nx"}
    _fake_gh(monkeypatch, [quoting, mine])
    assert ctr.find_comment("o/r", 7)["id"] == 2


def test_find_pr_matches_the_head_repo_and_prefers_open(monkeypatch):
    run = {"head_repository": {"owner": {"login": "me"}, "full_name": "me/r"}, "head_branch": "a&b"}
    pulls = [
        {"number": 1, "state": "closed", "head": {"repo": {"full_name": "me/r"}}},
        {"number": 2, "state": "open", "head": {"repo": {"full_name": "other/r"}}},
        {"number": 3, "state": "open", "head": {"repo": {"full_name": "me/r"}}},
    ]
    paths = []
    monkeypatch.setattr(ctr, "gh_json", lambda path, **kw: paths.append(path) or pulls)
    assert ctr.find_pr("o/r", run)["number"] == 3
    assert "head=me%3Aa%26b&" in paths[0]
    assert ctr.find_pr("o/r", {"head_repository": None}) is None


def test_read_json_artifact_survives_garbage(monkeypatch):
    monkeypatch.setattr(ctr, "gh_bytes", lambda path: b"not a zip")
    assert ctr.read_json_artifact("o/r", {"id": 1, "name": "durations-data-0"}) == {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("d.json", "[" * 100000)
    monkeypatch.setattr(ctr, "gh_bytes", lambda path: buf.getvalue())
    assert ctr.read_json_artifact("o/r", {"id": 1, "name": "durations-data-0"}) == {}
