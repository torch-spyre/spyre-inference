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

"""CPU-only tests for the JUnit aggregator (no hardware needed).

Pin the report shape the run Summary depends on: totals, one flat searchable row
per case across all shards, no per-shard rollup table and no per-file headings,
and the size-budget behaviour that keeps failures while shedding passes.
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "aggregate_junit_summary.py"
_spec = importlib.util.spec_from_file_location("aggregate_junit_summary", _SCRIPT)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


def _write(path: Path, cases: str, name: str = "pytest") -> str:
    path.write_text(
        f'<?xml version="1.0" encoding="utf-8"?>\n'
        f'<testsuites><testsuite name="{name}" time="1.0">{cases}</testsuite></testsuites>',
        encoding="utf-8",
    )
    return str(path)


def _tc(name: str, inner: str = "") -> str:
    return f'<testcase classname="tests.m" name="{name}" time="0.1">{inner}</testcase>'


PASS = _tc("test_pass")
SKIP = _tc("test_skip", '<skipped type="pytest.skip" message="no hw"/>')
XFAIL = _tc("test_xfail", '<skipped type="pytest.xfail" message="known"/>')
FAIL = _tc("test_fail", '<failure message="assert 1 == 2">boom</failure>')
ERROR = _tc("test_err", '<error message="setup blew up">trace</error>')
XPASS = _tc("test_xpass", '<failure message="[XPASS(strict)] should fail"/>')


def _render(paths):
    return agg.render(agg.build_report(paths))


def test_flat_index_no_per_shard_rollup(tmp_path):
    out = _render([_write(tmp_path / "junit-test-smoke-shard-0.xml", PASS + SKIP + XFAIL)])

    # Totals line + a single flat case table.
    assert "**1/3 passed**" in out
    assert "### All test cases" in out
    assert "| Shard | Test | Status | Time | Details |" in out
    # No per-shard rollup table, no per-file `## <file>.xml` headings, no xml names.
    assert "| Shard | Tests |" not in out
    assert "## junit" not in out
    assert ".xml" not in out
    # Every case is present as its own row with a ctrl+f-able status word.
    assert "`tests.m::test_pass` | ✅ pass" in out
    assert "`tests.m::test_skip` | ⏭️ skip" in out
    assert "`tests.m::test_xfail` | 🟡 xfail" in out
    # Shard provenance column carries the leg name (junit- prefix stripped).
    assert "test-smoke-shard-0" in out


def test_mixed_counts_and_worst_first(tmp_path):
    paths = [
        _write(tmp_path / "junit-test-a.xml", PASS + FAIL + ERROR),
        _write(tmp_path / "junit-test-b.xml", PASS + XPASS + SKIP),
    ]
    out = _render(paths)

    assert "**2/6 passed**" in out
    assert "1 failed, 1 errored, 1 xpassed" in out
    # Worst-first: error/fail/xpass rows precede the passing rows.
    body = out.split("### All test cases", 1)[1]
    assert body.index("🔥 error") < body.index("✅ pass")
    assert body.index("❌ fail") < body.index("✅ pass")
    assert body.index("⚠️ xpass") < body.index("✅ pass")


def test_empty_input_no_ops():
    # No paths at all: main() emits the skip message rather than a table.
    assert agg.build_report([]).cases == []


def test_malformed_file_skipped_others_kept(tmp_path, capsys):
    bad = tmp_path / "junit-test-bad.xml"
    bad.write_text("<not-xml", encoding="utf-8")
    good = _write(tmp_path / "junit-test-good.xml", PASS)

    out = _render([str(bad), good])

    assert "Skipping unreadable" in capsys.readouterr().err
    assert "**1/1 passed**" in out


def test_budget_sheds_passes_keeps_failures(tmp_path, monkeypatch):
    # Tiny budget so a couple of passing rows must be dropped, but the failure stays.
    monkeypatch.setattr(agg, "SUMMARY_BUDGET", 400)
    many_pass = "".join(
        f'<testcase classname="tests.m" name="test_pass_{i}" time="0.0"/>' for i in range(50)
    )
    out = _render([_write(tmp_path / "junit-test-a.xml", FAIL + many_pass)])

    assert "`tests.m::test_fail` | ❌ fail" in out  # failures never dropped
    assert "passing cases omitted to fit the summary size limit" in out
