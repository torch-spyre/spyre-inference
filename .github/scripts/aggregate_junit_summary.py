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

"""
Fan-in of every sharded leg's JUnit XML into one aggregate report.

Walks the junit-<target>.xml files the test matrix's legs produce and writes a
single summary to $GITHUB_STEP_SUMMARY (stdout when unset): overall totals, a
per-shard table, and the full list of failing/erroring cases, each labelled by
the shard it ran in. The shard label comes from the file name, matching
_test_matrix.yaml's artifact naming and the Makefile test targets.

Usage:
    python3 aggregate_junit_summary.py junit/*.xml
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Suite:
    label: str
    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    time: float = 0.0

    @property
    def passed(self) -> int:
        return self.tests - self.failures - self.errors - self.skipped


@dataclass
class Failure:
    suite: str
    test: str
    kind: str  # "fail" or "error"
    message: str


@dataclass
class Report:
    suites: list[Suite] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)

    @property
    def totals(self) -> Suite:
        agg = Suite(label="TOTAL")
        for s in self.suites:
            agg.tests += s.tests
            agg.failures += s.failures
            agg.errors += s.errors
            agg.skipped += s.skipped
            agg.time += s.time
        return agg


def _suite_label(path: str) -> str:
    stem = Path(path).stem
    return stem[len("junit-") :] if stem.startswith("junit-") else stem


def _clean(text: str, limit: int = 240) -> str:
    one_line = " ".join(text.split())
    one_line = one_line.replace("|", "\\|")
    return one_line[: limit - 1] + "…" if len(one_line) > limit else one_line


def _case_id(case: ET.Element) -> str:
    classname = case.get("classname", "")
    name = case.get("name", "")
    return f"{classname}::{name}" if classname else name


def _failure_message(child: ET.Element) -> str:
    return _clean(child.get("message") or (child.text or "").strip() or "(no message)")


def parse_file(path: str) -> tuple[Suite, list[Failure]] | None:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as e:
        print(f"Skipping unreadable {path}: {e}", file=sys.stderr)
        return None

    label = _suite_label(path)
    suite = Suite(label=label)
    failures: list[Failure] = []
    # Root is either <testsuites><testsuite>…</testsuites> or a bare <testsuite>.
    testsuites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    for ts in testsuites:
        suite.time += float(ts.get("time") or 0.0)
        for case in ts.findall("testcase"):
            suite.tests += 1
            fail = case.find("failure")
            err = case.find("error")
            if case.find("skipped") is not None:
                suite.skipped += 1
            elif fail is not None:
                suite.failures += 1
                failures.append(Failure(label, _case_id(case), "fail", _failure_message(fail)))
            elif err is not None:
                suite.errors += 1
                failures.append(Failure(label, _case_id(case), "error", _failure_message(err)))

    return suite, failures


def build_report(paths: list[str]) -> Report:
    report = Report()
    for path in sorted(paths):
        parsed = parse_file(path)
        if parsed is None:
            continue
        suite, failures = parsed
        report.suites.append(suite)
        report.failures.extend(failures)
    return report


def render(report: Report) -> str:
    t = report.totals
    broken = t.failures + t.errors
    status = "❌" if broken else ("⚠️" if t.tests == 0 else "✅")
    lines: list[str] = []
    lines.append("## Aggregate test report")
    lines.append("")
    lines.append(
        f"{status} **{t.passed}/{t.tests} passed** across {len(report.suites)} "
        f"shards — {t.failures} failed, {t.errors} errored, {t.skipped} skipped "
        f"({t.time:.0f}s)."
    )
    lines.append("")

    # Worst (most broken) shard first.
    lines.append("| Shard | Tests | ✅ | ❌ | 🔥 | ⏭️ | ⏱️ |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for s in sorted(report.suites, key=lambda s: (-(s.failures + s.errors), s.label)):
        lines.append(
            f"| {s.label} | {s.tests} | {s.passed} | {s.failures} | "
            f"{s.errors} | {s.skipped} | {s.time:.0f}s |"
        )
    lines.append("")

    if report.failures:
        lines.append(f"### Failures ({len(report.failures)})")
        lines.append("")
        lines.append("| Shard | Test | Message |")
        lines.append("|---|---|---|")
        for f in report.failures:
            mark = "🔥" if f.kind == "error" else "❌"
            lines.append(f"| {f.suite} | {mark} `{f.test}` | {f.message} |")
    else:
        lines.append("All tests passed 🎉")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", help="JUnit XML files (globbed)")
    args = parser.parse_args()

    if not args.inputs:
        msg = "No JUnit XML to aggregate (all suites skipped or failed early)."
        print(msg)
        _emit(msg + "\n")
        return

    report = build_report(args.inputs)
    _emit(render(report))


def _emit(markdown: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(markdown)
    else:
        print(markdown)


if __name__ == "__main__":
    main()
