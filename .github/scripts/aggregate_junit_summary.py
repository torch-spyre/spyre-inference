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

"""Aggregate the sharded legs' JUnit XML into one $GITHUB_STEP_SUMMARY report:
a per-shard summary plus a searchable row for every test case.

Each shard is labelled by its junit-<target>.xml file name.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# emoji, word (word is plain text so a browser ctrl+f for "xpass" hits the row).
STATUS_META = {
    "error": ("🔥", "error"),
    "fail": ("❌", "fail"),
    "xpass": ("⚠️", "xpass"),
    "xfail": ("🟡", "xfail"),
    "skip": ("⏭️", "skip"),
    "pass": ("✅", "pass"),
}
# Worst first, so a truncated table keeps the rows that matter.
SEVERITY = {s: i for i, s in enumerate(STATUS_META)}

# GitHub drops a step summary over 1 MiB; stay under with headroom.
SUMMARY_BUDGET = 900_000


@dataclass
class Case:
    shard: str
    test: str
    status: str
    time: float
    detail: str


@dataclass
class Report:
    cases: list[Case]
    shard_time: dict[str, float]


def _suite_label(path: str) -> str:
    stem = Path(path).stem
    return stem[len("junit-") :] if stem.startswith("junit-") else stem


def _clean(text: str, limit: int = 240) -> str:
    one_line = " ".join(text.split()).replace("|", "\\|")
    return one_line[: limit - 1] + "…" if len(one_line) > limit else one_line


def _classify(case: ET.Element) -> tuple[str, str]:
    """(status, detail) for one <testcase>, mirroring pytest 9's JUnit shapes:
    xfail -> <skipped type=pytest.xfail>; strict xpass -> <failure> whose message
    is "[XPASS(strict)]…" (no type). Non-strict xpass is a bare <testcase>,
    indistinguishable from a pass, so it can only ever read as "pass"."""
    err = case.find("error")
    if err is not None:
        return "error", _detail(err)
    fail = case.find("failure")
    if fail is not None:
        ftype = (fail.get("type") or "").lower()
        if "xfail" in ftype or (fail.get("message") or "").startswith("[XPASS"):
            return "xpass", _detail(fail)
        return "fail", _detail(fail)
    skipped = case.find("skipped")
    if skipped is not None:
        kind = "xfail" if "xfail" in (skipped.get("type") or "").lower() else "skip"
        return kind, _detail(skipped)
    return "pass", ""


def _detail(child: ET.Element) -> str:
    return _clean(child.get("message") or (child.text or "").strip() or "(no message)")


def _case_id(case: ET.Element) -> str:
    classname = case.get("classname", "")
    name = case.get("name", "")
    return f"{classname}::{name}" if classname else name


def parse_file(path: str) -> tuple[str, float, list[Case]] | None:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as e:
        print(f"Skipping unreadable {path}: {e}", file=sys.stderr)
        return None

    label = _suite_label(path)
    # Root is either <testsuites><testsuite>… or a bare <testsuite>.
    testsuites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    total_time = 0.0
    cases: list[Case] = []
    for ts in testsuites:
        total_time += float(ts.get("time") or 0.0)
        for case in ts.findall("testcase"):
            status, detail = _classify(case)
            cases.append(
                Case(label, _case_id(case), status, float(case.get("time") or 0.0), detail)
            )
    return label, total_time, cases


def build_report(paths: list[str]) -> Report:
    cases: list[Case] = []
    shard_time: dict[str, float] = {}
    for path in sorted(paths):
        parsed = parse_file(path)
        if parsed is None:
            continue
        label, total_time, shard_cases = parsed
        shard_time[label] = total_time
        cases.extend(shard_cases)
    return Report(cases, shard_time)


def _shard_table(report: Report) -> list[str]:
    lines = ["| Shard | Tests | ✅ | ❌ | 🔥 | ⚠️ | 🟡 | ⏭️ | ⏱️ |"]
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    by_shard: dict[str, list[Case]] = {}
    for c in report.cases:
        by_shard.setdefault(c.shard, []).append(c)
    for shard in sorted(by_shard, key=lambda s: (-_broken(by_shard[s]), s)):
        n = Counter(c.status for c in by_shard[shard])
        lines.append(
            f"| {shard} | {len(by_shard[shard])} | {n['pass']} | {n['fail']} | "
            f"{n['error']} | {n['xpass']} | {n['xfail']} | {n['skip']} | "
            f"{report.shard_time.get(shard, 0.0):.0f}s |"
        )
    return lines


def _broken(cases: list[Case]) -> int:
    return sum(c.status in ("fail", "error", "xpass") for c in cases)


def _case_table(report: Report) -> list[str]:
    def row(c: Case) -> str:
        emoji, word = STATUS_META[c.status]
        return f"| {c.shard} | `{c.test}` | {emoji} {word} | {c.time:.1f}s | {c.detail} |"

    ordered = sorted(report.cases, key=lambda c: (SEVERITY[c.status], c.shard, c.test))
    header = ["| Shard | Test | Status | Time | Details |", "|---|---|---|--:|---|"]
    # Non-passing rows always survive; passes fill whatever budget is left.
    lines = header + [row(c) for c in ordered if c.status != "pass"]
    used = len("\n".join(lines).encode())
    omitted = 0
    for c in (c for c in ordered if c.status == "pass"):
        r = row(c)
        if used + len(r.encode()) + 1 > SUMMARY_BUDGET:
            omitted += 1
            continue
        lines.append(r)
        used += len(r.encode()) + 1
    if omitted:
        lines.append("")
        lines.append(f"> {omitted} passing cases omitted to fit the summary size limit.")
    return lines


def render(report: Report) -> str:
    n = Counter(c.status for c in report.cases)
    total = len(report.cases)
    status = "❌" if _broken(report.cases) else ("⚠️" if total == 0 else "✅")
    time = sum(report.shard_time.values())

    lines = [
        "## Aggregate test report",
        "",
        f"{status} **{n['pass']}/{total} passed** across {len(report.shard_time)} "
        f"shards — {n['fail']} failed, {n['error']} errored, {n['xpass']} xpassed, "
        f"{n['xfail']} xfailed, {n['skip']} skipped ({time:.0f}s).",
        "",
        *_shard_table(report),
        "",
        "### All test cases",
        "",
        *_case_table(report),
    ]
    if n["xfail"] or n["xpass"]:
        lines += [
            "",
            "> A non-strict `xfail` that passes reads as a plain pass in JUnit XML "
            "and is counted as passed; only strict xpass shows as `xpass`.",
        ]
    return "\n".join(lines) + "\n"


def _emit(markdown: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(markdown)
    else:
        print(markdown)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", help="JUnit XML files (globbed)")
    args = parser.parse_args()

    if not args.inputs:
        msg = "No JUnit XML to aggregate (all suites skipped or failed early)."
        print(msg)
        _emit(msg + "\n")
        return

    _emit(render(build_report(args.inputs)))


if __name__ == "__main__":
    main()
