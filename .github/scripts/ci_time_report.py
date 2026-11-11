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
Comment on a PR when its new tests and matrix legs add more than THRESHOLD_PCT of the
regression tier's CI test time. Informational only, never blocks a PR.

Usage:
    GH_TOKEN=... python3 ci_time_report.py report --run-id <id> --post
    GH_TOKEN=... python3 ci_time_report.py report --pr <number>
    python3 ci_time_report.py replay <test-durations.json>... (oldest first)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import statistics
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import yaml

THRESHOLD_PCT = 3.0
BASELINE_RUNS = 5
REMOVED_MIN_RUNS = 3
# Median setup cost of one matrix leg (job wall clock minus its test time).
LEG_OVERHEAD_S = 432.0
TIER = "regression"
WORKFLOW = "test_each_commit.yaml"
MATRIX_PATH = ".github/workflows/_test_matrix.yaml"
MARKER = "<!-- ci-time-report -->"
BOT_LOGIN = "github-actions[bot]"
RETRY_SUITE = "(pod-level retry)"

_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
_MAX_NODEID_CHARS = 300
_TOP_N = 5
_SLOW_S = 30.0
_SUMMARY_ROWS = 200
_MAX_BASE_COMMITS = 30
_MAX_BASELINE_PAGES = 5


@dataclass
class NewTest:
    nodeid: str
    seconds: float
    suite: str


@dataclass
class Report:
    baseline_s: float
    baseline_runs: int
    new: list[NewTest]
    removed: dict[str, float]
    leg_delta: dict[str, int]
    on_base: list[NewTest] = field(default_factory=list)

    @property
    def new_s(self) -> float:
        return sum(t.seconds for t in self.new)

    @property
    def removed_s(self) -> float:
        return sum(self.removed.values())

    @property
    def legs(self) -> int:
        return sum(self.leg_delta.values())

    @property
    def net_s(self) -> float:
        return self.new_s - self.removed_s + self.legs * LEG_OVERHEAD_S

    @property
    def pct(self) -> float:
        return 100.0 * self.net_s / self.baseline_s if self.baseline_s else 0.0

    @property
    def over(self) -> bool:
        return self.pct >= THRESHOLD_PCT


def clean_durations(data: object) -> dict[str, float]:
    # A fork PR writes these artifacts: keep only sane {str: seconds} pairs.
    if not isinstance(data, dict):
        return {}
    out = {}
    for nodeid, seconds in data.items():
        if not isinstance(nodeid, str) or isinstance(seconds, bool):
            continue
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            continue
        if 0 <= seconds < 1e6:
            out[nodeid[:_MAX_NODEID_CHARS]] = seconds
    return out


class _NoDTD(ET.TreeBuilder):
    # JUnit XML has no DTD; refusing one closes entity-expansion attacks.
    def doctype(self, name, pubid, system):
        raise ValueError("DTD refused")


def junit_cases(xml: bytes) -> set[tuple[str, str]]:
    """(classname, name) of every collected case, skipped ones included."""
    root = ET.fromstring(xml, parser=ET.XMLParser(target=_NoDTD()))
    return {(c.get("classname") or "", c.get("name") or "") for c in root.iter("testcase")}


def junit_key(nodeid: str) -> tuple[str, str]:
    # pytest's junitxml: classname is the dotted module path plus any classes.
    path, *rest = nodeid.split("::")
    module = path[:-3].replace("/", ".") if path.endswith(".py") else path
    return ".".join(p for p in [module, *rest[:-1]] if p), rest[-1] if rest else path


def _plain(text: str) -> str:
    # cfg comes from the PR head and lands in a comment: no mentions, links or pipes.
    return re.sub(r"[^A-Za-z0-9 ()/=.,_+-]", "", text).strip()[:80] or "(unnamed leg)"


def suite_name(cfg: str) -> str:
    return _plain(re.sub(r"\s*\(shard \d+/\d+\)\s*$", "", cfg))


def matrix_legs(matrix_yaml: str) -> list[dict]:
    # Include order is strategy.job-index.
    doc = yaml.safe_load(matrix_yaml)
    try:
        legs = doc["jobs"]["test"]["strategy"]["matrix"]["include"]
    except (KeyError, TypeError):
        return []
    return [leg for leg in legs if isinstance(leg, dict) and isinstance(leg.get("cfg"), str)]


def tier_leg_counts(legs: list[dict], tier: str = TIER) -> dict[str, int]:
    counts: dict[str, int] = {}
    for leg in legs:
        if tier in str(leg.get("test_types", "")).split():
            name = suite_name(leg["cfg"])
            counts[name] = counts.get(name, 0) + 1
    return counts


def leg_delta(head: dict[str, int], base: dict[str, int]) -> dict[str, int]:
    return {
        s: head.get(s, 0) - base.get(s, 0)
        for s in sorted(set(head) | set(base))
        if head.get(s, 0) != base.get(s, 0)
    }


def slot_suites(legs: list[dict], failed_suite_names: list[str]) -> dict[str, str]:
    # retry-<i> indexes collect_failed_suites' retry matrix, built in sorted descriptor order.
    out = {str(i): suite_name(leg["cfg"]) for i, leg in enumerate(legs)}
    for i, name in enumerate(sorted(failed_suite_names)):
        idx = name.rsplit("-", 1)[-1]
        if idx.isdigit() and int(idx) < len(legs):
            out[f"retry-{i}"] = suite_name(legs[int(idx)]["cfg"])
    return out


def merge_pr_legs(
    slices: dict[str, dict[str, float]], suites: dict[str, str]
) -> dict[str, tuple[float, str]]:
    out: dict[str, tuple[float, str]] = {}
    # First attempts before retries, so a retried test keeps its original suite.
    for slot in sorted(slices, key=lambda s: s.startswith("retry-")):
        suite = suites.get(slot, RETRY_SUITE if slot.startswith("retry-") else "(unknown leg)")
        for nodeid, seconds in slices[slot].items():
            prev = out.get(nodeid)
            if prev is None:
                out[nodeid] = (seconds, suite)
            elif seconds > prev[0]:
                out[nodeid] = (seconds, prev[1])
    return out


def compute(
    pr: dict[str, tuple[float, str]],
    baselines: list[dict[str, float]],
    legs: dict[str, int] | None = None,
    collected: set[tuple[str, str]] | None = None,
    base_only_files: set[str] | None = None,
) -> Report:
    """Compare a PR run with baseline runs, newest first.

    collected: the PR run's JUnit cases. Durations omit skipped tests (e.g. trunk-only
    ones, in this regression-tier run), so only a test JUnit lacks counts as removed.
    """
    baselines = baselines[:BASELINE_RUNS]
    base_only_files = base_only_files or set()
    seen: dict[str, list[float]] = {}
    for b in baselines:
        for nodeid, seconds in b.items():
            seen.setdefault(nodeid, []).append(seconds)
    need = min(REMOVED_MIN_RUNS, len(baselines))
    # A test the base branch itself dropped lingers in older baselines.
    newest = baselines[0] if baselines else {}
    removed = {
        n: statistics.median(v)
        for n, v in seen.items()
        if n not in pr
        and n in newest
        and len(v) >= need
        and (collected is None or junit_key(n) not in collected)
    }
    new, on_base = [], []
    for nodeid, (seconds, suite) in pr.items():
        if nodeid in seen:
            continue
        t = NewTest(nodeid, seconds, suite)
        (on_base if nodeid.split("::")[0] in base_only_files else new).append(t)
    new.sort(key=lambda t: -t.seconds)
    on_base.sort(key=lambda t: -t.seconds)
    total = statistics.median([sum(b.values()) for b in baselines]) if baselines else 0.0
    return Report(total, len(baselines), new, removed, legs or {}, on_base)


def _min(seconds: float) -> str:
    return f"{seconds / 60:.1f} min"


def _code(nodeid: str) -> str:
    # PR-controlled: no backticks or newlines, and pipes break tables.
    return "`" + re.sub(r"[`\r\n]", "", nodeid).replace("|", "\\|") + "`"


def _suite_rows(report: Report) -> list[str]:
    per: dict[str, list[NewTest]] = {}
    for t in report.new:
        per.setdefault(t.suite, []).append(t)
    rows = ["| Suite | Tests | Time in this run |", "|---|--:|--:|"]
    for suite, tests in sorted(per.items(), key=lambda kv: -sum(t.seconds for t in kv[1])):
        rows.append(f"| {suite} | {len(tests)} | {_min(sum(t.seconds for t in tests))} |")
    if report.removed:
        rows.append(f"| Removed tests | −{len(report.removed)} | −{_min(report.removed_s)} |")
    for suite, d in report.leg_delta.items():
        rows.append(
            f"| {'New' if d > 0 else 'Removed'} matrix legs: {suite} | {d:+d} | "
            f"{'+' if d > 0 else '−'}{_min(abs(d) * LEG_OVERHEAD_S)} (setup) |"
        )
    return rows


def _headline(report: Report) -> str:
    if abs(report.net_s) < 30:
        return (
            f"This PR leaves CI test time about unchanged ({report.pct:+.1f}% of the {TIER} tier)."
        )
    sign = "adds" if report.net_s > 0 else "removes"
    return (
        f"This PR {sign} about {abs(report.net_s) / 60:.1f} minutes of CI test time "
        f"({report.pct:+.1f}% of the {TIER} tier's {report.baseline_s / 3600:.1f} h)."
    )


def _footnote(report: Report) -> str:
    return (
        f"New tests usually compile cold in a PR run, so their times can be higher than they "
        f"will be on the base branch. Baseline: median of the last {report.baseline_runs} green "
        f"push runs on the base branch. A new matrix leg counts {_min(LEG_OVERHEAD_S)} of setup. "
        f"This check is informational and does not block the merge."
    )


def render_comment(report: Report, pr_run_url: str = "", summary_url: str = "") -> str:
    lines = [MARKER]
    if report.over:
        lines += [
            f"**{_headline(report)}** That is above the {THRESHOLD_PCT:g}% this check flags. "
            "Please confirm this is intended, or say whether the new tests can be trimmed, "
            "folded into an existing parametrization, or moved out of the "
            f"{TIER} tier.",
            "",
        ]
    else:
        lines += [
            f"Update: {_headline(report)} That is below the {THRESHOLD_PCT:g}% this check flags, "
            "so nothing to confirm.",
            "",
        ]
    lines += _suite_rows(report)
    slow = [t for t in report.new[:_TOP_N] if t.seconds >= _SLOW_S]
    if slow:
        lines += ["", "Slowest new tests:", ""]
        lines += [f"- {_code(t.nodeid)}: {_min(t.seconds)}" for t in slow]
    links = []
    if pr_run_url:
        links.append(f"[PR run]({pr_run_url})")
    if summary_url:
        links.append(f"[full list of new and removed tests]({summary_url})")
    if links:
        lines += ["", " · ".join(links)]
    lines += ["", f"<sub>{_footnote(report)}</sub>"]
    return "\n".join(lines) + "\n"


def render_summary(report: Report, run_url: str = "") -> str:
    lines = ["## CI time report", "", _headline(report)]
    verdict = "above: the PR gets a comment" if report.over else "below: no new comment"
    lines.append(f"Comment threshold: {THRESHOLD_PCT:g}% ({verdict}).")
    if run_url:
        lines.append(f"PR run: {run_url}")
    lines += ["", *_suite_rows(report), ""]
    if report.new:
        lines += ["### New tests", "", "| Test | Suite | Time |", "|---|---|--:|"]
        lines += [
            f"| {_code(t.nodeid)} | {t.suite} | {_min(t.seconds)} |"
            for t in report.new[:_SUMMARY_ROWS]
        ]
        if len(report.new) > _SUMMARY_ROWS:
            lines.append(f"| … {len(report.new) - _SUMMARY_ROWS} more | | |")
        lines.append("")
    if report.removed:
        lines += ["### Removed tests (baseline time)", "", "| Test | Time |", "|---|--:|"]
        rem = sorted(report.removed.items(), key=lambda kv: -kv[1])
        lines += [f"| {_code(n)} | {_min(s)} |" for n, s in rem[:_SUMMARY_ROWS]]
        lines.append("")
    if report.on_base:
        lines += [
            "### Not counted: possibly the base branch's",
            "",
            f"{len(report.on_base)} new tests "
            f"({_min(sum(t.seconds for t in report.on_base))}) are in files the base branch "
            "changed after the newest baseline run, so they may be its tests, not this PR's.",
            "",
        ]
    lines += [_footnote(report), ""]
    return "\n".join(lines)


def gh_json(path: str, paginate: bool = False):
    cmd = ["gh", "api", path]
    if paginate:
        cmd += ["--paginate", "--slurp"]
    result = subprocess.run(cmd, capture_output=True, check=True, text=True)
    return json.loads(result.stdout)


def gh_bytes(path: str) -> bytes:
    return subprocess.run(["gh", "api", path], capture_output=True, check=True).stdout


def run_artifacts(repo: str, run_id: int) -> list[dict]:
    pages = gh_json(f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100", paginate=True)
    return [a for page in pages for a in page.get("artifacts", []) if not a.get("expired")]


def read_artifact(repo: str, artifact: dict, suffix: str, parse):
    """parse() of the artifact's first *suffix member, or None if unreadable."""
    if artifact.get("size_in_bytes", 0) > _MAX_ARTIFACT_BYTES:
        print(f"::warning::skipping oversized artifact {_plain(artifact['name'])}")
        return None
    blob = gh_bytes(f"/repos/{repo}/actions/artifacts/{artifact['id']}/zip")
    # Read in memory, never extracted; a malformed (fork-written) artifact is skipped.
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            members = [m for m in zf.infolist() if m.filename.endswith(suffix)]
            if not members or members[0].file_size > _MAX_ARTIFACT_BYTES:
                return None
            return parse(zf.read(members[0]))
    except Exception as e:
        name = _plain(artifact["name"])
        print(f"::warning::skipping unreadable artifact {name}: {type(e).__name__}")
        return None


def _durations(blob: bytes) -> dict[str, float]:
    return clean_durations(json.loads(blob))


def read_json_artifact(repo: str, artifact: dict) -> dict[str, float]:
    return read_artifact(repo, artifact, ".json", _durations) or {}


def missing_legs(legs: list[dict], junit_meta: dict[str, dict]) -> list[str]:
    """cfg of every TIER leg without a junit-<test_target> artifact (retries overwrite it)."""
    return [
        _plain(leg["cfg"])
        for leg in legs
        if TIER in str(leg.get("test_types", "")).split()
        and f"junit-{leg.get('test_target', '')}" not in junit_meta
    ]


def newest_by_name(artifacts: list[dict], prefix: str) -> dict[str, dict]:
    # A re-run attempt re-uploads the same names.
    out: dict[str, dict] = {}
    for a in sorted(artifacts, key=lambda a: a.get("created_at", "")):
        if a["name"].startswith(prefix):
            out[a["name"]] = a
    return out


def file_at(repo: str, path: str, ref: str) -> str:
    data = gh_json(f"/repos/{repo}/contents/{path}?ref={ref}")
    return base64.b64decode(data["content"]).decode()


def find_pr(repo: str, run: dict) -> dict | None:
    if not run.get("head_repository"):  # the fork was deleted
        return None
    owner = run["head_repository"]["owner"]["login"]
    head = urllib.parse.quote(f"{owner}:{run['head_branch']}", safe="")
    pulls = gh_json(f"/repos/{repo}/pulls?state=all&head={head}&per_page=100")
    head_repo = run["head_repository"]["full_name"]
    for pr in sorted(pulls, key=lambda p: p["state"] != "open"):
        if (pr["head"].get("repo") or {}).get("full_name") == head_repo:
            return pr
    return None


def latest_pr_run(repo: str, pr: dict) -> dict | None:
    runs = gh_json(
        f"/repos/{repo}/actions/workflows/{WORKFLOW}/runs?event=pull_request&status=success"
        f"&branch={urllib.parse.quote(pr['head']['ref'], safe='')}&per_page=30"
    )["workflow_runs"]
    head_repo = (pr["head"].get("repo") or {}).get("full_name")
    mine = [r for r in runs if (r.get("head_repository") or {}).get("full_name") == head_repo]
    return next((r for r in mine if r["head_sha"] == pr["head"]["sha"]), mine[0] if mine else None)


def baseline_runs(repo: str, branch: str, before: str) -> list[tuple[dict, dict[str, float]]]:
    out = []
    query = (
        f"/repos/{repo}/actions/workflows/{WORKFLOW}/runs"
        f"?branch={urllib.parse.quote(branch, safe='')}&event=push&status=success"
        f"&created={urllib.parse.quote('<=' + before, safe='')}&per_page=30"
    )
    for page in range(1, _MAX_BASELINE_PAGES + 1):
        runs = gh_json(f"{query}&page={page}")["workflow_runs"]
        for run in runs:
            art = newest_by_name(run_artifacts(repo, run["id"]), "test-durations").get(
                "test-durations"
            )
            data = read_json_artifact(repo, art) if art else {}
            if data:
                out.append((run, data))
            if len(out) == BASELINE_RUNS:
                return out
        if len(runs) < 30:
            break
    return out


def pr_changed_files(repo: str, number: int) -> set[str]:
    pages = gh_json(f"/repos/{repo}/pulls/{number}/files?per_page=100", paginate=True)
    return {
        name
        for page in pages
        for f in page
        for name in (f["filename"], f.get("previous_filename"))
        if name
    }


def base_only_test_files(
    repo: str, newest_base_sha: str, pr_run: dict, pr: dict
) -> set[str] | None:
    """Test files the base branch changed after the newest baseline, up to the PR run.

    None when the base branch moved too far to list. The PR run tests the merge with the
    base tip, so a new test in one of these files may be the base branch's, not the PR's.
    """
    try:
        cmp = gh_json(f"/repos/{repo}/compare/{newest_base_sha}...{pr['base']['ref']}")
    except subprocess.CalledProcessError:
        return None
    listed = cmp.get("commits", [])
    commits = [c for c in listed if c["commit"]["committer"]["date"] <= pr_run["created_at"]]
    # The listing is oldest first and capped; it must reach past the PR run to be complete.
    complete = cmp.get("total_commits", 0) <= len(listed) or len(commits) < len(listed)
    if not complete or len(commits) > _MAX_BASE_COMMITS:
        return None
    files: set[str] = set()
    for c in commits:
        detail = gh_json(f"/repos/{repo}/commits/{c['sha']}")
        files |= {f["filename"] for f in detail.get("files", [])}
    return {f for f in files if f.startswith("tests/")}


def build_report(repo: str, run: dict, pr: dict) -> Report | str:
    artifacts = run_artifacts(repo, run["id"])
    slices_meta = newest_by_name(artifacts, "durations-data-")
    if not slices_meta:
        return "the PR run recorded no test durations (matrix skipped or handed to Jenkins)"

    # Slots index the matrix the run executed: its merge commit's, which
    # referenced_workflows pins.
    ran = [
        w["sha"]
        for w in run.get("referenced_workflows") or []
        if w.get("path", "").split("@")[0].endswith(MATRIX_PATH)
    ]
    try:
        run_legs = matrix_legs(file_at(repo, MATRIX_PATH, ran[0])) if ran else []
    except subprocess.CalledProcessError:
        run_legs = []
    if not run_legs:
        return "the matrix this run executed could not be read"
    pr_files = pr_changed_files(repo, pr["number"])
    legs = {}
    if MATRIX_PATH in pr_files:
        # Against the merge base, so legs the base branch added since don't read as removed.
        cmp = gh_json(f"/repos/{repo}/compare/{pr['base']['ref']}...{run['head_sha']}")
        head_legs = matrix_legs(file_at(repo, MATRIX_PATH, run["head_sha"]))
        fork_legs = matrix_legs(file_at(repo, MATRIX_PATH, cmp["merge_base_commit"]["sha"]))
        legs = leg_delta(tier_leg_counts(head_legs), tier_leg_counts(fork_legs))
    failed = list(newest_by_name(artifacts, f"failed-suite-{run.get('run_attempt', 1)}-"))
    suites = slot_suites(run_legs, failed)

    junit_meta = newest_by_name(artifacts, "junit-")
    # A green run can still miss legs (e.g. a runner that never came up, which the
    # pod-level retry doesn't catch); their tests would read as removed.
    missing = missing_legs(run_legs, junit_meta)
    if missing:
        return f"{len(missing)} {TIER} legs uploaded no JUnit report ({', '.join(missing[:5])})"
    with ThreadPoolExecutor(max_workers=8) as pool:
        data = list(
            pool.map(lambda a: read_artifact(repo, a, ".json", _durations), slices_meta.values())
        )
        cases = list(
            pool.map(lambda a: read_artifact(repo, a, ".xml", junit_cases), junit_meta.values())
        )
    if any(d is None for d in data) or any(c is None for c in cases):
        return "some of the PR run's duration or JUnit artifacts could not be read"
    slices = {name[len("durations-data-") :]: d for name, d in zip(slices_meta, data)}
    pr_tests = merge_pr_legs(slices, suites)
    collected = set().union(*cases)

    bases = baseline_runs(repo, pr["base"]["ref"], run["created_at"])
    if not bases:
        return f"no green push run with test durations on {pr['base']['ref']} to compare against"
    base_only = base_only_test_files(repo, bases[0][0]["head_sha"], run, pr)
    if base_only is None:
        return f"{pr['base']['ref']} moved too far past its newest green baseline run"
    return compute(pr_tests, [d for _, d in bases], legs, collected, base_only)


def find_comment(repo: str, number: int) -> dict | None:
    pages = gh_json(f"/repos/{repo}/issues/{number}/comments?per_page=100", paginate=True)
    for page in pages:
        for c in page:
            if c["user"]["login"] == BOT_LOGIN and c.get("body", "").startswith(MARKER):
                return c
    return None


def upsert_comment(repo: str, number: int, body: str, report: Report) -> None:
    existing = find_comment(repo, number)
    if existing is None and not report.over:
        return
    if existing is not None:
        cmd = ["gh", "api", "-X", "PATCH", f"/repos/{repo}/issues/comments/{existing['id']}"]
    else:
        cmd = ["gh", "api", "-X", "POST", f"/repos/{repo}/issues/{number}/comments"]
    subprocess.run(cmd + ["-f", f"body={body}"], capture_output=True, check=True)
    print(f"{'Updated' if existing else 'Posted'} the CI time comment on PR #{number}.")


def publish(args, text: str, status: str = "") -> None:
    # In Actions the runner parses `::`/`##[` commands anywhere in stdout, and the text
    # holds PR-controlled node ids: there it goes only to files, stdout gets status.
    paths = [p for p in (os.environ.get("GITHUB_STEP_SUMMARY"), args.summary_file) if p]
    for path in paths:
        with open(path, "a") as f:
            f.write(text)
    if not os.environ.get("GITHUB_ACTIONS"):
        print(text)
    elif status:
        print(status)


def cmd_report(args) -> None:
    note = ""
    repo = args.repo
    if args.run_id:
        run = gh_json(f"/repos/{repo}/actions/runs/{args.run_id}")
        # workflow_run matches by name, which a fork PR can reuse.
        if (
            run.get("path") != f".github/workflows/{WORKFLOW}"
            or (run.get("repository") or {}).get("full_name") != repo
        ):
            print(f"Run {args.run_id} is not this repo's {WORKFLOW}; nothing to report.")
            return
        pr = find_pr(repo, run)
        if pr is None:
            print(f"No PR has head branch {run['head_branch']}; nothing to report.")
            return
        if args.post and pr["head"]["sha"] != run["head_sha"]:
            print(f"PR #{pr['number']} moved on to {pr['head']['sha'][:12]}; skipping stale run.")
            return
    else:
        pr = gh_json(f"/repos/{repo}/pulls/{args.pr}")
        run = latest_pr_run(repo, pr)
        if run is None:
            publish(args, f"PR #{args.pr} has no green {WORKFLOW} run yet.\n")
            return
        if run["head_sha"] != pr["head"]["sha"]:
            note = (
                f"Note: the head {pr['head']['sha'][:12]} has no green run yet; reporting the "
                f"run of the older commit {run['head_sha'][:12]}.\n\n"
            )

    report = build_report(repo, run, pr)
    if isinstance(report, str):
        publish(args, f"{note}## CI time report\n\nSkipped: {report}.\n")
        return
    publish(args, note + render_summary(report, run.get("html_url", "")), _headline(report))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "pr": pr["number"],
                    "run_id": run["id"],
                    "net_s": report.net_s,
                    "pct": report.pct,
                    "over": report.over,
                    "baseline_s": report.baseline_s,
                    "legs": report.leg_delta,
                    "new": [t.__dict__ for t in report.new],
                    "removed": report.removed,
                },
                f,
                indent=1,
            )
    if args.post:
        summary_url = ""
        if os.environ.get("GITHUB_RUN_ID"):
            server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
            summary_url = f"{server}/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        body = render_comment(report, run.get("html_url", ""), summary_url)
        upsert_comment(repo, pr["number"], body, report)


def cmd_replay(args) -> None:
    runs = []
    for path in args.files:
        with open(path) as f:
            runs.append(clean_durations(json.load(f)))
    fired = 0
    for i in range(BASELINE_RUNS, len(runs)):
        pr = {n: (s, "") for n, s in runs[i].items()}
        r = compute(pr, runs[i - BASELINE_RUNS : i][::-1])
        fired += r.over
        print(
            f"{args.files[i]}: new {len(r.new)} ({_min(r.new_s)}) removed {len(r.removed)} "
            f"({_min(r.removed_s)}) net {r.pct:+.1f}%{'  <- comment' if r.over else ''}"
        )
    print(
        f"Would comment on {fired}/{max(0, len(runs) - BASELINE_RUNS)} runs at {THRESHOLD_PCT:g}%."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("report", help="Report on one PR run")
    who = rep.add_mutually_exclusive_group(required=True)
    who.add_argument("--run-id", type=int, help="A pull_request test_each_commit run")
    who.add_argument("--pr", type=int, help="A PR number; uses its latest green run")
    rep.add_argument(
        "--repo", default=os.environ.get("GITHUB_REPOSITORY") or "torch-spyre/spyre-inference"
    )
    rep.add_argument("--post", action="store_true", help="Create or update the PR comment")
    rep.add_argument("--json", default="", help="Also write the report as JSON here")
    rep.add_argument("--summary-file", default="", help="Also append the Markdown report here")
    rep.set_defaults(func=cmd_report)
    rpl = sub.add_parser("replay", help="Replay the rule over consecutive test-durations.json")
    rpl.add_argument("files", nargs="+", help="Oldest first")
    rpl.set_defaults(func=cmd_replay)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
