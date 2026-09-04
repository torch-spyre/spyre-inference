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
Recommend a shard count per sharded suite from recorded per-test durations.

For each suite the plugin balances shards by measured runtime (see
tests/plugin/spyre_testing_plugin/sharding.py); this picks the *number* of shards
so the busiest shard's measured test time stays under a wall-clock budget. It only
reports -- the `rebalance-test-shards` skill applies the numbers to the Makefile
(`*_SHARDS`) and the CI matrix, which the count cross-check test keeps in lockstep.

Durations come from the newest *passing* run on the main branch (same resolver the
CI pin uses), or a local file via --durations. Suite membership is resolved by real
`pytest --collect-only -m "<expr>"`, so this needs an environment where collection
imports succeed (CI image or full local checkout; the upstream suite also needs the
upstream tests, added with --upstream).

Usage:
    python3 .github/scripts/rebalance_test_shards.py            # pull main durations
    python3 .github/scripts/rebalance_test_shards.py --durations test-durations.json
    python3 .github/scripts/rebalance_test_shards.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import tempfile

import resolve_durations_run

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The sharded suites. `makefile_name` names both the Makefile shard target
# (test-<name>-shard, source of the marker expression) and the *_SHARDS var; the
# marker expression itself is parsed from that target so the Makefile stays the one
# source of truth. `collect_upstream` marks the suite whose collection needs the
# upstream tests cloned (the others exclude upstream, so must not pull them in).
SUITES = [
    {"key": "smoke", "makefile_name": "smoke", "var": "SMOKE_SHARDS", "collect_upstream": False},
    {"key": "attn", "makefile_name": "attention", "var": "ATTN_SHARDS", "collect_upstream": False},
    {
        "key": "upstream",
        "makefile_name": "upstream",
        "var": "UPSTREAM_SHARDS",
        "collect_upstream": True,
    },
    {
        "key": "dist",
        "makefile_name": "distributed",
        "var": "DIST_SHARDS",
        "collect_upstream": False,
    },
    {"key": "probe", "makefile_name": "probes", "var": "PROBE_SHARDS", "collect_upstream": False},
]


def parse_makefile() -> dict[str, dict]:
    with open(os.path.join(_REPO_ROOT, "Makefile")) as f:
        text = f.read()
    out = {}
    for suite in SUITES:
        name = suite["makefile_name"]
        count = re.search(rf"^{suite['var']}\s*\?=\s*(\d+)", text, re.MULTILINE)
        # First MARK_OVERRIDE after the `test-<name>-shard:` target header is that
        # target's; non-greedy stops before the next target's recipe.
        expr = re.search(
            rf"^test-{name}-shard:.*?MARK_OVERRIDE='([^']*)'", text, re.DOTALL | re.MULTILINE
        )
        if not count or not expr:
            raise SystemExit(f"Makefile: could not parse {suite['var']} / test-{name}-shard marker")
        out[suite["key"]] = {"current": int(count.group(1)), "expr": expr.group(1)}
    return out


def load_durations(args) -> dict[str, float]:
    if args.durations:
        with open(args.durations) as f:
            raw = json.load(f)
    else:
        artifacts = resolve_durations_run.list_durations_artifacts(args.repo)
        run_id = resolve_durations_run.pick_run_id(args.repo, artifacts, args.branch, "")
        if not run_id:
            raise SystemExit(
                f"No passing test-durations artifact on '{args.branch}'. "
                "Pass --durations <file> to use a local durations JSON instead."
            )
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(
                [
                    "gh",
                    "run",
                    "download",
                    run_id,
                    "--repo",
                    args.repo,
                    "-n",
                    "test-durations",
                    "-D",
                    d,
                ],
                check=True,
            )
            with open(os.path.join(d, "test-durations.json")) as f:
                raw = json.load(f)
    return {str(k): float(v) for k, v in raw.items()}


def collect_nodeids(expr: str, want_upstream: bool) -> set[str]:
    # --active --no-sync mirrors the Makefile's run-one: plain `uv run` re-syncs from
    # pyproject and clobbers a locally built torch-spyre (see CLAUDE.md).
    cmd = ["uv", "run", "--active", "--no-sync", "pytest", "--collect-only", "-q", "-m", expr]
    if want_upstream:
        cmd.append("--upstream")
    result = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"`pytest --collect-only -m {expr!r}` failed (exit {result.returncode}). "
            "Run this where test collection imports succeed (CI image or full local env).\n"
            + result.stderr[-2000:]
        )
    # -q --collect-only prints one nodeid per line, then a blank line and a summary.
    return {
        ln.strip() for ln in result.stdout.splitlines() if "::" in ln and not ln.startswith(" ")
    }


def busiest_shard(weights: list[float], n: int) -> float:
    """Max shard load under the same greedy longest-processing-time packing CI uses."""
    loads = [0.0] * n
    for w in sorted(weights, reverse=True):
        i = min(range(n), key=lambda k: loads[k])
        loads[i] += w
    return max(loads)


def recommend(weights: list[float], budget_s: float) -> tuple[int, float, bool]:
    """Smallest shard count whose busiest shard fits budget_s; (n, busiest, over_budget)."""
    if not weights:
        return 1, 0.0, False
    max_item = max(weights)
    if max_item > budget_s:
        # One test alone exceeds the budget: no partition can fit. Best effort by total.
        n = max(1, math.ceil(sum(weights) / budget_s))
        n = min(n, len(weights))
        return n, busiest_shard(weights, n), True
    for n in range(1, len(weights) + 1):
        b = busiest_shard(weights, n)
        if b <= budget_s:
            return n, b, False
    return len(weights), busiest_shard(weights, len(weights)), False


def _fmt(seconds: float) -> str:
    return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--durations", help="Local durations JSON; skips the gh download.")
    ap.add_argument("--repo", default=os.environ.get("REPO", "torch-spyre/spyre-inference"))
    ap.add_argument("--branch", default="main", help="Branch whose newest passing run is used.")
    ap.add_argument("--max-minutes", type=float, default=10.0, help="Per-shard test-time ceiling.")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = ap.parse_args()

    makefile = parse_makefile()
    durations = load_durations(args)
    budget_s = args.max_minutes * 60

    rows = []
    for suite in SUITES:
        key = suite["key"]
        expr = makefile[key]["expr"]
        nodeids = collect_nodeids(expr, suite["collect_upstream"])
        weights = [durations[n] for n in nodeids if n in durations]
        unmeasured = len(nodeids) - len(weights)
        rec_n, busiest, over = recommend(weights, budget_s)
        rows.append(
            {
                "suite": key,
                "current": makefile[key]["current"],
                "measured_total_s": round(sum(weights), 1),
                "max_single_s": round(max(weights), 1) if weights else 0.0,
                "recommended": rec_n,
                "busiest_at_recommended_s": round(busiest, 1),
                "unmeasured": unmeasured,
                "over_budget": over,
            }
        )

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print(f"\nBudget: busiest shard <= {args.max_minutes:g} min of measured test time.\n")
    hdr = f"{'suite':<10}{'cur':>4}{'total':>9}{'max1':>8}{'REC':>5}{'busiest':>9}{'new':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = "  <- single test over budget" if r["over_budget"] else ""
        change = "" if r["recommended"] == r["current"] else f"  (was {r['current']})"
        print(
            f"{r['suite']:<10}{r['current']:>4}{_fmt(r['measured_total_s']):>9}"
            f"{_fmt(r['max_single_s']):>8}{r['recommended']:>5}"
            f"{_fmt(r['busiest_at_recommended_s']):>9}{r['unmeasured']:>5}{change}{flag}"
        )
    print(
        "\ncur=current shards, total=measured suite time, max1=slowest single test, "
        "REC=recommended shards, busiest=predicted busiest shard at REC, new=collected "
        "tests with no recorded duration (excluded from the estimate)."
    )
    if any(r["recommended"] != r["current"] for r in rows):
        print("\nApply the REC column to the Makefile *_SHARDS and the CI matrix (see the skill).")


if __name__ == "__main__":
    main()
