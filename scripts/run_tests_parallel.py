#!/usr/bin/env python3
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

"""Run the CI shard targets of `make tests` concurrently, one shard per card set.

`make tests TEST_TYPE=regression|trunk PARALLEL=auto|N` calls this with the same target
list that the serial run walks (SINGLE_CARD_TARGETS + MULTI_CARD_TARGETS). Each target's
card count comes from its `spyre_pf_x<N>` runner label in _test_matrix.yaml, so a shard
gets the cards CI gives it. Two shards never share a card.

Before the run, every target is collected (`--collect-only`). The run fails when a shard
collects 0 tests, when two shards collect the same test, or when the shards of a family
do not add up to the unsharded collection. After the run, each shard's JUnit file must
hold as many test cases as the shard collected.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

CARD_SCOPE_ENV = "SPYRE_TEST_CARD_SCOPE"  # read by spyre_testing_plugin.vfio_reaper
WORKER_ENV = "SPYRE_PARALLEL_WORKER"  # tells the Makefile's AIU_SETUP_CMD not to re-run setup

# Sharded families whose unsharded target runs the same marker expression without the
# shard flags (tests/test_run_tests_parallel.py checks that against the Makefile).
FAMILIES = {
    "test-smoke-shard": "test-smoke",
    "test-quality-shard": "test-quality",
    "test-attention-shard": "test-attention",
    "test-upstream-shard": "test-upstream",
    "test-distributed-shard": "test-distributed",
    "test-probes-shard": "test-probes",
}
_SHARD_RE = re.compile(r"^(test-[a-z-]+-shard)-\d+$")
# A node id: a path, then "::". Excludes "::error::" style CI annotations.
_NODE_ID_RE = re.compile(r"^[^\s:]\S*::\S")


def log(msg: str) -> None:
    print(f"[parallel] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Inputs: card needs from the CI matrix, cards on this host
# ---------------------------------------------------------------------------


def card_needs_from_matrix(matrix_path: Path) -> dict[str, int]:
    """Map each x86_64 test_target in _test_matrix.yaml to its spyre_pf_x<N> card count."""
    import yaml

    doc = yaml.safe_load(matrix_path.read_text())
    needs: dict[str, int] = {}
    for entry in doc["jobs"]["test"]["strategy"]["matrix"]["include"]:
        labels = entry.get("runs_on", [])
        if "x86_64" not in labels:
            continue
        for label in labels:
            m = re.fullmatch(r"spyre_pf_x(\d+)", label)
            if m:
                needs[entry["test_target"]] = int(m.group(1))
    return needs


def detect_devices(env: dict[str, str], requested: str) -> list[str]:
    """Spyre device indices to spread shards over, or raise ValueError.

    Order: SPYRE_DEVICES (a caller's restriction, kept as is), then the device plugin's
    PCIDEVICE_IBM_COM_AIU_PF list, then AIU_WORLD_SIZE (set by ibm-aiu-setup.sh)."""
    if env.get("SPYRE_DEVICES", "").strip():
        found = [d.strip() for d in env["SPYRE_DEVICES"].split(",") if d.strip()]
    elif env.get("PCIDEVICE_IBM_COM_AIU_PF", "").strip():
        n = len([p for p in env["PCIDEVICE_IBM_COM_AIU_PF"].split(",") if p.strip()])
        found = [str(i) for i in range(n)]
    else:
        try:
            n = int(env.get("AIU_WORLD_SIZE", "0") or 0)
        except ValueError:
            n = 0
        found = [str(i) for i in range(n)]
    if not found:
        raise ValueError(
            "no Spyre cards detected (SPYRE_DEVICES, PCIDEVICE_IBM_COM_AIU_PF and "
            "AIU_WORLD_SIZE are all empty)"
        )
    if requested in ("", "auto"):
        return found
    if not requested.isdigit() or int(requested) < 1:
        raise ValueError(f"PARALLEL must be 'auto' or a positive integer, got {requested!r}")
    if int(requested) > len(found):
        raise ValueError(
            f"PARALLEL={requested} but only {len(found)} card(s) detected: {','.join(found)}"
        )
    return found[: int(requested)]


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


class CardPool:
    """Hands out disjoint sets of device indices; blocks until enough are free."""

    def __init__(self, devices: list[str]):
        self.devices = list(devices)
        self._free = list(devices)
        self._cv = threading.Condition()

    def effective_need(self, need: int) -> int:
        # As in a serial run, a job that needs more cards than the host has gets them all
        # (its tests skip on the missing cards).
        return max(1, min(need, len(self.devices)))

    def acquire(self, need: int) -> list[str]:
        need = self.effective_need(need)
        with self._cv:
            self._cv.wait_for(lambda: len(self._free) >= need)
            taken, self._free = self._free[:need], self._free[need:]
            return taken

    def release(self, cards: list[str]) -> None:
        with self._cv:
            self._free.extend(cards)
            self._free.sort(key=self.devices.index)
            self._cv.notify_all()


def run_order(targets: list[str], needs: dict[str, int]) -> list[str]:
    """Widest jobs first, so a 4-card job is not starved by a queue of 1-card jobs."""
    return sorted(targets, key=lambda t: -needs[t])


def inductor_cache_dir(base: dict[str, str]) -> str:
    # torch's default when TORCHINDUCTOR_CACHE_DIR is unset.
    return base.get("TORCHINDUCTOR_CACHE_DIR") or os.path.join(
        tempfile.gettempdir(), f"torchinductor_{getpass.getuser()}"
    )


def card_cache_dir(base: dict[str, str], card: str) -> str:
    # Concurrent jobs sharing one Inductor cache reused mismatched graphs (CI fix, PR 591),
    # so each job uses the cache of its first card, which no other running job holds.
    return f"{inductor_cache_dir(base)}__card_{card}"


def seed_card_caches(base: dict[str, str], devices: list[str]) -> None:
    """Copy the existing Inductor cache to each card's cache once, so shards start warm."""
    src = Path(inductor_cache_dir(base))
    if not src.is_dir():
        return
    for card in devices:
        dest = Path(card_cache_dir(base, card))
        if not dest.exists():
            shutil.copytree(src, dest, symlinks=True)


def job_env(base: dict[str, str], cards: list[str]) -> dict[str, str]:
    env = dict(base)
    env["SPYRE_DEVICES"] = ",".join(cards)
    env["AIU_WORLD_SIZE"] = str(len(cards))
    env[CARD_SCOPE_ENV] = f"cards-{'-'.join(cards)}-{os.getpid()}"
    env[WORKER_ENV] = "1"
    env["TORCHINDUCTOR_CACHE_DIR"] = card_cache_dir(base, cards[0])
    return env


# ---------------------------------------------------------------------------
# Parity guard
# ---------------------------------------------------------------------------


def parse_collected(output: str) -> list[str]:
    """Node ids from `pytest --collect-only -q` output."""
    return [line.strip() for line in output.splitlines() if _NODE_ID_RE.match(line.strip())]


def shard_family(target: str) -> str | None:
    """The unsharded target for a shard target, or None."""
    m = _SHARD_RE.match(target)
    return FAMILIES.get(m.group(1)) if m else None


def check_collection(collected: dict[str, list[str]], families: dict[str, list[str]]) -> list[str]:
    """Errors for empty shards, a test in two shards, or shards that miss family tests.

    `collected` maps every shard target and every family's unsharded target to its ids;
    `families` maps each unsharded target to its shard targets."""
    errors = []
    for target, ids in collected.items():
        if not ids:
            errors.append(f"{target}: collected 0 test IDs")
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        if dupes:
            errors.append(f"{target}: {len(dupes)} test ID(s) collected twice")
    for family, shards in families.items():
        owner: dict[str, str] = {}
        for shard in shards:
            for node in collected.get(shard, []):
                if node in owner and owner[node] != shard:
                    errors.append(f"{node}: in both {owner[node]} and {shard}")
                owner.setdefault(node, shard)
        want = set(collected.get(family, []))
        missing = want - owner.keys()
        extra = owner.keys() - want
        if missing:
            errors.append(
                f"{family}: {len(missing)} test(s) in no shard, e.g. {sorted(missing)[0]}"
            )
        if extra:
            errors.append(
                f"{family}: {len(extra)} shard test(s) not in the unsharded run, "
                f"e.g. {sorted(extra)[0]}"
            )
    return errors


def junit_case_count(path: Path) -> int:
    return sum(1 for _ in ET.parse(path).getroot().iter("testcase"))


def check_results(expected: dict[str, int], junit_paths: dict[str, Path]) -> list[str]:
    errors = []
    for target, want in expected.items():
        path = junit_paths[target]
        if not path.is_file():
            errors.append(f"{target}: no JUnit file at {path}")
            continue
        try:
            got = junit_case_count(path)
        except ET.ParseError as e:
            errors.append(f"{target}: unreadable JUnit file {path}: {e}")
            continue
        if got != want:
            errors.append(f"{target}: collected {want} test(s) but JUnit has {got}")
    return errors


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class Result:
    target: str
    rc: int
    seconds: float
    cards: list[str] = field(default_factory=list)


def collect(make: list[str], target: str, env: dict[str, str]) -> list[str]:
    cmd = [*make, target, "PYTEST_ARGS=--collect-only -q -p no:cacheprovider", "JUNIT_XML="]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)  # noqa: S603
    ids = parse_collected(proc.stdout)
    # pytest exits 5 when nothing is collected; check_collection reports that case.
    if proc.returncode not in (0, 5):
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-20:])
        raise RuntimeError(f"collecting {target} failed (rc={proc.returncode}):\n{tail}")
    return ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", nargs="+", help="make targets, as the serial run lists them")
    ap.add_argument("--cards", default="auto", help="auto, or how many cards to use")
    ap.add_argument("--results-dir", type=Path, default=Path("."))
    ap.add_argument("--matrix", type=Path, default=Path(".github/workflows/_test_matrix.yaml"))
    ap.add_argument("--make", default=os.environ.get("MAKE", "make"))
    args = ap.parse_args(argv)

    make = [args.make, "--no-print-directory"]
    needs_all = card_needs_from_matrix(args.matrix)
    unknown = [t for t in args.targets if t not in needs_all]
    if unknown:
        log(f"ERROR: no spyre_pf_x<N> runner in {args.matrix} for: {' '.join(unknown)}")
        return 2
    needs = {t: needs_all[t] for t in args.targets}
    try:
        devices = detect_devices(dict(os.environ), args.cards)
    except ValueError as e:
        log(f"ERROR: {e}")
        return 2
    pool = CardPool(devices)
    log(f"{len(args.targets)} target(s) across {len(devices)} card(s): {','.join(devices)}")

    # 1. Collect. The upstream family goes first, alone: a cold run clones vLLM into a
    #    shared cache, and concurrent clones would race.
    # A family is checked against its unsharded run only when all its shards run here.
    families: dict[str, list[str]] = {}
    for t in needs_all:
        fam = shard_family(t)
        if fam:
            families.setdefault(fam, []).append(t)
    families = {f: s for f, s in families.items() if set(s) <= set(args.targets)}
    to_collect = list(dict.fromkeys([*families, *args.targets]))
    to_collect.sort(key=lambda t: not t.startswith("test-upstream"))
    base_env = dict(os.environ)
    base_env[WORKER_ENV] = "1"
    start = time.monotonic()
    collected: dict[str, list[str]] = {}
    try:
        collected[to_collect[0]] = collect(make, to_collect[0], base_env)
        with ThreadPoolExecutor(max_workers=len(devices)) as ex:
            futures = {t: ex.submit(collect, make, t, base_env) for t in to_collect[1:]}
            for t, fut in futures.items():
                collected[t] = fut.result()
    except RuntimeError as e:
        log(f"ERROR: {e}")
        return 2
    errors = check_collection(collected, families)
    total = sum(len(collected[t]) for t in args.targets)
    log(f"collected {total} test(s) in {time.monotonic() - start:.0f}s")
    for t in args.targets:
        log(f"  {t}: {len(collected[t])} test(s), {needs[t]} card(s)")
    if errors:
        for e in errors:
            log(f"ERROR: {e}")
        log("parity guard failed; not running the tests")
        return 3

    # 2. Run, widest first, each on its own cards.
    t0 = time.monotonic()
    seed_card_caches(dict(os.environ), devices)
    log(f"seeded per-card Inductor caches in {time.monotonic() - t0:.0f}s")
    logs_dir = args.results_dir / "parallel-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    junit = {t: args.results_dir / f"junit-{t}" / f"junit-{t}.xml" for t in args.targets}

    def run(target: str) -> Result:
        cards = pool.acquire(needs[target])
        t0 = time.monotonic()
        try:
            junit[target].parent.mkdir(parents=True, exist_ok=True)
            log(f"start {target} on card(s) {','.join(cards)}")
            with open(logs_dir / f"{target}.log", "w") as out:
                rc = subprocess.call(  # noqa: S603
                    [*make, target, f"JUNIT_XML={junit[target].resolve()}"],
                    env=job_env(dict(os.environ), cards),
                    stdout=out,
                    stderr=subprocess.STDOUT,
                )
        finally:
            pool.release(cards)
        res = Result(target, rc, time.monotonic() - t0, cards)
        log(f"done  {target} rc={rc} in {res.seconds:.0f}s (log: {logs_dir / (target + '.log')})")
        return res

    start = time.monotonic()
    width = len(devices)  # never more jobs than cards; CardPool keeps them disjoint
    with ThreadPoolExecutor(max_workers=width) as ex:
        results = list(ex.map(run, run_order(args.targets, needs)))
    wall = time.monotonic() - start

    # 3. Every shard ran what it collected.
    errors = check_results({t: len(collected[t]) for t in args.targets}, junit)
    failed = [r for r in results if r.rc != 0]
    log(f"wall time {wall:.0f}s; sum of shard times {sum(r.seconds for r in results):.0f}s")
    for r in failed:
        log(f"FAILED {r.target} rc={r.rc}")
    for e in errors:
        log(f"ERROR: {e}")
    return 1 if failed or errors else 0


if __name__ == "__main__":
    sys.exit(main())
