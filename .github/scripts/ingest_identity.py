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

"""Runtime guard on the shared identity library, for both v2 ingests.

Four writers -- these two scripts, torch-spyre's ingest_xml.py and the orchestrator's
pushToClickhouse -- must hash a run to the SAME uuid with nothing threaded between them.
That holds only while all four execute the same library code, and every one of them
installs it from `@main` at its own job-start time. A library change landing mid-run gives
the orchestrator's artifact_results row and this ingest's test_case_runs two run_ids for one
run: no error is raised, and the orphan is indistinguishable downstream from "no tests ran".

The golden tests beside this file pin whichever snapshot CI installed, not the one the
production job resolved, so the check has to run in the job that writes. Each consumer
declares the goldens IT depends on and refuses its v2 write when the installed library no
longer mints them -- v1 is unaffected, so a drift costs visibility, never data.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, distribution

_DIST = "spyre-clickhouse-ingest"


def library_provenance() -> str:
    """Which build of the shared library this process is about to mint ids with.

    Recorded by the installer under PEP 610, which is the only place the commit behind a
    floating `@main` survives -- so a row written today can still be attributed to the code
    that wrote it after main has moved.
    """
    try:
        dist = distribution(_DIST)
        origin = json.loads(dist.read_text("direct_url.json") or "{}")
    except (PackageNotFoundError, ValueError, OSError):
        return f"{_DIST} (provenance unavailable)"
    vcs = origin.get("vcs_info") or {}
    if vcs.get("commit_id"):
        return (
            f"{_DIST} {vcs['commit_id']} ({vcs.get('requested_revision') or origin.get('url', '')})"
        )
    return f"{_DIST} {dist.version} ({origin.get('url', 'no recorded origin')})"


def golden_drift(goldens) -> list[str]:
    """The identities this writer was built against that the installed library no longer
    mints, as human-readable lines. Empty means the contract still holds.

    `goldens` is (name, callable, args, expected). `name` is a caller-supplied label, not
    `callable.__name__`: the library re-exports these as bound methods (`RunId.derive`, ...),
    so every one of them reports `__name__ == "derive"` -- introspecting the callable would
    make every drift message identical and useless for telling which identity moved.
    """
    drift = []
    for name, fn, args, want in goldens:
        got = str(fn(*args))
        if got != want:
            drift.append(f"{name}{args!r} -> {got}, expected {want}")
    return drift
