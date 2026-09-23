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

"""The runtime guard on the shared identity library.

ingest_vllm_benchmarks installs that library from a floating `@main`, so what CI verified
and what the production job resolved are two different snapshots. These tests cover the
guard that closes the gap: the golden it carries, and that a drift takes the v2 write out
rather than writing rows nothing can join.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import types

import pytest
from ingest_identity import golden_drift, library_provenance

_SCRIPTS = pathlib.Path(__file__).resolve().parent


def _load(name, **stubs):
    for mod_name, mod in stubs.items():
        sys.modules.setdefault(mod_name, mod)
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except ModuleNotFoundError as exc:
        pytest.skip(f"ingest deps unavailable: {exc}")
    return m


@pytest.fixture(scope="module")
def vllm_mod():
    utils = types.ModuleType("utils")
    utils.read_benchmark_results = lambda *a, **k: []
    return _load("ingest_vllm_benchmarks", utils=utils)


# --- the goldens each writer carries ----------------------------------------------------
# ingest_xml_si no longer carries its own goldens: schema-v2 for JUnit XML is written by
# push-to-clickhouse.yaml calling torch-spyre's ingest-xml-to-clickhouse action, so the
# identity contract for that path is torch-spyre's own to guard, not duplicated here.


def test_vllm_goldens_hold_against_the_installed_library(vllm_mod):
    # The tripwire, checked against whatever snapshot is installed here. It is the same
    # constant the job evaluates, so CI cannot pass on goldens the ingest does not use.
    assert golden_drift(vllm_mod.IDENTITY_GOLDENS) == []


def test_every_golden_names_its_function_and_both_values(vllm_mod):
    # The message is the whole output of a drift: it has to say which hash moved and to what,
    # or the on-call reader cannot tell a rename from a re-key. The NAME is a caller-supplied
    # label, not fn.__name__: the library re-exports every identity function as a bound
    # `<Class>.derive`, so introspecting the callable would make every drift message read
    # "derive" and tell the reader nothing.
    (line,) = golden_drift(
        [("run_id_of", vllm_mod.run_id_of, ("gha", "1", "amd64", "perf"), "nope")]
    )
    assert "run_id_of" in line and "nope" in line
    assert str(vllm_mod.run_id_of("gha", "1", "amd64", "perf")) in line


def test_a_renamed_or_re_keyed_function_is_drift(vllm_mod):
    assert golden_drift([("canonical_arch", vllm_mod.canonical_arch, ("amd64",), "x86_64")]) == []
    assert golden_drift([("canonical_arch", vllm_mod.canonical_arch, ("amd64",), "amd64")]) != []


# --- provenance -------------------------------------------------------------------------


def test_provenance_is_reported_and_never_raises():
    # Logged on every v2 run, so it must degrade to a string rather than take the ingest out
    # when the metadata is missing (a vendored copy, a path install, a stripped image).
    got = library_provenance()
    assert isinstance(got, str) and got


# --- what a drift costs -----------------------------------------------------------------


class _Client:
    """Accepts every table and records what was sent, keyed by (database, table)."""

    def __init__(self):
        self.sent: dict[tuple, list] = {}

    def command(self, _q):
        return 1

    def query(self, q, parameters=None):
        if "system.columns" in q:
            from spyre_clickhouse_ingest import schema

            cols = schema.TABLES[parameters["t"]].columns
            return types.SimpleNamespace(result_rows=[(c,) for c in cols])
        return types.SimpleNamespace(result_rows=[])

    def insert(self, table, rows, column_names=None, database=None):
        self.sent.setdefault((database, table), []).extend(rows)


def _flat():
    extra = {
        "device": "spyre",
        "arch": "x86_64",
        "hardware_type": "IBM_Spyre",
        "model": "granite",
        "test_name": "latency_tp1_in64_out64",
        "head_sha": "deadbeef",
    }
    return {
        "timestamp": 1,
        "schema_version": "v3",
        "name": "spyre_e2e_benchmark",
        "metric": "avg_latency",
        "actual": 1.5,
        "target": 0.0,
        "repo": "spyre-inference",
        "head_branch": "main",
        "workflow_id": 7,
        "job_id": 9,
        "run_attempt": 1,
        "extra": json.dumps(extra),
    }


def _run_ingest(vllm_mod, monkeypatch, goldens):
    client = _Client()
    monkeypatch.setattr(vllm_mod, "IDENTITY_GOLDENS", goldens)
    monkeypatch.setattr(
        vllm_mod, "clickhouse_connect", types.SimpleNamespace(get_client=lambda **kw: client)
    )
    for k, v in dict(
        CLICKHOUSE_HOST="h",
        CLICKHOUSE_USER="u",
        CLICKHOUSE_PASS="p",
        CLICKHOUSE_DB="spyre",
        CLICKHOUSE_DB_V2="spyre_v2",
    ).items():
        monkeypatch.setitem(os.environ, k, v)
    vllm_mod.insert_to_clickhouse([_flat()], "dab2a67f-14bf-53be-b6e4-fc9642086e47")
    return {db for db, _t in client.sent}


def test_intact_goldens_let_the_v2_write_through(vllm_mod, monkeypatch):
    assert "spyre_v2" in _run_ingest(vllm_mod, monkeypatch, vllm_mod.IDENTITY_GOLDENS)


def test_drift_takes_out_v2_and_leaves_the_flat_tables(vllm_mod, monkeypatch):
    # Refusing is the point: rows minted by a library nobody else is running would be
    # orphans, and an orphan reads downstream as "no perf ran".
    broken = (("run_id_of", vllm_mod.run_id_of, ("gha", "1", "amd64", "perf"), "not-the-id"),)
    written = _run_ingest(vllm_mod, monkeypatch, broken)
    assert "spyre_v2" not in written
    assert None in written, "results_v3 / run_metadata must still be written"
