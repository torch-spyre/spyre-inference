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

"""The vLLM v2 write path: the entry shaping, and what the shared writer makes of it.

benchmark_runs is the ONLY perf fact written -- the HUD's oss_ci_benchmark_v3 pair is a
materialized view over it -- so everything the dashboard reads has to survive this shaping.
The collapse and merge assertions run through the real insert_benchmarks rather than a local
copy of its rules, because a reimplementation here would stay green while the library moved.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parent / "ingest_vllm_benchmarks.py"
_RUN = "dab2a67f-14bf-53be-b6e4-fc9642086e47"


@pytest.fixture(scope="module")
def mod():
    stub = types.ModuleType("utils")
    stub.read_benchmark_results = lambda *a, **k: []
    sys.modules.setdefault("utils", stub)
    spec = importlib.util.spec_from_file_location("ingest_vllm_benchmarks", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except ModuleNotFoundError as exc:
        pytest.skip(f"ingest deps unavailable: {exc}")
    return m


class _Client:
    """Captures what the writer would send. `query` answers the identity-dedup probe with
    no known ids, so every identity row this leg derives is offered for insert."""

    def __init__(self):
        self.inserted: dict[str, list] = {}

    def query(self, *_a, **_k):
        return types.SimpleNamespace(result_rows=[])

    def insert(self, table, rows, column_names=None, database=None):
        self.inserted.setdefault(table, []).extend(
            [dict(zip(column_names, r, strict=True)) for r in rows]
        )


def _flat(metric="latency", actual=1.5, test_name="latency_tp1_in64_out64", **extra):
    """One flat results_v3 row, the only input shape the v2 write path accepts."""
    base = {
        "test_name": test_name,
        "head_sha": "abc123",
        "model": "granite",
        "device": "spyre",
        "arch": "x86_64",
        "hardware_type": "IBM_Spyre",
    }
    base.update(extra)
    return {
        "timestamp": 1,
        "repo": "torch-spyre/spyre-inference",
        "head_branch": "main",
        "workflow_id": 7,
        "run_attempt": 1,
        "job_id": 9,
        "metric": metric,
        "actual": actual,
        "target": 2.0,
        "extra": json.dumps(base),
    }


def _write(mod, rows):
    """Run the real writer over these flat rows and return (benchmarks, benchmark_runs)."""
    from spyre_clickhouse_ingest import insert_benchmarks

    client = _Client()
    insert_benchmarks(
        client, "v2", mod.BENCH_COMPONENT, _RUN, mod._bench_entries(rows), report_kind="vllm"
    )
    return client.inserted.get("benchmarks", []), client.inserted.get("benchmark_runs", [])


# --- _parse_input_shapes ---------------------------------------------------------------


def test_parse_input_shapes_reads_all_three_discriminators(mod):
    assert mod._parse_input_shapes("latency_tp4_in128_out256") == {
        "tensor_parallel": "4",
        "input_len": "128",
        "output_len": "256",
    }


@pytest.mark.parametrize("name", ["", "latency", "serve_tpX_inY", "throughput_tp_in_out"])
def test_parse_input_shapes_ignores_non_numeric_tokens(mod, name):
    # A bare prefix with no digits is not a shape; emitting one would fabricate a
    # discriminator and split one benchmark's history in two.
    assert mod._parse_input_shapes(name) == {}


def test_shapes_discriminate_two_runs_of_one_benchmark(mod):
    # tp1 and tp4 were indistinguishable in the flat table, and they are hash inputs here.
    idents_a, _ = _write(mod, [_flat(test_name="latency_tp1_in64_out64")])
    idents_b, _ = _write(mod, [_flat(test_name="latency_tp4_in64_out64")])
    assert idents_a[0]["benchmark_id"] != idents_b[0]["benchmark_id"]


# --- entry shaping ---------------------------------------------------------------------


def test_entries_carry_what_the_flat_write_dropped(mod):
    (entry,) = mod._bench_entries([_flat()])
    assert entry["name"] == "latency_tp1_in64_out64", "the flat write hardcodes one constant"
    assert entry["backend"] == "spyre", "backend is the HUD's pivot axis"
    assert entry["props"]["model"] == "granite"
    assert entry["measurements"] == {"latency": [1.5]}, "samples are an Array, not a scalar"


def test_run_props_carry_the_ci_coordinates_the_hud_view_reads(mod):
    # oss_ci_benchmark_v3_mv reads these off benchmark_runs.props; it cannot see them
    # otherwise, and a guess would put a wrong commit on a chart.
    (entry,) = mod._bench_entries([_flat()])
    assert entry["run_props"] == {
        "repo": "torch-spyre/spyre-inference",
        "head_branch": "main",
        "workflow_id": "7",
        "run_attempt": "1",
        "job_id": "9",
        "head_sha": "abc123",
        "arch": "x86_64",
        "hardware_type": "IBM_Spyre",
    }


def test_metric_samples_stay_floats(mod):
    (entry,) = mod._bench_entries([_flat(actual="2.5")])
    (samples,) = entry["measurements"].values()
    assert samples == [2.5] and all(isinstance(s, float) for s in samples)


def test_run_mode_comes_from_the_test_name_prefix(mod):
    for name, mode in (("serve_tp1", "serve"), ("throughput_tp1", "throughput")):
        (entry,) = mod._bench_entries([_flat(test_name=name)])
        assert entry["props"]["run_mode"] == mode


def test_which_file_reported_it_does_not_split_the_benchmark(mod):
    # The writer reads both the native json and the .pytorch.json copy, and a benchmark_id is
    # a content hash of the name -- so the two must converge before they reach the hash.
    assert mod._test_name("latency_tp1.json") == mod._test_name("latency_tp1.pytorch.json")
    a = _flat(test_name="latency_tp1")
    b = _flat(test_name="latency_tp1", metric="p90")
    idents, _facts = _write(mod, [a, b])
    assert len(idents) == 1


def test_unnamed_benchmarks_are_skipped_not_merged(mod):
    # An id over a blank name would collide every unidentifiable benchmark onto one identity.
    assert mod._bench_entries([_flat(test_name="")]) == []


def test_iterations_stays_zero(mod):
    # 0 is the column's "the producer did not say": vLLM reports a pre-averaged value per
    # metric and never the n behind it. insert_benchmarks SUMS iterations across the merged
    # entries, so any per-metric placeholder would add up to the metric count.
    flat = [_flat(metric=m) for m in ("p50", "p90", "p99")]
    _idents, (fact,) = _write(mod, flat)
    assert fact["iterations"] == 0


# --- what the shared writer makes of them ----------------------------------------------


def test_many_metrics_of_one_benchmark_collapse_to_one_row(mod):
    # 26 metrics of one benchmark are ONE measurement. A row per metric would multiply
    # every trend point by the metric count.
    flat = [_flat(metric=m, actual=i) for i, m in enumerate(["p50", "p90", "p99", "mean"])]
    idents, facts = _write(mod, flat)
    assert len(facts) == 1, f"expected one collapsed row, got {len(facts)}"
    assert set(facts[0]["measurements"]) == {"p50", "p90", "p99", "mean"}
    assert len(idents) == 1


def test_backend_splits_facts_but_not_identity(mod):
    # backend is a COLUMN, never a hash input: it is the axis a cross-backend comparison
    # pivots on, so folding it into identity would make the two sides different benchmarks.
    idents, facts = _write(mod, [_flat(device="spyre"), _flat(device="cpu")])
    assert len(idents) == 1, "one benchmark identity across backends"
    assert {f["backend"] for f in facts} == {"spyre", "cpu"}
    assert len({f["benchmark_id"] for f in facts}) == 1


def test_two_files_reporting_one_benchmark_merge_richer_props(mod):
    # The merge must not lose a field when the same benchmark arrives twice.
    a = _flat(test_name="latency_tp1_in64_out64", model="latency_tp1_in64_out64")
    b = _flat(test_name="latency_tp1_in64_out64", model="granite-3b")
    (ident,), _facts = _write(mod, [a, b])
    assert ident["props"]["tensor_parallel"] == "1"
    assert ident["props"]["model"] == "granite-3b"


def test_repeated_metrics_of_one_benchmark_become_samples(mod):
    # Map(String, Array(Float64)) exists to keep both: overwriting froze variance at zero.
    _idents, (fact,) = _write(
        mod, [_flat(metric="p50", actual=1.0), _flat(metric="p50", actual=2.0)]
    )
    assert fact["measurements"]["p50"] == [1.0, 2.0]


def test_run_id_and_report_kind_are_stamped_on_every_fact_row(mod):
    # report_kind scopes the dedup probe, so it must reach the row the probe reads.
    flat = [_flat(metric="p50"), _flat(metric="p90", test_name="serve_tp1_in8_out8")]
    _idents, facts = _write(mod, flat)
    assert facts and all(f["run_id"] == _RUN for f in facts)
    assert all(f["props"]["report_kind"] == "vllm" for f in facts)
