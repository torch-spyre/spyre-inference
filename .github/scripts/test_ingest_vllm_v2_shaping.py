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

"""The vLLM v2 reshaping path: Array/Map shaping, the per-metric collapse, and dedup.

These functions derive the nested tables from the SAME rows the flat table gets, so a bug
here shows up as two tables disagreeing about shape while agreeing about numbers -- which no
value assertion on the flat write can catch.
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


def _flat(metric="latency", actual=1.5, test_name="latency_tp1_in64_out64", **extra):
    """One flat results_v3 row, the only input shape the reshaping path accepts."""
    base = {"test_name": test_name, "head_sha": "abc123", "model": "granite", "device": "spyre"}
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


# --- _parse_input_shapes ---------------------------------------------------------------


def test_parse_input_shapes_reads_all_three_discriminators(mod):
    got = mod._parse_input_shapes("latency_tp4_in128_out256")
    assert got == {
        "tensor_parallel": ("int", {"value": "4"}),
        "input_len": ("int", {"value": "128"}),
        "output_len": ("int", {"value": "256"}),
    }


@pytest.mark.parametrize("name", ["", "latency", "serve_tpX_inY", "throughput_tp_in_out"])
def test_parse_input_shapes_ignores_non_numeric_tokens(mod, name):
    # A bare prefix with no digits is not a shape; emitting one would fabricate a
    # discriminator and split one benchmark's history in two.
    assert mod._parse_input_shapes(name) == {}


def test_shapes_discriminate_two_runs_of_one_benchmark(mod):
    # The whole point of the Map: tp1 and tp4 were indistinguishable in the flat table.
    assert mod._parse_input_shapes("latency_tp1_in64_out64") != mod._parse_input_shapes(
        "latency_tp4_in64_out64"
    )


# --- to_vllm_v3_rows ------------------------------------------------------------------


def test_v3_rows_recover_what_the_flat_write_dropped(mod):
    (row,) = mod.to_vllm_v3_rows([_flat()], _RUN)
    assert row["run_id"] == _RUN
    assert row["head_sha"] == "abc123", "blank on every flat row; recovered here"
    assert row["name"] == "latency_tp1_in64_out64", "flat write hardcodes one constant"
    assert row["model"][2] == "spyre", "model.backend is the HUD pivot axis"
    assert row["metric"][1] == [1.5], "samples must be an Array, not a scalar"


def test_v3_metric_samples_stay_a_list_of_floats(mod):
    (row,) = mod.to_vllm_v3_rows([_flat(actual="2.5")], _RUN)
    samples = row["metric"][1]
    assert isinstance(samples, list) and all(isinstance(s, float) for s in samples)


def test_v3_mode_comes_from_the_test_name_prefix(mod):
    for name, mode in (("serve_tp1", "serve"), ("throughput_tp1", "throughput"), ("", "")):
        (row,) = mod.to_vllm_v3_rows([_flat(test_name=name)], _RUN)
        assert row["benchmark"][1] == mode


def test_v3_extra_does_not_duplicate_first_class_columns(mod):
    # Storing a value twice lets the two copies drift.
    (row,) = mod.to_vllm_v3_rows([_flat()], _RUN)
    assert "head_sha" not in row["extra"]


# --- v2_benchmark_id ------------------------------------------------------------------


def test_benchmark_id_ignores_which_file_reported_it(mod):
    # The writer reads both the native json and the .pytorch.json copy; one benchmark must
    # not split by reporter.
    a = mod.v2_benchmark_id("spyre-inference", "latency_tp1.json", [], {})
    b = mod.v2_benchmark_id("spyre-inference", "latency_tp1.pytorch.json", [], {})
    assert a == b != ""


def test_benchmark_id_refuses_an_empty_name(mod):
    # An id here would collide every unidentifiable benchmark onto one identity.
    assert mod.v2_benchmark_id("spyre-inference", "", [], {}) == ""
    assert mod.v2_benchmark_id("spyre-inference", "   ", [], {}) == ""


def test_benchmark_id_varies_with_the_discriminators(mod):
    base = mod.v2_benchmark_id("spyre-inference", "latency", [], {"tensor_parallel": "1"})
    other = mod.v2_benchmark_id("spyre-inference", "latency", [], {"tensor_parallel": "4"})
    assert base != other


def test_benchmark_id_is_tag_order_independent(mod):
    a = mod.v2_benchmark_id("spyre-inference", "latency", ["x", "y"], {})
    b = mod.v2_benchmark_id("spyre-inference", "latency", ["y", "x"], {})
    assert a == b


# --- to_v2_benchmark_rows: the per-metric collapse ------------------------------------


def test_many_metrics_of_one_benchmark_collapse_to_one_row(mod):
    # 26 metrics of one benchmark are ONE measurement. A row per metric would multiply
    # every trend point by the metric count -- the bug this collapse exists to prevent.
    flat = [_flat(metric=m, actual=i) for i, m in enumerate(["p50", "p90", "p99", "mean"])]
    idents, facts = mod.to_v2_benchmark_rows(mod.to_vllm_v3_rows(flat, _RUN), _RUN)
    assert len(facts) == 1, f"expected one collapsed row, got {len(facts)}"
    assert set(facts[0]["measurements"]) == {"p50", "p90", "p99", "mean"}
    assert len(idents) == 1


def test_backend_splits_facts_but_not_identity(mod):
    # backend is a COLUMN, never a hash input: it is the axis a cross-backend comparison
    # pivots on, so folding it into identity would make the two sides different benchmarks.
    flat = [_flat(device="spyre"), _flat(device="cpu")]
    idents, facts = mod.to_v2_benchmark_rows(mod.to_vllm_v3_rows(flat, _RUN), _RUN)
    assert len(idents) == 1, "one benchmark identity across backends"
    assert {f["backend"] for f in facts} == {"spyre", "cpu"}
    assert len({f["benchmark_id"] for f in facts}) == 1


def test_rows_with_no_measurements_are_dropped(mod):
    # chk_measurements refuses an empty map: nothing measured is a parse failure.
    v3 = mod.to_vllm_v3_rows([_flat()], _RUN)
    v3[0]["metric"] = (v3[0]["metric"][0], [], 0.0, {})
    _idents, facts = mod.to_v2_benchmark_rows(v3, _RUN)
    assert facts == []


def test_unnamed_benchmarks_are_skipped_not_merged(mod):
    v3 = mod.to_vllm_v3_rows([_flat()], _RUN)
    v3[0]["name"] = ""
    idents, facts = mod.to_v2_benchmark_rows(v3, _RUN)
    assert (idents, facts) == ({}, [])


def test_two_files_reporting_one_benchmark_merge_richer_props(mod):
    # The merge must not lose a field when the same benchmark arrives twice.
    a = _flat(test_name="latency_tp1_in64_out64")
    b = _flat(test_name="latency_tp1_in64_out64", model="granite-3b")
    idents, _facts = mod.to_v2_benchmark_rows(mod.to_vllm_v3_rows([a, b], _RUN), _RUN)
    (props,) = [row[4] for row in idents.values()]
    assert props.get("tensor_parallel") == "1"


def test_iterations_tracks_the_widest_sample_array(mod):
    v3 = mod.to_vllm_v3_rows([_flat(metric="p50"), _flat(metric="p90")], _RUN)
    v3[0]["metric"] = ("p50", [1.0, 2.0, 3.0], 0.0, {})
    _idents, facts = mod.to_v2_benchmark_rows(v3, _RUN)
    assert facts[0]["iterations"] == 3


def test_run_id_is_stamped_on_every_fact_row(mod):
    flat = [_flat(metric="p50"), _flat(metric="p90", test_name="serve_tp1_in8_out8")]
    _idents, facts = mod.to_v2_benchmark_rows(mod.to_vllm_v3_rows(flat, _RUN), _RUN)
    assert facts and all(f["run_id"] == _RUN for f in facts)
