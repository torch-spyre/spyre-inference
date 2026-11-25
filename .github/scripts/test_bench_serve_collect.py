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

"""vllm-bench-serve's collect.py: the server log lines it parses, the per-config gate and
the Δ colour thresholds. Kept here so the pre-commit ingest-script-tests job runs it.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / ".claude/skills/vllm-bench-serve/scripts/collect.py"
)

# ruff: noqa: E501
# Two TP workers; each line as the Spyre warmup, spyre_attn.py and vLLM core.py log it.
SERVER_LOG = """\
INFO 10-13 10:00:00 vLLM API server version 0.0
(TP0) INFO 10-13 10:01:00 Recording 12 per-seq + 4 batched-decode attention variants for layer...
(TP0) INFO 10-13 10:02:00 Attention graph recording complete: 16 graphs in 50.000s.
(TP1) INFO 10-13 10:02:01 Attention graph recording complete: 16 graphs in 52.500s.
(TP0) INFO 10-13 10:05:00 Warmup complete in 180.000s for 6 buckets.
(TP1) INFO 10-13 10:05:01 Warmup complete in 181.000s for 6 buckets.
INFO 10-13 10:05:02 init engine (profile, create kv cache, warmup model) took 300.00 s (compilation: 240.00 s)
INFO 10-13 10:05:30 Application startup complete.
"""


@pytest.fixture(scope="module")
def mod():
    """Load collect.py by path."""
    spec = importlib.util.spec_from_file_location("bench_serve_collect", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_parse_server_log(mod):
    assert mod.parse_server_log(SERVER_LOG) == {
        "startup_s": 330.0,
        "init_engine_s": 300.0,
        "compilation_s": 240.0,
        "model_warmup_s": 181.0,
        "buckets": 6,
        "attn_record_s": 52.5,
        "attn_graphs": 16,
        "attn_split": "12 + 4",
    }


def _write_config(tmp_path: pathlib.Path, result: dict) -> pathlib.Path:
    results = tmp_path / "results"
    results.mkdir()
    (results / "cfg_server.log").write_text(SERVER_LOG)
    (results / "cfg_bench.log").write_text("")
    (results / "cfg.json").write_text(json.dumps(result))
    return tmp_path


RESULT = {
    "num_prompts": 6,
    "completed": 6,
    "failed": 0,
    "server_startup_sec": 333.3,
    "mean_itl_ms": 50.0,
    "median_itl_ms": 50.0,
}


def test_check_config_valid(mod, tmp_path):
    leg = _write_config(tmp_path, RESULT)
    cfg = {"test_name": "cfg", "environment_variables": {"SPYRE_DEVICES": 0}}
    rec = mod.check_config(leg, cfg, {"SPYRE_DEVICES": "0,1"})
    assert rec["valid"] and rec["errors"] == []
    assert rec["startup_s"] == 333.3  # from the result JSON, not the log
    assert rec["env"]["SPYRE_DEVICES"] == "0"  # config env overrides env.txt
    assert rec["default_state"] == "unknown"  # no startup record
    assert rec["compile_leak"] is False


def test_check_config_invalid(mod, tmp_path):
    result = {**RESULT, "completed": 5, "failed": 1, "mean_itl_ms": 60.0}
    leg = _write_config(tmp_path, result)
    rec = mod.check_config(leg, {"test_name": "cfg"}, {"SPYRE_KERNEL_CACHE": "1"})
    assert not rec["valid"]
    assert rec["errors"][:2] == ["completed 5 of 6 prompts", "1 requests failed"]
    assert rec["compile_leak"] is True


def test_delta_classes(mod):
    assert mod.delta(100, 101, "lower")[1] == ""
    assert mod.delta(100, 90, "lower")[1] == "d-good"
    assert mod.delta(100, 105, "lower")[1] == "d-warn"
    assert mod.delta(100, 120, "lower")[1] == "d-bad"
    assert mod.delta(100, 80, "higher")[1] == "d-bad"
    assert mod.delta(4, 4, "count") == ("", "", None)


def _start(state: str, hits: int, misses: int, **extra) -> dict:
    return {
        "seconds": 100.0,
        "state": state,
        "fxgraph_hits": hits,
        "fxgraph_misses": misses,
        "aotautograd_hits": 0,
        "aotautograd_misses": 0,
        "spyre_kernel_hits": 0,
        "spyre_kernel_misses": 0,
        **extra,
    }


def _write_cold_warm(leg: pathlib.Path, warm: dict) -> None:
    results = leg / "results"
    (results / "cfg_cold_server.log").write_text(SERVER_LOG)
    startup = {"mode": "cold-warm", "starts": {"cold": _start("cold", 0, 9), "warm": warm}}
    (results / "cfg_startup.json").write_text(json.dumps(startup))


def test_cold_warm_splits_the_startup_rows(mod, tmp_path):
    leg = _write_config(tmp_path, RESULT)
    _write_cold_warm(leg, _start("warm", 9, 0, fxgraph_new_files=0))
    rec = mod.check_config(leg, {"test_name": "cfg"}, {})
    assert rec["compile_cache_mode"] == "cold-warm"
    assert (rec["cold_startup_s"], rec["warm_startup_s"]) == (100.0, 100.0)
    assert rec["cold_compilation_s"] == 240.0
    assert rec["warm_state_label"] == "warm (9 hit, 0 miss)"
    assert rec["warnings"] == []
    legs = [{"configs": {"cfg": rec}}]
    keys = [r[0] for r in mod.all_rows(legs, ["cfg"]) if not isinstance(r, str)]
    assert keys[:2] == ["cold_startup_s", "warm_startup_s"]
    assert "warm_fxgraph_misses" in keys and "startup_s" not in keys


def test_cold_warm_warm_start_that_misses_is_a_warning(mod, tmp_path):
    leg = _write_config(tmp_path, RESULT)
    _write_cold_warm(leg, _start("partial", 8, 1, spyre_kernel_new_files=2))
    rec = mod.check_config(leg, {"test_name": "cfg"}, {})
    assert rec["valid"]
    assert rec["warnings"] == [
        {"state": "partial", "misses": {"fxgraph": 1}, "new_files": {"spyre_kernel": 2}}
    ]
    assert "fxgraph 1 miss, spyre_kernel 2 new files" in mod.warning_text(rec["warnings"][0])


def test_default_partial_start_is_only_a_label(mod, tmp_path):
    leg = _write_config(tmp_path, RESULT)
    startup = {"mode": "default", "starts": {"default": _start("partial", 5, 1)}}
    (leg / "results" / "cfg_startup.json").write_text(json.dumps(startup))
    rec = mod.check_config(leg, {"test_name": "cfg"}, {})
    assert rec["default_state_label"] == "partial (5 hit, 1 miss)"
    assert rec["warnings"] == []


@pytest.mark.parametrize(
    ("base", "target", "flagged"),
    [("warm", "warm", False), ("cold", "cold", False), ("warm", "cold", True)]
    + [("partial", "partial", True), ("warm", "unknown", True)],
)
def test_startup_delta_is_unreliable_only_across_cache_states(mod, base, target, flagged):
    def leg(state):
        return {"configs": {"cfg": {"default_state": state}}}

    assert mod.unreliable(leg(base), leg(target), "cfg", "default") is flagged
    assert mod.unreliable(leg(base), leg(target), "cfg", None) is False
