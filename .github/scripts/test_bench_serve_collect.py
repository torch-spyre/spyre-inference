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
# Two TP workers; each line as spyre_model_runner.py, spyre_attn.py and vLLM core.py log it.
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


CACHE_OFF = {"TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1", "VLLM_DISABLE_COMPILE_CACHE": "1"}
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
    rec = mod.check_config(leg, cfg, {"SPYRE_DEVICES": "0,1", **CACHE_OFF})
    assert rec["valid"] and rec["errors"] == []
    assert rec["startup_s"] == 333.3  # from the result JSON, not the log
    assert rec["env"]["SPYRE_DEVICES"] == "0"  # config env overrides env.txt
    assert rec["compile_cache"] is False
    assert rec["compile_leak"] is False


def test_check_config_invalid(mod, tmp_path):
    result = {**RESULT, "completed": 5, "failed": 1, "mean_itl_ms": 60.0}
    leg = _write_config(tmp_path, result)
    rec = mod.check_config(leg, {"test_name": "cfg"}, {"SPYRE_KERNEL_CACHE": "1", **CACHE_OFF})
    assert not rec["valid"]
    assert rec["errors"][:2] == ["completed 5 of 6 prompts", "1 requests failed"]
    assert rec["compile_cache"] is True
    assert rec["compile_leak"] is True


def test_delta_classes(mod):
    assert mod.delta(100, 101, "lower")[1] == ""
    assert mod.delta(100, 90, "lower")[1] == "d-good"
    assert mod.delta(100, 105, "lower")[1] == "d-warn"
    assert mod.delta(100, 120, "lower")[1] == "d-bad"
    assert mod.delta(100, 80, "higher")[1] == "d-bad"
    assert mod.delta(4, 4, "count") == ("", "", None)
