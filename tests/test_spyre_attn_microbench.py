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

"""CPU-only tests for the attention microbench's import-time contract (no hardware).

The harness resolves --timing/--threads from argv and writes the environment at
module scope — before any spyre_inference import, because spyre_attn reads
_ATTN_PROFILING at import time — so importing it in-process here would leak those
writes into this pytest run. A subprocess loads it under several argv/env
combinations and exercises measure_wall, whose 5-tuple run_config unpacks on every
wall row.

What a typo would silently break: the mode-to-SPYRE_ATTN_PROFILING mapping (an
inverted value drops the spans every profiler row depends on), the assigned-not-
setdefault thread vars (an inherited OMP_NUM_THREADS would decide the thread count
while the report records another), the argv forms, and measure_wall's shape.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "microbench" / "spyre_attn_microbench.py"

_PROBE = """
import gc
import importlib.util
import json
import os
import sys
import types


def load(argv):
    sys.argv = ["microbench"] + argv
    spec = importlib.util.spec_from_file_location("mb", os.environ["MB_SCRIPT"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


out = {}

# Loaded first, against inherited thread vars: a setdefault in the harness would
# keep the shell's value while the report recorded the requested one.
mod = load(["--timing", "wall", "--threads", "4"])
out["wall4"] = {
    "timing": mod.TIMING,
    "threads": mod._THREADS,
    "profiling": os.environ["SPYRE_ATTN_PROFILING"],
    "spyre_num_cpus": os.environ["SPYRE_NUM_CPUS"],
    "omp": os.environ["OMP_NUM_THREADS"],
    "numexpr": os.environ["NUMEXPR_NUM_THREADS"],
}

mod = load([])
out["default"] = {
    "timing": mod.TIMING,
    "threads": mod._THREADS,
    "profiling": os.environ["SPYRE_ATTN_PROFILING"],
    "omp": os.environ["OMP_NUM_THREADS"],
}

mod = load(["--timing=wall"])
out["wall_eq_form"] = {"timing": mod.TIMING, "threads": mod._THREADS}

# A trailing --timing has no value to read; the bound check must fall back to the
# default rather than raise. argparse reports the missing value later, in main().
mod = load(["--config", "x", "--timing"])
out["trailing_timing"] = {"timing": mod.TIMING}

try:
    load(["--timing", "bogus"])
    out["bogus"] = None
except SystemExit as exc:
    out["bogus"] = str(exc)

# measure_wall with a stubbed fence: the sync calls bound every run, and the
# 5-tuple must match what measure() returns and run_config unpacks.
syncs = []
mod.torch.spyre = types.SimpleNamespace(synchronize=lambda *a: syncs.append(1))
runs = []
dev_us, mem_us, cpu_us, n_kernels, by_op = mod.measure_wall(lambda: runs.append(1), 3)
out["measure_wall"] = {
    "n_samples": len(dev_us),
    "samples_positive": all(s > 0 for s in dev_us),
    "mem_zeroed": mem_us == [0.0] * 3,
    "kernels_zeroed": n_kernels == [0.0] * 3,
    "cpu_is_dev_copy": cpu_us == dev_us and cpu_us is not dev_us,
    "by_op_empty": by_op == {},
    "n_runs": len(runs),
    "n_syncs": len(syncs),
    "gc_enabled": gc.isenabled(),
}

print("MBPROBE" + json.dumps(out))
"""


@pytest.fixture(scope="module")
def probe():
    env = dict(os.environ)
    env.update(
        MB_SCRIPT=str(_SCRIPT),
        OMP_NUM_THREADS="96",
        SPYRE_NUM_CPUS="32",
    )
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("MBPROBE")]
    assert lines, proc.stdout + proc.stderr
    return json.loads(lines[-1][len("MBPROBE") :])


def test_wall_timing_disables_spans_and_overrides_inherited_threads(probe):
    assert probe["wall4"] == {
        "timing": "wall",
        "threads": "4",
        "profiling": "0",
        "spyre_num_cpus": "4",
        "omp": "4",
        "numexpr": "4",
    }


def test_default_is_profiler_with_one_thread(probe):
    assert probe["default"] == {
        "timing": "profiler",
        "threads": "1",
        "profiling": "1",
        "omp": "1",
    }


def test_timing_accepts_equals_form(probe):
    assert probe["wall_eq_form"] == {"timing": "wall", "threads": "8"}


def test_trailing_timing_falls_back_to_default(probe):
    assert probe["trailing_timing"] == {"timing": "profiler"}


def test_invalid_timing_is_refused_before_imports(probe):
    assert probe["bogus"] == "--timing must be wall or profiler, got 'bogus'"


def test_measure_wall_return_contract(probe):
    assert probe["measure_wall"] == {
        "n_samples": 3,
        "samples_positive": True,
        "mem_zeroed": True,
        "kernels_zeroed": True,
        "cpu_is_dev_copy": True,
        "by_op_empty": True,
        "n_runs": 3,
        "n_syncs": 6,
        "gc_enabled": True,
    }
