# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Profiled benchmark runs (`--profile-level light|full`) and the trace bundle they leave.

The bundle, `<results-dir>/<test_name>/trace/`, is what torch-spyre's `vllm-trace` ingest
reads: manifest.json (label, artifact, env), e2e.json (workload), phases.jsonl, and the
analysed res.json / roof.json / host.json, beside the raw traces they came from.
A profiled run writes nothing to `<results-dir>/*.json`: it is never a regression input.
"""

import contextlib
import glob
import json
import logging
import os
import time
from pathlib import Path

from . import roofline, steps

log = logging.getLogger(__name__)

LEVELS = ("off", "light", "full")
DEVMEM_INTERVAL_S = 0.1
# One per run, written by each worker: across TP ranks only the first counts.
LIFECYCLE = ("weight_load", "compile_warmup", "ready")


def profiler_config(trace_dir: Path, level: str) -> dict:
    """vLLM `--profiler-config` for a level: no stack capture, no table dump at stop."""
    full = level == "full"
    return {
        "profiler": "torch",
        "torch_profiler_dir": str((trace_dir / "raw").resolve()),
        "torch_profiler_activities": ["CPU", "PrivateUse1"],
        "torch_profiler_with_stack": False,
        "torch_profiler_dump_cuda_time_total": False,
        "torch_profiler_record_shapes": full,
        "torch_profiler_with_memory": full,
        "ignore_frontend": True,
    }


def child_env(trace_dir: Path, level: str) -> dict[str, str]:
    """Env vars that point the benchmarked process at the bundle."""
    env = {"SPYRE_PROFILE_DIR": str((trace_dir / "markers").resolve())}
    if level == "full":
        env["TORCH_SPYRE_KERNEL_COST_DIR"] = str((trace_dir / "kernel_cost").resolve())
        env["SPYRE_PROFILE_MEMORY_INTERVAL"] = str(DEVMEM_INTERVAL_S)
        env["SPYRE_ATTN_PROFILING"] = "1"
    return env


def _jsonl(paths) -> list[dict]:
    rows = []
    for p in paths:
        try:
            lines = Path(p).read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            with contextlib.suppress(ValueError):
                rows.append(json.loads(line))
    return rows


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))


def _timeline(marks: list[dict], t0: float, t_end: float) -> list[tuple[str, float, float]]:
    """(phase, start, end) spans from every process's markers, merged by wall time.

    Time before the first marker (interpreter, vLLM import, engine start) is `startup`.
    """
    seen: set[str] = set()
    kept = []
    for m in sorted(marks, key=lambda m: m["t"]):
        if m["phase"] in LIFECYCLE:
            if m["phase"] in seen:
                continue
            seen.add(m["phase"])
        kept.append(m)
    marks = kept
    spans: list[tuple[str, float, float]] = []
    if marks and t0 < marks[0]["t"]:
        spans.append(("startup", t0, marks[0]["t"]))
    for i, m in enumerate(marks):
        end = marks[i + 1]["t"] if i + 1 < len(marks) else t_end
        if spans and spans[-1][0] == m["phase"]:
            spans[-1] = (m["phase"], spans[-1][1], end)
        else:
            spans.append((m["phase"], m["t"], end))
    return spans


def _phase_at(spans, t: float) -> str:
    for name, a, b in spans:
        if a <= t < b:
            return name
    return spans[-1][0] if spans and t >= spans[-1][2] else "startup"


def _peak(rows, key):
    return max((r[key] for r in rows if r.get(key) is not None), default=None)


def resource_report(label, spans, samples, devmem) -> dict:
    """res.json: per-phase wall, CPU, host and device memory, then whole-run device memory."""
    wall: dict[str, float] = {}
    for name, a, b in spans:
        wall[name] = wall.get(name, 0.0) + (b - a)
    by: dict[str, list] = {n: [] for n in wall}
    for s in samples:
        by.setdefault(_phase_at(spans, s["t"]), []).append(s)
    dev_by: dict[str, list] = {}
    for d in devmem:
        if "allocated" in d:
            dev_by.setdefault(_phase_at(spans, d["t"]), []).append(d)
    phases = {}
    for name, ss in by.items():
        the = [s["throttled_cum_s"] for s in ss if s.get("throttled_cum_s") is not None]
        nthr = [s["nr_throttled"] for s in ss if s.get("nr_throttled") is not None]
        cores = [s["cpu_cores"] for s in ss]
        dd = dev_by.get(name, [])
        phases[name] = {
            "wall_s": round(wall.get(name, 0.0), 3),
            "n_samples": len(ss),
            "cpu_s": round(sum(s["cpu_cores"] * s["dt_s"] for s in ss), 3) if ss else None,
            "cpu_cores_mean": round(sum(cores) / len(cores), 3) if cores else None,
            "cpu_cores_peak": round(max(cores), 3) if cores else None,
            "throttled_s": round(the[-1] - the[0], 3) if len(the) > 1 else None,
            "nr_throttled": int(nthr[-1] - nthr[0]) if len(nthr) > 1 else None,
            "rss_peak_b": _peak(ss, "rss_b"),
            "cg_mem_peak_b": _peak(ss, "cgroup_b"),
            "dev_allocated_peak_b": _peak(dd, "allocated"),
            "dev_max_allocated_b": _peak(dd, "max_allocated"),
        }
    dev = [d for d in devmem if "allocated" in d]
    device_memory = None
    if dev:
        loaded = [d["allocated"] for d in dev_by.get("weight_load", [])]
        device_memory = {
            "total_b": dev[-1].get("total"),
            "weights_b": loaded[-1] if loaded else None,
            "peak_b": _peak(dev, "max_allocated"),
            "method": "max over ranks; weights = allocated at the end of weight_load",
        }
    return {"schema": 1, "label": label, "phases": phases, "device_memory": device_memory}


def workload(bench_type: str, parameters: dict) -> dict:
    """e2e.json `args`: the workload keys the ingest hashes, from the test's vllm bench flags."""
    p = {k.replace("_", "-"): v for k, v in parameters.items()}
    if bench_type == "latency":
        batch, inp, out = p.get("batch-size"), p.get("input-len"), p.get("output-len")
    else:
        batch = p.get("max-num-seqs") or p.get("num-prompts") or p.get("max-concurrency")
        inp = p.get("input-len") or p.get("random-input-len")
        out = p.get("output-len") or p.get("random-output-len")
    return {
        "model": p.get("model"),
        "tp": p.get("tensor-parallel-size", 1),
        "batch_size": batch,
        "input_len": inp,
        "output_len": out,
        "max_model_len": p.get("max-model-len"),
        "budget": p.get("max-num-batched-tokens", 0),
        "dtype": p.get("dtype"),
        "bench_type": bench_type,
    }


def write_bundle(
    trace_dir: Path,
    *,
    test_name: str,
    level: str,
    bench_type: str,
    parameters: dict,
    env_fingerprint: dict,
    samples: list[dict],
) -> dict:
    """Analyse the raw traces and write the bundle files; never raises. Returns the manifest.

    `samples` are host resource rows with wall-clock `t` and the `dt_s` they cover.
    """
    manifest: dict = {
        "schema": 1,
        "label": test_name,
        "test_name": test_name,
        "profile_level": level,
        "bench_type": bench_type,
        "artifact": env_fingerprint.get("artifact"),
        "env": {k: v for k, v in env_fingerprint.items() if k != "artifact"},
        "profiler_config": profiler_config(trace_dir, level),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "absent": {},
    }
    try:
        _analyse(trace_dir, manifest, bench_type, parameters, samples)
    except Exception as e:
        log.warning("Trace analysis for %s failed: %s", test_name, e)
        manifest["absent"]["analysis"] = f"{type(e).__name__}: {e}"
    result = trace_dir / f"{test_name}.json"
    try:
        _write(trace_dir / "manifest.json", manifest)
        if result.is_file():
            _write(result, {**json.loads(result.read_text()), "profile_level": level})
    except (OSError, ValueError) as e:
        log.warning("Could not write the trace manifest for %s: %s", test_name, e)
    return manifest


def _analyse(trace_dir, manifest, bench_type, parameters, samples) -> None:
    absent = manifest["absent"]
    marks = _jsonl(sorted(glob.glob(str(trace_dir / "markers" / "phases.*.jsonl"))))
    devmem = _jsonl(sorted(glob.glob(str(trace_dir / "markers" / "devmem.*.jsonl"))))
    times = [s["t"] for s in samples] + [m["t"] for m in marks]
    spans = _timeline(marks, min(times, default=0.0), max(times, default=0.0))
    with open(trace_dir / "phases.jsonl", "w") as f:
        f.writelines(json.dumps(m) + "\n" for m in sorted(marks, key=lambda m: m["t"]))
    if not marks:
        absent["phases"] = "no phase markers: the spyre_inference plugin did not load"
    if manifest["profile_level"] == "full" and not any("allocated" in d for d in devmem):
        err = next((d["err"] for d in devmem if "err" in d), "no devmem samples from the workers")
        absent["device_memory"] = err

    e2e = {"args": workload(bench_type, parameters), "profile_level": manifest["profile_level"]}
    _write(trace_dir / "res.json", resource_report(manifest["label"], spans, samples, devmem))

    cost_files = sorted(Path(p) for p in glob.glob(str(trace_dir / "kernel_cost" / "*.jsonl")))
    costs = roofline.load_costs(cost_files)
    if manifest["profile_level"] == "full" and not cost_files:
        absent["kernel_cost"] = "torch-spyre wrote no kernel_cost.*.jsonl"
    elif cost_files and not costs:
        absent["kernel_cost"] = "every kernel_cost record has cost null (graph cache replay)"

    traces = sorted(
        glob.glob(str(trace_dir / "raw" / "*.pt.trace.json*")),
        key=lambda p: ("_rank0." not in os.path.basename(p), p),
    )
    manifest["traces"] = [os.path.relpath(t, trace_dir) for t in traces]
    if not traces:
        absent["trace"] = "vLLM wrote no torch profiler trace"
        _write(trace_dir / "e2e.json", e2e)
        return
    host = []
    for i, t in enumerate(traces):
        ev = steps.load(t)
        win = steps.windows(ev)
        host.append(steps.host_report(os.path.relpath(t, trace_dir), ev, win))
        if i == 0:
            if win:
                # First step start to last step end of the profiled window, in the trace's clock.
                e2e["e2e_s"] = round((win[-1]["ts"] + win[-1]["dur"] - win[0]["ts"]) / 1e6, 6)
            else:
                absent["steps"] = f"{os.path.basename(t)}: no spyre:: or execute_context_ regions"
            if not any(e.get("cat") in steps.DEVICE_CATS for e in ev):
                absent["device_events"] = "no device rows: torch-spyre built without AIUPTI?"
            _write(trace_dir / "roof.json", roofline.build(ev, win, costs))
    _write(trace_dir / "host.json", host)
    _write(trace_dir / "e2e.json", e2e)
