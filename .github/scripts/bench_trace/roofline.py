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

"""Per-kernel roofline: device kernel events (args.provenance_key) joined to the torch-spyre
kernel-cost sidecar, measured time against max(bytes / HBM bw, MACs / (MAC rate x cores)).

Bytes, MACs and cores are compiler-model values, not counters, so `bound` is the larger floor
term, not a measurement. Without a sidecar the kernels carry only their measured times.
"""

import collections
import json
import statistics
from pathlib import Path

# torch-spyre CostParams defaults, used when a sidecar record carries no floor_inputs.
DEFAULT_BW_GBPS = 150.0
DEFAULT_MACS_PER_NS_CORE = 1140.0


def load_costs(paths: list[Path]) -> dict[str, dict]:
    """provenance_key -> cost dict of every sidecar line that has one."""
    out = {}
    for path in paths:
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("provenance_key") and isinstance(r.get("cost"), dict):
                out[r["provenance_key"]] = {**r["cost"], "aten": r.get("aten_ops") or []}
    return out


def _floors(c: dict) -> tuple[float, float]:
    """(memory, compute) floor in ns."""
    fi = c.get("floor_inputs") or {}
    bw = fi.get("bw_peak_gbps") or DEFAULT_BW_GBPS
    mpn = fi.get("mac_peak_per_core_ns") or DEFAULT_MACS_PER_NS_CORE
    hbm = c.get("hbm_read_bytes", 0) + c.get("hbm_write_bytes", 0)
    return hbm / bw, c.get("macs", 0) / (mpn * max(c.get("cores") or 1, 1))


def build(ev: list[dict], steps: list[dict], costs: dict[str, dict], skip_first: int = 1) -> dict:
    """roof.json: one row per (kernel, step kind), plus the keys no sidecar line covers."""
    kern = sorted(
        (e for e in ev if e.get("cat") == "kernel" and e.get("args", {}).get("provenance_key")),
        key=lambda e: e["ts"],
    )
    by: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    unjoined, decodes = set(), 0
    for s in steps:
        decodes += s["kind"] == "decode"
        if s["kind"] == "decode" and decodes <= skip_first:
            continue
        lo, hi = s["ts"], s["ts"] + s["dur"]
        for e in kern:
            if lo <= e["ts"] < hi:
                key = e["args"]["provenance_key"]
                # A bundle of several ComputeOnDevice steps (#N suffix) is one call.
                by[(key, s["kind"])][e["name"].rsplit("#", 1)[-1]].append(e["dur"])
                if key not in costs:
                    unjoined.add(key)
    kernels = []
    for (key, kind), grouped in by.items():
        n = min(len(v) for v in grouped.values())
        per_call = [sum(v[i] for v in grouped.values()) for i in range(n)]
        med = statistics.median(per_call)
        c = costs.get(key)
        row = {
            "key": key, "phase": kind, "aten": c["aten"] if c else [], "calls": n,
            "steps_per_call": len(grouped), "median_us": round(med, 2),
            "total_ms": round(sum(sum(v) for v in grouped.values()) / 1e3, 3),
        }  # fmt: skip
        if c:
            mem, cmp_ = _floors(c)
            floor = max(mem, cmp_) / 1e3
            hbm = c.get("hbm_read_bytes", 0) + c.get("hbm_write_bytes", 0)
            row.update(
                hbm_bytes=hbm, macs=c.get("macs", 0), cores=c.get("cores"),
                mem_floor_us=round(mem / 1e3, 2), cmp_floor_us=round(cmp_ / 1e3, 2),
                floor_us=round(floor, 2), ratio=round(med / floor, 2) if floor else None,
                bound="memory" if mem >= cmp_ else "compute",
                gbps=round(hbm / (med * 1e3), 1) if med else None,
                mac_per_ns=round(c.get("macs", 0) / (med * 1e3), 1) if med else None,
            )  # fmt: skip
        kernels.append(row)
    kernels.sort(key=lambda r: -r["total_ms"])
    return {
        "schema": 1,
        "params": {"skip_first_decode": skip_first, "cost_records": len(costs)},
        "kernels": kernels,
        "unjoined_keys": sorted(unjoined),
    }
