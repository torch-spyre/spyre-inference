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

Output keys follow spyre-perf-suite `spsprofile.roofline`. Bytes, MACs and cores are
compiler-model values, not counters, so `bound` is the larger floor term, not a measurement.
"""

import collections
import json
import statistics
from pathlib import Path

# torch-spyre CostParams defaults, used when a sidecar record carries no floor_inputs.
HBM_GBPS = 150.0
MACS_PER_NS_PER_CORE = 1140.0


def load_costs(paths: list[Path]) -> dict[str, dict]:
    """provenance_key -> {key, kernel, aten, hbm_bytes, macs, cores, predicted_us, bw, mpn}.

    Lines with `cost: null` (replayed from the FX graph cache) are left out.
    """
    out: dict[str, dict] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            c = r.get("cost")
            if not (r.get("provenance_key") and isinstance(c, dict)):
                continue
            fi = c.get("floor_inputs") or {}
            out.setdefault(
                r["provenance_key"],
                {
                    "key": r["provenance_key"],
                    "kernel": r.get("kernel"),
                    "aten": r.get("aten_ops"),
                    "hbm_bytes": c.get("hbm_read_bytes", 0) + c.get("hbm_write_bytes", 0),
                    "macs": c.get("macs", 0),
                    "cores": c.get("cores") or 1,
                    "predicted_us": c.get("predicted_us"),
                    "bw": fi.get("bw_peak_gbps"),
                    "mpn": fi.get("mac_peak_per_core_ns"),
                },  # fmt: skip
            )
    return out


def floor_us(c: dict) -> tuple[float, float]:
    """(memory, compute) floor in microseconds."""
    mem_ns = c["hbm_bytes"] / (c.get("bw") or HBM_GBPS)  # bytes / (GB/s) = ns
    cmp_ns = c["macs"] / ((c.get("mpn") or MACS_PER_NS_PER_CORE) * max(c["cores"], 1))
    return mem_ns / 1e3, cmp_ns / 1e3


def build(ev: list[dict], steps: list[dict], costs: dict[str, dict], skip_first: int = 1) -> dict:
    """roof.json, spsprofile.roofline.kernel_roofline's shape: one row per (kernel, step kind)."""
    kern = sorted(
        (e for e in ev if e.get("cat") == "kernel" and (e.get("args") or {}).get("provenance_key")),
        key=lambda e: e["ts"],
    )
    by: dict = collections.defaultdict(list)
    timeline, unjoined, decodes = [], set(), 0
    for s in steps:
        kind = s["kind"]
        decodes += kind == "decode"
        skip = kind == "decode" and decodes <= skip_first
        lo, hi = s["ts"], s["ts"] + s["dur"]
        segs = []
        for e in kern:
            if not lo <= e["ts"] < hi:
                continue
            key = e["args"]["provenance_key"]
            c = costs.get(key)
            bound = pct = None
            if c is None:
                unjoined.add(key)
            else:
                mem, cmp_ = floor_us(c)
                bound = "memory" if mem >= cmp_ else "compute"
                pct = 100 * max(mem, cmp_) / e["dur"] if e["dur"] else None
            segs.append(
                {
                    "t_us": round(e["ts"] - lo, 2),
                    "dur_us": round(e["dur"], 2),
                    "key": key,
                    "bound": bound,
                    "pct_floor": round(pct, 1) if pct is not None else None,
                }  # fmt: skip
            )
            if not skip:
                # A bundle of several ComputeOnDevice steps (#N suffix) is one call.
                sn = e["name"].rsplit("#", 1)[-1] if "#" in e["name"] else ""
                by[(key, kind)].append((sn, e["dur"]))
        timeline.append({"step": s["name"], "kind": kind, "skipped": skip, "segs": segs})
    kernels = []
    for (key, kind), evs in by.items():
        c = costs.get(key)
        grouped = collections.defaultdict(list)
        for sn, d in evs:
            grouped[sn].append(d)
        n = min(len(v) for v in grouped.values())
        med = statistics.median([sum(v[i] for v in grouped.values()) for i in range(n)])
        row = {
            "key": key, "phase": kind, "aten": c and c["aten"], "kernel": c and c["kernel"],
            "calls": n, "steps_per_call": len(grouped), "median_us": round(med, 2),
            "total_ms": round(sum(d for _, d in evs) / 1e3, 3),
        }  # fmt: skip
        if c:
            mem, cmp_ = floor_us(c)
            fl = max(mem, cmp_)
            row.update(
                hbm_bytes=c["hbm_bytes"], macs=c["macs"], cores=c["cores"],
                predicted_us=c["predicted_us"], mem_floor_us=round(mem, 2),
                cmp_floor_us=round(cmp_, 2), floor_us=round(fl, 2),
                ratio=round(med / fl, 2) if fl else None,
                bound="memory" if mem >= cmp_ else "compute",
                gbps=round(c["hbm_bytes"] / (med * 1e3), 1) if med else None,
                mac_per_ns=round(c["macs"] / (med * 1e3), 1) if med else None,
            )  # fmt: skip
        kernels.append(row)
    kernels.sort(key=lambda r: -r["total_ms"])
    return {
        "schema": 1,
        "params": {
            "hbm_gbps": HBM_GBPS,
            "macs_per_ns_per_core": MACS_PER_NS_PER_CORE,
            "skip_first_decode": skip_first,
        },  # fmt: skip
        "kernels": kernels,
        "timeline": timeline,
        "unjoined_keys": sorted(unjoined),
    }
