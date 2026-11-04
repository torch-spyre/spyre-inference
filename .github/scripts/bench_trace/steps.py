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

"""Per-step host timeline of a Kineto trace: device busy, the host work in its idle gaps.

Device time comes only from device-row events (kernel, gpu_memcpy, gpu_memset), never from
host `launch::*` markers or runtime spans, which mirror device time.
"""

import bisect
import collections
import gzip
import json
import re

DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
HOST_CATS = ("cpu_op", "user_annotation", "python_function")
DYNAMO = ("TorchDynamo", "Torch-Compiled Region", "## Call CompiledFxGraph", "AOTDispatcher")
# torch_spyre/ops/fallbacks.py: register_fallback_default plus the spyre::*_fallback ops.
FALLBACK_ATEN = {
    "cumsum", "arange", "isin", "tril", "triu", "bitwise_xor", "bitwise_or", "argmax",
    "argmin", "where", "normal_", "random_",
}  # fmt: skip
FALLBACK_SPYRE = {
    "spyre::triu_cpu", "spyre::max_dim_int64_fallback", "spyre::min_dim_int64_fallback",
    "spyre::max_default_int64_fallback",
}  # fmt: skip
COPY_CHILD = ("launch::D2H", "launch::H2D", "aten::_to_copy", "aten::to", "aten::copy_")
COLL = ("allreduce", "allgather", "all_gather", "reduce_scatter", "wait_work", "collective")

SPYRE_STEP = "spyre::step"
SPYRE_TAG = re.compile(r"^spyre::(prefill|decode|mixed):(\d+)$")
# vLLM's Worker.annotate_profile, simple and detailed forms.
VLLM_TAG = re.compile(
    r"^execute_(?:\d+_)?context_(\d+)\((?:sq)?(\d+).*?_generation_(\d+)\((?:sq)?(\d+)"
)


def load(path: str) -> list[dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]


def _merge(iv):
    out: list[list[float]] = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def windows(ev: list[dict]) -> list[dict]:
    """Engine steps as {name, kind, tokens, ts, dur, tid}, in time order.

    The harness's `spyre::<kind>:<tokens>` tag inside its `spyre::step` region wins; a trace
    without them (TP>1 worker processes) falls back to vLLM's `execute_context_*` regions.
    """
    ann = [e for e in ev if e.get("cat") == "user_annotation"]
    outer = sorted((e for e in ann if e["name"] == SPYRE_STEP), key=lambda e: e["ts"])
    starts = [e["ts"] for e in outer]
    out = []
    for e in ann:
        m = SPYRE_TAG.match(e["name"])
        if not m:
            continue
        win = e
        i = bisect.bisect_right(starts, e["ts"]) - 1
        if i >= 0 and outer[i]["ts"] + outer[i]["dur"] >= e["ts"] + e["dur"]:
            win = outer[i]
        out.append(
            {
                "name": e["name"],
                "kind": m[1],
                "tokens": int(m[2]),
                "ts": win["ts"],
                "dur": win["dur"],
                "tid": win["tid"],
            }  # fmt: skip
        )
    if not out:
        for e in ann:
            m = VLLM_TAG.match(e["name"])
            if not m:
                continue
            ctx, gen = int(m[1]), int(m[3])
            kind = "mixed" if ctx and gen else "prefill" if ctx else "decode"
            out.append(
                {
                    "name": e["name"],
                    "kind": kind,
                    "tokens": int(m[2]) + int(m[4]),
                    "ts": e["ts"],
                    "dur": e["dur"],
                    "tid": e["tid"],
                }  # fmt: skip
            )
    return sorted(out, key=lambda w: w["ts"])


def _is_fallback(name: str) -> bool:
    return name in FALLBACK_SPYRE or (
        name.startswith("aten::") and name[6:].split(".")[0] in FALLBACK_ATEN
    )


def _fallback_ids(host: list[dict]) -> set[int]:
    """Outermost registered-fallback ops holding a host<->device copy child.

    The copy separates a real fallback from a CPU-tensor op of the same name (input prep).
    """
    hs = sorted(host, key=lambda e: (e["ts"], -e["dur"]))
    out, fb_end = set(), -1.0
    for i, e in enumerate(hs):
        if e["ts"] < fb_end or not _is_fallback(e["name"]):
            continue
        end = e["ts"] + e["dur"]
        j = i + 1
        while j < len(hs) and hs[j]["ts"] < end:
            if hs[j]["name"].startswith(COPY_CHILD):
                out.add(id(e))
                fb_end = end
                break
            j += 1
    return out


def _classify(name: str, in_fallback: bool) -> str:
    if in_fallback:
        return "cpu_fallback"
    if name == "host::synchronize":
        return "sync_wait"
    if name == "launch::H2D":
        return "h2d_copy"
    if name == "launch::D2H":
        return "d2h_copy"
    if name.startswith(("launch::", "launch_jobplan:")):
        return "launch"
    if any(k in name for k in COLL):
        return "collective"
    if name.startswith(DYNAMO):
        return "dynamo_guard"
    if name.startswith("spyre_attn::"):
        return "attn_host"
    if name.startswith(("aten::", "vllm::")):
        return "torch_op"
    return "python"


def _self_segments(evs, lo, hi, fb_ids):
    """(start, end, class) self-time segments of one thread's nested events in [lo, hi]."""
    evs = sorted(
        (e for e in evs if e["ts"] < hi and e["ts"] + e["dur"] > lo),
        key=lambda e: (e["ts"], -e["dur"]),
    )
    segs, stack, cur = [], [], lo

    def emit(upto):
        nonlocal cur
        a, b = max(cur, lo), min(upto, hi)
        if b > a:
            if stack:
                in_fb = any(id(s[0]) in fb_ids for s in stack)
                segs.append((a, b, _classify(stack[-1][0]["name"], in_fb)))
            else:
                segs.append((a, b, "python"))
        cur = max(cur, upto)

    for e in evs:
        while stack and stack[-1][1] <= e["ts"]:
            emit(stack[-1][1])
            stack.pop()
        emit(e["ts"])
        stack.append((e, e["ts"] + e["dur"]))
    while stack:
        emit(stack[-1][1])
        stack.pop()
    emit(hi)
    return segs


def analyse(ev: list[dict], steps: list[dict]) -> list[dict]:
    """One row per step: wall, device busy and idle, idle time by host class, copies."""
    if not steps:
        return []
    tid = steps[0]["tid"]
    dev = sorted((e for e in ev if e.get("cat") in DEVICE_CATS), key=lambda e: e["ts"])
    dev_ts = [e["ts"] for e in dev]
    step_names = {s["name"] for s in steps} | {SPYRE_STEP}
    host = sorted(
        (
            e
            for e in ev
            if e.get("cat") in HOST_CATS and e["tid"] == tid and e["name"] not in step_names
        ),
        key=lambda e: e["ts"],
    )
    host_ts = [e["ts"] for e in host]
    fb_ids = _fallback_ids(host)
    rows = []
    for s in steps:
        lo, hi = s["ts"], s["ts"] + s["dur"]
        d = dev[bisect.bisect_left(dev_ts, lo) : bisect.bisect_left(dev_ts, hi)]
        busy_iv = _merge([(e["ts"], e["ts"] + e["dur"]) for e in d])
        gaps, c = [], lo
        for a, b in busy_iv:
            if a > c:
                gaps.append((c, a))
            c = max(c, b)
        if c < hi:
            gaps.append((c, hi))
        segs = _self_segments(host[: bisect.bisect_right(host_ts, hi)], lo, hi, fb_ids)
        seg_a = [x[0] for x in segs]
        attrib: collections.Counter = collections.Counter()
        for a, b in gaps:
            for i in range(max(0, bisect.bisect_right(seg_a, a) - 1), len(segs)):
                sa, sb, cl = segs[i]
                if sa >= b:
                    break
                attrib[cl] += max(0.0, min(b, sb) - max(a, sa))
        inside = host[bisect.bisect_left(host_ts, lo) : bisect.bisect_left(host_ts, hi)]
        fb = [e for e in inside if id(e) in fb_ids]
        fb_ops: dict = collections.defaultdict(lambda: [0, 0.0])
        for e in fb:
            fb_ops[e["name"]][0] += 1
            fb_ops[e["name"]][1] += e["dur"]
        copies = collections.defaultdict(lambda: [0, 0])
        for m in (e for e in d if e["cat"] == "gpu_memcpy"):
            kind = m.get("args", {}).get("call", "?")
            copies[kind][0] += 1
            copies[kind][1] += m.get("args", {}).get("bytes", 0)
        busy = sum(b - a for a, b in busy_iv)
        rows.append(
            {
                "step": s["name"],
                "kind": s["kind"],
                "tokens": s["tokens"],
                "wall_us": s["dur"],
                "busy_us": busy,
                "idle_us": s["dur"] - busy,
                "n_gaps": len(gaps),
                "gap_attrib_us": dict(attrib),
                "launches": sum(e["name"] == "launch::Compute" for e in inside),
                "h2d": copies["HtoD"][0],
                "h2d_bytes": copies["HtoD"][1],
                "d2h": copies["DtoH"][0],
                "d2h_bytes": copies["DtoH"][1],
                "sync_us": sum(e["dur"] for e in inside if e["name"] == "host::synchronize"),
                "fallback_n": len(fb),
                "fallback_us": sum(e["dur"] for e in fb),
                "fallback_ops": {k: {"n": v[0], "us": v[1]} for k, v in fb_ops.items()},
            }  # fmt: skip
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    """Means per step; the field names kernel/ingest readers key on."""
    if not rows:
        return {}
    n = len(rows)

    def mean(f):
        return sum(f(r) for r in rows) / n

    cls: collections.Counter = collections.Counter()
    fbo: dict = collections.defaultdict(lambda: [0, 0.0])
    for r in rows:
        cls.update(r["gap_attrib_us"])
        for k, v in r["fallback_ops"].items():
            fbo[k][0] += v["n"]
            fbo[k][1] += v["us"]
    wall = mean(lambda r: r["wall_us"])
    return {
        "n": n,
        "wall_ms": wall / 1e3,
        "busy_ms": mean(lambda r: r["busy_us"]) / 1e3,
        "busy_pct": 100 * mean(lambda r: r["busy_us"]) / wall if wall else 0.0,
        "host_exposed_ms": mean(lambda r: r["idle_us"]) / 1e3,
        "gaps_per_step": mean(lambda r: r["n_gaps"]),
        "launches_per_step": mean(lambda r: r["launches"]),
        "h2d_per_step": mean(lambda r: r["h2d"]),
        "h2d_kb_per_step": mean(lambda r: r["h2d_bytes"]) / 1024,
        "d2h_per_step": mean(lambda r: r["d2h"]),
        "d2h_kb_per_step": mean(lambda r: r["d2h_bytes"]) / 1024,
        "sync_ms_per_step": mean(lambda r: r["sync_us"]) / 1e3,
        "fallback_per_step": mean(lambda r: r["fallback_n"]),
        "fallback_ms_per_step": mean(lambda r: r["fallback_us"]) / 1e3,
        "fallback_ops": {
            k: {"calls_per_step": v[0] / n, "ms_per_step": v[1] / n / 1e3} for k, v in fbo.items()
        },
        "waterfall_ms_per_step": {k: v / n / 1e3 for k, v in cls.most_common()},
    }


def host_report(trace: str, ev: list[dict], steps: list[dict], skip_first: int = 1) -> dict:
    """host.json entry; the first `skip_first` decode steps (shape transition) are dropped."""
    rows = analyse(ev, steps)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    by["decode"] = by["decode"][skip_first:]
    return {
        "trace": trace,
        **{f"{k}_summary": summarize(by[k]) for k in ("prefill", "decode", "mixed")},
        "steps": rows,
    }
