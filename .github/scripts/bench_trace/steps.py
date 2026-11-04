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

Rows and summaries carry spyre-perf-suite `spsprofile.host`'s keys. Device time comes only
from device-row events (kernel, gpu_memcpy, gpu_memset), never from host `launch::*` markers
or runtime spans, which mirror device time.
"""

import bisect
import collections
import gzip
import json
import re

DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
HOST_CATS = ("cpu_op", "user_annotation", "python_function")
DYNAMO = (
    "TorchDynamo", "Torch-Compiled Region", "## Call CompiledFxGraph", "AOTDispatcher", "Pregraph",
)  # fmt: skip
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
COLL_KEYS = ("coll_group", "coll_algo", "coll_bytes", "collective_id")

STEP = re.compile(r"^(ttft|itl_\d+)$")
SPYRE_TAG = re.compile(r"^spyre::(prefill|decode|mixed):(\d+)$")
# vLLM's Worker.annotate_profile, simple and detailed forms: the only step marks at TP>1.
VLLM_TAG = re.compile(
    r"^execute_(?:\d+_)?context_(\d+)\((?:sq)?(\d+).*?_generation_(\d+)\((?:sq)?(\d+)"
)


def load(path: str) -> list[dict]:
    """Complete ('X') events of a trace file."""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        doc = json.load(f)
    ev = doc.get("traceEvents", []) if isinstance(doc, dict) else doc
    return [e for e in ev if e.get("ph") == "X" and "dur" in e]


def _merge(iv):
    out: list[list[float]] = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _covered(iv) -> float:
    return sum(b - a for a, b in _merge(iv))


def windows(ev: list[dict]) -> list[dict]:
    """Engine steps as {name, kind, tag, tokens, ts, dur, tid}, in time order.

    `ttft` / `itl_NNNN` regions are the steps (kind prefill / decode, as spsprofile reads them);
    the `spyre::<tag>:<tokens>` region inside one gives the batch it ran. A trace without them
    (TP>1 worker processes) falls back to vLLM's `execute_context_*` regions.
    """
    ann = sorted((e for e in ev if e.get("cat") == "user_annotation"), key=lambda e: e["ts"])
    tags = [(e["ts"], SPYRE_TAG.match(e["name"])) for e in ann if SPYRE_TAG.match(e["name"])]
    tag_ts = [t for t, _ in tags]
    out = []
    for e in ann:
        if not STEP.match(e["name"]):
            continue
        tag = tokens = None
        i = bisect.bisect_left(tag_ts, e["ts"])
        if i < len(tags) and tags[i][0] < e["ts"] + e["dur"]:
            tag, tokens = tags[i][1][1], int(tags[i][1][2])
        kind = "prefill" if e["name"] == "ttft" else "decode"
        out.append(
            {
                "name": e["name"],
                "kind": kind,
                "tag": tag or kind,
                "tokens": tokens,
                "ts": e["ts"],
                "dur": e["dur"],
                "tid": e["tid"],
            }  # fmt: skip
        )
    if out:
        return out
    for e in ann:
        m = VLLM_TAG.match(e["name"])
        if not m:
            continue
        ctx, gen = int(m[1]), int(m[3])
        tag = "mixed" if ctx and gen else "prefill" if ctx else "decode"
        out.append(
            {
                "name": e["name"],
                "kind": "prefill" if ctx else "decode",
                "tag": tag,
                "tokens": int(m[2]) + int(m[4]),
                "ts": e["ts"],
                "dur": e["dur"],
                "tid": e["tid"],
            }  # fmt: skip
        )
    return out


def _is_fallback(name: str) -> bool:
    return name in FALLBACK_SPYRE or (
        name.startswith("aten::") and name[6:].split(".")[0] in FALLBACK_ATEN
    )


def _fallbacks(host: list[dict]) -> dict[int, bool]:
    """id -> in_compiled_region for each outermost registered fallback holding a copy child.

    The copy separates a real fallback from a CPU-tensor op of the same name (input prep).
    """
    hs = sorted(host, key=lambda e: (e["ts"], -e["dur"]))
    out: dict[int, bool] = {}
    compiled: list[float] = []
    fb_end = -1.0
    for i, e in enumerate(hs):
        while compiled and compiled[-1] <= e["ts"]:
            compiled.pop()
        if e["name"].startswith(("Torch-Compiled Region", "## Call CompiledFxGraph")):
            compiled.append(e["ts"] + e["dur"])
        if e["ts"] < fb_end or not _is_fallback(e["name"]):
            continue
        end = e["ts"] + e["dur"]
        j = i + 1
        while j < len(hs) and hs[j]["ts"] < end:
            if hs[j]["name"].startswith(COPY_CHILD):
                out[id(e)] = bool(compiled)
                fb_end = end
                break
            j += 1
    return out


def _classify(name: str, in_jobplan: bool, in_fallback: bool) -> str:
    if in_fallback:
        return "cpu_fallback"
    if name == "host::synchronize":
        return "sync_wait"
    if name == "launch::H2D":
        return "correction_upload" if in_jobplan else "h2d_copy"
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
                jp = any(s[0]["name"].startswith("launch_jobplan:") for s in stack)
                fb = any(id(s[0]) in fb_ids for s in stack)
                segs.append((a, b, _classify(stack[-1][0]["name"], jp, fb)))
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


def _is_coll(e: dict) -> bool:
    args = e.get("args") or {}
    return any(args.get(k) for k in COLL_KEYS)


def analyse(ev: list[dict], steps: list[dict], skip_first: int = 1) -> dict:
    """{audit, steps, decode, prefill}: one row per step, microseconds.

    The first `skip_first` decode steps (shape transition) are left out of `decode`.
    """
    rows = []
    if steps:
        rows = _rows(ev, steps)

    def rowsum(cat):
        return sum(e["dur"] for e in ev if e.get("cat") == cat)

    dev = sorted((e for e in ev if e.get("cat") in DEVICE_CATS), key=lambda e: e["ts"])
    audit = {
        "kernel_ms": rowsum("kernel") / 1e3,
        "memcpy_ms": rowsum("gpu_memcpy") / 1e3,
        "memset_ms": rowsum("gpu_memset") / 1e3,
        "launch_marker_ms": sum(e["dur"] for e in ev if e["name"].startswith("launch::")) / 1e3,
        "runtime_fetch_ms": sum(e["dur"] for e in ev if e["name"] == "aiuFetchResponseBlocks")
        / 1e3,
        "runtime_barrier_ms": sum(e["dur"] for e in ev if e["name"] == "aiuIssueBarrier") / 1e3,
        "device_overlaps": sum(1 for a, b in zip(dev, dev[1:]) if b["ts"] < a["ts"] + a["dur"]),
    }
    return {
        "audit": audit,
        "steps": rows,
        "decode": [r for r in rows if r["kind"] == "decode"][skip_first:],
        "prefill": [r for r in rows if r["kind"] == "prefill"],
    }


def _rows(ev, steps):
    tid = steps[0]["tid"]
    dev = sorted((e for e in ev if e.get("cat") in DEVICE_CATS), key=lambda e: e["ts"])
    dev_ts = [e["ts"] for e in dev]
    skip = {s["name"] for s in steps}
    host = sorted(
        (
            e
            for e in ev
            if e.get("cat") in HOST_CATS
            and e["tid"] == tid
            and e["name"] not in skip
            and not SPYRE_TAG.match(e["name"])
        ),
        key=lambda e: e["ts"],
    )
    host_ts = [e["ts"] for e in host]
    runtime = sorted(
        (e for e in ev if e.get("cat") == "privateuse1_runtime"), key=lambda e: e["ts"]
    )
    launch_by_ext = {
        (e.get("args") or {}).get("External id"): e
        for e in host
        if e["name"].startswith("launch::")
    }
    jobplans = sorted(
        (e["ts"], e["ts"] + e["dur"]) for e in host if e["name"].startswith("launch_jobplan:")
    )
    jp_start = [s for s, _ in jobplans]

    def in_jobplan(ts):
        i = bisect.bisect_right(jp_start, ts) - 1
        return i >= 0 and jobplans[i][1] >= ts

    fb = _fallbacks(host)
    rows = []
    for s in steps:
        lo, hi = s["ts"], s["ts"] + s["dur"]
        d = dev[bisect.bisect_left(dev_ts, lo) : bisect.bisect_left(dev_ts, hi)]
        busy_iv = _merge([(e["ts"], e["ts"] + e["dur"]) for e in d])
        busy = sum(b - a for a, b in busy_iv)
        gaps, c = [], lo
        for a, b in busy_iv:
            if a > c:
                gaps.append((c, a))
            c = max(c, b)
        if c < hi:
            gaps.append((c, hi))
        segs = _self_segments(host[: bisect.bisect_right(host_ts, hi)], lo, hi, fb)
        seg_a = [x[0] for x in segs]
        attrib: collections.Counter = collections.Counter()
        for a, b in gaps:
            for i in range(max(0, bisect.bisect_right(seg_a, a) - 1), len(segs)):
                sa, sb, cl = segs[i]
                if sa >= b:
                    break
                attrib[cl] += max(0.0, min(b, sb) - max(a, sa))
        rt_gap: collections.Counter = collections.Counter()
        for e in runtime:
            if e["ts"] >= hi:
                break
            if e["ts"] + e["dur"] <= lo or e["tid"] == tid:
                continue
            for a, b in gaps:
                ov = min(b, e["ts"] + e["dur"]) - max(a, e["ts"])
                if ov > 0:
                    rt_gap[e["name"]] += ov
        inside = host[bisect.bisect_left(host_ts, lo) : bisect.bisect_left(host_ts, hi)]
        jp = [e["dur"] for e in inside if e["name"].startswith("launch_jobplan:")]
        sync = [e["dur"] for e in inside if e["name"] == "host::synchronize"]
        copies = collections.defaultdict(lambda: [0, 0])
        for m in (e for e in d if e["cat"] == "gpu_memcpy"):
            args = m.get("args") or {}
            kind = args.get("call", "?")
            if kind == "HtoD":
                le = launch_by_ext.get(args.get("External id"))
                if le is not None and in_jobplan(le["ts"]):
                    kind = "HtoD_correction"
            copies[kind][0] += 1
            copies[kind][1] += args.get("bytes", 0)
        fb_in = [e for e in inside if id(e) in fb]
        fb_ops: dict = collections.defaultdict(lambda: [0, 0.0, 0])
        for e in fb_in:
            o = fb_ops[e["name"]]
            o[0] += 1
            o[1] += e["dur"]
            o[2] += fb[id(e)]
        rows.append(
            {
                "step": s["name"],
                "kind": s["kind"],
                "tag": s["tag"],
                "tokens": s["tokens"],
                "wall_us": s["dur"],
                "busy_us": busy,
                "idle_us": s["dur"] - busy,
                "kernel_us": _covered(
                    [(e["ts"], e["ts"] + e["dur"]) for e in d if e["cat"] == "kernel"]
                ),
                "collective_us": _covered(
                    [(e["ts"], e["ts"] + e["dur"]) for e in d if _is_coll(e)]
                ),
                "n_gaps": len(gaps),
                "gap_attrib_us": dict(attrib),
                "gap_runtime_us": dict(rt_gap),
                "fallback_n": len(fb_in),
                "fallback_us": sum(e["dur"] for e in fb_in),
                "fallback_ops": {
                    k: {"n": v[0], "us": v[1], "in_compiled": v[2]} for k, v in fb_ops.items()
                },
                "launches": sum(e["name"] == "launch::Compute" for e in inside),
                "launch_marker_us": sum(
                    e["dur"]
                    for e in inside
                    if e["name"] in ("launch::Compute", "launch::HostCompute")
                ),
                "jobplan_n": len(jp),
                "jobplan_us": sum(jp),
                "h2d": copies["HtoD"][0] + copies["HtoD_correction"][0],
                "h2d_bytes": copies["HtoD"][1] + copies["HtoD_correction"][1],
                "corr": copies["HtoD_correction"][0],
                "corr_bytes": copies["HtoD_correction"][1],
                "d2h": copies["DtoH"][0],
                "d2h_bytes": copies["DtoH"][1],
                "sync_n": len(sync),
                "sync_us": sum(sync),
            }  # fmt: skip
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    """Mean per step in ms, spsprofile.host.summarize's keys; {} for no rows."""
    if not rows:
        return {}
    n = len(rows)

    def mean(f):
        return sum(f(r) for r in rows) / n

    cls: collections.Counter = collections.Counter()
    rtc: collections.Counter = collections.Counter()
    fbo: dict = collections.defaultdict(lambda: {"n": 0, "us": 0.0, "in_compiled": 0})
    for r in rows:
        cls.update(r["gap_attrib_us"])
        rtc.update(r["gap_runtime_us"])
        for k, v in r["fallback_ops"].items():
            for f in ("n", "us", "in_compiled"):
                fbo[k][f] += v[f]
    launches = sum(r["launches"] for r in rows)
    jobplans = sum(r["jobplan_n"] for r in rows)
    wall = mean(lambda r: r["wall_us"])
    return {
        "n": n,
        "fallback_per_step": mean(lambda r: r["fallback_n"]),
        "fallback_ms_per_step": mean(lambda r: r["fallback_us"]) / 1e3,
        "fallback_ops": {
            k: {
                "calls_per_step": v["n"] / n,
                "ms_per_step": v["us"] / n / 1e3,
                "in_compiled": v["in_compiled"],
            }  # fmt: skip
            for k, v in fbo.items()
        },
        "wall_ms": wall / 1e3,
        "busy_ms": mean(lambda r: r["busy_us"]) / 1e3,
        "host_exposed_ms": mean(lambda r: r["idle_us"]) / 1e3,
        "busy_pct": 100 * mean(lambda r: r["busy_us"]) / wall if wall else 0.0,
        "collective_ms": mean(lambda r: r["collective_us"]) / 1e3,
        "gaps_per_step": mean(lambda r: r["n_gaps"]),
        "launches_per_step": launches / n,
        "launch_marker_us_per_launch": sum(r["launch_marker_us"] for r in rows) / max(launches, 1),
        "jobplan_us_per_launch": sum(r["jobplan_us"] for r in rows) / max(jobplans, 1),
        "h2d_per_step": mean(lambda r: r["h2d"]),
        "h2d_kb_per_step": mean(lambda r: r["h2d_bytes"]) / 1024,
        "corr_per_step": mean(lambda r: r["corr"]),
        "corr_kb_per_step": mean(lambda r: r["corr_bytes"]) / 1024,
        "d2h_per_step": mean(lambda r: r["d2h"]),
        "d2h_kb_per_step": mean(lambda r: r["d2h_bytes"]) / 1024,
        "sync_n_per_step": mean(lambda r: r["sync_n"]),
        "sync_ms_per_step": mean(lambda r: r["sync_us"]) / 1e3,
        "waterfall_ms_per_step": {k: v / n / 1e3 for k, v in cls.most_common()},
        "runtime_threads_in_gaps_ms_per_step": {k: v / n / 1e3 for k, v in rtc.most_common()},
    }


def host_report(trace: str, h: dict) -> dict:
    """host.json entry: analyse() summaries, plus steps whose batch was mixed."""
    return {
        "trace": trace,
        "audit": h["audit"],
        "prefill_summary": summarize(h["prefill"]),
        "decode_summary": summarize(h["decode"]),
        "mixed_summary": summarize([r for r in h["steps"] if r["tag"] == "mixed"]),
        "steps": h["steps"],
    }
