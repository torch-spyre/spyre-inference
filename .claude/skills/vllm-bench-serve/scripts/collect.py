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

"""Check the legs of a vllm-bench-serve comparison and render their results.

validate BENCH LEG
    Checks one leg run by run_leg.sh: every config of BENCH/serve-tests.yaml has a result
    JSON with completed == num_prompts and failed == 0, and its server became ready.
    Exit 0 = valid, 1 = invalid.

summarize BENCH --legs TARGET[,TARGET...],BASELINE
    Writes BENCH/summary.json (every metric, server startup, init and compile times, the
    compile cache state and counts of every start, validity, warnings, the server env and
    deltas per leg and config) and two HTML fragments in the classes of
    report-example.html: BENCH/results-table.html and BENCH/leg-validity.html. The last
    leg is the baseline. A leg without BENCH/<leg>/leg.json is reported as not run.
"""

import argparse
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / ".github" / "scripts"))
from run_vllm_benchmarks import _load_configs  # noqa: E402

# (key, label, kind, start). kind: lower / higher is better, count (Δ only when the values
# differ, never coloured) or text. start: the server start whose cache state a timing row
# depends on (default, cold or warm), None for a row that does not depend on it.
LOG_ROWS = [
    ("init_engine_s", "Init engine (s)", "lower", True),
    ("compilation_s", "Total compilation (s)", "lower", True),
    ("model_warmup_s", "Model graph warmup (s)", "lower", True),
    ("buckets", "Model graph buckets", "count", False),
    ("attn_record_s", "Attention graph recording (s)", "lower", True),
    ("attn_graphs", "Attention graphs", "count", False),
    ("attn_split", "Per layer: per-seq + batched-decode", "text", False),
]
CACHE_LAYERS = [
    ("fxgraph", "FX graph"),
    ("aotautograd", "AOTAutograd"),
    ("spyre_kernel", "Spyre kernel"),
    ("output_code", "Output code"),
    ("vllm", "vLLM"),
]
SUMMARY_ROWS = [
    ("completed", "Successful requests", "count", None),
    ("failed", "Failed requests", "count", None),
    ("max_concurrency", "Max request concurrency", "count", None),
    ("duration", "Benchmark duration (s)", "lower", None),
    ("total_input_tokens", "Total input tokens", "count", None),
    ("total_output_tokens", "Total generated tokens", "count", None),
    ("request_throughput", "Request throughput (req/s)", "higher", None),
    ("output_throughput", "Output token throughput (tok/s)", "higher", None),
    ("max_output_tokens_per_s", "Peak output token throughput (tok/s)", "higher", None),
    ("max_concurrent_requests", "Peak concurrent requests", "count", None),
    ("total_token_throughput", "Total token throughput (tok/s)", "higher", None),
]
LATENCY_BANDS = [
    ("ttft", "Time to first token", "TTFT"),
    ("tpot", "Time per output token (excl. first)", "TPOT"),
    ("itl", "Inter-token latency", "ITL"),
    ("e2el", "End-to-end latency", "E2EL"),
]
SECONDS = {"startup_s", "init_engine_s", "compilation_s", "model_warmup_s", "attn_record_s"}
ITL_LEAK_RATIO = 1.1

TS = re.compile(r"\b(\d\d-\d\d \d\d:\d\d:\d\d)\b")
INIT = re.compile(
    r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"
    r"(?: \(compilation: ([\d.]+) s)?"
)
WARM = re.compile(r"Warmup complete in ([\d.]+)s for (\d+) buckets")
ATTN = re.compile(r"Attention graph recording complete: (\d+) graphs in ([\d.]+)s")
SPLIT = re.compile(r"Recording (\d+) per-seq \+ (\d+) batched-decode attention variants")
READY = "Application startup complete"
ERROR_LINE = re.compile(r"\b(Error|Traceback)\b")


def configs(bench: Path) -> list[dict]:
    loaded = _load_configs(bench / "serve-tests.yaml")
    if loaded is None:
        sys.exit(f"collect.py: cannot read {bench / 'serve-tests.yaml'}")
    return loaded


def config_label(cfg: dict) -> str:
    server = cfg.get("server_parameters", {})
    model = str(server.get("model", "?")).rstrip("/").rsplit("/", 1)[-1]
    path = str(cfg.get("parameters", {}).get("dataset-path", ""))
    m = re.search(r"SPYRE_(\w+)_DATASET", path)
    dataset = m.group(1).lower() if m else Path(path).stem
    mml = server.get("max-model-len")
    ctx = f" {int(mml) // 1024}k" if mml else ""
    return (
        f"{model} · tp{server.get('tensor-parallel-size', 1)} · {dataset}{ctx}"
        f" · bs{server.get('max-num-seqs', '?')}"
    )


def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def startup_s(log: str) -> float | None:
    """First timestamp of the server log to the last one before startup completes.

    Misses the imports before vLLM logs its first line; the harness's own timing (the
    startup record, or the result JSON's server_startup_sec) is preferred when present.
    """
    ts = first = None
    for line in log.splitlines():
        m = TS.search(line)
        if m:
            # A leap year, so that 02-29 parses. The log has no year.
            ts = datetime.strptime(f"2000-{m.group(1)}", "%Y-%m-%d %H:%M:%S")
            first = first or ts
        if READY in line:
            if not (ts and first) or ts < first:
                return None
            return (ts - first).total_seconds()
    return None


def max_of(pat: re.Pattern, log: str, group: int) -> float | None:
    """Max across TP workers, as vLLM reports compilation."""
    vals = [float(m.group(group)) for m in pat.finditer(log) if m.group(group)]
    return max(vals) if vals else None


def parse_server_log(log: str) -> dict:
    splits = sorted(set(SPLIT.findall(log)))
    buckets = max_of(WARM, log, 2)
    graphs = max_of(ATTN, log, 1)
    return {
        "startup_s": startup_s(log),
        "init_engine_s": max_of(INIT, log, 1),
        "compilation_s": max_of(INIT, log, 2),
        "model_warmup_s": max_of(WARM, log, 1),
        "buckets": int(buckets) if buckets is not None else None,
        "attn_record_s": max_of(ATTN, log, 2),
        "attn_graphs": int(graphs) if graphs is not None else None,
        "attn_split": " / ".join(f"{p} + {d}" for p, d in splits) or None,
    }


def cache_label(start: dict) -> str:
    """`warm (1558 hit, 0 miss)`: the state of one start with its total hits and misses."""
    if start.get("state", "unknown") == "unknown":
        return "unknown (no cache line)"
    hits = sum(v for k, v in start.items() if k.endswith("_hits"))
    misses = sum(v for k, v in start.items() if k.endswith("_misses"))
    return f"{start['state']} ({hits} hit, {misses} miss)"


def warm_start_warning(warm: dict) -> dict | None:
    """The misses and new files of a cold-warm warm start, or None for a full hit.

    The warm start reads the cache its own cold start just filled, so any miss or new
    file means a cache key that is not stable across restarts.
    """
    misses = {k[: -len("_misses")]: v for k, v in warm.items() if k.endswith("_misses") and v}
    new = {k[: -len("_new_files")]: v for k, v in warm.items() if k.endswith("_new_files") and v}
    if not (misses or new or warm.get("state") == "unknown"):
        return None
    return {"state": warm.get("state"), "misses": misses, "new_files": new}


def add_starts(rec: dict, results: Path, name: str, server_log: str) -> None:
    """Every start of <name>_startup.json in rec, as `{start}_{field}`.

    The default mode's one start keeps the unprefixed timing keys of today's table. In
    cold-warm, the serving start's log gives the warm_ timing keys and
    <name>_cold_server.log the cold_ ones.
    """
    path = results / f"{name}_startup.json"
    startup = json.loads(path.read_text()) if path.exists() else {}
    mode = startup.get("mode", "default")
    starts = startup.get("starts", {})
    rec["compile_cache_mode"] = mode
    if mode == "cold-warm":
        cold_path = results / f"{name}_cold_server.log"
        cold_log = cold_path.read_text(errors="replace") if cold_path.exists() else ""
        for prefix, log in (("cold_", cold_log), ("warm_", server_log)):
            for key, v in parse_server_log(log).items():
                rec[prefix + key] = v
    for start, srec in starts.items():
        for key, v in srec.items():
            rec[f"{start}_{'startup_s' if key == 'seconds' else key}"] = v
        rec[f"{start}_state_label"] = cache_label(srec)
    if "default" not in starts and mode == "default":
        rec["default_state"] = "unknown"
        rec["default_state_label"] = "unknown (no startup record)"
    if mode == "cold-warm" and "warm" in starts:
        warning = warm_start_warning(starts["warm"])
        if warning:
            rec["warnings"].append(warning)


def first_errors(*logs: str, limit: int = 5) -> list[str]:
    lines = [ln.strip() for log in logs for ln in log.splitlines() if ERROR_LINE.search(ln)]
    return lines[:limit]


def check_config(leg_dir: Path, cfg: dict, leg_env: dict) -> dict:
    name = cfg["test_name"]
    results = leg_dir / "results"
    server_log_path = results / f"{name}_server.log"
    bench_log_path = results / f"{name}_bench.log"
    server_log = server_log_path.read_text(errors="replace") if server_log_path.exists() else ""
    bench_log = bench_log_path.read_text(errors="replace") if bench_log_path.exists() else ""
    rec = {"test_name": name, "errors": [], "warnings": [], **parse_server_log(server_log)}

    # The server env: the config's environment_variables (SPYRE_DEVICES and AIU_WORLD_SIZE
    # included) override the leg's env.txt.
    env = {**leg_env, **{k: str(v) for k, v in cfg.get("environment_variables", {}).items()}}
    rec["env"] = env
    add_starts(rec, results, name, server_log)

    result_path = results / f"{name}.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    rec["result"] = result
    if result and result.get("server_startup_sec") is not None:
        rec["startup_s"] = result["server_startup_sec"]
    ready = READY in server_log
    if not server_log:
        rec["errors"].append(f"no {server_log_path.name}")
    elif not ready:
        rec["errors"].append("server never became ready")
    if result is None:
        rec["errors"].append(f"no {result_path.name}")
    else:
        want = result.get("num_prompts")
        if result.get("completed") != want:
            rec["errors"].append(f"completed {result.get('completed')} of {want} prompts")
        if result.get("failed", 0):
            rec["errors"].append(f"{result['failed']} requests failed")
        mean, median = result.get("mean_itl_ms"), result.get("median_itl_ms")
        if mean and median:
            rec["itl_ratio"] = round(mean / median, 3)
            rec["compile_leak"] = rec["itl_ratio"] > ITL_LEAK_RATIO

    rec["valid"] = not rec["errors"]
    if not rec["valid"]:
        rec["errors"] += first_errors(server_log, bench_log)
        tail_log = bench_log if ready else server_log
        rec["log_tail"] = tail_log.splitlines()[-50:]
    return rec


def check_leg(bench: Path, leg: str) -> dict:
    leg_dir = bench / leg
    meta_path = leg_dir / "leg.json"
    if not meta_path.exists():
        return {"name": leg, "label": leg, "run": False, "valid": False, "configs": {}}
    meta = json.loads(meta_path.read_text())
    env = read_env(leg_dir / "env.txt")
    recs = {c["test_name"]: check_config(leg_dir, c, env) for c in configs(bench)}
    valid = not meta.get("error") and all(r["valid"] for r in recs.values())
    return {**meta, "run": True, "valid": valid, "configs": recs}


def warning_text(warning: dict) -> str:
    parts = [f"{layer} {n} miss" for layer, n in warning["misses"].items()]
    parts += [f"{layer} {n} new files" for layer, n in warning["new_files"].items()]
    return f"warm start not a full cache hit ({warning['state']}): " + (
        ", ".join(parts) or "no cache line"
    )


def validate(args) -> int:
    leg = check_leg(args.bench, args.leg)
    if not leg["run"]:
        print(f"{args.leg}: not run (no leg.json)")
        return 1
    if leg.get("error"):
        print(f"{args.leg}: {leg['error']}")
    for name, rec in leg["configs"].items():
        status = "valid" if rec["valid"] else "INVALID"
        ratio = rec.get("itl_ratio")
        extra = f", ITL mean/median {ratio}" if ratio else ""
        print(f"{args.leg} {name}: {status}{extra}")
        for err in rec["errors"]:
            print(f"    {err}")
        for warning in rec["warnings"]:
            print(f"    WARNING: {warning_text(warning)}")
    return 0 if leg["valid"] else 1


def value(leg: dict, cfg: str, key: str):
    rec = leg["configs"].get(cfg)
    if not rec:
        return None
    if key in rec:
        return rec[key]
    return (rec.get("result") or {}).get(key)


def unreliable(base: dict, target: dict, cfg: str, start: str | None) -> bool:
    """A startup Δ compares like with like only when both starts had the same cache state
    and neither was partial (or unknown)."""
    if start is None:
        return False
    states = {value(leg, cfg, f"{start}_state") or "unknown" for leg in (base, target)}
    return len(states) > 1 or bool(states & {"partial", "unknown"})


def delta(base, target, kind: str) -> tuple[str, str, float | None]:
    """(text, css class, fraction) of (target - baseline) / baseline."""
    if base is None or target is None:
        return "", "", None
    if kind == "text":
        return ("≠", "", None) if base != target else ("", "", None)
    if kind == "count" and base == target:
        return "", "", None
    if not base:
        return "", "", None
    d = (target - base) / base
    text = f"{d * 100:+.1f}%".replace("-", "−")
    if kind == "count":
        return text, "", d
    better = d if kind == "higher" else -d
    if better >= 0.02:
        cls = "d-good"
    elif better > -0.02:
        cls = ""
    elif better >= -0.10:
        cls = "d-warn"
    else:
        cls = "d-bad"
    return text, cls, d


def fmt(v, key: str) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return html.escape(v)
    if re.sub(r"^(cold|warm)_", "", key) in SECONDS:
        return f"{v:.1f}"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def latency_rows(legs: list[dict], cfg_names: list[str]) -> list:
    keys = {
        k
        for leg in legs
        for n in cfg_names
        for k in ((leg["configs"].get(n) or {}).get("result") or {})
    }
    pcts = {m.group(1) for k in keys if (m := re.match(r"p([\d.]+)_\w+_ms$", k))}
    rows = []
    for metric, band, short in LATENCY_BANDS:
        rows.append(band)
        rows.append((f"mean_{metric}_ms", f"Mean {short} (ms)", "lower", None))
        rows.append((f"median_{metric}_ms", f"Median {short} (ms)", "lower", None))
        for p in sorted(pcts, key=float):
            rows.append((f"p{p}_{metric}_ms", f"P{p} {short} (ms)", "lower", None))
    return rows


def cold_warm(legs: list[dict], cfg_names: list[str]) -> bool:
    return any(
        (leg["configs"].get(n) or {}).get("compile_cache_mode") == "cold-warm"
        for leg in legs
        for n in cfg_names
    )


def startup_rows(legs: list[dict], cfg_names: list[str]) -> list:
    """The startup rows: one start in the default mode, cold and warm side by side in
    cold-warm."""
    if not cold_warm(legs, cfg_names):
        rows = [
            ("startup_s", "Server startup (s)", "lower", "default"),
            "Startup and compilation",
            ("default_state_label", "Compile cache state", "text", None),
            *[(k, label, kind, "default" if t else None) for k, label, kind, t in LOG_ROWS],
        ]
        starts = [("default", "")]
    else:
        rows = [
            ("cold_startup_s", "Server startup, cold (s)", "lower", "cold"),
            ("warm_startup_s", "Server startup, warm (s)", "lower", "warm"),
            "Startup and compilation",
            ("cold_state_label", "Compile cache state, cold", "text", None),
            ("warm_state_label", "Compile cache state, warm", "text", None),
        ]
        for key, label, kind, timing in LOG_ROWS:
            if timing:
                stem = label.removesuffix(" (s)")
                rows.append((f"cold_{key}", f"{stem}, cold (s)", kind, "cold"))
                rows.append((f"warm_{key}", f"{stem}, warm (s)", kind, "warm"))
            else:
                rows.append((f"warm_{key}", label, kind, None))
        starts = [("cold", ", cold"), ("warm", ", warm")]
    rows.append("Compile cache")
    for start, suffix in starts:
        for layer, name in CACHE_LAYERS:
            rows.append((f"{start}_{layer}_hits", f"{name} hits{suffix}", "count", None))
            rows.append((f"{start}_{layer}_misses", f"{name} misses{suffix}", "count", None))
            rows.append((f"{start}_{layer}_new_files", f"{name} new files{suffix}", "count", None))
    return rows


def all_rows(legs, cfg_names) -> list:
    rows = startup_rows(legs, cfg_names)
    rows += ["Request summary", *SUMMARY_ROWS]
    return rows + latency_rows(legs, cfg_names)


def render_table(legs, base, cfgs) -> str:
    targets = [leg for leg in legs if leg is not base]
    names = [c["test_name"] for c in cfgs]
    ncols = 1 + len(names) * (1 + 2 * len(targets))
    head1 = ['<th class="k"></th>'] + [
        f'<th colspan="{1 + 2 * len(targets)}">{html.escape(config_label(c))}</th>' for c in cfgs
    ]
    head2 = ['<th class="k">Metric</th>']
    for _ in cfgs:
        head2.append(f"<th>{html.escape(base['label'])}</th>")
        for t in targets:
            head2 += [f"<th>{html.escape(t['label'])}</th>", "<th>&Delta;</th>"]

    body = []
    for row in all_rows(legs, names):
        if isinstance(row, str):
            body.append(f'<tr class="band"><td colspan="{ncols}">{html.escape(row)}</td></tr>')
            continue
        key, label, kind, start = row
        if all(value(leg, n, key) is None for leg in legs for n in names):
            continue
        cells = []
        for n in names:
            b = value(base, n, key)
            cells.append(f"<td>{fmt(b, key)}</td>")
            for t in targets:
                v = value(t, n, key)
                text, cls, _ = delta(b, v, kind)
                cells.append(f"<td>{fmt(v, key)}</td>")
                if text and unreliable(base, t, n, start):
                    cells.append(
                        f'<td class="unreliable">{text}'
                        '<span class="caveat">unreliable: cache states differ</span></td>'
                    )
                else:
                    cells.append(f'<td class="{cls}">{text}</td>' if cls else f"<td>{text}</td>")
        body.append(f'<tr><td class="k">{html.escape(label)}</td>{"".join(cells)}</tr>')

    return "\n".join(
        [
            '<div class="scroll">',
            "  <table>",
            "    <thead>",
            f'      <tr class="grp">{"".join(head1)}</tr>',
            f"      <tr>{''.join(head2)}</tr>",
            "    </thead>",
            "    <tbody>",
            *(f"      {line}" for line in body),
            "    </tbody>",
            "  </table>",
            "</div>",
        ]
    )


def render_validity(legs, cfgs) -> str:
    cards, alerts = [], []
    for leg in legs:
        for c in cfgs:
            name = c["test_name"]
            short = html.escape(config_label(c).split(" · ", 1)[-1])
            who = f"{html.escape(leg['label'])} &middot; {short}"
            rec = leg["configs"].get(name)
            if not leg["run"] or rec is None:
                cards.append(
                    f'<div class="leg pending"><span class="who">{who}</span>'
                    '<span class="st wt">pending</span><span class="det">not run</span></div>'
                )
                continue
            res = rec.get("result") or {}
            det = [f"{res.get('completed', 0)}/{res.get('num_prompts', '?')}"]
            det.append(f"{res.get('failed', '?')} failed")
            if rec.get("itl_ratio"):
                leak = " (compile leak?)" if rec.get("compile_leak") else ""
                det.append(f"ITL mean/median {rec['itl_ratio']:.2f}{leak}")
            ok = rec["valid"] and not leg.get("error")
            cls, st = (
                ("", '<span class="st ok">valid</span>')
                if ok
                else (
                    " bad",
                    '<span class="st no">invalid</span>',
                )
            )
            warn = "".join(
                f'<span class="det caveat">{html.escape(warning_text(w))}</span>'
                for w in rec.get("warnings", [])
            )
            cards.append(
                f'<div class="leg{cls}"><span class="who">{who}</span>{st}'
                f'<span class="det">{" &middot; ".join(det)}</span>{warn}</div>'
            )
            if not ok:
                errs = ([leg["error"]] if leg.get("error") else []) + rec["errors"]
                items = "".join(f"<li><code>{html.escape(e)}</code></li>" for e in errs)
                alerts.append(
                    f'<div class="note alert"><h3>{who} failed</h3><ul>{items}</ul></div>'
                )
    return "\n".join(['<div class="legs">', *(f"  {c}" for c in cards), "</div>", *alerts])


def summarize(args) -> int:
    names = [n for n in args.legs.split(",") if n]
    if len(names) < 2:
        sys.exit("collect.py: --legs needs at least one target and the baseline")
    cfgs = configs(args.bench)
    legs = [check_leg(args.bench, n) for n in names]
    base = legs[-1]

    deltas = {}
    for t in legs[:-1]:
        for c in cfgs:
            n = c["test_name"]
            for row in all_rows(legs, [n]):
                if isinstance(row, str):
                    continue
                key, _, kind, start = row
                text, cls, d = delta(value(base, n, key), value(t, n, key), kind)
                if text:
                    deltas.setdefault(t["name"], {}).setdefault(n, {})[key] = {
                        "delta": d,
                        "text": text,
                        "class": cls,
                        "unreliable": unreliable(base, t, n, start),
                    }

    warnings = [
        {"leg": leg["name"], "config": n, **w}
        for leg in legs
        for n, rec in leg["configs"].items()
        for w in rec.get("warnings", [])
    ]
    (args.bench / "results-table.html").write_text(render_table(legs, base, cfgs) + "\n")
    (args.bench / "leg-validity.html").write_text(render_validity(legs, cfgs) + "\n")
    summary = {
        "legs": legs,
        "baseline": base["name"],
        "compile_cache": sorted(
            {leg.get("compile_cache", "default") for leg in legs if leg["run"]}
        ),
        "configs": {c["test_name"]: config_label(c) for c in cfgs},
        "deltas": deltas,
        "warnings": warnings,
    }
    (args.bench / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for leg in legs:
        state = "valid" if leg["valid"] else ("not run" if not leg["run"] else "INVALID")
        print(f"{leg['name']}: {state}")
    for w in warnings:
        print(f"WARNING: {w['leg']} {w['config']}: {warning_text(w)}")
    print(f"wrote {args.bench}/summary.json, results-table.html, leg-validity.html")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("bench", type=Path)
    v.add_argument("leg")
    s = sub.add_parser("summarize")
    s.add_argument("bench", type=Path)
    s.add_argument("--legs", required=True)
    args = p.parse_args()
    return validate(args) if args.cmd == "validate" else summarize(args)


if __name__ == "__main__":
    sys.exit(main())
