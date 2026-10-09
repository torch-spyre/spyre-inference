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

"""Shareable Markdown report of a run_1102.sh output directory, or of a run_suite.sh suite.

TTFT (the issue's workload): `vllm bench latency --output-len 1` average latency against the
hf arm's median, ttft.py's metric. ITL (--itl, the method of the granite-3.3-8b TP4 ITL
report of 2026-10-02): per arm and replicate, ITL = (lat64 - lat1) / 63 from the average and
from the p50 latencies, and TTFT = lat1's average. Every summary is the median over
replicates. A suite adds tensor-parallel scaling: each arm at TP N against its own TP1.

Usage: report.py <out-dir>
       report.py --suite <suite-dir>    (its run_1102.sh output subdirectories)
"""

import argparse
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path

# A timed window holding a recompile has one iteration far above the rest.
LEAK_SPREAD = 1.10
TTFT_RE = re.compile(r"input_len=\d+ TTFT median ([\d.]+) s \(min ([\d.]+), max ([\d.]+)\)")
THREADS_RE = re.compile(r"Setting each threading configuration to (\d+) for (\d+) worker")
PROCESS_RE = re.compile(r"^\((\S+) pid=\d+\)")
ARMS = ("hf", "main", "branch")
PERCENTILES = (10, 50, 90, 99)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _percentile(values: list[float], q: float) -> float:
    """numpy's default (linear) percentile, which vllm bench latency reports."""
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _vllm_recompiles(log: str) -> int | None:
    """Post-warmup recompiles of the busiest process: every TP worker logs its own."""
    if "[__recompiles]" not in log:
        return None
    _, _, timed = log.partition("\nWarming up...")
    per_process = collections.Counter(
        m.group(1) if (m := PROCESS_RE.match(line)) else ""
        for line in timed.splitlines()
        if "[__recompiles] Recompiling" in line
    )
    return max(per_process.values(), default=0)


def _row(run_dir: Path) -> dict:
    meta = _read_json(run_dir / "meta.json")
    log_path = run_dir / "run.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    arm = "branch" if meta.get("arm") == "vllm" else meta.get("arm")
    row = {
        "arm": arm,
        "rep": meta.get("rep"),
        "rc": meta.get("rc"),
        "output_len": meta.get("output_len", 1),
        "dir": run_dir.name,
    }
    data = _read_json(run_dir / "latency.json")
    if lat := data.get("latencies"):
        row.update(
            n=len(lat),
            avg=statistics.mean(lat),
            min=min(lat),
            max=max(lat),
            **{f"p{q}": _percentile(lat, q) for q in PERCENTILES},
        )
        # The issue's metric: vLLM's average latency; hf's median, as ttft.py prints.
        row["metric"] = row["p50"] if arm == "hf" else row["avg"]
        if steps := [s for t in data.get("per_token", []) if t for s in t[1:]]:
            row["decode_step"] = statistics.mean(steps)
    elif arm == "hf" and (m := TTFT_RE.search(log)):
        median, low, high = map(float, m.groups())
        row.update(metric=median, p50=median, min=low, max=high)
    if arm != "hf":
        # The last line is the engine's, which sizes the workers.
        if found := THREADS_RE.findall(log):
            row["threads"], row["workers"] = map(int, found[-1])
        row["recompiles"] = _vllm_recompiles(log)
    if "min" in row:
        row["spread"] = row["max"] / row["min"]
    return row


def _summarize(rows: list[dict], workload: dict) -> dict:
    lo, hi = min(workload["output_lens"]), max(workload["output_lens"])
    summary = {}
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm]
        if not mine:
            continue
        s: dict = {"failed": [r["dir"] for r in mine if "metric" not in r]}
        if not workload["itl"]:
            if ok := [r["metric"] for r in mine if "metric" in r]:
                s.update(ttft=statistics.median(ok), n=len(ok))
        else:
            itl, itl_p50, ttft, steps = [], [], [], []
            for rep in sorted({r["rep"] for r in mine}):
                by_len = {r["output_len"]: r for r in mine if r["rep"] == rep and "avg" in r}
                if lo in by_len:
                    ttft.append(by_len[lo]["avg"])
                if lo in by_len and hi in by_len:
                    itl.append((by_len[hi]["avg"] - by_len[lo]["avg"]) / (hi - lo))
                    itl_p50.append((by_len[hi]["p50"] - by_len[lo]["p50"]) / (hi - lo))
                if "decode_step" in by_len.get(hi, {}):
                    steps.append(by_len[hi]["decode_step"])
            if ttft:
                s["ttft"] = statistics.median(ttft)
            if itl:
                s.update(
                    itl=statistics.median(itl),
                    itl_p50=statistics.median(itl_p50),
                    n=len(itl),
                    itl_reps=itl,
                )
            if steps:
                s["decode_step"] = statistics.median(steps)
        summary[arm] = s
    return summary


def load(out: Path) -> dict:
    rows = sorted(
        (_row(d) for d in out.glob("*_*/") if (d / "meta.json").exists()),
        key=lambda r: (
            r["rep"] or 0,
            ARMS.index(r["arm"]) if r["arm"] in ARMS else 9,
            r["output_len"],
        ),
    )
    workload = {
        "input_len": 1984,
        "output_lens": [1],
        "tp": 1,
        "iters": 3,
        "hf_iters": 5,
        "hf_script": "ttft.py",
        "hf_devices": "",
        "itl": False,
        "vllm_env": "",
    } | _read_json(out / "workload.json")
    envs: dict = {}
    for name in ("hf", "main", "branch", "vllm"):
        if env := _read_json(out / f"env_{name}.json"):
            envs.setdefault("branch" if name == "vllm" else name, env)
    prov_path, plan_path = out / "provenance.txt", out / "plan.txt"
    return {
        "dir": out,
        "rows": rows,
        "workload": workload,
        "envs": envs,
        "prov": prov_path.read_text().splitlines() if prov_path.exists() else [],
        "plan": plan_path.read_text().rstrip() if plan_path.exists() else "",
        "summary": _summarize(rows, workload),
    }


def _any_env(run: dict) -> dict:
    return next(iter(run["envs"].values()), {})


def _si(run: dict) -> tuple[dict, dict]:
    env = run["envs"].get("branch") or run["envs"].get("main") or _any_env(run)
    return env.get("spyre-inference") or {}, env.get("spyre-inference-main") or {}


def labels(run: dict) -> dict:
    hf = (run["envs"].get("hf") or _any_env(run)).get("hf-adapters-spyre") or {}
    si, main = _si(run)
    branch = f"`{si['branch']}` " if si.get("branch") else "branch "
    main_commit = f" `{main['commit'][:8]}`" if main.get("commit") else ""
    return {
        "hf": f"hf-adapters `{(hf.get('commit') or '?')[:7]}`",
        "main": f"spyre-inference main{main_commit}",
        "branch": f"spyre-inference {branch}`{(si.get('commit') or '?')[:8]}`",
    }


def _ratio(a: float | None, b: float | None) -> str:
    return f"{a / b:.2f}x" if a and b else "-"


def _change(a: float | None, b: float | None) -> str:
    return f"{(a / b - 1) * 100:+.1f}%" if a and b else "-"


def _s(v: float | None, digits: int = 3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _ms(v: float | None) -> str:
    return "-" if v is None else f"{v * 1000:.1f}"


def _title(run: dict) -> str:
    w = run["workload"]
    what = "decode latency (ITL)" if w["itl"] else "TTFT, spyre-inference#1102 recipe"
    return f"Gemma-4 26B-A4B {what}, TP{w['tp']}"


def headline(run: dict, h: str) -> list[str]:
    w, summary, names = run["workload"], run["summary"], labels(run)
    hf, main = summary.get("hf", {}), summary.get("main", {})
    out = [f"{h} Headline", ""]
    if w["itl"]:
        out += [
            "| arm | ITL (ms/token) | ITL from p50s | tok/s | TTFT (s) | ITL vs hf-adapters "
            "| ITL vs main |",
            "|---|---|---|---|---|---|---|",
        ]
        for arm, s in summary.items():
            tps = f"{1 / s['itl']:.1f}" if s.get("itl") else "-"
            vs_main = _change(s.get("itl"), main.get("itl")) if arm == "branch" else "-"
            out.append(
                f"| {names[arm]} | **{_ms(s.get('itl'))}** | {_ms(s.get('itl_p50'))} | {tps} | "
                f"{_s(s.get('ttft'))} | {_ratio(s.get('itl'), hf.get('itl'))} | {vs_main} |"
            )
    else:
        out += ["| arm | TTFT (s) | vs hf-adapters | vs main |", "|---|---|---|---|"]
        for arm, s in summary.items():
            vs_main = _change(s.get("ttft"), main.get("ttft")) if arm == "branch" else "-"
            out.append(
                f"| {names[arm]} | **{_s(s.get('ttft'))}** | "
                f"{_ratio(s.get('ttft'), hf.get('ttft'))} | {vs_main} |"
            )
    out.append("")
    for arm, s in summary.items():
        if s["failed"]:
            out.append(
                f"**{names[arm]}: {len(s['failed'])} run(s) FAILED** "
                f"({', '.join(s['failed'])}): see their run.log; no number is reported for them."
            )
    reps = [s["n"] for s in summary.values() if s.get("n")]
    if reps and min(reps) < 3:
        out.append(f"{min(reps)} replicate(s) per arm: indicative, not an equivalence claim.")
    return out


def _tp_method(w: dict) -> str:
    tp = w["tp"]
    if tp == 1:
        return "TP1, one card per arm."
    return (
        f"TP{tp}: vLLM with `--tensor-parallel-size {tp}`; hf-adapters under torchrun with {tp} "
        f'ranks, `tp_plan="auto"` and `SPYRE_DEVICES={w["hf_devices"] or "?"}`.'
    )


def method(run: dict, h: str) -> list[str]:
    w = run["workload"]
    out = [f"{h} Method", ""]
    if w["itl"]:
        lo, hi = min(w["output_lens"]), max(w["output_lens"])
        out += [
            f"Each arm runs twice per replicate, changing only the output length: `{lo}` gives "
            f"an end-to-end latency of about TTFT, `{hi}` gives TTFT + {hi - lo} decode steps, "
            f"so ITL = (lat{hi} - lat{lo}) / {hi - lo}, from the averages (headline) and from "
            f"the p50s. vLLM: `vllm bench latency` under `{w['vllm_env']}`, {w['iters']} timed "
            f"iterations after 2 warmups. hf-adapters: `hf_latency.py`, ttft.py's timing at the "
            f"same lengths, {w['hf_iters']} timed calls after 2 warmups (`min_new_tokens` = "
            "output length, like vLLM's ignored EOS)."
        ]
    elif w["hf_script"] == "ttft.py":
        out += [
            "The issue's two commands: `vllm bench latency --output-len 1` average latency "
            "against the median TTFT of the maintainer's `ttft.py` (5 timed calls after 2 "
            "warmups)."
        ]
    else:
        out += [
            f"The issue's commands at TP{w['tp']}: `vllm bench latency --output-len 1` average "
            f"latency against the median of `hf_latency.py --output-len 1`, which is ttft.py's "
            f"timing on a sharded model ({w['hf_iters']} timed calls after 2 warmups)."
        ]
    out += [
        f"{_tp_method(w)} Batch 1, {w['input_len']}-token random prompt, max-model-len 2048, "
        f"prefix caching off (bench default). Arms alternate, {w.get('repeat', '?')} "
        "replicate(s), summarized by their median; one process on the cards at a time. The "
        "main arm runs an export of the main commit through PYTHONPATH in the same "
        "environment, so only spyre-inference differs between main and the branch.",
        "",
    ]
    if run["plan"]:
        out += ["```text", run["plan"], "```", ""]
    return out


def raw(run: dict, h: str) -> list[str]:
    names = labels(run)
    out = [
        f"{h} Raw results (seconds per request)",
        "",
        "| arm | rep | output-len | avg | p10 | p50 | p90 | p99 | notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in run["rows"]:
        name = names.get(r["arm"], r["arm"])
        if "metric" not in r:
            out.append(
                f"| {name} | {r['rep']} | {r['output_len']} | FAILED rc={r['rc']} | | | | | "
                f"see `{r['dir']}/run.log` |"
            )
            continue
        notes = []
        if r["spread"] > LEAK_SPREAD:
            notes.append("LEAK: max/min > 1.10, trust the p50 only")
        if "avg" not in r:
            notes.append(f"ttft.py median; min {r['min']:.3f}, max {r['max']:.3f}")
        if r.get("rc"):
            notes.append(f"exited rc={r['rc']} after writing its results")
        if r.get("threads"):
            workers = f" x {r['workers']} workers" if r.get("workers", 1) > 1 else ""
            notes.append(f"{r['threads']} threads{workers}")
        if r.get("recompiles") is not None:
            worst = " (busiest worker)" if r.get("workers", 1) > 1 else ""
            notes.append(f"{r['recompiles']} post-warmup recompiles{worst}")
        if "decode_step" in r:
            notes.append(f"in-run decode step {r['decode_step'] * 1000:.1f} ms")
        out.append(
            f"| {name} | {r['rep']} | {r['output_len']} | {_s(r.get('avg'))} | "
            f"{_s(r.get('p10'))} | {_s(r.get('p50'))} | {_s(r.get('p90'))} | "
            f"{_s(r.get('p99'))} | {'; '.join(notes)} |"
        )
    out.append("")
    return out


def environment(run: dict, h: str) -> list[str]:
    env = _any_env(run)
    si, main = _si(run)
    out = [f"{h} Environment", ""]
    if si:
        rel = f", main + {main['branch_ahead']} commit(s)" if "branch_ahead" in main else ""
        tree = f"{si['dirty']} modified tracked file(s)" if si.get("dirty") else "clean tree"
        branch = f"`{si['branch']}` at " if si.get("branch") else ""
        subject = f': "{si["subject"]}"' if si.get("subject") else ""
        out.append(
            f"- **spyre-inference, branch under test:** {branch}`{si.get('commit')}` "
            f"({tree}{rel}){subject}"
        )
    if main:
        current = ", = GitHub main at run time" if main.get("github_main") == main["commit"] else ""
        out.append(
            f"- **spyre-inference main:** `{main['ref']}` at `{main['commit']}`{current}: "
            f'"{main.get("subject")}"'
        )
    hf = (run["envs"].get("hf") or env).get("hf-adapters-spyre") or {}
    if hf:
        out.append(f"- **hf-adapters:** `{hf.get('commit')}` (`{hf.get('version')}`)")
    ts = env.get("torch-spyre") or {}
    out.append(
        f"- **torch-spyre:** `{ts.get('commit')}` (installed `{ts.get('version')}`), "
        "one build for all arms"
    )
    versions = [
        f"**{d}** {(env.get(d) or {}).get('version', '?')}"
        for d in ("torch", "vllm", "transformers")
    ]
    out.append(f"- {', '.join(versions)}; Python `{env.get('python')}`")
    for line in run["prov"]:
        if line.startswith(("host ", "cpu ", "cards ")):
            out.append(f"- {line}")
    rpms = [p for p in run["prov"] if p.startswith("ibm-")]
    if rpms:
        out += ["", "Spyre RPMs (`rpm -qa 'ibm-*'`):", "", "| package |", "|---|"]
        out += [f"| `{p}` |" for p in rpms]
    out.append("")
    return out


def findings(runs: list[dict], h: str) -> list[str]:
    seen = {
        (f["level"], f["message"]): f
        for run in runs
        for env in run["envs"].values()
        for f in env.get("findings", [])
    }
    if not seen:
        return []
    order = ["ERROR", "WARN", "INFO"]
    out = [
        f"{h} Environment findings",
        "",
        "WARN deviates from the issue's recipe and was accepted for this run; INFO is recorded "
        "only.",
        "",
    ]
    for (level, message), f in sorted(seen.items(), key=lambda kv: order.index(kv[0][0])):
        out.append(f"- **{level}** [{f['arm']}] {message}")
    if any("libaiupti" in m for _, m in seen):
        out.append("\n**A profiler build was timed (--allow-profiler): latencies are inflated.**")
    out.append("")
    return out


def scaling(runs: list[dict], h: str) -> list[str]:
    """Each arm's summary at every TP of a workload against the same arm's TP1."""
    names = labels(runs[0])
    out = []
    for itl, key, what, fmt, flag in (
        (False, "ttft", "TTFT (s), #1102 recipe", _s, "--ttft-tps"),
        (True, "itl", "ITL (ms/token)", _ms, "--itl-tps"),
    ):
        by_tp = {r["workload"]["tp"]: r["summary"] for r in runs if r["workload"]["itl"] == itl}
        if len(by_tp) < 2:
            continue
        if 1 not in by_tp:
            out += [f"{what}: no TP1 run to scale from; include 1 in {flag}.", ""]
            continue
        tps = sorted(by_tp)
        out += [
            f"{h} {what}",
            "",
            "| arm | " + " | ".join(f"TP{tp}" for tp in tps) + " |",
            "|---" * (len(tps) + 1) + "|",
        ]
        for arm in ARMS:
            base = by_tp[1].get(arm, {}).get(key)
            cells = []
            for tp in tps:
                v = by_tp[tp].get(arm, {}).get(key)
                if v is None:
                    cells.append("-")
                elif tp == 1 or not base:
                    cells.append(fmt(v))
                else:
                    cells.append(f"{fmt(v)} ({base / v:.2f}x, {base / v / tp:.0%})")
            if any(c != "-" for c in cells):
                out.append(f"| {names[arm]} | " + " | ".join(cells) + " |")
        out.append("")
    if out:
        out = [
            f"{h[:-1]} Tensor-parallel scaling",
            "",
            "Each cell: the arm's median, then its speedup over its own TP1 and the parallel "
            "efficiency (speedup / TP). TP sizes ran one after another, not alternated, so "
            "session drift enters here, unlike in the arm comparisons.",
            "",
            *out,
        ]
    return out


def single(run: dict) -> str:
    prov = run["prov"]
    date = next((p.split("date ")[-1][:10] for p in prov if " date " in p), "?")
    lines = [f"# {_title(run)}", "", f"**Date:** {date}  ", f"**Run:** `{run['dir']}`", ""]
    lines += headline(run, "##") + [""] + method(run, "##") + raw(run, "##")
    lines += environment(run, "##") + findings([run], "##")
    return "\n".join(lines)


def suite(top: Path) -> tuple[str, list[dict]]:
    runs = [load(d) for d in sorted(top.iterdir()) if (d / "workload.json").exists()]
    if not runs:
        raise SystemExit(f"report.py: no run_1102.sh output under {top}")
    runs.sort(key=lambda r: (r["workload"]["itl"], r["workload"]["tp"]))
    first = runs[0]
    names = labels(first)
    date = next((p.split("date ")[-1][:10] for p in first["prov"] if " date " in p), "?")
    si, _ = _si(first)
    branch = f"`{si['branch']}`" if si.get("branch") else "the branch under test"
    lines = [
        f"# Gemma-4 26B-A4B on Spyre: hf-adapters vs spyre-inference main vs {branch}",
        "",
        f"**Date:** {date}  ",
        f"**Suite:** `{top}`",
        "",
        "## Headline",
        "",
        "| metric | "
        + " | ".join(names[a] for a in ARMS)
        + " | branch vs main | branch vs hf-adapters |",
        "|---" * (len(ARMS) + 3) + "|",
    ]
    for run in runs:
        w, summary = run["workload"], run["summary"]
        if w["itl"]:
            keys = [("itl", "ITL (ms/token)", _ms), ("ttft", "TTFT (s), lat1 of the ITL runs", _s)]
        else:
            keys = [("ttft", "TTFT (s), #1102 recipe", _s)]
        for key, what, fmt in keys:
            vals = {a: summary.get(a, {}).get(key) for a in ARMS}
            if not any(vals.values()):
                continue
            cells = " | ".join(fmt(vals[a]) for a in ARMS)
            lines.append(
                f"| TP{w['tp']} {what} | {cells} | {_change(vals['branch'], vals['main'])} | "
                f"{_ratio(vals['branch'], vals['hf'])} |"
            )
    lines += [
        "",
        "ITL = (lat64 - lat1) / 63 from `vllm bench latency` averages (the granite TP4 ITL "
        "report's method); TTFT rows of the ITL runs are their `--output-len 1` averages. "
        "Every value is the median over replicates. Per-workload details follow.",
        "",
    ]
    lines += scaling(runs, "###")
    for run in runs:
        lines += [f"## {_title(run)}", ""] + headline(run, "###")[2:] + [""]
        lines += method(run, "###") + raw(run, "###")
    lines += environment(first, "##") + findings(runs, "##")
    return "\n".join(lines), runs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path, nargs="?")
    p.add_argument("--suite", type=Path)
    args = p.parse_args()
    if args.suite:
        text, runs = suite(args.suite)
        print(text)
        (args.suite / "report.json").write_text(
            json.dumps(
                [
                    {
                        "dir": str(r["dir"]),
                        "workload": r["workload"],
                        "summary": r["summary"],
                        "rows": r["rows"],
                    }
                    for r in runs
                ],
                indent=2,
            )
        )
        return 0 if all("metric" in row for r in runs for row in r["rows"]) else 1
    if args.out is None:
        p.error("give an output directory or --suite")
    run = load(args.out)
    print(single(run))
    (args.out / "report.json").write_text(
        json.dumps(
            {
                "workload": run["workload"],
                "summary": run["summary"],
                "rows": run["rows"],
                "envs": run["envs"],
            },
            indent=2,
        )
    )
    return 0 if run["rows"] and all("metric" in r for r in run["rows"]) else 1


if __name__ == "__main__":
    sys.exit(main())
