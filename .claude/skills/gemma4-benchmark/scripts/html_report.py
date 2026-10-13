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

"""Shareable HTML page of a run_suite.sh suite or of one run_1102.sh directory.

Every number comes from the run directories through report.py's loader, the same data
report.md shows. The page works without any prose; a narrative JSON (default:
<dir>/narrative.json) adds what needs judgement after reading the results. Its keys, all
optional: title, eyebrow, h1, lede, refs, verdict [{big, lab}], recommendation, itl_sub,
ttft_sub, itl_notes, ttft_notes, scaling_notes, findings [{sev: major|minor|nit, title, loc,
dl: [[term, html]]}], checks, method, not_covered, footer. Strings are HTML. `checks` and
`method` add to the generated ones; every other key replaces its default.

An ITL workload with a LEAK run (max/min > 1.10) is shown from p50s, the skill's gate rule,
with the average-based value alongside.

Usage: html_report.py <suite-or-run-dir> [--narrative FILE] > report.html
"""

import argparse
import html
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report  # noqa: E402

HERE = Path(__file__).resolve().parent
COLORS = {"hf": "--s-hf", "main": "--s-main", "branch": "--s-branch"}
SHORT = {"hf": "hf-adapters", "main": "main", "branch": "branch"}


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def pct(a, b):
    return (a / b - 1) * 100 if a and b else None


def fmt_pct(v) -> str:
    return "-" if v is None else f"{v:+.1f}%".replace("-", "−")


def overlap(a: list, b: list) -> bool:
    return bool(a and b) and min(a) <= max(b) and min(b) <= max(a)


def color(arm: str) -> str:
    return f"var({COLORS.get(arm, '--ink-3')})"


def swatch(arm: str) -> str:
    return f'<i class="sw" style="background:{color(arm)}"></i>'


def arms_of(run: dict) -> list[str]:
    present = {r["arm"] for r in run["rows"]}
    return [a for a in report.ARMS if a in present]


def all_arms(runs: list[dict]) -> list[str]:
    present = {a for run in runs for a in arms_of(run)}
    return [a for a in report.ARMS if a in present]


def names_of(runs: list[dict]) -> tuple[dict, dict]:
    """HTML label (commit in <code>) and plain label of every arm, from report.labels."""
    out_html, out_plain = {}, {}
    for run in runs:
        md = report.labels(run)
        for arm in arms_of(run):
            label = md.get(arm, arm)
            out_html.setdefault(arm, re.sub(r"`([^`]*)`", r"<code>\1</code>", esc(label)))
            out_plain.setdefault(arm, label.replace("`", ""))
    return out_html, out_plain


def _itl_reps(run: dict, arm: str, field: str) -> list[float]:
    lo, hi = min(run["workload"]["output_lens"]), max(run["workload"]["output_lens"])
    rows = {(r["rep"], r["output_len"]): r for r in run["rows"] if r["arm"] == arm and field in r}
    return [
        (rows[(k, hi)][field] - rows[(k, lo)][field]) / (hi - lo)
        for k in sorted({rep for rep, _ in rows})
        if (k, lo) in rows and (k, hi) in rows
    ]


def _ttft_reps(run: dict, arm: str) -> list[float]:
    """Per-replicate TTFT (s): the issue's metric, or lat1's average in an ITL workload."""
    w = run["workload"]
    lo = min(w["output_lens"])
    field = "avg" if w["itl"] else "metric"
    rows = sorted(
        (r for r in run["rows"] if r["arm"] == arm and r["output_len"] == lo and field in r),
        key=lambda r: r["rep"] or 0,
    )
    return [r[field] for r in rows]


def metrics(runs: list[dict]) -> list[dict]:
    """One entry per (metric, TP): every arm's median and replicates, and its comparison step.

    Each spyre-inference arm is compared with the arm before it: main and a lone branch with
    hf-adapters (as a ratio), the branch with main (as a change).
    """
    out = []
    for run in sorted(runs, key=lambda r: (not r["workload"]["itl"], r["workload"]["tp"])):
        w, summary, order = run["workload"], run["summary"], arms_of(run)
        leak = any(r.get("spread", 1) > report.LEAK_SPREAD for r in run["rows"])
        if w["itl"]:
            keys = [
                ("itl", "ITL from p50s (LEAK runs)" if leak else "ITL", "ms / token", 1000, 1),
                ("ttft", "TTFT, lat1 of the ITL runs", "s", 1, 3),
            ]
        else:
            keys = [("ttft", "TTFT, #1102 recipe", "s", 1, 3)]
        for key, what, unit, scale, digits in keys:
            arms = []
            for arm in order:
                s = summary.get(arm, {})
                entry = {
                    "arm": arm,
                    "value": s[key] * scale if s.get(key) else None,
                    "reps": [v * scale for v in _ttft_reps(run, arm)],
                }
                if key == "itl":
                    avg, p50 = _itl_reps(run, arm, "avg"), _itl_reps(run, arm, "p50")
                    shown, other = (p50, avg) if leak else (avg, p50)
                    entry.update(
                        value=statistics.median(shown) * scale if shown else None,
                        reps=[v * scale for v in shown],
                        alt=statistics.median(other) * scale if other else None,
                        alt_basis="averages" if leak else "p50s",
                    )
                arms.append(entry)
            for prev, a in zip(arms, arms[1:]):
                if a["arm"] == "hf" or not (a["value"] and prev["value"]):
                    continue
                a.update(
                    vs=SHORT.get(prev["arm"], prev["arm"]),
                    vs_hf=prev["arm"] == "hf",
                    change=pct(a["value"], prev["value"]),
                    ratio=a["value"] / prev["value"],
                    overlap=overlap(a["reps"], prev["reps"]),
                )
            out.append(
                {
                    "id": f"{key}{'_itl' if w['itl'] and key == 'ttft' else ''}_tp{w['tp']}",
                    "key": key,
                    "itl": w["itl"],
                    "tp": w["tp"],
                    "what": what,
                    "unit": unit,
                    "digits": digits,
                    "arms": arms,
                }
            )
    return out


def chart_data(ms: list[dict], plain: dict, ids: list[str]) -> list[dict]:
    return [
        {
            "label": f"TP{m['tp']}",
            "unit": m["unit"],
            "digits": m["digits"],
            "sub": m["what"],
            "rows": [
                {
                    "arm": a["arm"],
                    "name": plain.get(a["arm"], a["arm"]),
                    "short": SHORT.get(a["arm"], a["arm"]),
                    "color": color(a["arm"]),
                    "value": a["value"],
                    "reps": a["reps"],
                    "vs": a.get("vs"),
                    "vs_hf": a.get("vs_hf"),
                    "change": a.get("change"),
                    "ratio": a.get("ratio"),
                    "overlap": a.get("overlap"),
                }
                for a in m["arms"]
            ],
        }
        for m in ms
        if m["id"] in ids
    ]


def table(head: str, rows: list[str]) -> str:
    return (
        f'<div class="tablewrap"><table><thead>{head}</thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def metric_table(ms: list[dict], names: dict, ids: list[str]) -> str:
    with_main = any(a["arm"] == "main" for m in ms for a in m["arms"])
    head = (
        '<tr><th>Metric</th><th>Arm</th><th class="num">Median</th><th class="num">Replicates</th>'
        '<th class="num">Other basis</th><th class="num">vs hf-adapters</th>'
        + ('<th class="num">vs main</th>' if with_main else "")
        + "</tr>"
    )
    body = []
    for m in ms:
        if m["id"] not in ids:
            continue
        by = {a["arm"]: a for a in m["arms"]}
        d = m["digits"]
        for i, a in enumerate(m["arms"]):
            label = (
                f"TP{m['tp']} {esc(m['what'])} <small>({esc(m['unit'])})</small>" if not i else ""
            )
            v = "FAILED" if a["value"] is None else f"{a['value']:.{d}f}"
            reps = ", ".join(f"{r:.{d}f}" for r in a["reps"]) or "-"
            alt = f"{a['alt']:.{d}f} <small>from {a['alt_basis']}</small>" if a.get("alt") else "-"
            hf = by.get("hf", {}).get("value")
            vs_hf = f"{a['value'] / hf:.2f}×" if a["value"] and hf and a["arm"] != "hf" else "-"
            cells = [
                f'<td class="num"><span class="mono"><b>{v}</b></span></td>',
                f'<td class="num"><span class="mono">{reps}</span></td>',
                f'<td class="num"><span class="mono">{alt}</span></td>',
                f'<td class="num">{vs_hf}</td>',
            ]
            if with_main:
                step = "-"
                if a["arm"] not in ("hf", "main") and by.get("main", {}).get("value"):
                    step = f"<b>{fmt_pct(pct(a['value'], by['main']['value']))}</b>"
                    if overlap(a["reps"], by["main"]["reps"]):
                        step += ' <span class="tag">ranges overlap</span>'
                cells.append(f'<td class="num">{step}</td>')
            first = ' class="first"' if not i else ""
            body.append(
                f"<tr{first}><td>{label}</td>"
                f"<td>{swatch(a['arm'])} {names.get(a['arm'], a['arm'])}</td>{''.join(cells)}</tr>"
            )
    return table(head, body)


def scaling_table(ms: list[dict], names: dict) -> str:
    rows = []
    for key, what, unit in (("itl", "ITL", "ms / token"), ("ttft", "TTFT, #1102 recipe", "s")):
        group = {m["tp"]: m for m in ms if m["key"] == key and m["itl"] == (key == "itl")}
        if 1 not in group or len(group) < 2:
            continue
        tps = sorted(group)
        for i, arm in enumerate(a["arm"] for a in group[1]["arms"]):
            base = next(a["value"] for a in group[1]["arms"] if a["arm"] == arm)
            cells = []
            for tp in tps:
                v = next((a["value"] for a in group[tp]["arms"] if a["arm"] == arm), None)
                d = group[tp]["digits"]
                if v is None:
                    cells.append('<td class="num">-</td>')
                elif tp == 1 or not base:
                    cells.append(f'<td class="num"><span class="mono">{v:.{d}f}</span></td>')
                else:
                    cells.append(
                        f'<td class="num"><span class="mono">{v:.{d}f}</span> '
                        f'<span class="sub">{base / v:.2f}× · {base / v / tp:.0%}</span></td>'
                    )
            label = f"{esc(what)} <small>({esc(unit)})</small>" if not i else ""
            first = ' class="first"' if not i else ""
            rows.append(
                f"<tr{first}><td>{label}</td>"
                f"<td>{swatch(arm)} {names.get(arm, arm)}</td>{''.join(cells)}</tr>"
            )
    if not rows:
        return ""
    tps = sorted({m["tp"] for m in ms})
    head = "<tr><th>Metric</th><th>Arm</th>" + "".join(f'<th class="num">TP{t}</th>' for t in tps)
    return table(head + "</tr>", rows)


def raw_table(runs: list[dict]) -> str:
    head = (
        '<tr><th>Workload</th><th>Arm</th><th class="num">Rep</th><th class="num">Out</th>'
        '<th class="num">avg</th><th class="num">p10</th><th class="num">p50</th>'
        '<th class="num">p90</th><th class="num">max/min</th><th>Notes</th></tr>'
    )
    body = []
    for run in sorted(runs, key=lambda r: (r["workload"]["itl"], r["workload"]["tp"])):
        w = run["workload"]
        workload = f"{'ITL' if w['itl'] else 'TTFT'} TP{w['tp']}"
        for r in run["rows"]:
            notes = []
            if "metric" not in r:
                notes.append(f"FAILED rc={r['rc']}")
            else:
                if r.get("spread", 1) > report.LEAK_SPREAD:
                    notes.append("LEAK: max/min > 1.10")
                if "avg" not in r:
                    notes.append("ttft.py median")
                if r.get("rc"):
                    notes.append(f"exited rc={r['rc']} after its results")
                if r.get("threads"):
                    workers = f" × {r['workers']}" if r.get("workers", 1) > 1 else ""
                    notes.append(f"{r['threads']} threads{workers}")
                if r.get("recompiles") is not None:
                    notes.append(f"{r['recompiles']} recompiles")
                if "decode_step" in r:
                    notes.append(f"in-run step {r['decode_step'] * 1000:.1f} ms")
            cells = "".join(
                f'<td class="num"><span class="mono">{report._s(r.get(k))}</span></td>'
                for k in ("avg", "p10", "p50", "p90")
            )
            spread = f"{r['spread']:.3f}" if "spread" in r else "-"
            body.append(
                f"<tr><td>{workload}</td>"
                f"<td>{swatch(r['arm'])} {esc(SHORT.get(r['arm'], r['arm']))}</td>"
                f'<td class="num">{r["rep"]}</td><td class="num">{r["output_len"]}</td>{cells}'
                f'<td class="num"><span class="mono">{spread}</span></td>'
                f"<td>{esc('; '.join(notes))}</td></tr>"
            )
    return table(head, body)


def auto_checks(runs: list[dict]) -> list[str]:
    rows = [r for run in runs for r in run["rows"]]
    vllm = [r for r in rows if r["arm"] != "hf"]
    ok = [r for r in rows if "metric" in r]
    out = [f"{len(ok)} of {len(rows)} runs produced their timings."]
    if crashed := [r for r in ok if r.get("rc")]:
        out[-1] += f" {len(crashed)} exited non-zero only after writing them."
    if rc := [r["recompiles"] for r in vllm if r.get("recompiles") is not None]:
        out.append(f"{rc.count(0)} of {len(rc)} vLLM runs logged 0 recompiles after warmup.")
    leaks = [
        (run["workload"], r)
        for run in runs
        for r in run["rows"]
        if "metric" in r and r.get("spread", 1) > report.LEAK_SPREAD
    ]
    if not leaks:
        out.append("Every run's timed iterations stay within max/min ≤ 1.10.")
    else:
        where = sorted({f"{'ITL' if w['itl'] else 'TTFT'} TP{w['tp']}" for w, _ in leaks})
        clean = all(r.get("recompiles") in (0, None) for _, r in leaks)
        out.append(
            f"{len(leaks)} run(s) exceed max/min 1.10, in {', '.join(where)}"
            + (", every vLLM one with 0 recompiles." if clean else ".")
            + " ITL there is quoted from p50s, with the averages alongside."
        )
    env = report._any_env(runs[0])
    main = env.get("spyre-inference-main") or {}
    if main.get("commit") and main.get("github_main") == main["commit"]:
        out.append(f"main was GitHub's main ({main['commit'][:8]}) when the run started.")
    if not any("libaiupti" in f["message"] for f in env.get("findings", [])):
        out.append("torch-spyre's _C.so does not link the profiler (libaiupti).")
    if threads := sorted({(r["threads"], r.get("workers", 1)) for r in vllm if r.get("threads")}):
        out.append(
            "vLLM threads per run: "
            + ", ".join(f"{t} × {w} worker{'s' if w > 1 else ''}" for t, w in threads)
            + "."
        )
    return out


def findings_html(items: list[dict]) -> str:
    out = []
    for f in items:
        dl = "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in f.get("dl", []))
        loc = f'<div class="loc">{esc(f["loc"])}</div>' if f.get("loc") else ""
        out.append(
            f'<article class="finding"><div class="fhead"><span class="sev {esc(f["sev"])}">'
            f"{esc(f['sev'])}</span><h3>{f['title']}</h3></div>{loc}<dl>{dl}</dl></article>"
        )
    return "".join(out)


def env_items(runs: list[dict]) -> list[str]:
    env = report._any_env(runs[0])
    si, main = report._si(runs[0])
    hf = (runs[0]["envs"].get("hf") or env).get("hf-adapters-spyre") or {}
    ts = env.get("torch-spyre") or {}
    arms = set(all_arms(runs))
    items = []
    if "hf" in arms and hf:
        items.append(
            f"<b>hf-adapters</b>: <code>{esc(hf.get('commit'))}</code> "
            f"(<code>{esc(hf.get('version'))}</code>{esc(report._upstream(hf))})"
        )
    if "main" in arms and main:
        items.append(
            f"<b>spyre-inference main</b>: <code>{esc(main.get('ref'))}</code> at "
            f"<code>{esc(main.get('commit'))}</code>, “{esc(main.get('subject'))}”"
        )
    if "branch" in arms and si:
        tree = f"{si['dirty']} modified tracked file(s)" if si.get("dirty") else "clean tree"
        items.append(
            f"<b>spyre-inference, branch under test</b>: <code>{esc(si.get('branch'))}</code> at "
            f"<code>{esc(si.get('commit'))}</code> ({tree}), “{esc(si.get('subject'))}”"
        )
    items.append(
        f"<b>torch-spyre</b>: <code>{esc(ts.get('commit'))}</code> "
        f"(<code>{esc(ts.get('version'))}</code>{esc(report._upstream(ts))}), "
        "one build for every arm"
    )
    items.append(
        ", ".join(
            f"<b>{d}</b> {esc((env.get(d) or {}).get('version', '?'))}"
            for d in ("torch", "vllm", "transformers")
        )
        + f"; Python <code>{esc(env.get('python'))}</code>"
    )
    items += [esc(p) for p in runs[0]["prov"] if p.startswith(("host ", "cpu ", "cards "))]
    if rpms := [p for p in runs[0]["prov"] if p.startswith("ibm-")]:
        items.append("Spyre RPMs: " + ", ".join(f"<code>{esc(p)}</code>" for p in rpms))
    return items


def env_findings(runs: list[dict]) -> str:
    seen = {
        (f["level"], f["message"]): f
        for run in runs
        for env in run["envs"].values()
        for f in env.get("findings", [])
    }
    order = ["ERROR", "WARN", "INFO"]
    items = [
        f'<li><span class="lvl {lvl.lower()}">{lvl}</span> '
        f'<span class="mono">[{esc(f["arm"])}]</span> '
        f"{esc(msg)}</li>"
        for (lvl, msg), f in sorted(seen.items(), key=lambda kv: order.index(kv[0][0]))
    ]
    return '<ul class="plain envf">' + "".join(items) + "</ul>" if items else ""


def spoken(items: list[str]) -> str:
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else "".join(items)


def defaults(runs: list[dict], plain: dict, ms: list[dict], top: Path) -> dict:
    """The page's prose when no narrative says otherwise, all derived from the runs."""
    si, _ = report._si(runs[0])
    arms = all_arms(runs)
    tps = sorted({r["workload"]["tp"] for r in runs})
    w0 = runs[0]["workload"]
    reps = max(r["workload"].get("repeat", 1) for r in runs)
    total = sum(len(r["rows"]) for r in runs)
    others = [plain[a] for a in arms if a != "branch"]
    subject = (
        f"<code>{esc(si['branch'])}</code>"
        if si.get("branch")
        else esc(plain.get("branch", "branch"))
    )
    date = next((p.split("date ")[-1][:10] for p in runs[0]["prov"] if " date " in p), "?")
    host = next((p.split()[1] for p in runs[0]["prov"] if p.startswith("host ")), "?")
    env = report._any_env(runs[0])
    ts = (env.get("torch-spyre") or {}).get("commit") or "?"
    refs = [f"{esc(SHORT.get(a, a))} <b>{esc(plain[a].split()[-1])}</b>" for a in arms]
    refs += [f"torch-spyre <b>{esc(ts[:8])}</b>", f"pod <b>{esc(host)}</b>"]
    verdict = []
    for m in ms:
        if m["key"] == "ttft" and m["itl"]:
            continue
        b = next((a for a in m["arms"] if a["arm"] == "branch"), None)
        ref = next((a for a in m["arms"] if a["arm"] == "main"), None) or next(
            (a for a in m["arms"] if a["arm"] == "hf"), None
        )
        if not (b and ref and b["value"] and ref["value"]):
            continue
        name = "decode" if m["key"] == "itl" else "TTFT"
        big = (
            fmt_pct(pct(b["value"], ref["value"]))
            if ref["arm"] == "main"
            else f"{b['value'] / ref['value']:.2f}×"
        )
        d, unit = m["digits"], m["unit"].replace(" / ", "/")
        verdict.append(
            {
                "big": big,
                "lab": f"{name} at TP{m['tp']}, branch vs {SHORT[ref['arm']]}<br>"
                f"{ref['value']:.{d}f} → {b['value']:.{d}f} {esc(unit)}",
            }
        )
    itl_runs = [r for r in runs if r["workload"]["itl"]]
    hi = max((max(r["workload"]["output_lens"]) for r in itl_runs), default=1)
    itl_env = itl_runs[0]["workload"].get("vllm_env") if itl_runs else ""
    return {
        "title": "Gemma-4 Spyre Benchmark",
        "eyebrow": "spyre-inference · gemma4-benchmark · " + spoken([f"TP{t}" for t in tps]),
        "h1": f"Gemma-4 26B-A4B on Spyre: {subject} against " + spoken([esc(o) for o in others]),
        "lede": (
            f"Batch 1, {w0['input_len']}-token prompt. Time to first token by the "
            "spyre-inference#1102 recipe and decode latency per token, "
            f"{reps} alternating replicate(s) per workload, {total} runs in total."
        ),
        "refs": refs,
        "verdict": verdict,
        "itl_sub": (
            f"ITL = (lat{hi} − lat1) / {hi - 1}: <code>vllm bench latency</code> at output "
            f"lengths 1 and {hi}, under <code>{esc(itl_env)}</code>; hf-adapters runs "
            "<code>hf_latency.py</code> at the same lengths. Median over replicates. Each panel "
            "has its own scale and bars start at zero. A pill compares an arm with the one above "
            "it; an outlined pill means their replicate ranges overlap."
        )
        if itl_runs
        else "",
        "ttft_sub": (
            "<code>vllm bench latency --output-len 1</code> average against the hf-adapters "
            "median (<code>ttft.py</code> at TP1, <code>hf_latency.py --output-len 1</code> "
            "above). Median over replicates."
        ),
        "method": [
            "Arms alternate "
            + " → ".join(esc(SHORT.get(a, a)) for a in arms)
            + f" inside every replicate, {reps} replicate(s) per workload; one process on the "
            "cards at a time. Commands, per workload:",
        ],
        "checks": [],
        "footer": f"Measured {esc(date)} on {esc(host)}. Directory <code>{esc(top)}</code>.",
    }


def page(top: Path, narrative: dict) -> str:
    if (top / "workload.json").exists():
        runs = [report.load(top)]
    else:
        runs = [report.load(d) for d in sorted(top.iterdir()) if (d / "workload.json").exists()]
    if not runs:
        raise SystemExit(f"html_report.py: no run_1102.sh output under {top}")
    names, plain = names_of(runs)
    ms = metrics(runs)
    ids = {
        "itl": [m["id"] for m in ms if m["key"] == "itl"],
        "lat1": [m["id"] for m in ms if m["key"] == "ttft" and m["itl"]],
        "ttft": [m["id"] for m in ms if m["key"] == "ttft" and not m["itl"]],
    }
    n = defaults(runs, plain, ms, top)
    n["checks"] = auto_checks(runs) + list(narrative.get("checks", []))
    n["method"] = (
        n["method"]
        + [
            f"{'ITL' if r['workload']['itl'] else 'TTFT'} TP{r['workload']['tp']}:"
            f"<pre>{esc(r['plan'])}</pre>"
            for r in sorted(runs, key=lambda r: (r["workload"]["itl"], r["workload"]["tp"]))
            if r["plan"]
        ]
        + list(narrative.get("method", []))
    )
    n |= {k: v for k, v in narrative.items() if k not in ("checks", "method")}

    def ul(key: str, cls: str = "plain") -> str:
        items = n.get(key) or []
        return (
            f'<ul class="{cls}">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
            if items
            else ""
        )

    legend = "".join(f"<span>{swatch(a)}{names[a]}</span>" for a in all_arms(runs))
    cells = n.get("verdict") or []
    cols = 4 if len(cells) == 4 else max(1, min(len(cells), 3))
    verdict = "".join(
        f'<div class="cell"><span class="big">{c["big"]}</span>'
        f'<span class="lab">{c["lab"]}</span></div>'
        for c in cells
    )
    rec = f'<p class="rec">{n["recommendation"]}</p>' if n.get("recommendation") else ""
    verdict_box = (
        f'<div class="verdict" style="--cols:{cols}" aria-label="Summary">{verdict}{rec}</div>'
        if verdict or rec
        else ""
    )

    def section(sid: str, eyebrow: str, title: str, body: str) -> str:
        return (
            f'<section id="{sid}"><div class="eyebrow">{eyebrow}</div>'
            f"<h2>{title}</h2>{body}</section>"
            if body.strip()
            else ""
        )

    def measured(kind: str, title: str, table_ids: list[str]) -> str:
        if not ids[kind]:
            return ""
        return section(
            kind,
            "Measured",
            title,
            f'<p class="sub">{n.get(kind + "_sub", "")}</p>'
            f'<div class="legend" aria-hidden="true">{legend}</div>'
            f'<div class="charts" data-chart="{kind}"></div>'
            f"{metric_table(ms, names, table_ids)}{ul(kind + '_notes')}",
        )

    scaling = scaling_table(ms, names)
    scaling_sec = section(
        "scaling",
        "Measured",
        "Tensor-parallel scaling",
        '<p class="sub">Each cell above TP1: the median, then its speedup over the same arm\'s '
        "TP1 and the parallel efficiency (speedup / TP). TP sizes ran one after another, not "
        "alternated, so session drift enters here; the arm comparisons above do not have that "
        "problem.</p>" + scaling + ul("scaling_notes")
        if scaling
        else "",
    )
    findings_sec = section(
        "findings",
        "Observed",
        "Findings",
        f'<div class="findings">{findings_html(n["findings"])}</div>' if n.get("findings") else "",
    )
    not_covered = "<h3>Not covered</h3>" + ul("not_covered") if n.get("not_covered") else ""
    env_list = "".join(f"<li>{i}</li>" for i in env_items(runs))
    total = sum(len(r["rows"]) for r in runs)
    data = {k: chart_data(ms, plain, ids[k]) for k in ("itl", "ttft")}
    css = (HERE / "html_report.css").read_text()
    js = (HERE / "html_report.js").read_text()
    refs = "".join(f"<span>{r}</span>" for r in n.get("refs", []))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{esc(n["title"])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
{css}
</style>
</head>
<body>
<div class="page">
  <header>
    <div class="eyebrow">{esc(n["eyebrow"])}</div>
    <h1>{n["h1"]}</h1>
    <p class="sub">{n["lede"]}</p>
    <div class="refs">{refs}</div>
  </header>
  {verdict_box}
  {measured("itl", "Decode latency (ITL)", ids["itl"] + ids["lat1"])}
  {measured("ttft", "Time to first token", ids["ttft"])}
  {scaling_sec}
  {findings_sec}
  {section("verified", "Checked", "Run integrity", ul("checks", "checks"))}
  <section id="method">
    <div class="eyebrow">Method</div>
    <h2>How this was measured</h2>
    {ul("method")}
    {not_covered}
    <h3>Environment</h3>
    <ul class="plain">{env_list}</ul>
    <h3>Environment check</h3>
    <p class="sub">WARN was reviewed and accepted for this run; INFO is recorded only.</p>
    {env_findings(runs)}
    <details>
      <summary>Every run ({total}), seconds per request</summary>
      {raw_table(runs)}
    </details>
  </section>
  <footer>{n["footer"]}</footer>
</div>
<div id="tip" hidden></div>
<script>
var CHARTS = {json.dumps(data)};
{js}
</script>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    p.add_argument("--narrative", type=Path)
    args = p.parse_args()
    path = args.narrative or args.dir / "narrative.json"
    narrative = json.loads(path.read_text()) if path.exists() else {}
    print(page(args.dir, narrative))
    return 0


if __name__ == "__main__":
    sys.exit(main())
