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

"""Run one vllm bench case and hand its numbers to the spyre-bench engine.

  run   executes the runner command given after `--` (unchanged: it still writes the native vLLM
        JSON to --raw-dir for the existing ingest), then reads that JSON and writes `metrics.json`
        into --repeat-dir, the sidecar the engine turns into the repeat's samples of a Bench result.

Only metrics the llm-serving pack registers are written. vLLM's `tokens_per_second` of the
throughput bench counts prompt and output tokens together, which no registered metric means, so it
is left out.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

_SERVE = {
    "request_throughput": ("llm.request_throughput", 1.0),
    "output_throughput": ("llm.output_throughput", 1.0),
    "mean_ttft_ms": ("llm.ttft.mean", 1e-3),
    "p99_ttft_ms": ("llm.ttft.p99", 1e-3),
    "median_itl_ms": ("llm.itl.median", 1e-3),
    "p99_itl_ms": ("llm.itl.p99", 1e-3),
    "p99_tpot_ms": ("llm.tpot.p99", 1e-3),
    "mean_e2el_ms": ("llm.e2el.mean", 1e-3),
    "p99_e2el_ms": ("llm.e2el.p99", 1e-3),
}


def samples_of(record: dict) -> dict[str, list[float]]:
    """Registered engine metrics of one native vLLM record; the three schemas are disjoint."""
    if "avg_latency" in record:
        pairs = {"llm.batch_latency.mean": (record["avg_latency"], 1.0)}
    elif "requests_per_second" in record or "tokens_per_second" in record:
        pairs = {"llm.request_throughput": (record.get("requests_per_second"), 1.0)}
    elif "request_throughput" in record or "output_throughput" in record:
        pairs = {name: (record.get(key), scale) for key, (name, scale) in _SERVE.items()}
    else:
        pairs = {}
    out = {
        k: [float(v) * scale]
        for k, (v, scale) in pairs.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    }
    # Pooling runs report 0 output tokens/s: not a measurement of this metric.
    if out.get("llm.output_throughput") == [0.0]:
        del out["llm.output_throughput"]
    return out


def find_result(raw_dir: Path, case: str) -> Path | None:
    for f in sorted(raw_dir.glob("*.json")):
        if f.name.lower() == f"{case.lower()}.json":
            return f
    return None


def cmd_run(args) -> int:
    if not args.argv:
        print("bench_perf_vllm: no runner command after `--`", file=sys.stderr)
        return 2
    rc = subprocess.run(args.argv, check=False).returncode
    if rc:
        return rc
    raw_dir = Path(args.raw_dir)
    result = find_result(raw_dir, args.case)
    if result is None:
        print(
            f"bench_perf_vllm: {args.case}: the runner left no {args.case}.json in {raw_dir}",
            file=sys.stderr,
        )
        return 2
    doc = json.loads(result.read_text())
    record = doc[0] if isinstance(doc, list) and doc else doc
    samples = samples_of(record) if isinstance(record, dict) else {}
    if not samples:
        print(
            f"bench_perf_vllm: {args.case}: no registered metric in {result.name}", file=sys.stderr
        )
        return 2
    repeat_dir = Path(args.repeat_dir)
    repeat_dir.mkdir(parents=True, exist_ok=True)
    (repeat_dir / "metrics.json").write_text(json.dumps({"samples": samples}))
    if not os.environ.get("BENCH_QUIET"):
        print(f"recorded {args.case}: {', '.join(sorted(samples))}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run the runner command after `--` and write metrics.json")
    run.add_argument("--case", required=True)
    run.add_argument("--repeat-dir", required=True)
    run.add_argument("--raw-dir", required=True)
    run.add_argument("argv", nargs=argparse.REMAINDER)
    run.set_defaults(fn=cmd_run)
    args = ap.parse_args(argv)
    if args.argv[:1] == ["--"]:
        args.argv = args.argv[1:]
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
