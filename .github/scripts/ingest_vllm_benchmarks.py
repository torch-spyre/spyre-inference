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

"""
Ingest vLLM benchmark JSON results into ClickHouse.

Expects the following environment variables:
  CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER,
  CLICKHOUSE_PASS, CLICKHOUSE_DB
"""

import glob
import json
import logging
import os
import sys
import time
from argparse import ArgumentParser
from typing import Any

import clickhouse_connect
from utils import read_benchmark_results

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

RESULTS_TABLE = "results_v3"
METADATA_TABLE = "run_metadata"


def parse_args() -> Any:
    parser = ArgumentParser("Ingest vLLM benchmark results into ClickHouse")

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="directory containing benchmark result JSON files",
    )
    parser.add_argument("--workflow", type=str, default="vLLM Benchmark")
    parser.add_argument("--branch", type=str, required=True)
    parser.add_argument("--sha", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--job-id", type=str, default="0")
    parser.add_argument("--pr-number", type=str, default="0")
    parser.add_argument(
        "--arch",
        type=str,
        default=os.environ.get("BENCHMARK_ARCH", "x86_64"),
        help="hardware architecture the benchmark ran on (e.g. x86_64, ppc64le, s390x)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print rows instead of inserting into ClickHouse",
    )

    return parser.parse_args()


# Scalar metrics per vLLM bench schema. Each entry is the JSON key vLLM writes
# to --output-json (latency, throughput) or --result-filename (serve). Only
# scalar (single-number) metrics are ingested; list fields (latencies, itls,
# ttfts, ...) are the raw samples behind these aggregates and are skipped.
_LATENCY_METRICS = ("avg_latency",)
_THROUGHPUT_METRICS = (
    "elapsed_time",
    "requests_per_second",
    "tokens_per_second",
)
_SERVE_METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p99_e2el_ms",
)


def extract_vllm_metrics(record: dict[str, Any]) -> list[tuple[str, float]]:
    """Return (metric_name, value) pairs from one vLLM-native benchmark record.

    Detects the vLLM bench schema (latency / throughput / serve) by the keys
    the record carries and pulls out the scalar metrics for each. The three
    schemas are disjoint on their signature keys, so a record maps to exactly
    one. `percentiles` (latency) is a nested {percentile: value} dict and is
    flattened to `p{percentile}_latency` metrics.
    """
    pairs: list[tuple[str, float]] = []

    def _add(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        pairs.append((name, float(value)))

    if "avg_latency" in record:
        for key in _LATENCY_METRICS:
            if key in record:
                _add(key, record[key])
        percentiles = record.get("percentiles")
        if isinstance(percentiles, dict):
            for pct, value in percentiles.items():
                _add(f"p{pct}_latency", value)
    elif "requests_per_second" in record or "tokens_per_second" in record:
        for key in _THROUGHPUT_METRICS:
            if key in record:
                _add(key, record[key])
    elif "request_throughput" in record or "output_throughput" in record:
        for key in _SERVE_METRICS:
            if key in record:
                _add(key, record[key])

    return pairs


def extract_pytorch_metrics(record: dict[str, Any]) -> list[tuple[str, float]]:
    """Return (metric_name, value) pairs from one PyTorch-format record.

    This is the schema `convert_to_pytorch_benchmark_format` writes to
    `*.pytorch.json`, produced only when SAVE_TO_PYTORCH_BENCHMARK_FORMAT is
    set. Kept for compatibility with runs that enable it.
    """
    if "benchmark" not in record or "metric" not in record:
        return []
    metric = record["metric"]
    metric_name = metric.get("name", "unknown")
    return [(metric_name, float(v)) for v in metric.get("benchmark_values", [])]


def _model_from_record(record: dict[str, Any], filename: str) -> str:
    """Best-effort model name from a benchmark record, falling back to the file."""
    # vLLM-native JSON writes "model" as a top-level string
    raw_model = record.get("model")
    if isinstance(raw_model, str) and raw_model:
        return raw_model

    benchmark = record.get("benchmark", {})
    if not isinstance(benchmark, dict):
        benchmark = {}
    model_info = raw_model if isinstance(raw_model, dict) else {}
    return (
        benchmark.get("model")
        or benchmark.get("model_name")
        or model_info.get("name")
        or record.get("model_id")
        or filename.replace(".pytorch.json", "").replace(".json", "")
    )


def extract_rows(
    results_dir: str,
    branch: str,
    sha: str,
    run_id: str,
    job_id: str,
    workflow: str,
    pr_number: int,
    arch: str = "x86_64",
) -> list[dict[str, Any]]:
    """Extract ClickHouse rows from vLLM benchmark JSON files.

    The vLLM benchmark runner writes native `{test_name}.json` files
    (latency / throughput / serve schemas). When SAVE_TO_PYTORCH_BENCHMARK_FORMAT
    is set it ALSO writes `{test_name}.pytorch.json`. This reads both: the
    PyTorch-format files via their `benchmark`/`metric` schema, and every other
    `*.json` via the native vLLM schema. A `.pytorch.json` file is not read
    twice (it is excluded from the native pass).
    """
    rows = []
    ts = int(time.time() * 1000)

    all_json = set(glob.glob(f"{results_dir}/*.json"))
    pytorch_files = set(glob.glob(f"{results_dir}/*.pytorch.json"))
    native_files = sorted(all_json - pytorch_files)
    log.info(
        "Found %d vLLM-native and %d PyTorch-format benchmark JSON files in %s",
        len(native_files),
        len(pytorch_files),
        results_dir,
    )

    def _emit(filename: str, model: str, metric_name: str, value: float) -> None:
        extra = json.dumps(
            {
                "device": "spyre",
                "arch": arch,
                "hardware_type": "IBM_Spyre",
                "model": model,
                "test_name": filename.replace(".pytorch.json", "").replace(".json", ""),
                "head_sha": sha,
                "pr_number": pr_number,
                "value": value,
            }
        )
        rows.append(
            {
                "timestamp": ts,
                "schema_version": "v3",
                "name": "spyre_e2e_benchmark",
                "metric": metric_name,
                "actual": float(value),
                "target": 0.0,
                "repo": "spyre-inference",
                "head_branch": branch,
                "workflow_id": int(run_id) if run_id.isdigit() else 0,
                "job_id": int(job_id) if job_id.isdigit() else 0,
                "run_attempt": 1,
                "extra": extra,
            }
        )

    for file, extractor in [
        *[(f, extract_pytorch_metrics) for f in sorted(pytorch_files)],
        *[(f, extract_vllm_metrics) for f in native_files],
    ]:
        filename = os.path.basename(file)

        try:
            records = read_benchmark_results(file)
        except Exception:
            log.exception("Failed to read benchmark results from %s", filename)
            continue

        if not records:
            log.warning("No results in %s", filename)
            continue

        before_rows = len(rows)

        for record in records:
            if not isinstance(record, dict):
                continue
            model = _model_from_record(record, filename)
            for metric_name, value in extractor(record):
                _emit(filename, model, metric_name, value)

        extracted = len(rows) - before_rows
        if extracted:
            log.info("Extracted %d rows from %s", extracted, filename)
        else:
            log.warning("No usable metrics in %s", filename)

    log.info("Total rows extracted: %d", len(rows))
    return rows


def insert_to_clickhouse(rows: list[dict[str, Any]]) -> None:
    """Insert rows into ClickHouse using environment-configured connection."""
    clickhouse_env_vars = {
        "CLICKHOUSE_HOST": os.environ.get("CLICKHOUSE_HOST"),
        "CLICKHOUSE_USER": os.environ.get("CLICKHOUSE_USER"),
        "CLICKHOUSE_PASS": os.environ.get("CLICKHOUSE_PASS"),
        "CLICKHOUSE_DB": os.environ.get("CLICKHOUSE_DB"),
    }
    missing = [k for k, v in clickhouse_env_vars.items() if not v]
    if missing:
        raise OSError(f"Missing required environment variables: {', '.join(missing)}")

    host = clickhouse_env_vars["CLICKHOUSE_HOST"]
    port = int(os.environ.get("CLICKHOUSE_PORT") or "8123")
    user = clickhouse_env_vars["CLICKHOUSE_USER"]
    password = clickhouse_env_vars["CLICKHOUSE_PASS"]
    database = clickhouse_env_vars["CLICKHOUSE_DB"]

    client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=user,
        password=password,
        database=database,
    )

    if not rows:
        log.warning("No rows to insert")
        return

    columns = list(rows[0].keys())
    data = [[row[col] for col in columns] for row in rows]

    client.insert(
        RESULTS_TABLE,
        data,
        column_names=columns,
    )
    log.info("Inserted %d rows into %s", len(rows), RESULTS_TABLE)

    # Insert metadata rows (required for dashboard commit picker)
    metadata_rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for row in rows:
        extra_data = json.loads(row["extra"])
        key = (row["workflow_id"], row["metric"], extra_data.get("model", ""))
        if key in seen:
            continue
        seen.add(key)
        metadata_rows.append(
            {
                "timestamp": row["timestamp"],
                "repo": row["repo"],
                "head_branch": row["head_branch"],
                "head_sha": extra_data.get("head_sha", ""),
                "workflow_id": row["workflow_id"],
                "benchmark_name": row["name"],
                "model_name": extra_data.get("model", ""),
                "metric_name": row["metric"],
                "device": extra_data.get("device", "spyre"),
                "arch": extra_data.get("arch", "x86_64"),
            }
        )

    if metadata_rows:
        meta_columns = list(metadata_rows[0].keys())
        meta_data = [[r[col] for col in meta_columns] for r in metadata_rows]
        client.insert(METADATA_TABLE, meta_data, column_names=meta_columns)
        log.info("Inserted %d rows into %s", len(metadata_rows), METADATA_TABLE)


def main() -> None:
    args = parse_args()

    pr_number = int(args.pr_number) if args.pr_number else 0

    rows = extract_rows(
        results_dir=args.results_dir,
        branch=args.branch,
        sha=args.sha,
        run_id=args.run_id,
        job_id=args.job_id,
        workflow=args.workflow,
        pr_number=pr_number,
        arch=args.arch,
    )

    if not rows:
        log.warning("No benchmark results found in %s", args.results_dir)
        sys.exit(1)

    if args.dry_run:
        log.info("Dry run: would insert %d rows:", len(rows))
        for row in rows[:5]:
            print(json.dumps(row, indent=2))
        if len(rows) > 5:
            print(f"... and {len(rows) - 5} more")
        return

    insert_to_clickhouse(rows)


if __name__ == "__main__":
    main()
