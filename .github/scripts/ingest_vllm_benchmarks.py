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
from json.decoder import JSONDecodeError
from typing import Any

import clickhouse_connect

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

RESULTS_TABLE = "results_v3"


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
        "--dry-run",
        action="store_true",
        help="print rows instead of inserting into ClickHouse",
    )

    return parser.parse_args()


def read_benchmark_results(filepath: str) -> list[dict[str, Any]]:
    """Read benchmark results from a JSON file (standard or JSONEachRow format)."""
    results = []
    with open(filepath) as f:
        try:
            r = json.load(f)
            if isinstance(r, dict):
                results.append(r)
            elif isinstance(r, list):
                results = r
        except JSONDecodeError:
            f.seek(0)
            for line in f:
                try:
                    r = json.loads(line)
                    if isinstance(r, dict):
                        results.append(r)
                    elif isinstance(r, list):
                        results.extend(r)
                except JSONDecodeError:
                    pass
    return results


def extract_rows(
    results_dir: str,
    branch: str,
    sha: str,
    run_id: str,
    job_id: str,
    workflow: str,
    pr_number: int,
) -> list[dict[str, Any]]:
    """Extract rows from benchmark JSON files."""
    rows = []
    ts = int(time.time() * 1000)

    for file in glob.glob(f"{results_dir}/*.json"):
        filename = os.path.basename(file)
        records = read_benchmark_results(file)

        if not records:
            log.warning("No results in %s", filename)
            continue

        for record in records:
            if "benchmark" not in record or "metric" not in record:
                continue

            benchmark = record["benchmark"]
            metric = record["metric"]

            test_name = benchmark.get("test_name", filename)
            model = benchmark.get("model", "unknown")
            metric_name = metric.get("name", "unknown")
            benchmark_values = metric.get("benchmark_values", [])

            for value in benchmark_values:
                extra = json.dumps(
                    {
                        "device": "spyre",
                        "arch": "x86_64",
                        "hardware_type": "IBM_Spyre",
                        "model": model,
                        "test_name": test_name,
                        "head_sha": sha,
                        "pr_number": pr_number,
                        "value": value,
                    }
                )

                rows.append(
                    {
                        "timestamp": ts,
                        "schema_version": "v3",
                        "name": workflow,
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

    return rows


def insert_to_clickhouse(rows: list[dict[str, Any]]) -> None:
    """Insert rows into ClickHouse using environment-configured connection."""
    host = os.environ["CLICKHOUSE_HOST"]
    port = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
    user = os.environ["CLICKHOUSE_USER"]
    password = os.environ["CLICKHOUSE_PASS"]
    database = os.environ["CLICKHOUSE_DB"]

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
    log.info("Inserted %d rows", len(rows))


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
