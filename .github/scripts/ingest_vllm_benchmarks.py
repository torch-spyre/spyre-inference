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

import datetime
import hashlib
import json
import logging
import os
import sys
import uuid
from argparse import ArgumentParser
from typing import Any

import clickhouse_connect
from ingest_identity import golden_drift, library_provenance
from spyre_clickhouse_ingest import (
    artifact_id_for,
    base_artifact_id,
    benchmark_id_for,
    ensure,
    gha_artifact_id,
    insert_artifact_result,
    run_id_of,
    schema,
    tables_present,
    target_database,
)
from spyre_clickhouse_ingest.vllm import (
    BENCH_COMPONENT,
    BENCH_ID_KEYS,
    duration_s,
    extract_rows,
    write_benchmarks,
)

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
    # --run-id is THIS RUN'S IDENTITY, and on the Jenkins path that is the orchestrator's uuid,
    # used verbatim. Named for what a CALLER means by "the run id" rather than for what one
    # column needs: spyre-frameworks' ingest_cmd already passes ${RUN_ID} (a uuid) here, so this
    # naming makes the cross-repo caller correct without it having to know our column layout.
    # The numeric GitHub run id is --gha-run-id, below, because it is GHA-specific and only
    # in-repo workflows have one.
    # Not required: a Jenkins standalone run may have neither, and the v2 write is then skipped
    # rather than landing an unjoinable row.
    parser.add_argument("--run-id", type=str, default="")
    # The numeric GitHub Actions run id -> upstream's workflow_id (Int64) and, when --run-id
    # carries no uuid, the external_run_id half of a DERIVED v2 run_id. GHA-only by nature:
    # only an in-repo workflow has a github.run_id, and a Jenkins leg legitimately has none.
    parser.add_argument("--gha-run-id", type=str, default=os.environ.get("GITHUB_RUN_ID", ""))
    parser.add_argument("--job-id", type=str, default="0")
    parser.add_argument("--pr-number", type=str, default="0")
    parser.add_argument(
        "--arch",
        type=str,
        default=os.environ.get("BENCHMARK_ARCH", "x86_64"),
        help="hardware architecture the benchmark ran on (e.g. x86_64, ppc64le, s390x)",
    )
    parser.add_argument(
        "--v2-run-id",
        type=str,
        default=os.environ.get("V2_RUN_ID", ""),
        help="An ALREADY-DERIVED v2 run_id (a uuid). Jenkins passes the orchestrator's own "
        "params.RUN_ID here and it is used VERBATIM -- re-hashing an already-hashed id "
        "mints a third identity that joins to nothing. Mutually exclusive with the "
        "derive-from-GHA path below.",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        default=os.environ.get("TRIGGER_TYPE", "perf"),
        help="Tier for the run_id hash. 'perf' for a benchmark leg.",
    )
    parser.add_argument(
        "--rpm-lock",
        type=str,
        default=os.environ.get("SPYRE_RPM_LOCK", "spyre-rpms.lock"),
        help="spyre-rpms.lock the leg installed on top of the image; its digest joins the "
        "installed delta hashed into the leg's artifact_id. Empty when the image ran as baked.",
    )
    parser.add_argument(
        "--installed",
        type=str,
        default="",
        help="Anything else installed on top of the image (override RPMs, a torch-spyre ref), "
        "space- or comma-separated.",
    )
    parser.add_argument(
        "--state",
        choices=("passed", "failed"),
        default="passed",
        help="The benchmark step's verdict for artifact_results; partial results are still "
        "ingested when it failed.",
    )
    parser.add_argument(
        "--repository",
        type=str,
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="owner/name, for the run url and the artifact's sources.",
    )
    parser.add_argument(
        "--ci-event",
        type=str,
        default=os.environ.get("GITHUB_EVENT_NAME", ""),
        help="The GHA event that built the leg's artifact; tagged as the library's ci_tags spells.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print rows instead of inserting into ClickHouse",
    )

    return parser.parse_args()


# ── GHA artifact identity ────────────────────────────────────────────────────────────────
# The same derivation torch-spyre's GHA legs use: the runner image's stamped artifact_id with
# this leg's installed delta chained onto it (gha_artifact_id). Computed inline, since this
# ingest runs on the card runner that can read the image's id file.


def leg_installed(sha: str, rpm_lock: str, extra: str) -> str:
    """What this leg installed on top of the image, as the token set gha_artifact_id hashes."""
    tokens = [f"{BENCH_COMPONENT}@{sha[:12]}"] if sha else []
    if rpm_lock:
        try:
            with open(rpm_lock, "rb") as fh:
                tokens.append(f"spyre-rpms.lock@{hashlib.sha256(fh.read()).hexdigest()[:12]}")
        except OSError:
            log.warning("%s unreadable — left out of the installed delta", rpm_lock)
    tokens += (extra or "").replace(",", " ").split()
    return " ".join(tokens)


def _is_uuid(value) -> bool:
    try:
        uuid.UUID((value or "").strip())
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def links_artifact(args) -> bool:
    """Does this leg own its artifact_results row?

    Not when a uuid was threaded in: the orchestrator already wrote that run_id's row, and a
    second writer would duplicate it. A NUMERIC --run-id is the old GHA wiring, which does.
    """
    return not (_is_uuid(getattr(args, "run_id", "")) or _is_uuid(getattr(args, "v2_run_id", "")))


def resolve_v2_run_id(args) -> str:
    """The v2 run_id for this leg, or "" when it cannot be derived.

    Two paths, kept explicit. Jenkins already HOLDS the orchestrator's v2 run_id
    (params.RUN_ID) -- use it verbatim. GHA holds only its own integer run id, so the id is
    derived from (gha, run id, arch, test_type). Empty means the v2 write is skipped rather
    than writing an unjoinable row, which downstream cannot tell apart from "no perf ran".
    """
    # --run-id first, then --v2-run-id: the latter is kept only so a caller already passing it
    # keeps working. Both mean the same thing -- an already-derived uuid, used verbatim.
    verbatim = (getattr(args, "run_id", "") or "").strip() or (
        getattr(args, "v2_run_id", "") or ""
    ).strip()
    if verbatim:
        try:
            uuid.UUID(verbatim)
        except (ValueError, AttributeError, TypeError):
            # A numeric value here is the old wiring (GitHub's run id in --run-id). Fall through
            # to the derive path rather than skipping: that is what the caller meant.
            if verbatim.isdigit():
                return run_id_of("gha", verbatim, args.arch, getattr(args, "test_type", "perf"))
            log.warning("--run-id %r is neither a uuid nor numeric; v2 rows skipped", verbatim)
            return ""
        return verbatim
    gha = (getattr(args, "gha_run_id", "") or "").strip()
    if not gha:
        return ""
    return run_id_of("gha", gha, args.arch, getattr(args, "test_type", "perf"))


def _ci_tags(leg) -> list:
    """The main/pr/nightly tags of the leg's artifact; none from a library that predates them."""
    try:
        from spyre_clickhouse_ingest import ci_tags
    except ImportError:
        return []
    day = datetime.datetime.now(datetime.UTC).date()
    return ci_tags(
        getattr(leg, "ci_event", ""), leg.repository, leg.branch, leg.sha,
        getattr(leg, "pr_number", ""), day=day,
    )  # fmt: skip


def _write_artifact_results(client, db: str, rows, run_id_value: str, leg) -> None:
    """This leg's artifacts row and its performance verdict in artifact_results.

    Contained: a failure here must not cost the benchmark rows already written.
    """
    try:
        tables = (schema.ARTIFACTS, schema.ARTIFACT_RESULTS)
        if not tables_present(client, db, tables=tables):
            log.info("artifacts/artifact_results absent in %s — artifact link skipped", db)
            return
        base = base_artifact_id()
        if not base:
            # Nothing to chain onto; a coordinate invented here would be shared by every such leg.
            log.info("runner image carries no artifact id — artifact link skipped")
            return
        installed = leg_installed(leg.sha, leg.rpm_lock, leg.installed)
        aid = gha_artifact_id(BENCH_COMPONENT, base, installed, leg.arch)
        repo, gha = leg.repository, leg.gha_run_id
        run_url = f"https://github.com/{repo}/actions/runs/{gha}" if repo and gha else ""
        # The derive-gha-artifact-id record, so the leg registers as every GHA leg does.
        ensure(
            client,
            db,
            f"gha:{aid}|{base}|{installed}",
            leg.arch,
            component=BENCH_COMPONENT,
            run_url=run_url,
            sources=[(repo, leg.branch, leg.sha)],
            tags=_ci_tags(leg),
            tag_props={"source": "gha"},
        )
        wrote = insert_artifact_result(
            client,
            db,
            artifact_id=aid,
            run_id=run_id_value,
            test_type=leg.test_type,
            state=leg.state,
            arch=leg.arch,
            result_kind="performance",
            # Suite wall clock: each throughput run's own elapsed_time.
            duration_s=duration_s(rows),
            props={"run_url": run_url, "source": "gha"},
        )
        if wrote:
            log.info("Linked artifact %s (base %s) to run_id=%s", aid, base, run_id_value)
    except Exception as exc:  # noqa: BLE001
        log.warning("artifact_results link failed, benchmark rows unaffected: %r", exc)


# ── v2 benchmarks / benchmark_runs ───────────────────────────────────────────────────────
# The perf surfaces of the v2 dashboard read this dimension+fact pair, and the HUD's
# oss_ci_benchmark_v3 / oss_ci_benchmark_metadata are materialized views over benchmark_runs
# (schema/70-vllm-hud-projection.sql in torch-spyre). So this is the ONLY perf write: the
# upstream-shaped rows are projected from it rather than inserted a second time, which is what
# keeps the two from disagreeing.
# The identities this script writes, pinned as literals against the library that mints
# them. Installed from a floating `@main`, so the job that WRITES has to check them --
# ingest_identity says why the test-time goldens are not enough. run_id and the artifact ids
# are the cross-writer contract; benchmark_id is this producer's own, and pinned for the same
# reason: benchmarks dedups across runs on it, so a re-key silently forks every trend line.
IDENTITY_GOLDENS = (
    (
        "run_id_of",
        run_id_of,
        ("gha", "12345", "amd64", "integration"),
        "dab2a67f-14bf-53be-b6e4-fc9642086e47",
    ),
    (
        "artifact_id_for",
        artifact_id_for,
        ("torch-spyre", "flex-rpm", "abc123def456", "amd64"),
        "86a5c6e3-bd2f-5d27-9a8f-9b8d23efc65b",
    ),
    (
        "gha_artifact_id",
        gha_artifact_id,
        (
            BENCH_COMPONENT,
            "6ecddb3f-1809-533f-9552-fafdba8a331d",
            "spyre-inference@abc123def456 spyre-rpms.lock@0123456789ab",
            "amd64",
        ),
        "932cc6a6-c3eb-5be2-8057-a4fe5303ffd3",
    ),
    (
        "benchmark_id_for",
        benchmark_id_for,
        (
            BENCH_COMPONENT,
            "latency_tp1_in64_out64",
            [],
            {
                "record_type": "model",
                "run_mode": "latency",
                "tensor_parallel": "1",
                "input_len": "64",
                "output_len": "64",
            },
            BENCH_ID_KEYS,
        ),
        "f07dc029-26b3-51d2-8109-7e32e0edc1b7",
    ),
)


def _write_v2_benchmarks(client, db: str, rows, run_id_value: str) -> None:
    """benchmarks + benchmark_runs for this leg; a failure must not cost the flat rows."""
    try:
        write_benchmarks(client, db, rows, run_id_value)
    except Exception as exc:  # noqa: BLE001
        log.warning("v2 perf write failed, %s unaffected: %r", RESULTS_TABLE, exc)


def insert_to_clickhouse(
    rows: list[dict[str, Any]],
    v2_run_id_value: str = "",
    leg: Any = None,
) -> None:
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

    # v2 rows, additive. One client serves both generations, so every v2 statement is
    # QUALIFIED with this database name; "" means v2 is not configured and the write is a
    # clean no-op rather than an error.
    v2db = target_database()
    if v2db and v2_run_id_value:
        # Logged whether or not it drifted: this is what attributes a row to the code that
        # wrote it once `main` has moved past it.
        log.info("v2 identity: %s", library_provenance())
        drift = golden_drift(IDENTITY_GOLDENS)
        if drift:
            # ::error:: so it is an annotation, not a line in a 10k-line log. results_v3 and
            # run_metadata still go in: the drift costs v2 visibility, and writing ids
            # nothing else can join costs more.
            print(
                "::error::v2 skipped — the shared identity library no longer mints the ids "
                "this ingest was built against, so its rows would not join any other "
                f"writer's: {'; '.join(drift)}",
                file=sys.stderr,
            )
            v2db = ""
    if v2_run_id_value and v2db:
        if leg is not None:
            _write_artifact_results(client, v2db, rows, v2_run_id_value, leg)
        _write_v2_benchmarks(client, v2db, rows, v2_run_id_value)
    elif v2_run_id_value:
        # Cause-agnostic: a drift has already said its piece as an ::error:: above, and this
        # is the only report when the database is simply not configured.
        log.warning(
            "no v2 database (CLICKHOUSE_DB_V2 unset, or the identity drift above) — "
            "v2 perf rows skipped, %s still written",
            RESULTS_TABLE,
        )
    else:
        # Loud: without a run_id the perf numbers cannot reach an artifact, and a blank
        # artifact page reads as "no perf ran" rather than "not linked".
        log.warning(
            "no v2 run_id (pass --v2-run-id on Jenkins, or --run-id + --arch on Actions) "
            "— v2 perf rows skipped, %s still written",
            RESULTS_TABLE,
        )

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
        # workflow_id's source: the NUMERIC id. --run-id may hold a uuid (Jenkins), which
        # int()s to 0 and would blank run_url and collapse run_metadata's dedup key.
        run_id=args.gha_run_id,
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

    insert_to_clickhouse(rows, resolve_v2_run_id(args), args if links_artifact(args) else None)


if __name__ == "__main__":
    main()
