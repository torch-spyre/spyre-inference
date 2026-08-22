#!/usr/bin/env python3
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

"""Classify a failed CI run's logs as a Spyre hardware fault vs a test failure.

The discriminator is `"category":"hardware"` in a RAS JSON blob, which torch-spyre
emits when a card aborts the process and is absent from ordinary pytest failures.
The GitHub logs zip lays out one file per step as `<job>/<step#>_<step>.txt`, so a
job's node echo and its RAS abort share a top-level folder, giving per-job
attribution. Always exits 0: this classifies a failure, it is not one.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Cheap gate before json-parsing; the blob is then sliced first-{ to last-} to
# strip GHA's timestamp prefix and ANSI wrappers.
RE_HW_CATEGORY = re.compile(r'"category"\s*:\s*"hardware"')

# The (?!\$) skips the command-echo line that prints the literal
# `$GHA_RUNNER_POD_NODE_NAME` before the resolved node value.
RE_NODE = re.compile(r"GHA_RUNNER_POD_NODE_NAME\s+(?!\$)([A-Za-z0-9][\w.-]*)")


def _extract_json_blob(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(line[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _iter_jobs(log_dir: Path) -> list[tuple[str, list[Path]]]:
    """Group .txt log files by job folder; loose files fall under '<root>'."""
    jobs: dict[str, list[Path]] = {}
    for path in sorted(log_dir.rglob("*.txt")):
        rel = path.relative_to(log_dir)
        job = rel.parts[0] if len(rel.parts) > 1 else "<root>"
        jobs.setdefault(job, []).append(path)
    return sorted(jobs.items())


def _scan_job(job: str, files: list[Path]) -> dict[str, Any] | None:
    """Return a hardware-fault event for this job, or None if not a hw fault."""
    node: str | None = None
    # Dedupe RAS events by (name, code): the same blob appears on both the
    # ras_base.hpp ERRR line and the C++ `what():` line.
    events: dict[tuple[str, str], dict[str, str]] = {}
    for path in files:
        text = path.read_text(errors="replace")
        for line in text.splitlines():
            if node is None:
                m = RE_NODE.search(line)
                if m:
                    node = m.group(1)
            if RE_HW_CATEGORY.search(line):
                blob = _extract_json_blob(line)
                if blob and blob.get("category") == "hardware":
                    name = str(blob.get("name", "unknown"))
                    code = str(blob.get("code", ""))
                    events.setdefault(
                        (name, code),
                        {
                            "ras_name": name,
                            "ras_code": code,
                            "message": str(blob.get("message", "")),
                            "raw": line[line.find("{") : line.rfind("}") + 1],
                        },
                    )
    if not events:
        return None
    first = next(iter(events.values()))
    return {
        "job": job,
        "node": node or "unknown",
        "ras_name": first["ras_name"],
        "ras_code": first["ras_code"],
        "message": first["message"],
        "ras_events": list(events.values()),
        "raw": first["raw"],
    }


def detect(log_dir: Path) -> list[dict[str, Any]]:
    """Return one hardware-fault event per affected job (empty if none)."""
    return [
        event for job, files in _iter_jobs(log_dir) if (event := _scan_job(job, files)) is not None
    ]


def build_check_output(events: list[dict[str, Any]], run_url: str | None) -> dict[str, str]:
    """Compose the GitHub check-run `output` (title + markdown summary)."""
    nodes = sorted({e["node"] for e in events})
    if len(events) == 1:
        e = events[0]
        title = f"⚠️ Hardware fault on {e['node']} ({e['ras_name']})"
    else:
        title = f"⚠️ Hardware fault detected in {len(events)} CI job(s)"

    lines = [
        "**This CI failure was a hardware fault, not a code issue.** A Spyre "
        "card raised a RAS error and aborted the test process.",
        "",
        "| Job | Node | RAS error | Code |",
        "| --- | --- | --- | --- |",
    ]
    for e in events:
        lines.append(f"| {e['job']} | `{e['node']}` | `{e['ras_name']}` | `{e['ras_code']}` |")
    lines += [
        "",
        f"Affected node(s): {', '.join(f'`{n}`' for n in nodes)} — the card likely "
        "needs a reset, reseat, or replace. **Escalate to the infrastructure team**; "
        "re-running the failed jobs on a fresh runner may also land on a healthy card.",
    ]
    if run_url:
        lines += ["", f"Failed run: {run_url}"]
    return {"title": title, "summary": "\n".join(lines)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path, help="Dir of unzipped run logs")
    parser.add_argument(
        "--run-url", default="", help="HTML URL of the failed run (for the summary)"
    )
    parser.add_argument("--github-output", type=Path, help="Append hw_fault=true|false here")
    parser.add_argument("--check-output", type=Path, help="Write the check-run output JSON here")
    args = parser.parse_args(argv)

    if not args.log_dir.is_dir():
        print(f"::error::log dir not found: {args.log_dir}", file=sys.stderr)
        return 1

    events = detect(args.log_dir)
    hw_fault = bool(events)
    report = {"hw_fault": hw_fault, "events": events}
    print(json.dumps(report, indent=2))

    if args.github_output:
        with args.github_output.open("a") as fh:
            fh.write(f"hw_fault={'true' if hw_fault else 'false'}\n")

    if hw_fault and args.check_output:
        args.check_output.write_text(json.dumps(build_check_output(events, args.run_url or None)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
