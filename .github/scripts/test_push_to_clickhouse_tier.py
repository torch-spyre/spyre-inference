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

"""push-to-clickhouse's tier step, run as the workflow runs it.

The tier is a run_id hash input, so an empty one makes the v2 ingest skip the whole run.
The env below is what GitHub actually delivers: workflow_run.name is the run-name.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest
import yaml

_WORKFLOW = pathlib.Path(__file__).resolve().parents[1] / "workflows" / "push-to-clickhouse.yaml"


def _tier_script() -> str:
    steps = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["ingest"]["steps"]
    return next(s["run"] for s in steps if s.get("id") == "extract-tier")


@pytest.mark.parametrize(
    ("workflow", "title", "tier"),
    [
        ("regression", "regression", "regression"),
        ("trunk", "trunk", "trunk"),
        ("integration-tests (integration)", "integration-tests (integration)", "integration"),
        ("integration-tests (perf)", "integration-tests (perf)", "perf"),
        ("test_each_commit", "regression", "regression"),
        ("integration-tests", "", ""),
    ],
)
def test_tier_from_display_title(tmp_path, workflow, title, tier):
    out = tmp_path / "out"
    env = {
        **os.environ,
        "TRIGGERING_WORKFLOW": workflow,
        "TRIGGERING_DISPLAY_TITLE": title,
        "GITHUB_OUTPUT": str(out),
    }
    subprocess.run(["bash", "-e", "-c", _tier_script()], env=env, check=True, capture_output=True)
    assert out.read_text().splitlines() == [f"tier={tier}"]


def _download_script() -> str:
    steps = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["ingest"]["steps"]
    return next(s["run"] for s in steps if s.get("id") == "download-artifacts")


def test_download_skips_non_ingest_artifacts(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gh.log"
    # Stub gh: log the artifact id and emit a one-file zip; id 3 (coverage) would fail.
    (bin_dir / "gh").write_text(
        "#!/bin/bash\n"
        f'echo "$2" >> {log}\n'
        '[[ "$2" == */3/zip ]] && exit 1\n'
        "python3 -c \"import sys,zipfile,io; b=io.BytesIO(); z=zipfile.ZipFile(b,'w'); "
        "z.writestr('f.txt','x'); z.close(); sys.stdout.buffer.write(b.getvalue())\"\n"
    )
    (bin_dir / "gh").chmod(0o755)
    artifacts = [
        {"id": 1, "name": "junit-test-smoke-shard-1"},
        {"id": 2, "name": "run-id"},
        {"id": 3, "name": "coverage-7"},
        {"id": 4, "name": "durations-data-2"},
        {"id": 5, "name": "artifact-id"},
    ]
    script = _download_script().replace(
        "${{ steps.list-artifacts.outputs.artifacts }}", json.dumps(artifacts)
    )
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "GITHUB_REPOSITORY": "o/r"}
    subprocess.run(
        ["bash", "-e", "-c", script], env=env, check=True, capture_output=True, cwd=tmp_path
    )
    fetched = [line.split("/")[-2] for line in log.read_text().splitlines()]
    assert fetched == ["1", "2", "5"]
