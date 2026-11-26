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

"""The composite ingest action installs spyre-clickhouse-ingest from a pinned rev.

The action's venv step runs with the ClickHouse login in scope, so an unpinned
``@main`` means any push to torch-spyre changes what this credentialed step
executes without a review here (#1541). The two workflow-file install sites
(pre-commit.yml, push-to-clickhouse.yaml) are outside this token tier's reach;
this test keeps the composite-action site pinned to a full 40-hex commit SHA.
"""

from __future__ import annotations

import pathlib
import re

import yaml

_ACTION = (
    pathlib.Path(__file__).resolve().parents[1]
    / "actions/ingest-vllm-benchmarks/action.yml"
)


def test_action_pinned_to_full_commit_sha():
    steps = yaml.safe_load(_ACTION.read_text())["runs"]["steps"]
    venv = next(s for s in steps if "uv pip install" in s.get("run", ""))
    sources = re.findall(r"git\+https://github\.com/torch-spyre/torch-spyre@(\S+?)#subdirectory", venv["run"])
    assert sources, "torch-spyre install not found in the action's venv step"
    for rev in sources:
        assert rev != "main", (
            "the ingest action installs spyre-clickhouse-ingest from torch-spyre@main; "
            "pin it to a reviewed rev instead (#1541)"
        )
        assert re.fullmatch(r"[0-9a-f]{40}", rev), (
            f"rev {rev!r} is not a full 40-hex commit SHA; a short rev or branch name "
            "still floats with the upstream repo"
        )
