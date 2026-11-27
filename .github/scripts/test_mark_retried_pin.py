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

"""The pod-retry leg fetches mark_retried.py from a pinned rev.

The step downloads the helper straight from raw.githubusercontent.com and runs
it with python3 on every test pod, so an unpinned ``/main/`` path means any
push to torch-spyre changes what CI executes without a review here (#1541).
The ingest installs are pinned separately; this test keeps the fetched-by-path
helper pinned to a full 40-hex commit SHA.
"""

from __future__ import annotations

import pathlib
import re

import yaml

_ACTION = pathlib.Path(__file__).resolve().parents[1] / "actions/run-matrix-config/action.yml"


def test_mark_retried_url_pinned_to_full_commit_sha():
    steps = yaml.safe_load(_ACTION.read_text())["runs"]["steps"]
    mark = next(s for s in steps if "MARK_RETRIED_URL" in s.get("env", {}))
    url = mark["env"]["MARK_RETRIED_URL"]
    m = re.fullmatch(
        r"https://raw\.githubusercontent\.com/torch-spyre/torch-spyre/(\S+)"
        r"/extensions/clickhouse-ingest/spyre_clickhouse_ingest/mark_retried\.py",
        url,
    )
    assert m, f"mark_retried.py URL not found or unexpected shape: {url!r}"
    rev = m.group(1)
    assert rev != "main", (
        "the retry leg fetches mark_retried.py from torch-spyre@main; pin it to a "
        "reviewed rev instead (#1541)"
    )
    assert re.fullmatch(r"[0-9a-f]{40}", rev), (
        f"rev {rev!r} is not a full 40-hex commit SHA; a short rev or branch name "
        "still floats with the upstream repo"
    )
