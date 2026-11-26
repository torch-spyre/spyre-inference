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

"""The benchmark ingest requirements install spyre-clickhouse-ingest from a pinned rev.

An unpinned ``@main`` resolves whatever torch-spyre's main holds at install time, so a
push to that repo changes what runs in CI without any review here (#1541). The
requirements file is the one install site outside workflow/composite files; this test
keeps it pinned to a full 40-hex commit SHA.
"""

from __future__ import annotations

import pathlib
import re

_REQ = pathlib.Path(__file__).resolve().parent / "requirements-benchmark.txt"


def test_spyre_clickhouse_ingest_is_pinned_to_a_full_commit_sha():
    text = _REQ.read_text(encoding="utf-8")
    m = re.search(r"^spyre-clickhouse-ingest\s*@\s*git\+(\S+)\s*$", text, re.MULTILINE)
    assert m, "spyre-clickhouse-ingest line not found in requirements-benchmark.txt"
    source = m.group(1)
    assert "@main" not in source, (
        "spyre-clickhouse-ingest installs from torch-spyre@main; pin it to a reviewed "
        "rev instead (#1541)"
    )
    rev = source.rsplit("@", 1)[1].split("#")[0]
    assert re.fullmatch(r"[0-9a-f]{40}", rev), (
        f"spyre-clickhouse-ingest rev {rev!r} is not a full 40-hex commit SHA; a short "
        "rev or branch name still floats with the upstream repo"
    )
