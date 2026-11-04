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

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import download_xml_artifacts as dxa  # noqa: E402


def test_safe_segment_accepts_plain_names():
    assert dxa.safe_segment("junit-shard-1")
    assert dxa.safe_segment("run-id")
    assert dxa.safe_segment("artifact-id")


def test_safe_segment_rejects_traversal_and_separators():
    for bad in (
        "",
        ".",
        "..",
        "junit-../../etc",
        "junit-/abs",
        "junit-sub/dir",
        "junit-back\\slash",
        ".hidden",
    ):
        assert not dxa.safe_segment(bad), bad


def test_wanted_keeps_prefix_filter_but_rejects_traversing_prefix():
    assert dxa.wanted({"name": "junit-shard-1", "id": 1})
    assert dxa.wanted({"name": "run-id", "id": 2})
    assert not dxa.wanted({"name": "junit-../../../../tmp/evil", "id": 3})
    assert not dxa.wanted({"name": "not-junit", "id": 4})


def test_wanted_destination_stays_within_out_dir(tmp_path):
    out = tmp_path / "xml_artifacts"
    out_root = out.resolve()
    for name in ("junit-shard-1", "run-id", "artifact-id"):
        assert dxa.wanted({"name": name, "id": 0})
        assert (out / name).resolve().parent == out_root
