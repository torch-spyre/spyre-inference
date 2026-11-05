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
import aggregate_junit_summary as agg  # noqa: E402

_BILLION_LAUGHS = """<?xml version="1.0"?>
<!DOCTYPE testsuites [
  <!ENTITY a "aaaaaaaaaa">
  <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
  <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">
  <!ENTITY d "&c;&c;&c;&c;&c;&c;&c;&c;&c;&c;">
]>
<testsuite name="s" time="1.0">
  <testcase classname="C" name="&d;" time="0.1"/>
</testsuite>
"""

_CLEAN = """<testsuite name="s" time="1.5">
  <testcase classname="C" name="t_ok" time="0.5"/>
  <testcase classname="C" name="t_fail" time="0.2"><failure message="boom"/></testcase>
</testsuite>
"""


def test_billion_laughs_dtd_is_refused(tmp_path):
    path = tmp_path / "junit-evil.xml"
    path.write_text(_BILLION_LAUGHS)
    assert agg.parse_file(str(path)) is None


def test_clean_junit_xml_still_parses(tmp_path):
    path = tmp_path / "junit-good.xml"
    path.write_text(_CLEAN)
    parsed = agg.parse_file(str(path))
    assert parsed is not None
    label, total_time, cases = parsed
    assert label == "good"
    assert total_time == 1.5
    names = {c.test: c.status for c in cases}
    assert names == {"C::t_ok": "pass", "C::t_fail": "fail"}
