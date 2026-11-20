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

import importlib.util
import pathlib
import sys
import types

import pytest
from lxml import etree

_SCRIPTS = pathlib.Path(__file__).resolve().parent


def _load(name, **stubs):
    for mod_name, mod in stubs.items():
        sys.modules.setdefault(mod_name, mod)
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except ModuleNotFoundError as exc:
        pytest.skip(f"ingest deps unavailable: {exc}")
    return m


@pytest.fixture(scope="module")
def xml_mod():
    stub = types.ModuleType("spyre_clickhouse_ingest")
    stub.extract_properties = lambda tc: {}
    stub.get_client = lambda *a, **k: None
    stub.promote_xpass = lambda *a, **k: None
    return _load("ingest_xml_si", spyre_clickhouse_ingest=stub)


def _write(tmp_path, body):
    xml = tmp_path / "junit.xml"
    xml.write_text(body)
    return xml


def test_billion_laughs_dtd_is_refused(xml_mod, tmp_path):
    xml = _write(
        tmp_path,
        """<?xml version="1.0"?>
<!DOCTYPE testsuites [
  <!ENTITY a "aaaaaaaaaa">
  <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
  <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">
]>
<testsuites>
  <testsuite name="s" tests="1" timestamp="2026-04-25T14:20:45Z">
    <testcase classname="c" name="&c;" time="0"/>
  </testsuite>
</testsuites>
""",
    )
    with pytest.raises(ValueError):
        xml_mod.parse_test_xml(xml)


def test_external_entity_dtd_is_refused(xml_mod, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET")
    xml = _write(
        tmp_path,
        f"""<?xml version="1.0"?>
<!DOCTYPE testsuites [
  <!ENTITY xxe SYSTEM "file://{secret}">
]>
<testsuites>
  <testsuite name="s" tests="1" timestamp="2026-04-25T14:20:45Z">
    <testcase classname="c" name="&xxe;" time="0"/>
  </testsuite>
</testsuites>
""",
    )
    with pytest.raises((ValueError, etree.LxmlError)):
        xml_mod.parse_test_xml(xml)


def test_clean_junit_xml_still_parses(xml_mod, tmp_path):
    xml = _write(
        tmp_path,
        """<?xml version="1.0"?>
<testsuites>
  <testsuite name="s" tests="2" timestamp="2026-04-25T14:20:45Z">
    <testcase classname="pkg.mod" name="test_ok" time="0.5"/>
    <testcase classname="pkg.mod" name="test_bad" time="0.1">
      <failure message="boom">trace</failure>
    </testcase>
  </testsuite>
</testsuites>
""",
    )
    run, cases = xml_mod.parse_test_xml(xml)
    assert run is not None
    names = {c["name"] for c in cases}
    assert {"test_ok", "test_bad"} <= names
    statuses = {c["name"]: c["status"] for c in cases}
    assert statuses["test_ok"] == "passed"
    assert statuses["test_bad"] == "failed"
