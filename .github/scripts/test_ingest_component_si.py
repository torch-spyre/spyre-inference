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

"""The component this ingest stamps on v2 rows.

`component` is a test_case_id hash input, so getting it wrong mints a DIFFERENT identity
rather than mislabelling a row -- which is why the default is asserted explicitly here and
not left to the shared library's own default.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import types

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parent / "ingest_xml_si.py"


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.fixture(scope="module")
def mod():
    sys.modules.setdefault("clickhouse_connect", types.ModuleType("clickhouse_connect"))
    spec = importlib.util.spec_from_file_location("ingest_xml_si", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except ModuleNotFoundError as exc:
        pytest.skip(f"ingest deps unavailable: {exc}")
    return m


def test_default_is_this_repo_not_the_library_default(mod):
    from spyre_clickhouse_ingest import v2_component

    assert mod.V2_COMPONENT_DEFAULT == "spyre-inference"
    # v2_component takes the default as a PARAMETER precisely so each repo stamps itself; a
    # bare call would silently yield the library's own default instead.
    assert v2_component(_Args(component=""), mod.V2_COMPONENT_DEFAULT) == "spyre-inference"
    assert v2_component(_Args(component="")) != "spyre-inference"


@pytest.mark.parametrize("args", [_Args(), _Args(component=""), _Args(component="   ")])
def test_absent_or_blank_flag_keeps_the_default(mod, args):
    from spyre_clickhouse_ingest import v2_component

    assert v2_component(args, mod.V2_COMPONENT_DEFAULT) == "spyre-inference"


def test_flag_overrides_and_that_changes_identity(mod):
    from spyre_clickhouse_ingest import v2_component, v2_test_case_id

    assert v2_component(_Args(component="torch-spyre"), mod.V2_COMPONENT_DEFAULT) == "torch-spyre"
    # The override is only useful because it re-owns the identity: a cell running another
    # component's suite must not hash its cases under spyre-inference.
    assert v2_test_case_id("torch-spyre", "T", "t", []) != v2_test_case_id(
        "spyre-inference", "T", "t", []
    )


def test_component_flag_is_declared():
    # pushToClickhouse probes `--help` for --component and only passes it when present, so an
    # undeclared flag degrades silently to the default rather than failing. Probe the same way.
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    assert "--component" in out.stdout


def test_identity_goldens():
    """Both halves of the identity contract, pinned locally.

    The library is installed from torch-spyre@main, so these are the in-repo tripwire for a
    change there re-minting ids: a drift orphans rows, which reads downstream as "no tests ran".
    """
    from spyre_clickhouse_ingest import v2_run_id, v2_test_case_id

    assert (
        str(v2_run_id("gha", "123", "x86_64", "regression"))
        == "1a6080e8-d061-547f-ab63-1af99b18ad0c"
    )
    assert (
        str(
            v2_test_case_id(
                "torch-spyre",
                "test_ops",
                "test_add",
                ["testtype__trunk", "platform__x86_64"],
            )
        )
        == "2f0e2626-3b33-56c7-9019-fb261450c7aa"
    )


def test_arch_is_folded_inside_the_hash():
    """amd64/x86/x86-64 must reach the same id as x86_64, or a leg labelled either way
    joins to nothing."""
    from spyre_clickhouse_ingest import v2_run_id

    canonical = v2_run_id("gha", "123", "x86_64", "regression")
    for alias in ("amd64", "x86", "x86-64", "X86_64", " x86_64 "):
        assert v2_run_id("gha", "123", alias, "regression") == canonical, alias
