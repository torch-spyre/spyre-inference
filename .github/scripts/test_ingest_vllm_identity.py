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

"""The v2 identities this writer derives must equal the ones every other writer derives.

These ids are DERIVED, not threaded: a GHA perf leg reads its own RPM lockfile and computes
the artifact_id the orchestrator would have written, with nothing passed between them. That
only works while all writers agree byte-for-byte, so the golden values below are pinned as
CONSTANTS rather than recomputed by the same code under test -- recomputing would pass even
if every writer drifted together.

The cross-writer contract these must match:
  * torch-spyre  .github/scripts/ingest_xml.py  artifact_id_for / run_id_of
  * frameworks   vars/pushToClickhouse.groovy   deriveArtifactIds / deriveRunIds
  * frameworks   pipelines/lib/run_identity.py

An earlier version of this file's canonical_arch folded only ('amd64', 'x86') and did not
lowercase, so 'AMD64' and 'x86-64' hashed differently here than in the other writers -- the
same run landing twice, joinable to neither. test_arch_aliases_all_fold pins that.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import uuid

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parent / "ingest_vllm_benchmarks.py"

# uuid5(NAMESPACE_DNS, "clickhouse-v2.spyre.ibm.com") -- the shared v2 namespace.
_NS = "cb0af9bf-2858-5eab-9211-f51190531bf3"

# Golden values, computed once from the agreed formula and pinned. Any change here is a
# schema-wide breaking change, not a test fix.
# gha|12345|amd64|integration
_GOLDEN_RUN_ID = "dab2a67f-14bf-53be-b6e4-fc9642086e47"
# torch-spyre|flex-rpm|abc123def456|x86_64
_GOLDEN_ARTIFACT_ID = "86a5c6e3-bd2f-5d27-9a8f-9b8d23efc65b"
# flex|ibm-flex|abc123def456|x86_64
_GOLDEN_FLEX_ID = "937c72dc-85e1-5c35-9093-4a18cac7cda3"


@pytest.fixture(scope="module")
def mod():
    """Load the ingest script by path; stub `utils`, which pulls in the repo's test deps."""
    stub = types.ModuleType("utils")
    stub.read_benchmark_results = lambda *a, **k: []
    sys.modules.setdefault("utils", stub)
    spec = importlib.util.spec_from_file_location("ingest_vllm_benchmarks", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except ModuleNotFoundError as exc:  # clickhouse_connect absent
        pytest.skip(f"ingest deps unavailable: {exc}")
    return m


# ── the shared namespace ────────────────────────────────────────────────────────


def test_namespace_is_the_shared_v2_namespace():
    """Hardcoded as a literal here for speed; it must still equal the derived value, or every
    id in this writer lands in a different space from every other writer's. Asserted on the
    library: this writer reaches the namespace only through the shared id functions."""
    import spyre_clickhouse_ingest as lib

    assert str(lib.ID_NAMESPACE) == _NS
    assert uuid.uuid5(uuid.NAMESPACE_DNS, "clickhouse-v2.spyre.ibm.com") == lib.ID_NAMESPACE


# ── run_id ─────────────────────────────────────────────────────────────────────


def test_run_id_matches_the_golden_value(mod):
    assert mod.run_id_of("gha", "12345", "amd64", "integration") == _GOLDEN_RUN_ID


def test_run_id_is_blank_on_incomplete_input(mod):
    """Blank, not a hash of defaults: an all-defaults key is a real uuid that every
    incomplete run would share, which is worse than no id."""
    assert mod.run_id_of("", "12345", "amd64", "integration") == ""
    assert mod.run_id_of("gha", "", "amd64", "integration") == ""
    assert mod.run_id_of("gha", "12345", "", "integration") == ""
    assert mod.run_id_of("gha", "12345", "amd64", "") == ""


# ── arch folding, the regression this file exists for ──────────────────────────


@pytest.mark.parametrize("alias", ["amd64", "x86", "x86-64", "x86_64", "AMD64", " amd64 "])
def test_arch_aliases_all_fold(mod, alias):
    assert mod.canonical_arch(alias) == "x86_64"


@pytest.mark.parametrize(
    ("raw", "expected"), [("S390X", "s390x"), ("ppc64le", "ppc64le"), ("PPC64LE", "ppc64le")]
)
def test_non_x86_arch_is_lowercased_not_folded(mod, raw, expected):
    assert mod.canonical_arch(raw) == expected


def test_arch_spelling_does_not_change_the_run_id(mod):
    """The point of folding INSIDE the hash: one machine, one id, however it is spelled."""
    ids = {
        mod.run_id_of("gha", "12345", a, "integration")
        for a in ("amd64", "x86_64", "AMD64", "x86-64")
    }
    assert ids == {_GOLDEN_RUN_ID}


# ── artifact_id ────────────────────────────────────────────────────────────────


def test_artifact_id_matches_the_golden_value(mod):
    assert (
        mod.artifact_id_for("torch-spyre", "flex-rpm", "abc123def456", "amd64")
        == _GOLDEN_ARTIFACT_ID
    )


def test_artifact_id_is_case_and_space_insensitive(mod):
    assert (
        mod.artifact_id_for("Torch-Spyre", " FLEX-RPM ", "ABC123DEF456", "AMD64")
        == _GOLDEN_ARTIFACT_ID
    )


def test_artifact_id_refuses_a_blank_component_or_arch(mod):
    assert mod.artifact_id_for("", "flex-rpm", "abc123def456", "amd64") == ""
    assert mod.artifact_id_for("flex", "flex-rpm", "abc123def456", "") == ""


def test_artifact_id_allows_a_blank_id12(mod):
    """A GHA-derived identity carries a base-image + installed-set hash in that slot, and
    some legs have neither -- blank is a legitimate value, unlike a blank component."""
    assert mod.artifact_id_for("flex", "ibm-flex", "", "amd64") != ""


def test_content_changes_the_identity(mod):
    a = mod.artifact_id_for("flex", "ibm-flex", "abc123def456", "amd64")
    for other in (
        mod.artifact_id_for("flex", "ibm-flex", "999999999999", "amd64"),
        mod.artifact_id_for("flex", "ibm-flex", "abc123def456", "s390x"),
        mod.artifact_id_for("deeptools", "ibm-flex", "abc123def456", "amd64"),
    ):
        assert other != a


# ── the lockfile -> artifact_id path ──────────────────────────────────────────


def _lock(tmp_path, *lines):
    p = tmp_path / "spyre-rpms.lock"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def test_lockfile_yields_the_golden_flex_id(mod, tmp_path):
    """The whole design in one assertion: a leg that only read its lockfile derives the id
    the orchestrator wrote, so the perf numbers reach the artifact page."""
    path = _lock(tmp_path, "ibm-flex-1.2.3-0.next.abc123def456.el10.x86_64.rpm")
    assert mod.rpm_artifact_ids(path, "amd64") == [_GOLDEN_FLEX_ID]


def test_devel_subpackage_collapses_onto_the_base_artifact(mod, tmp_path):
    """-devel/-headers ship from ONE build and share its id12; the builder registers only the
    base name, so a per-subpackage id would name a row that cannot exist."""
    path = _lock(
        tmp_path,
        "ibm-flex-1.2.3-0.next.abc123def456.el10.x86_64.rpm",
        "ibm-flex-devel-1.2.3-0.next.abc123def456.el10.x86_64.rpm",
        "ibm-flex-headers-1.2.3-0.next.abc123def456.el10.x86_64.rpm",
    )
    assert mod.rpm_artifact_ids(path, "amd64") == [_GOLDEN_FLEX_ID]


def test_every_id_is_a_uuid_not_a_delimited_string(mod, tmp_path):
    """artifact_results.artifact_id is UUID in v2; a `component|name|id12|arch` string would
    be rejected at insert."""
    path = _lock(
        tmp_path,
        "ibm-flex-1.2.3-0.next.abc123def456.el10.x86_64.rpm",
        "ibm-deeptools-2.0.0-0.next.fedcba987654.el10.x86_64.rpm",
    )
    ids = mod.rpm_artifact_ids(path, "amd64")
    assert len(ids) == 2
    for got in ids:
        uuid.UUID(got)  # raises if not a uuid
        assert "|" not in got


# ── which flag carries which id ────────────────────────────────────────────────
#
# --run-id is THIS RUN'S IDENTITY (a uuid on the Jenkins path, used verbatim); --gha-run-id is
# GitHub's NUMERIC run id, which becomes upstream's workflow_id and derives a v2 run_id when no
# uuid was supplied. Named that way so spyre-frameworks' ingest_cmd -- which already passes
# ${RUN_ID}, a uuid, as --run-id -- is correct without knowing this script's column layout.


def _args(mod, **kw):
    import argparse

    base = dict(run_id="", gha_run_id="", v2_run_id="", arch="amd64", test_type="perf", rpm_lock="")
    base.update(kw)
    return argparse.Namespace(**base)


def test_uuid_in_run_id_is_used_verbatim(mod):
    """The cross-repo caller's shape. Re-hashing an already-hashed id would mint a third
    identity that joins to nothing -- the v1 failure this whole scheme exists to avoid."""
    u = "dab2a67f-14bf-53be-b6e4-fc9642086e47"
    assert mod.resolve_v2_run_id(_args(mod, run_id=u)) == u


def test_numeric_gha_run_id_derives(mod):
    """The in-repo GHA shape: no uuid to hand over, so the id is derived from the run id."""
    got = mod.resolve_v2_run_id(_args(mod, gha_run_id="34958223121"))
    assert got == mod.run_id_of("gha", "34958223121", "amd64", "perf")


def test_uuid_wins_over_a_numeric_id(mod):
    """Both present (a Jenkins-dispatched leg that also has a GHA run): the uuid is the
    orchestrator's own identity, so it must win -- deriving instead would fork the join."""
    u = "dab2a67f-14bf-53be-b6e4-fc9642086e47"
    assert mod.resolve_v2_run_id(_args(mod, run_id=u, gha_run_id="34958223121")) == u


def test_a_numeric_run_id_still_derives(mod):
    """Back-compat: the OLD wiring put GitHub's numeric id in --run-id. Fall through to the
    derive path rather than rejecting it -- that is what such a caller meant."""
    got = mod.resolve_v2_run_id(_args(mod, run_id="34958223121"))
    assert got == mod.run_id_of("gha", "34958223121", "amd64", "perf")


def test_legacy_v2_run_id_still_honoured(mod):
    u = "dab2a67f-14bf-53be-b6e4-fc9642086e47"
    assert mod.resolve_v2_run_id(_args(mod, v2_run_id=u)) == u


def test_no_id_at_all_skips_the_v2_write(mod):
    """'' rather than a minted id: an unjoinable row is indistinguishable downstream from
    "no perf ran"."""
    assert mod.resolve_v2_run_id(_args(mod)) == ""


# ── the RPM->artifact link is the DERIVED path's alone ────────────────────────
#
# Jenkins passes --v2-run-id verbatim and the orchestrator has already written an
# artifact_results row for that run_id, so linking again would add one row per pinned RPM on
# top of it. The dedup guard inside the writer cannot catch that -- it only skips when a row
# is ALREADY present, so whichever writer lands first wins and the other duplicates.


def test_workflow_id_comes_from_the_numeric_flag(mod):
    """upstream's workflow_id is Int64 and built as `int(x) if x.isdigit() else 0`, so it must be
    sourced from --gha-run-id. Taking it from --run-id would yield 0 on the Jenkins path, which
    blanks run_url, collapses run_metadata's dedup key and hands the HUD a non-existent run.

    Asserted against main()'s own wiring rather than a copy of it: the source line is read out of
    the file, so re-pointing it at args.run_id fails here.
    """
    src = _SCRIPT.read_text(encoding="utf-8")
    body = src[src.index("def main(") :]
    assert "run_id=args.gha_run_id," in body, (
        "extract_rows must take the NUMERIC id for workflow_id"
    )
    assert "run_id=args.run_id," not in body, (
        "workflow_id must not be sourced from --run-id, which may hold a uuid"
    )


def _lock_for(mod, **kw):
    """Call the REAL gate (effective_rpm_lock), never a local reimplementation of it.

    An earlier version of these tests re-derived the rule inline, so mutating the production
    logic left them green -- they were testing the test.
    """
    return mod.effective_rpm_lock(_args(mod, **kw))


def test_jenkins_uuid_in_run_id_does_not_link_rpms(mod):
    """The cross-repo caller's shape: Jenkins owns the artifact_results row for this run."""
    assert (
        _lock_for(mod, run_id="dab2a67f-14bf-53be-b6e4-fc9642086e47", rpm_lock="spyre-rpms.lock")
        == ""
    )


def test_jenkins_uuid_in_legacy_flag_does_not_link_rpms(mod):
    assert (
        _lock_for(mod, v2_run_id="dab2a67f-14bf-53be-b6e4-fc9642086e47", rpm_lock="spyre-rpms.lock")
        == ""
    )


def test_gha_path_links_rpms(mod):
    assert _lock_for(mod, gha_run_id="34958223121", rpm_lock="spyre-rpms.lock") == "spyre-rpms.lock"


def test_a_numeric_run_id_still_links_rpms(mod):
    """The old wiring put GitHub's numeric id in --run-id. That is a GHA leg, which DOES own
    the link -- keying the gate on "any value present" would wrongly skip it."""
    assert _lock_for(mod, run_id="34958223121", rpm_lock="spyre-rpms.lock") == "spyre-rpms.lock"


def test_gha_path_honours_an_explicit_empty_lock(mod):
    """Empty stays the documented opt-out on the path that does own the link."""
    assert _lock_for(mod, gha_run_id="34958223121", rpm_lock="") == ""


def test_unknown_arch_yields_nothing(mod, tmp_path):
    path = _lock(tmp_path, "ibm-flex-1.2.3-0.next.abc123def456.el10.x86_64.rpm")
    assert mod.rpm_artifact_ids(path, "") == []


def test_identity_comes_from_the_shared_library_not_a_local_copy(mod):
    # The golden constants above pin the VALUES; this pins the SOURCE. A local redefinition
    # that agrees today satisfies every assertion in this file while drifting later, so assert
    # object identity: editing the library must change what this writer executes.
    import spyre_clickhouse_ingest as lib

    for name in ("canonical_arch", "run_id_of", "artifact_id_for"):
        assert getattr(mod, name) is getattr(lib, name), name
