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

"""CPU-only tests for the JUnit result tags (no hardware needed).

The invariant that matters: `testtype__` carries the tiers a test BELONGS to, read
from the matrix entry's declared set, and is never inferred from a tier ladder. A
ladder over-claims coverage, and over-claimed coverage silently skips real tests --
so the ladder tests below are the ones that must not regress.
"""

import re
from pathlib import Path

import yaml

# Imported as a MODULE, deliberately: `test_tier`/`test_tiers` start with `test_`, so
# importing them by name makes pytest collect the production functions themselves as
# (vacuously passing) tests.
from spyre_testing_plugin import tags

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MATRIX = _REPO_ROOT / ".github" / "workflows" / "_test_matrix.yaml"


def _tags(pairs):
    """Just the tag values; every pair's name is the literal 'tag'."""
    assert {n for n, _ in pairs} <= {"tag"}
    return [v for _, v in pairs]


# ── the declared set ────────────────────────────────────────────────────────────


def test_tiers_reads_the_declared_set(monkeypatch):
    monkeypatch.setenv("SPYRE_TEST_TIERS", "unit integration regression trunk")
    assert tags.test_tiers() == ["integration", "regression", "trunk", "unit"]


def test_tiers_dedups_and_sorts(monkeypatch):
    # Sorted and deduped so two writers agree on the same set regardless of the
    # order it was declared in -- the tags feed a content hash downstream.
    monkeypatch.setenv("SPYRE_TEST_TIERS", "trunk unit trunk  unit")
    assert tags.test_tiers() == ["trunk", "unit"]


def test_every_declared_tier_becomes_a_tag(monkeypatch):
    monkeypatch.setenv("SPYRE_TEST_TIERS", "unit integration regression trunk")
    monkeypatch.delenv("SPYRE_TEST_TIER", raising=False)
    assert _tags(tags.result_tags({})) == [
        "testtype__integration",
        "testtype__regression",
        "testtype__trunk",
        "testtype__unit",
    ]


# ── the fallback: a caller that does not pass the set still tags ────────────────


def test_falls_back_to_the_invoked_tier(monkeypatch):
    """A local `make test` and any un-updated caller keep emitting a tag."""
    monkeypatch.delenv("SPYRE_TEST_TIERS", raising=False)
    monkeypatch.setenv("SPYRE_TEST_TIER", "regression")
    assert tags.test_tiers() == ["regression"]


def test_blank_tiers_falls_back_too(monkeypatch):
    # The action exports "" when the input is unset, so whitespace must not read as
    # a one-element set containing the empty string.
    monkeypatch.setenv("SPYRE_TEST_TIERS", "   ")
    monkeypatch.setenv("SPYRE_TEST_TIER", "unit")
    assert tags.test_tiers() == ["unit"]


def test_no_tier_anywhere_emits_no_tier_tag(monkeypatch):
    monkeypatch.delenv("SPYRE_TEST_TIERS", raising=False)
    monkeypatch.delenv("SPYRE_TEST_TIER", raising=False)
    assert tags.test_tiers() == []
    assert _tags(tags.result_tags({})) == []


# ── membership vs invocation are different facts ────────────────────────────────


def test_the_invocation_is_never_a_case_tag(monkeypatch):
    """The invocation must NOT reach the tags, because case tags are hashed into
    test_case_id: tagging it would give one test a different identity per invoking
    tier. It is a property of the run (si_test_runs.test_type), not of the test."""
    monkeypatch.setenv("SPYRE_TEST_TIERS", "unit regression trunk")
    monkeypatch.setenv("SPYRE_TEST_TIER", "regression")
    values = _tags(tags.result_tags({}))
    assert not any("invoked_as" in v for v in values)
    assert values == [
        "testtype__regression",
        "testtype__trunk",
        "testtype__unit",
    ]
    # Still readable for whoever records the run row.
    assert tags.test_tier() == "regression"


def test_identity_does_not_vary_by_invoking_tier(monkeypatch):
    """The concrete regression: the same test, same declared membership, invoked as
    two different tiers, must produce the SAME tag list -- otherwise its hashed
    test_case_id splits and every trend query fragments."""
    monkeypatch.setenv("SPYRE_TEST_TIERS", "unit regression trunk")
    monkeypatch.setenv("SPYRE_TEST_TIER", "regression")
    as_regression = _tags(tags.result_tags({}))
    monkeypatch.setenv("SPYRE_TEST_TIER", "unit")
    as_unit = _tags(tags.result_tags({}))
    assert as_regression == as_unit


def test_model_tag_still_rides_along(monkeypatch):
    monkeypatch.setenv("SPYRE_TEST_TIERS", "unit")
    monkeypatch.delenv("SPYRE_TEST_TIER", raising=False)
    assert _tags(tags.result_tags({"model": "ibm/granite"})) == [
        "model__ibm/granite",
        "testtype__unit",
    ]


# ── the ladder must never be inferred ──────────────────────────────────────────

# Ordered weakest-to-strongest. Only used to PROVE the code does not apply it.
_LADDER = ("unit", "integration", "regression", "trunk")


def _declared_sets():
    """Every `test_types` set declared in the real matrix."""
    doc = yaml.safe_load(_MATRIX.read_text(encoding="utf-8"))
    jobs = doc["jobs"]
    include = jobs["test"]["strategy"]["matrix"]["include"]
    return [entry["test_types"].split() for entry in include if entry.get("test_types")]


def test_the_matrix_still_declares_membership_sets():
    """Guards the source of truth itself: if `test_types` ever disappears from the
    matrix, the tags silently degrade to the invoked tier via the fallback."""
    sets = _declared_sets()
    assert len(sets) >= 20, f"expected the full shard matrix, got {len(sets)} entries"
    assert any(len(s) > 1 for s in sets), "no multi-tier entry left to reuse"


def test_a_ladder_would_over_claim_this_repos_matrix():
    """The reason membership is read and not inferred, re-derived from the matrix.

    If this ever finds zero over-claims the ladder has become safe FOR THESE FILES,
    which is a property of the files and not a rule -- do not start inferring it.
    """
    over = []
    for declared in _declared_sets():
        idx = [_LADDER.index(t) for t in declared if t in _LADDER]
        if not idx:
            continue
        implied = set(_LADDER[min(idx) :])
        missing = implied - set(declared)
        if missing:
            over.append((sorted(declared), sorted(missing)))
    assert over, (
        "a tier ladder no longer over-claims any matrix entry; this is a property of "
        "the current files, not a licence to infer the ladder"
    )


def test_tags_are_exactly_the_declared_set_not_the_ladder_closure(monkeypatch):
    """The end-to-end assertion: a leg declaring unit/regression/trunk must NOT be
    tagged integration, even though integration sits between them in the ladder."""
    monkeypatch.setenv("SPYRE_TEST_TIERS", "unit regression trunk")
    monkeypatch.delenv("SPYRE_TEST_TIER", raising=False)
    values = _tags(tags.result_tags({}))
    assert "testtype__integration" not in values
    assert sorted(values) == [
        "testtype__regression",
        "testtype__trunk",
        "testtype__unit",
    ]


# ── the tag shape the ingest parses ────────────────────────────────────────────


def test_tag_values_match_the_ingest_namespace_form(monkeypatch):
    """`namespace__value`, which is what ingest_xml_si.v2_tags_for_case reads off
    `<property name="tag" value="..."/>`."""
    monkeypatch.setenv("SPYRE_TEST_TIERS", "unit trunk")
    monkeypatch.setenv("SPYRE_TEST_TIER", "unit")
    for name, value in tags.result_tags({"model": "ibm/granite"}):
        assert name == "tag"
        assert re.fullmatch(r"[a-z_]+__\S+", value), value
