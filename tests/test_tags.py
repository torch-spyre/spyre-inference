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

"""Tier tags emitted into JUnit properties, which the ingest hashes into test_case_id."""

import re

import pytest
from spyre_testing_plugin import tags


def _tags(pairs):
    assert {n for n, _ in pairs} <= {"tag"}
    return [v for _, v in pairs]


def test_declared_tiers_dedups_and_sorts(monkeypatch):
    monkeypatch.setenv("SPYRE_TEST_TIERS", "trunk unit trunk regression")
    assert tags.declared_tiers() == ["regression", "trunk", "unit"]


@pytest.mark.parametrize("raw", ["", "   "])
def test_unset_or_blank_tiers_falls_back_to_the_invoked_tier(monkeypatch, raw):
    monkeypatch.setenv("SPYRE_TEST_TIERS", raw)
    monkeypatch.setenv("SPYRE_TEST_TIER", "regression")
    assert tags.declared_tiers() == ["regression"]


def test_no_tier_anywhere_emits_no_tier_tag(monkeypatch):
    monkeypatch.delenv("SPYRE_TEST_TIERS", raising=False)
    monkeypatch.delenv("SPYRE_TEST_TIER", raising=False)
    assert tags.declared_tiers() == []
    assert _tags(tags.result_tags({})) == []


def test_every_declared_tier_becomes_a_tag(monkeypatch):
    monkeypatch.setenv("SPYRE_TEST_TIERS", "unit regression trunk")
    monkeypatch.setenv("SPYRE_TEST_TIER", "regression")
    assert _tags(tags.result_tags({})) == [
        "testtype__regression",
        "testtype__trunk",
        "testtype__unit",
    ]


def test_identity_does_not_vary_by_invoking_tier(monkeypatch):
    # Case tags are hashed into test_case_id, so the invoked tier must not reach them:
    # otherwise one test gets a different identity per tier that launched it.
    monkeypatch.setenv("SPYRE_TEST_TIERS", "unit regression trunk")
    monkeypatch.setenv("SPYRE_TEST_TIER", "regression")
    as_regression = _tags(tags.result_tags({}))
    monkeypatch.setenv("SPYRE_TEST_TIER", "unit")
    assert _tags(tags.result_tags({})) == as_regression
    assert tags.invoked_tier() == "unit"


def test_membership_is_the_declared_set_not_a_ladder_closure(monkeypatch):
    monkeypatch.setenv("SPYRE_TEST_TIERS", "unit regression trunk")
    monkeypatch.delenv("SPYRE_TEST_TIER", raising=False)
    assert "testtype__integration" not in _tags(tags.result_tags({}))


def test_model_tag_still_rides_along(monkeypatch):
    monkeypatch.setenv("SPYRE_TEST_TIERS", "unit")
    monkeypatch.delenv("SPYRE_TEST_TIER", raising=False)
    assert _tags(tags.result_tags({"model": "ibm/granite"})) == [
        "model__ibm/granite",
        "testtype__unit",
    ]


def test_tag_values_match_the_ingest_namespace_form(monkeypatch):
    # `namespace__value` is what v2_tags_for_case reads off the JUnit property.
    monkeypatch.setenv("SPYRE_TEST_TIERS", "unit trunk")
    monkeypatch.setenv("SPYRE_TEST_TIER", "unit")
    for name, value in tags.result_tags({"model": "ibm/granite"}):
        assert name == "tag"
        assert re.fullmatch(r"[a-z_]+__\S+", value), value
