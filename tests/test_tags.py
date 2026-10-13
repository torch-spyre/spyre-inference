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
import xml.etree.ElementTree as ET

import conftest as production_conftest
import pytest
from spyre_testing_plugin import tags

pytest_plugins = ["pytester"]


def _tags(pairs):
    assert {n for n, _ in pairs} <= {"tag"}
    return [v for _, v in pairs if not v.startswith("platform__")]


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


@pytest.mark.parametrize(
    "machine, expected",
    [("x86_64", "platform__x86_64"), ("ppc64le", "platform__ppc64le"), ("", "platform__unknown")],
)
def test_platform_tag_matches_torch_spyre_normalization(monkeypatch, machine, expected):
    monkeypatch.setattr(tags.platform, "machine", lambda: machine)
    assert tags.platform_tag() == expected


def test_platform_tag_is_emitted_even_without_tier_or_model(monkeypatch):
    monkeypatch.delenv("SPYRE_TEST_TIERS", raising=False)
    monkeypatch.delenv("SPYRE_TEST_TIER", raising=False)
    monkeypatch.setattr(tags.platform, "machine", lambda: "s390x")
    assert tags.result_tags({}) == [("tag", "platform__s390x")]


def _tags_in(junit_path):
    """{testcase name: [tag property values]} from a JUnit XML file."""
    tree = ET.parse(junit_path)
    return {
        tc.get("name"): [p.get("value") for p in tc.iter("property") if p.get("name") == "tag"]
        for tc in tree.iter("testcase")
    }


def test_collection_time_tagging_survives_skip_and_setup_error(pytester, monkeypatch):
    """Registers the REAL tests/conftest.py as a plugin (not a copy): a
    FUNCTION-scoped autouse fixture never runs for a marked skip or when an
    earlier fixture already failed, so tagging has to happen in
    pytest_collection_modifyitems, as tests/conftest.py now does, or these
    cases silently lose their tags -- and a revert back to the fixture would
    make this test fail, since it runs the actual hook, not a reimplementation.
    """
    # Pin a deterministic tag set (just platform__) regardless of the outer run's
    # own SPYRE_TEST_TIER(S), since those leak into this in-process nested run.
    monkeypatch.delenv("SPYRE_TEST_TIER", raising=False)
    monkeypatch.delenv("SPYRE_TEST_TIERS", raising=False)
    expected = [tags.platform_tag()]

    pytester.makeconftest(
        """
        import pytest

        def pytest_addoption(parser):
            parser.addoption("--boom", action="store_true", default=False)

        @pytest.fixture(scope="session", autouse=True)
        def _fail_once(request):
            if request.config.getoption("--boom"):
                raise RuntimeError("boom")
            yield
        """
    )
    pytester.makepyfile(
        """
        import pytest

        def test_passes():
            assert True

        @pytest.mark.skip(reason="marker skip")
        def test_marker_skip():
            assert True
        """
    )

    result = pytester.runpytest("--junitxml=result.xml", plugins=[production_conftest])
    result.assert_outcomes(passed=1, skipped=1)
    assert _tags_in(pytester.path / "result.xml") == {
        "test_passes": expected,
        "test_marker_skip": expected,
    }

    result = pytester.runpytest(
        "--junitxml=result_boom.xml", "--boom", plugins=[production_conftest]
    )
    result.assert_outcomes(errors=1, skipped=1)
    assert _tags_in(pytester.path / "result_boom.xml") == {
        "test_passes": expected,
        "test_marker_skip": expected,
    }
