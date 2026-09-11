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

"""Shared JUnit result-tagging helpers for both the tests/conftest.py autouse
fixture and this plugin's collection hook (which tags upstream vLLM tests,
collected outside tests/ where the fixture never binds).

Tags emit as JUnit `<property name="tag" value="key__value"/>`, the convention
the ClickHouse ingest reads.
"""

import os

# Parametrize argnames whose value names the model: a scalar id, a vLLM
# model-info object (.name), or a (model_id, ...) tuple.
MODEL_PARAM_NAMES = ("model", "model_path", "model_info", "model_ref_output")


def model_from_params(params):
    """Best-effort model id from a test's parametrization, or None."""
    for name in MODEL_PARAM_NAMES:
        if name not in params:
            continue
        value = params[name]
        if value is None:
            continue
        # (model_id, ...) tuple: the id is the first element.
        if isinstance(value, (tuple, list)) and value:
            value = value[0]
        name_attr = getattr(value, "name", None)
        return name_attr if name_attr is not None else str(value)
    return None


def test_tier():
    """How this run was INVOKED, from SPYRE_TEST_TIER (exported by CI's
    run-matrix-config; empty on a local run). Read live rather than at import so tests
    that monkeypatch the env still see it.

    One value, and NOT the same fact as test_tiers() below: a leg invoked as
    `regression` may hold tests that also belong to `unit` and `integration`.

    Deliberately NOT emitted as a case tag. The invocation is a property of the RUN,
    while case tags are hashed into test_case_id (see ingest_xml_si.v2_test_case_id) --
    tagging it would give one test three identities depending on which tier happened to
    launch it, which is the fragmentation the derived identity exists to prevent. It
    belongs on the run row instead, where si_test_runs.test_type already records it.
    """
    return os.environ.get("SPYRE_TEST_TIER", "")


def test_tiers():
    """Every tier this leg's tests BELONG to, from SPYRE_TEST_TIERS.

    Declared per matrix entry as `test_types` in _test_matrix.yaml and threaded
    through run-matrix-config; whitespace-separated, e.g.
    "unit integration regression trunk".

    This is the set that makes coverage reusable. The invocation tier alone cannot:
    a `regression` run of a leg whose tests are also `unit` and `integration` members
    would record only `regression`, so a later `integration` run finds no coverage and
    re-executes identical work.

    Membership is read from the declaration and never inferred from a tier ladder.
    Measured on this repo's 32 matrix entries: 13 declare `unit regression trunk`
    WITHOUT `integration`, so a ladder ("in unit => in everything above") would claim
    coverage for 13 legs that never ran integration and silently skip real tests.

    Falls back to the single invocation tier when unset, so a local `make test` and any
    caller not yet passing test_types keep emitting a tag rather than none.
    """
    raw = os.environ.get("SPYRE_TEST_TIERS", "")
    if not raw.strip():
        tier = test_tier()
        return [tier] if tier else []
    return sorted(set(raw.split()))


def result_tags(params):
    """The (name, value) JUnit property pairs for these params; empty when no
    model param is recognized and no tier is set, so callers append
    unconditionally.
    """
    tags = []
    model = model_from_params(params)
    if model:
        tags.append(("tag", f"model__{model}"))
    for tier in test_tiers():
        tags.append(("tag", f"testtype__{tier}"))
    return tags
