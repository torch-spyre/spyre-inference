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
    """Suite tier from SPYRE_TEST_TIER (exported by CI's run-matrix-config; empty
    on a local run). Read live rather than at import so tests that monkeypatch the
    env still see it."""
    return os.environ.get("SPYRE_TEST_TIER", "")


def result_tags(params):
    """The (name, value) JUnit property pairs for these params; empty when no
    model param is recognized and no tier is set, so callers append
    unconditionally.
    """
    tags = []
    model = model_from_params(params)
    if model:
        tags.append(("tag", f"model__{model}"))
    tier = test_tier()
    if tier:
        tags.append(("tag", f"testtype__{tier}"))
    return tags
