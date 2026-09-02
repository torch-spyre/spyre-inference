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

import os

import pytest

from spyre_inference import envs

# Suite tier for this run, forwarded by CI (run-matrix-config resolves it via
# `make print-test-type` and exports it here). Empty on a local run.
_TEST_TIER = os.environ.get("SPYRE_TEST_TIER", "")


@pytest.fixture(autouse=True)
def _clear_env_cache():
    """envs.py caches each SPYRE_* value on first read; drop the cache around
    every test so monkeypatched vars take effect and don't leak between tests."""
    envs.clear_env_cache()
    yield
    envs.clear_env_cache()


@pytest.fixture(autouse=True)
def _emit_result_tags(request, record_property):
    """Stamp `model__<name>` and `testtype__<tier>` onto each test as JUnit
    `<property name="tag" .../>` elements, matching the tag convention the
    ClickHouse ingest reads (a single `tag` property whose value is
    `key__value`). The model comes from the test's own parametrization; the
    tier from the CI-resolved SPYRE_TEST_TIER. Both are optional: a test with
    no model param and a local run with no tier just emit nothing."""
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    model = params.get("model") or params.get("model_path")
    if model is None and "model_info" in params:
        info = params["model_info"]
        model = getattr(info, "name", None) or str(info)
    if model:
        record_property("tag", f"model__{model}")
    if _TEST_TIER:
        record_property("tag", f"testtype__{_TEST_TIER}")
