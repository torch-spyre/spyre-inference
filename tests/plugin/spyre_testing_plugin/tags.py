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

"""Shared JUnit result-tagging helpers for the collection hooks in
tests/conftest.py (local tests) and this plugin (upstream vLLM tests).

Tags emit as JUnit `<property name="tag" value="key__value"/>`, the convention
the ClickHouse ingest reads.
"""

import os
import platform
import re

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


def invoked_tier():
    """The tier this run was invoked as, from SPYRE_TEST_TIER. Read live so tests that
    monkeypatch the env still see it."""
    return os.environ.get("SPYRE_TEST_TIER", "")


def declared_tiers():
    """Every tier this leg's tests belong to, from SPYRE_TEST_TIERS (whitespace-separated,
    declared as `test_types` per matrix entry).

    A declared SET, never inferred from a tier ladder: legs here declare `unit regression
    trunk` without `integration`, so a ladder would claim coverage that never ran.
    """
    raw = os.environ.get("SPYRE_TEST_TIERS", "")
    if not raw.strip():
        tier = invoked_tier()
        return [tier] if tier else []
    return sorted(set(raw.split()))


def platform_tag():
    """`platform__<arch>`, normalized like torch-spyre's oot_framework so arches match."""
    arch = re.sub(r"[^a-zA-Z0-9_]", "_", platform.machine() or "unknown").strip("_")
    return f"platform__{arch or 'unknown'}"


def result_tags(params):
    """JUnit (name, value) tag pairs; `platform__`/`testtype__` are run context, never hashed."""
    tags = [("tag", platform_tag())]
    model = model_from_params(params)
    if model:
        tags.append(("tag", f"model__{model}"))
    for tier in declared_tiers():
        tags.append(("tag", f"testtype__{tier}"))
    return tags
