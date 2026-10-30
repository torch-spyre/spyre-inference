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

"""Per-architecture model exclusion for upstream tests (CI is x86-only, so patch the arch)."""

from types import SimpleNamespace

import pytest
from spyre_testing_plugin import pytest_plugin


def _item(path="upstream/tests/models/test_x.py", **params):
    return SimpleNamespace(fspath=path, callspec=SimpleNamespace(params=params))


@pytest.fixture
def ppc64le(monkeypatch):
    monkeypatch.setattr(pytest_plugin.platform, "machine", lambda: "ppc64le")


def test_qwen_param_is_unsupported_on_ppc64le(ppc64le):
    assert pytest_plugin._is_unsupported_on_current_platform(_item(model="Qwen/Qwen2.5-0.5B"))


def test_qwen_in_path_is_unsupported_on_ppc64le(ppc64le):
    assert pytest_plugin._is_unsupported_on_current_platform(
        _item(path="upstream/tests/models/test_qwen.py")
    )


def test_gsm8k_qwen_config_filename_is_unsupported_on_ppc64le(ppc64le):
    item = _item(
        path="upstream/tests/evals/gsm8k/test_gsm8k_correctness.py",
        config_filename="qwen2.5-0.5b-instruct.yaml",
    )
    assert pytest_plugin._is_unsupported_on_current_platform(item)


def test_granite_param_is_supported_on_ppc64le(ppc64le):
    assert not pytest_plugin._is_unsupported_on_current_platform(
        _item(model="ibm-granite/granite-4.1-3b")
    )


def test_gsm8k_non_qwen_config_filename_is_supported_on_ppc64le(ppc64le):
    item = _item(
        path="upstream/tests/evals/gsm8k/test_gsm8k_correctness.py",
        config_filename="granite-4.1-3b.yaml",
    )
    assert not pytest_plugin._is_unsupported_on_current_platform(item)


def test_item_without_callspec_is_supported_on_ppc64le(ppc64le):
    item = SimpleNamespace(fspath="upstream/tests/models/test_x.py")
    assert not pytest_plugin._is_unsupported_on_current_platform(item)


def test_qwen_is_supported_on_x86_64(monkeypatch):
    monkeypatch.setattr(pytest_plugin.platform, "machine", lambda: "x86_64")
    assert not pytest_plugin._is_unsupported_on_current_platform(_item(model="Qwen/Qwen2.5-0.5B"))
