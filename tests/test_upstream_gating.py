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

"""Upstream vLLM tests are opt-in: collected only when the -m expression names the
`upstream` marker, or --upstream is passed."""

import re
from pathlib import Path

import pytest

from spyre_testing_plugin.pytest_plugin import _markexpr_selects_upstream

REQUESTS_UPSTREAM = [
    "upstream",
    "upstream and distributed",
    "upstream or distributed",
    "upstream and model and not distributed",
    "model and upstream",
]

DOES_NOT_REQUEST_UPSTREAM = [
    "",
    "not upstream",
    "attention and not upstream",
    "distributed and not upstream",
    "not (distributed or upstream or attention)",
    # Could match upstream tests, but keeping the clone off the default path is the point.
    "attention",
    "not distributed",
]


@pytest.mark.parametrize("markexpr", REQUESTS_UPSTREAM)
def test_markexpr_requests_upstream(markexpr):
    assert _markexpr_selects_upstream(markexpr)


@pytest.mark.parametrize("markexpr", DOES_NOT_REQUEST_UPSTREAM)
def test_markexpr_does_not_request_upstream(markexpr):
    assert not _markexpr_selects_upstream(markexpr)


def test_makefile_upstream_targets_are_classified_correctly(pytestconfig):
    """The Makefile's combos all mention `upstream`, most negatively; only the
    test-upstream* targets should trigger a clone.
    """
    makefile = (Path(pytestconfig.rootpath) / "Makefile").read_text()
    overrides = dict(re.findall(r"^(test-[\w-]+):.*\n\t.*MARK_OVERRIDE='([^']*)'", makefile, re.M))
    assert len(overrides) >= 8, f"failed to parse MARK_OVERRIDEs out of the Makefile: {overrides}"

    for target, markexpr in overrides.items():
        assert _markexpr_selects_upstream(markexpr) == target.startswith("test-upstream"), (
            f"{target} ({markexpr!r}) is on the wrong side of the upstream gate"
        )


def test_plugin_is_loaded_via_addopts(pytestconfig):
    """Guards pyproject.toml's `-p` addopts entry, which is what scopes the plugin to
    this repo and nowhere else.
    """
    assert pytestconfig.pluginmanager.hasplugin("spyre_testing_plugin.pytest_plugin")
