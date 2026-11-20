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

"""Credential-transport guards for .github/cache_config/manage_cache.py.

manage_cache.py is not on pytest's testpaths; the ingest-script-tests CI job
collects .github/scripts, so this test lives here and loads the script by path,
mirroring test_resolve_rpms.py. The two guards assert that the Artifactory
bearer token is never attached to a cleartext request and never replayed to a
host the origin redirects to.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_MC_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "cache_config"
    / "manage_cache.py"
)
_spec = importlib.util.spec_from_file_location("manage_cache", _MC_PATH)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)


def test_non_https_url_refused_before_token_sent():
    # artifactory_open must raise before it ever opens a socket, so a cleartext
    # base URL cannot leak the bearer token on the wire.
    with pytest.raises(ValueError):
        mc.artifactory_open("http://artifactory.example/foo.tar", "s3cr3t", 1)


def test_cross_host_redirect_strips_authorization():
    import urllib.request

    handler = mc._StripAuthOnCrossHostRedirect()
    req = urllib.request.Request(
        "https://artifactory.example/a",
        headers={"Authorization": "Bearer s3cr3t"},
    )
    # A redirect to a different host must drop Authorization.
    new = handler.redirect_request(
        req, fp=None, code=302, msg="Found", headers={},
        newurl="https://evil.example/a",
    )
    assert new is not None
    assert new.get_header("Authorization") is None


def test_same_host_redirect_keeps_authorization():
    import urllib.request

    handler = mc._StripAuthOnCrossHostRedirect()
    req = urllib.request.Request(
        "https://artifactory.example/a",
        headers={"Authorization": "Bearer s3cr3t"},
    )
    new = handler.redirect_request(
        req, fp=None, code=302, msg="Found", headers={},
        newurl="https://artifactory.example/b",
    )
    assert new is not None
    assert new.get_header("Authorization") == "Bearer s3cr3t"
