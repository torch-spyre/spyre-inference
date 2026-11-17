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

"""Credential-transport hardening for fetch_bench_datasets.download().

The Artifactory bearer token must never leave on anything but https: a non-https
base URL is refused before the token is sent, and the curl invocation pins every
redirect hop to https so a server-returned redirect to http:// cannot carry the
Authorization header in cleartext.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fetch_bench_datasets as fbd  # noqa: E402


def test_download_refuses_non_https_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACTORY_BASE_URL", "http://artifactory.example")
    monkeypatch.setenv("ARTIFACTORY_DATA_PATH", "data")
    monkeypatch.setenv("ARTIFACTORY_TOKEN", "s3cr3t")

    # The token must not reach curl at all when the scheme is not https.
    def _fail(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("curl was invoked with a non-https base URL")

    monkeypatch.setattr(fbd.subprocess, "run", _fail)

    with pytest.raises(fbd.FetchUnavailable, match="https"):
        fbd.download("some/file.bin", tmp_path / "file.bin", "deadbeef")


def test_download_pins_curl_to_https_only(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACTORY_BASE_URL", "https://artifactory.example")
    monkeypatch.setenv("ARTIFACTORY_DATA_PATH", "data")
    monkeypatch.setenv("ARTIFACTORY_TOKEN", "s3cr3t")

    seen = {}

    class _Result:
        returncode = 1  # fail fast so download() stops after the curl call

    def _capture(argv, *args, **kwargs):
        seen["argv"] = argv
        return _Result()

    monkeypatch.setattr(fbd.subprocess, "run", _capture)

    with pytest.raises(fbd.FetchUnavailable):
        fbd.download("some/file.bin", tmp_path / "file.bin", "deadbeef")

    argv = seen["argv"]
    assert "--proto" in argv and argv[argv.index("--proto") + 1] == "=https"
    assert "--proto-redir" in argv and argv[argv.index("--proto-redir") + 1] == "=https"
