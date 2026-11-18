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

"""Credential-transport hardening for download_rpms.sh.

The Artifactory bearer token must never leave on anything but https: a
non-https ARTIFACTORY_LOCATION is refused before the token is sent, and the
curl invocation pins every redirect hop to https so a server-returned
redirect to http:// cannot carry the Authorization header in cleartext.
"""

from __future__ import annotations

import pathlib
import subprocess
import textwrap

SCRIPT = pathlib.Path(__file__).resolve().parent / "download_rpms.sh"


def _run(env_extra: dict[str, str], tmp_path: pathlib.Path) -> subprocess.CompletedProcess:
    import os

    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ARTIFACTORY_TOKEN": "s3cr3t",
        "RPM_NAMES": "some-rpm",
        "RPMS_DOWNLOAD_DIR": str(tmp_path / "rpms"),
        **env_extra,
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )


def test_refuses_non_https_location(tmp_path):
    result = _run({"ARTIFACTORY_LOCATION": "http://artifactory.example/repo"}, tmp_path)
    assert result.returncode != 0
    assert "https" in result.stderr
    # The token must not reach curl: nothing was downloaded.
    assert not (tmp_path / "rpms" / "some-rpm.x86_64.rpm").exists()


def test_refuses_non_https_base_url(tmp_path):
    result = _run(
        {
            "ARTIFACTORY_BASE_URL": "http://artifactory.example",
            "ARTIFACTORY_RPM_PATH": "repo",
        },
        tmp_path,
    )
    assert result.returncode != 0
    assert "https" in result.stderr


def test_curl_argv_pins_https(tmp_path, monkeypatch):
    """A stub curl records its argv; the proto pins must be present.

    The script is run with a PATH whose `curl` is a stub that fails
    immediately, so the loop's first invocation exits before any network
    traffic. The stub writes its argv to a file for the assertion.
    """
    stub = tmp_path / "bin"
    stub.mkdir()
    argv_file = tmp_path / "argv"
    (stub / "curl").write_text(
        textwrap.dedent(
            f"""
            #!/usr/bin/env bash
            printf '%s\\n' "$@" > {argv_file}
            exit 1
            """
        )
    )
    (stub / "curl").chmod(0o755)
    import os

    monkeypatch.setenv("PATH", f"{stub}:{os.environ['PATH']}")

    result = _run({"ARTIFACTORY_LOCATION": "https://artifactory.example/repo"}, tmp_path)
    assert result.returncode != 0
    argv = (tmp_path / "argv").read_text().splitlines()
    assert "--proto" in argv and argv[argv.index("--proto") + 1] == "=https"
    assert "--proto-redir" in argv
    assert argv[argv.index("--proto-redir") + 1] == "=https"
