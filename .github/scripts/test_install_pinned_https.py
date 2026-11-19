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

"""Credential-transport hardening for scripts/install-pinned-rpms.sh.

The Artifactory bearer token must never leave on anything but https, and must
never appear in a process's argv (world-readable via /proc/<pid>/cmdline on a
shared host): a non-https base URL is refused before any download, and the
curl invocation pins every redirect hop to https while taking the
Authorization header on stdin.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import textwrap

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "install-pinned-rpms.sh"


def _run(env_extra: dict[str, str], tmp_path: pathlib.Path) -> subprocess.CompletedProcess:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ARTIFACTORY_TOKEN": "s3cr3t",
        "ARTIFACTORY_BASE_URL": "https://artifactory.example",
        "ARTIFACTORY_RPM_PATH": "repo",
        "SPYRE_LIBS": str(tmp_path / "spyre-libs"),
        "RPM_CACHE_DIR": str(tmp_path / "cache"),
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


def _stub_bin(tmp_path: pathlib.Path, name: str, body: str) -> pathlib.Path:
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    exe = stub / name
    exe.write_text(textwrap.dedent(body))
    exe.chmod(0o755)
    return stub


def _stub_python_and_curl(tmp_path: pathlib.Path) -> tuple[pathlib.Path, dict[str, str]]:
    """A python3 stub that resolves one RPM offline, and a curl stub that
    records argv+stdin then fails, so the run stops at the first download."""
    argv_file = tmp_path / "curl-argv"
    stdin_file = tmp_path / "curl-stdin"
    _stub_bin(
        tmp_path,
        "python3",
        f"""
        #!/usr/bin/env bash
        if [[ "$1" == *resolve_rpms.py ]]; then
            echo "x86_64/some-rpm-1.x86_64.rpm\t{"a" * 40}"
            exit 0
        fi
        exec /usr/bin/python3 "$@"
        """,
    )
    _stub_bin(
        tmp_path,
        "curl",
        f"""
        #!/usr/bin/env bash
        printf '%s\\n' "$@" > {argv_file}
        cat > {stdin_file}
        exit 1
        """,
    )
    env = {
        "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
    }
    return argv_file, {**env}


def test_refuses_non_https_base_url(tmp_path):
    argv_file, stub_env = _stub_python_and_curl(tmp_path)
    result = _run(
        {**stub_env, "ARTIFACTORY_BASE_URL": "http://artifactory.example"},
        tmp_path,
    )
    assert result.returncode != 0
    assert "https" in result.stderr
    # The token never reached curl: no invocation was recorded.
    assert not argv_file.exists()


def test_refuses_non_https_url_env(tmp_path):
    argv_file, stub_env = _stub_python_and_curl(tmp_path)
    result = _run(
        {
            **stub_env,
            "ARTIFACTORY_BASE_URL": "",
            "ARTIFACTORY_URL": "http://artifactory.example",
        },
        tmp_path,
    )
    assert result.returncode != 0
    assert "https" in result.stderr
    assert not argv_file.exists()


def test_curl_takes_token_on_stdin_not_argv(tmp_path):
    (
        argv_file,
        stdin_file,
    ) = (
        tmp_path / "curl-argv",
        tmp_path / "curl-stdin",
    )
    _, stub_env = _stub_python_and_curl(tmp_path)
    result = _run(stub_env, tmp_path)
    assert result.returncode != 0
    argv = (argv_file).read_text().splitlines()
    # No Authorization header on the command line.
    joined = " ".join(argv)
    assert "Authorization" not in joined
    assert "s3cr3t" not in joined
    # The header is fed via a stdin config file instead.
    stdin = (stdin_file).read_text()
    assert 'header = "Authorization: Bearer s3cr3t"' in stdin


def test_curl_argv_pins_https(tmp_path):
    argv_file = tmp_path / "curl-argv"
    _, stub_env = _stub_python_and_curl(tmp_path)
    result = _run(stub_env, tmp_path)
    assert result.returncode != 0
    argv = (argv_file).read_text().splitlines()
    assert "--proto" in argv and argv[argv.index("--proto") + 1] == "=https"
    assert "--proto-redir" in argv and argv[argv.index("--proto-redir") + 1] == "=https"
