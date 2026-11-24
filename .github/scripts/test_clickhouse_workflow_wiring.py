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

"""The v2 ClickHouse connection expressions and the no-login gate, as the workflows spell them.

GitHub's `&&` / `||` return an operand like Python's `and` / `or`, so each expression is
evaluated by translating it to Python.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

import pytest
import yaml

_WF = pathlib.Path(__file__).resolve().parents[1] / "workflows"
_ACTION = pathlib.Path(__file__).resolve().parents[1] / "actions/ingest-vllm-benchmarks/action.yml"


def _expr(text: str, secrets: dict, inputs: dict | None = None) -> str:
    m = re.fullmatch(r"\s*\$\{\{(.*)\}\}\s*", text, re.DOTALL)
    assert m, text
    py = m.group(1).replace("&&", " and ").replace("||", " or ")
    py = re.sub(r"!(?!=)", " not ", py).replace("!=", "!=")
    py = re.sub(r"\b(secrets|inputs)\.(\w+)", r"\1['\2']", py)
    ns = {"secrets": _Ctx(secrets), "inputs": _Ctx(inputs or {})}
    return eval(py, {}, ns)  # noqa: S307 - our own workflow text


class _Ctx(dict):
    def __missing__(self, key):
        return ""


def _dispatch_env() -> dict:
    jobs = yaml.safe_load((_WF / "vllm-benchmark.yml").read_text())["jobs"]
    for job in jobs.values():
        for step in job["steps"]:
            if step.get("name") == "Ingest benchmark results into ClickHouse":
                return step["env"]
    raise AssertionError("ingest step not found")


def _job_env(name: str, job: str) -> dict:
    return yaml.safe_load((_WF / name).read_text())["jobs"][job]["env"]


_SECRET_CASES = [
    ({"CLICKHOUSE_HOST": "legacy", "CLICKHOUSE_PORT": "8123"}, ("legacy", "8123")),
    (
        {"CLICKHOUSE_HOST": "legacy", "CLICKHOUSE_PORT": "8123", "CLICKHOUSE_V2_HOST": "ha"},
        ("ha", ""),
    ),
    (
        {
            "CLICKHOUSE_HOST": "legacy",
            "CLICKHOUSE_PORT": "8123",
            "CLICKHOUSE_V2_HOST": "ha",
            "CLICKHOUSE_V2_PORT": "443",
        },
        ("ha", "443"),
    ),
    ({}, ("", "")),
]


@pytest.mark.parametrize(("secrets", "want"), _SECRET_CASES)
@pytest.mark.parametrize(
    ("workflow", "job"),
    [("push-to-clickhouse.yaml", "ingest"), ("push-hw-diagnostics-to-clickhouse.yaml", "ingest")],
)
def test_host_and_port_follow_the_v2_host(workflow, job, secrets, want):
    env = _job_env(workflow, job)
    assert (_expr(env["CH_HOST"], secrets), _expr(env["CH_PORT"], secrets)) == want


@pytest.mark.parametrize(("secrets", "want"), _SECRET_CASES)
def test_dispatch_prod_host_and_port_follow_the_v2_host(secrets, want):
    env = _dispatch_env()
    prod = {"clickhouse_env": "prod"}
    assert (
        _expr(env["CLICKHOUSE_HOST"], secrets, prod),
        _expr(env["CLICKHOUSE_PORT"], secrets, prod),
    ) == want


def test_dispatch_dev_never_falls_through_to_prod():
    env = _dispatch_env()
    dev = {"clickhouse_env": "dev"}
    prod_only = {
        "CLICKHOUSE_HOST": "p",
        "CLICKHOUSE_PORT": "1",
        "CLICKHOUSE_V2_USER": "u",
        "CLICKHOUSE_V2_PASSWORD": "pw",
    }
    for key in (
        "CLICKHOUSE_HOST",
        "CLICKHOUSE_PORT",
        "CLICKHOUSE_USER",
        "CLICKHOUSE_PASS",
        "CLICKHOUSE_DB_V2",
    ):
        assert _expr(env[key], prod_only, dev) == "", key
    assert _expr(env["CLICKHOUSE_DB_V2"], {"CLICKHOUSE_DB_V2_DEV": "d"}, dev) == "d"
    assert _expr(env["CLICKHOUSE_DB_V2"], {}, {"clickhouse_env": "prod"}) == "spyre_v2"
    assert _expr(env["CLICKHOUSE_USER"], {"CLICKHOUSE_USER_DEV": "du"}, dev) == "du"


def _gate_script() -> str:
    steps = yaml.safe_load(_ACTION.read_text())["runs"]["steps"]
    return next(s["run"] for s in steps if s.get("id") == "gate")


@pytest.mark.parametrize(
    ("host", "user", "pw", "rc", "configured"),
    [
        ("", "", "", 0, "false"),
        ("h", "", "pw", 1, None),
        ("h", "u", "", 1, None),
        ("h", "u", "pw", 0, "true"),
    ],
)
def test_gate_fails_loudly_on_a_host_without_login(tmp_path, host, user, pw, rc, configured):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "x.json").write_text("{}")
    out = tmp_path / "out"
    env = {
        **os.environ,
        "CH_HOST": host,
        "CH_USER": user,
        "CH_PASS": pw,
        "RESULTS_DIR": str(tmp_path / "r"),
        "GITHUB_OUTPUT": str(out),
    }
    p = subprocess.run(
        ["bash", "-e", "-c", _gate_script()], env=env, capture_output=True, text=True
    )
    assert p.returncode == rc
    if configured is None:
        assert "::error::" in p.stdout
    else:
        assert out.read_text().strip() == f"configured={configured}"


def test_guard_step_fails_on_host_without_login():
    steps = yaml.safe_load((_WF / "push-to-clickhouse.yaml").read_text())["jobs"]["ingest"]["steps"]
    script = next(s["run"] for s in steps if s.get("name") == "Require the v2 login")
    base = {**os.environ, "CH_HOST": "h", "CH_USER": "u", "CH_PASS": "p"}
    assert subprocess.run(["bash", "-e", "-c", script], env=base).returncode == 0
    assert subprocess.run(["bash", "-e", "-c", script], env={**base, "CH_PASS": ""}).returncode == 1
    assert (
        subprocess.run(
            ["bash", "-e", "-c", script], env={**base, "CH_HOST": "", "CH_PASS": ""}
        ).returncode
        == 0
    )
