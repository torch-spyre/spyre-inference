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

"""The csrc/ host kernels match upstream's Triton kernels on triton-cpu, output
for output and buffer for buffer. One subprocess for every case: the gate has to
open real Triton before vLLM imports, which this process did with it closed."""

import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from tests.mrv2.kernels import parity_cases, spec_decode_cases  # noqa: F401  (registers cases)

pytestmark = pytest.mark.skipif(
    platform.machine() != "x86_64" or importlib.util.find_spec("triton") is None,
    reason="the parity reference is triton-cpu, which is x86-only and not installed",
)


@pytest.fixture(scope="module")
def parity_report() -> dict[str, list[str]]:
    result = subprocess.run(
        [sys.executable, str(Path(parity_cases.__file__))],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "SPYRE_MODEL_RUNNER": "v2",
            "SPYRE_MRV2_KERNELS": "triton",
            # The script imports its sibling case modules as tests.mrv2.kernels.*.
            "PYTHONPATH": os.pathsep.join(
                [str(Path(__file__).parents[3]), os.environ.get("PYTHONPATH", "")]
            ),
        },
        timeout=1800,
    )
    report: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        if line.startswith(("ok ", "FAIL ")):
            name = line.split()[1]
            report.setdefault(name, [])
            current = name
        elif line.startswith("    "):
            report[current].append(line.strip())
    assert report, result.stderr[-4000:]
    return report


@pytest.mark.triton_cpu
@pytest.mark.parametrize("name", sorted(parity_cases.CASES))
def test_host_kernel_matches_triton(parity_report: dict[str, list[str]], name: str) -> None:
    assert name in parity_report, f"{name} did not run"
    assert not parity_report[name], "\n".join(parity_report[name])
