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

"""triton-cpu is the host kernels' parity reference, so only the dev group installs it;
a runtime dependency that starts requesting triton fails here."""

import tomllib
from pathlib import Path

_LOCK = Path(__file__).resolve().parents[2] / "uv.lock"


def test_only_the_dev_group_and_darwin_torch_request_triton() -> None:
    packages = tomllib.loads(_LOCK.read_text())["package"]
    requesters = set()
    for package in packages:
        for dep in package.get("dependencies", []):
            if dep["name"] == "triton":
                requesters.add((package["name"], package.get("version")))
        for group, deps in package.get("dev-dependencies", {}).items():
            if any(dep["name"] == "triton" for dep in deps):
                requesters.add((f"{package['name']}[{group}]", None))
    # PyPI's torch wheel is only locked for macOS, which Spyre does not run on.
    assert requesters == {("spyre-inference[dev]", None), ("torch", "2.14.1")}, requesters
