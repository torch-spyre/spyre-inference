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

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import resolve_rpms as rr  # noqa: E402


def _data(name="ibm-flex", version="2.1.0-0.main.1+754.8267eae_700", tree=""):
    return {
        "defaults": {"tree": tree},
        "packages": {name: {"version": version}},
    }


def test_clean_lock_resolves_to_specs():
    specs = rr.resolve_specs(_data(), "x86_64")
    assert specs == [("ibm-flex", "2.1.0-0.main.1+754.8267eae_700", "")]


def test_override_version_wildcard_is_allowed():
    data = _data(version="2.1.0-0.main.1+754.8267eae_700")
    data["overrides"] = {"ppc64le": {"ibm-flex": {"version": "2.1.0-0.main.1+754.8267eae_*"}}}
    name, version, tree = rr.resolve_specs(data, "ppc64le")[0]
    assert version.endswith("_*")


@pytest.mark.parametrize(
    "tree",
    [
        'x","$or":[{"repo":{"$match":"*"',
        'a" b',
        "a$b",
        "..",
        "a/../b",
    ],
)
def test_injection_in_tree_is_rejected(tree):
    with pytest.raises(SystemExit):
        rr.resolve_specs(_data(tree=tree), "x86_64")


def test_legit_nested_tree_with_dots_is_allowed():
    assert rr.resolve_specs(_data(tree="dev.snap/sub"), "x86_64")[0][2] == "dev.snap/sub"


@pytest.mark.parametrize("name", ['ibm"flex', "ibm flex", "ibm$flex", 'a"}}],"x":"'])
def test_injection_in_name_is_rejected(name):
    with pytest.raises(SystemExit):
        rr.resolve_specs(_data(name=name), "x86_64")


@pytest.mark.parametrize("version", ['2.0"', "2.0 0", 'x"}}]'])
def test_injection_in_version_is_rejected(version):
    with pytest.raises(SystemExit):
        rr.resolve_specs(_data(version=version), "x86_64")


def test_package_sha1_returns_pinned_digest():
    data = _data()
    data["packages"]["ibm-flex"]["sha1"] = "0" * 40
    assert rr.package_sha1(data, "ibm-flex") == "0" * 40


def test_package_sha1_absent_is_none():
    # An override-arch entry (or a lock predating digests) pins no sha1.
    assert rr.package_sha1(_data(), "ibm-flex") is None


@pytest.mark.parametrize(
    "sha1",
    [
        "not-hex",
        "0" * 39,  # too short
        "0" * 41,  # too long
        "A" * 40,  # uppercase is not Artifactory's actual_sha1 form
        '0" $(touch pwned) "',  # shell-special must never reach the installer
    ],
)
def test_malformed_sha1_is_rejected(sha1):
    data = _data()
    data["packages"]["ibm-flex"]["sha1"] = sha1
    with pytest.raises(SystemExit):
        rr.package_sha1(data, "ibm-flex")
