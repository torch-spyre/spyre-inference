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

"""Offline tests for lock_spyre_rpms.py; not run by CI. Run them after changing the tool:

uv run --no-project --with pytest pytest -o addopts="" scripts/test_lock_spyre_rpms.py
"""

import sys
import tomllib
import urllib.error
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / ".github" / "scripts")]
import lock_spyre_rpms as lock  # noqa: E402
from resolve_rpms import resolve_specs  # noqa: E402

FLEX_SHA = "e066296616826cbf3f32b13a5a5627408e7c65d9"
SENLIB_SHA = "9b2385fa8aea16c23bfd2838b691ab92224f4d87"
FLEX_V = "2.0.0-0.main.1+700.e066296_608"
SENLIB_V = "2.0.0-0.main.1+316.9b2385f_451"

CFG = {
    "bundle": "nightly",
    "packages": ["ibm-flex", "ibm-flex-devel", "ibm-senlib-core"],
    "devel": {"ibm-flex-devel": "ibm-flex"},
}


def rpm(name, version, arch="x86_64"):
    return f"{name}-{version}.el10.{arch}.rpm"


def bundle_files():
    return {
        rpm("ibm-flex", FLEX_V): "a1",
        rpm("ibm-flex-test", FLEX_V): "a2",
        rpm("ibm-senlib-core", SENLIB_V): "b1",
    }


def published_files():
    files = {"x86_64": {**bundle_files(), rpm("ibm-flex-devel", FLEX_V): "a3"}}
    for arch, build in (("ppc64le", "563"), ("s390x", "716")):
        files[arch] = {
            rpm(n, v.rsplit("_", 1)[0] + "_" + build, arch): "x"
            for n, v in (
                ("ibm-flex", FLEX_V),
                ("ibm-flex-devel", FLEX_V),
                ("ibm-senlib-core", SENLIB_V),
            )
        }
    return files


class FakeArtifactory:
    def __init__(self, bundled=None, published=None, build_hashes=None):
        self.bundled = bundle_files() if bundled is None else bundled
        self.published = published_files() if published is None else published
        self.build_hashes = build_hashes or {
            "flex": FLEX_SHA,
            "senlib": SENLIB_SHA,
            "branch": "main",
        }

    def get_json(self, path):
        assert path == "bundles/zsp/x86_64/prod/nightly/build_hashes.json"
        if self.build_hashes == "missing":
            raise urllib.error.HTTPError(path, 404, "Not Found", None, None)
        return self.build_hashes

    def listing(self, repo, path):
        if repo == "bundles":
            assert path == "zsp/x86_64/prod/nightly"
            return self.bundled
        assert repo == "rpms"
        return self.published[path]


def run(client, cfg=CFG):
    return lock.resolve(cfg, client, "bundles/zsp", "rpms")


def test_lock_round_trips_through_ci_resolver():
    build_hashes, versions = run(FakeArtifactory())
    assert versions == {"ibm-flex": FLEX_V, "ibm-flex-devel": FLEX_V, "ibm-senlib-core": SENLIB_V}

    data = tomllib.loads(lock.render("nightly", build_hashes, versions))
    assert data["bundle"] == {"id": "nightly", "build_hashes": build_hashes}
    assert resolve_specs(data, "x86_64") == [
        ("ibm-flex", FLEX_V, ""),
        ("ibm-flex-devel", FLEX_V, ""),
        ("ibm-senlib-core", SENLIB_V, ""),
    ]
    for arch in ("ppc64le", "s390x"):
        assert [v for _, v, _ in resolve_specs(data, arch)] == [
            "2.0.0-0.main.1+700.e066296_*",
            "2.0.0-0.main.1+700.e066296_*",
            "2.0.0-0.main.1+316.9b2385f_*",
        ]


def test_sha1_mismatch_fails():
    published = published_files()
    published["x86_64"][rpm("ibm-flex", FLEX_V)] = "other"
    with pytest.raises(lock.LockError, match="ibm-flex: .*sha1 other, bundle has a1"):
        run(FakeArtifactory(published=published))


def test_bundled_rpm_missing_from_rpm_repo_fails():
    published = published_files()
    del published["x86_64"][rpm("ibm-senlib-core", SENLIB_V)]
    with pytest.raises(lock.LockError, match="ibm-senlib-core: .* missing"):
        run(FakeArtifactory(published=published))


def test_devel_missing_from_rpm_repo_fails():
    published = published_files()
    del published["x86_64"][rpm("ibm-flex-devel", FLEX_V)]
    with pytest.raises(lock.LockError, match="ibm-flex-devel: .* missing"):
        run(FakeArtifactory(published=published))


def test_stale_build_at_unbundled_commit_is_skipped():
    stale = "2.0.0-0.main.1+690.1234567_600"
    bundled = {**bundle_files(), rpm("ibm-flex", stale): "old"}
    published = published_files()
    published["x86_64"][rpm("ibm-flex", stale)] = "old"
    _, versions = run(FakeArtifactory(bundled=bundled, published=published))
    assert versions["ibm-flex"] == FLEX_V


def test_two_builds_at_bundled_commit_fail():
    bundled = {**bundle_files(), rpm("ibm-flex", "2.0.0-0.main.1+700.e066296_609"): "a9"}
    with pytest.raises(lock.LockError, match="ibm-flex: expected one build .* found 2"):
        run(FakeArtifactory(bundled=bundled))


def test_commit_missing_on_other_arch_fails():
    published = published_files()
    published["s390x"] = {k: v for k, v in published["s390x"].items() if "senlib" not in k}
    with pytest.raises(
        lock.LockError, match=r"ibm-senlib-core: nothing matches rpms/s390x/.*9b2385f_\*"
    ):
        run(FakeArtifactory(published=published))


def test_bundle_without_build_hashes_fails():
    with pytest.raises(lock.LockError, match="no build_hashes.json for bundle nightly"):
        run(FakeArtifactory(build_hashes="missing"))


def test_devel_must_be_a_listed_package(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.spyre-rpms]\nbundle = "b"\npackages = ["ibm-flex"]\n'
        '[tool.spyre-rpms.devel]\nibm-flex-devel = "ibm-flex"\n'
    )
    with pytest.raises(lock.LockError, match="ibm-flex-devel"):
        lock.load_config(pyproject)


def _lock_text():
    return lock.render("nightly", *run(FakeArtifactory()))


def test_verify_accepts_rendered_lock():
    lock.verify(CFG, _lock_text())


@pytest.mark.parametrize(
    "cfg, edit, match",
    [
        ({**CFG, "bundle": "newer"}, None, "from bundle 'nightly', pyproject.toml names 'newer'"),
        ({**CFG, "packages": CFG["packages"][:2]}, None, "lock packages .* differ"),
        (CFG, ("700.e066296_*", "700.e066296_607"), "differs from what this tool renders"),
        (
            CFG,
            ('_608" }\nibm-senlib', '_607" }\nibm-senlib'),
            "ibm-flex-devel .* differs from its base",
        ),
        (CFG, ("+316.9b2385f_451", "+317.1234567_451"), "ibm-senlib-core .* not at a commit"),
    ],
)
def test_verify_rejects(cfg, edit, match):
    text = _lock_text()
    if edit:
        assert edit[0] in text
        text = text.replace(edit[0], edit[1], 1)
    with pytest.raises(lock.LockError, match=match):
        lock.verify(cfg, text)
