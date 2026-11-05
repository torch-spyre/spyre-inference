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
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / ".github" / "scripts")]
import lock_spyre_rpms as lock  # noqa: E402
from resolve_rpms import resolve_specs  # noqa: E402

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
    def __init__(self, bundled=None, published=None):
        self.bundled = bundle_files() if bundled is None else bundled
        self.published = published_files() if published is None else published

    def listing(self, repo, path):
        if repo == "bundles":
            assert path == "zsp/x86_64/prod/nightly"
            return self.bundled
        assert repo == "rpms"
        return self.published[path]


def run(client, cfg=CFG):
    return lock.resolve(cfg, client, "bundles/zsp", "rpms")


def test_lock_round_trips_through_ci_resolver():
    versions = run(FakeArtifactory())
    assert versions == {"ibm-flex": FLEX_V, "ibm-flex-devel": FLEX_V, "ibm-senlib-core": SENLIB_V}

    data = tomllib.loads(lock.render(CFG, versions))
    assert data["bundle"] == {"id": "nightly"}
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


@pytest.mark.parametrize(
    "extra", ["2.0.0-0.main.1+690.1234567_600", "2.0.0-0.main.1+700.e066296_609"]
)
def test_two_builds_in_bundle_fail(extra):
    bundled = {**bundle_files(), rpm("ibm-flex", extra): "a9"}
    with pytest.raises(lock.LockError, match="ibm-flex: expected one build .* found 2"):
        run(FakeArtifactory(bundled=bundled))


def test_package_missing_from_bundle_fails():
    bundled = {k: v for k, v in bundle_files().items() if "senlib" not in k}
    with pytest.raises(lock.LockError, match="ibm-senlib-core: expected one build .* found 0"):
        run(FakeArtifactory(bundled=bundled))


def test_commit_missing_on_other_arch_fails():
    published = published_files()
    published["s390x"] = {k: v for k, v in published["s390x"].items() if "senlib" not in k}
    with pytest.raises(
        lock.LockError, match=r"ibm-senlib-core: nothing matches rpms/s390x/.*9b2385f_\*"
    ):
        run(FakeArtifactory(published=published))


def test_empty_bundle_fails():
    with pytest.raises(lock.LockError, match="bundle nightly has no RPMs"):
        run(FakeArtifactory(bundled={}))


SENLIB_OLD = "2.0.0-0.main.1+315.0123abc"
STALE_CFG = {
    **CFG,
    "packages": [*CFG["packages"], "ibm-senlib-dd2", "ibm-senlib-headers"],
    "devel": {**CFG["devel"], "ibm-senlib-headers": "ibm-senlib-core"},
    "stale": {"s390x": {"ibm-senlib-core": SENLIB_OLD}},
}


def stale_artifactory(z_has_bundled=False):
    bundled = {**bundle_files(), rpm("ibm-senlib-dd2", SENLIB_V): "b2"}
    published = published_files()
    published["x86_64"] |= {
        rpm("ibm-senlib-dd2", SENLIB_V): "b2",
        rpm("ibm-senlib-headers", SENLIB_V): "b3",
    }
    published["ppc64le"] |= {
        rpm(n, SENLIB_V.replace("_451", "_563"), "ppc64le"): "x"
        for n in ("ibm-senlib-dd2", "ibm-senlib-headers")
    }
    published["s390x"] = {k: v for k, v in published["s390x"].items() if "senlib" not in k}
    for n in ("ibm-senlib-core", "ibm-senlib-dd2", "ibm-senlib-headers"):
        published["s390x"][rpm(n, f"{SENLIB_OLD}_700", "s390x")] = "x"
        if z_has_bundled:
            published["s390x"][rpm(n, SENLIB_V.replace("_451", "_716"), "s390x")] = "x"
    return FakeArtifactory(bundled=bundled, published=published)


def test_stale_pin_holds_back_every_package_at_its_commit():
    versions = run(stale_artifactory(), STALE_CFG)
    data = tomllib.loads(lock.render(STALE_CFG, versions))
    z = {name: v for name, v, _ in resolve_specs(data, "s390x")}
    for name in ("ibm-senlib-core", "ibm-senlib-dd2", "ibm-senlib-headers"):
        assert data["packages"][name]["version"] == SENLIB_V
        assert data["overrides"]["ppc64le"][name]["version"] == "2.0.0-0.main.1+316.9b2385f_*"
        assert z[name] == f"{SENLIB_OLD}_*"
    assert z["ibm-flex"] == "2.0.0-0.main.1+700.e066296_*"
    text = lock.render(STALE_CFG, versions)
    assert text.count("# stale: bundle has 2.0.0-0.main.1+316.9b2385f") == 3
    lock.verify(STALE_CFG, text)
    with pytest.raises(lock.LockError, match="differs from what this tool renders"):
        lock.verify(CFG | {"packages": STALE_CFG["packages"], "devel": STALE_CFG["devel"]}, text)


def test_stale_pin_fails_once_arch_has_bundled_commit():
    with pytest.raises(lock.LockError, match=r"ibm-senlib-core: s390x now has .*drop it"):
        run(stale_artifactory(z_has_bundled=True), STALE_CFG)


def test_stale_pin_must_exist_on_its_arch():
    cfg = {**STALE_CFG, "stale": {"s390x": {"ibm-senlib-core": "2.0.0-0.main.1+314.fedcba9"}}}
    with pytest.raises(lock.LockError, match=r"ibm-senlib-dd2: nothing matches rpms/s390x/.*314"):
        run(stale_artifactory(), cfg)


def test_stale_pin_must_be_older_than_bundle():
    cfg = {**STALE_CFG, "stale": {"s390x": {"ibm-senlib-core": "2.0.0-0.main.1+316.9b2385f"}}}
    with pytest.raises(lock.LockError, match="is not older than the bundle's"):
        run(stale_artifactory(), cfg)


@pytest.mark.parametrize(
    "stale, match",
    [
        (
            '[tool.spyre-rpms.stale.x86_64]\nibm-flex = "2.0.0-0.main.1+1.abcdef0"\n',
            "x86_64 is not",
        ),
        (
            '[tool.spyre-rpms.stale.s390x]\nibm-flex-devel = "2.0.0-0.main.1+1.abcdef0"\n',
            "non-devel",
        ),
        ('[tool.spyre-rpms.stale.s390x]\nibm-flex = "2.0.0-0.main.1+1.abcdef0_5"\n', "is not `<v"),
    ],
)
def test_stale_config_is_checked(tmp_path, stale, match):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.uv.extra-build-variables.torch-spyre]\nSPYRE_RPM_BUNDLE = "b"\n'
        '[tool.spyre-rpms]\nbundle = "b"\npackages = ["ibm-flex", "ibm-flex-devel"]\n'
        '[tool.spyre-rpms.devel]\nibm-flex-devel = "ibm-flex"\n' + stale
    )
    with pytest.raises(lock.LockError, match=match):
        lock.load_config(pyproject)


def test_devel_must_be_a_listed_package(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.uv.extra-build-variables.torch-spyre]\nSPYRE_RPM_BUNDLE = "b"\n'
        '[tool.spyre-rpms]\nbundle = "b"\npackages = ["ibm-flex"]\n'
        '[tool.spyre-rpms.devel]\nibm-flex-devel = "ibm-flex"\n'
    )
    with pytest.raises(lock.LockError, match="ibm-flex-devel"):
        lock.load_config(pyproject)


@pytest.mark.parametrize("stamp", ["", 'SPYRE_RPM_BUNDLE = "older"\n'])
def test_torch_spyre_build_stamp_must_match_bundle(tmp_path, stamp):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        f"[tool.uv.extra-build-variables.torch-spyre]\n{stamp}"
        '[tool.spyre-rpms]\nbundle = "b"\npackages = ["ibm-flex"]\n'
    )
    with pytest.raises(lock.LockError, match="SPYRE_RPM_BUNDLE"):
        lock.load_config(pyproject)


def _lock_text():
    return lock.render(CFG, run(FakeArtifactory()))


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
        (CFG, ("+316.9b2385f_451", "+316.9b2385f"), "ibm-senlib-core .* no `.<commit>_<build>`"),
    ],
)
def test_verify_rejects(cfg, edit, match):
    text = _lock_text()
    if edit:
        assert edit[0] in text
        text = text.replace(edit[0], edit[1], 1)
    with pytest.raises(lock.LockError, match=match):
        lock.verify(cfg, text)
