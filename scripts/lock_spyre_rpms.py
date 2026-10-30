#!/usr/bin/env python3
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

"""Regenerate spyre-rpms.lock from the RPM bundle named in [tool.spyre-rpms].

The bundle folder must hold exactly one build of each package: its filename is the pin.
CI downloads from ARTIFACTORY_RPM_PATH, so every bundled RPM must have a bit-exact
copy there. Bundles don't ship -devel packages yet; those take their base's build.

Power and Z pin the bundle's commits too. When one hasn't built a bundled commit yet,
[tool.spyre-rpms.stale.<arch>] holds that package, and every package built from the same
commit, at an older one until it does:

  [tool.spyre-rpms.stale.s390x]
  ibm-senlib-core = "2.0.0-0.main.1+340.e2cc280"

  python3 scripts/lock_spyre_rpms.py
  python3 scripts/lock_spyre_rpms.py --verify   # offline, run by CI

Env: ARTIFACTORY_BASE_URL (or ARTIFACTORY_URL), ARTIFACTORY_RPM_PATH (or
ARTIFACTORY_RPM_REPO), ARTIFACTORY_BUNDLE_PATH (<repo>/<subpath>), ARTIFACTORY_TOKEN.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import tomllib
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))
from resolve_rpms import ARCHES, _env, _match_pattern  # noqa: E402

BUNDLE_ARCH = "x86_64"
# `+<count>.<sha7>_<build>` ends every version this tool can pin.
COMMIT_BUILD = re.compile(r"\+(\d+)\.([0-9a-f]{7,40})_(\d+)$")
STALE_COMMIT = re.compile(r"\+(\d+)\.[0-9a-f]{7,40}$")


class LockError(Exception):
    pass


class Artifactory:
    def __init__(self, base_url, token):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _request(self, path, data=None):
        req = urllib.request.Request(
            f"{self.base_url}/artifactory/{path}",
            data=data,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "text/plain"},
            method="POST" if data else "GET",
        )
        with urllib.request.urlopen(req) as resp:
            return json.load(resp)

    def listing(self, repo, path):
        """{filename: sha1} of the RPMs directly in <repo>/<path>."""
        query = (
            'items.find({"repo":%s,"path":%s,"name":{"$match":"*.rpm"}})'
            '.include("repo","path","name","actual_sha1")' % (json.dumps(repo), json.dumps(path))
        )
        results = self._request("api/search/aql", query.encode())["results"]
        return {r["name"]: r["actual_sha1"] for r in results}


def load_config(pyproject):
    with open(pyproject, "rb") as f:
        tool = tomllib.load(f).get("tool", {})
    cfg = tool.get("spyre-rpms")
    if not cfg or "bundle" not in cfg or "packages" not in cfg:
        raise LockError(f"{pyproject} needs [tool.spyre-rpms] with `bundle` and `packages`")
    stamp = tool.get("uv", {}).get("extra-build-variables", {}).get("torch-spyre", {})
    if stamp.get("SPYRE_RPM_BUNDLE") != cfg["bundle"]:
        raise LockError(
            f"set [tool.uv.extra-build-variables.torch-spyre] SPYRE_RPM_BUNDLE "
            f"(now {stamp.get('SPYRE_RPM_BUNDLE')!r}) to the bundle {cfg['bundle']!r}"
        )
    unknown = set(cfg.get("devel", {})) - set(cfg["packages"])
    if unknown:
        raise LockError(
            f"[tool.spyre-rpms.devel] names packages not in `packages`: {sorted(unknown)}"
        )
    errors = []
    for arch, pins in cfg.get("stale", {}).items():
        where = f"[tool.spyre-rpms.stale.{arch}]"
        if arch not in ARCHES or arch == BUNDLE_ARCH:
            errors.append(
                f"{where}: {arch} is not one of {[a for a in ARCHES if a != BUNDLE_ARCH]}"
            )
        for name, version in pins.items():
            if name not in cfg["packages"] or name in cfg.get("devel", {}):
                errors.append(f"{where}: {name} is not a non-devel entry of `packages`")
            if not STALE_COMMIT.search(str(version)):
                errors.append(f"{where}: {name} = {version!r} is not `<version>+<count>.<sha7>`")
    if errors:
        raise LockError("\n".join(errors))
    return cfg


def _version(name, filename):
    prefix, suffix = f"{name}-", f".el10.{BUNDLE_ARCH}.rpm"
    if not (filename.startswith(prefix) and filename.endswith(suffix)):
        return None
    version = filename[len(prefix) : -len(suffix)]
    # `ibm-flex-` also prefixes `ibm-flex-test-2.0.0...`; a version starts with a digit.
    return version if version[:1].isdigit() else None


def _bundled_version(name, bundled):
    found = [v for v in (_version(name, f) for f in bundled) if v and COMMIT_BUILD.search(v)]
    # A folder re-published in place (`main`, `main_snap`) can keep stale builds beside the
    # current one, and nothing in it says which is current.
    if len(found) != 1:
        raise LockError(
            f"{name}: expected one build in the bundle, found {len(found)}: {sorted(found)}"
        )
    return found[0]


def resolve(cfg, client, bundle_path, rpm_repo):
    """{name: x86_64 version} for cfg's bundle."""
    repo, _, subpath = bundle_path.strip("/").partition("/")
    folder = "/".join(p for p in (subpath, BUNDLE_ARCH, "prod", cfg["bundle"]) if p)
    bundled = client.listing(repo, folder)
    if not bundled:
        raise LockError(f"bundle {cfg['bundle']} has no RPMs ({repo}/{folder})")
    published = {arch: client.listing(rpm_repo, arch) for arch in ARCHES}
    devel = cfg.get("devel", {})

    errors = []
    versions = {}
    for name in cfg["packages"]:
        if name in devel:
            continue
        try:
            versions[name] = _bundled_version(name, bundled)
        except LockError as e:
            errors.append(str(e))
            continue
        filename = _match_pattern(name, versions[name], BUNDLE_ARCH)
        sha1 = published[BUNDLE_ARCH].get(filename)
        if sha1 != bundled[filename]:
            what = "missing" if sha1 is None else f"sha1 {sha1}, bundle has {bundled[filename]}"
            errors.append(f"{name}: {rpm_repo}/{BUNDLE_ARCH}/{filename} {what}")
    for name, base in devel.items():
        if base not in versions:
            errors.append(f"{name}: its base package {base} did not resolve")
            continue
        versions[name] = versions[base]
        filename = _match_pattern(name, versions[name], BUNDLE_ARCH)
        if filename not in published[BUNDLE_ARCH]:
            errors.append(f"{name}: {rpm_repo}/{BUNDLE_ARCH}/{filename} missing")
    if errors:
        raise LockError("\n".join(errors))
    for arch, pins in overrides(cfg, versions).items():
        for name, pattern in pins.items():
            filename = _match_pattern(name, pattern, arch)
            if not fnmatch.filter(published[arch], filename):
                errors.append(f"{name}: nothing matches {rpm_repo}/{arch}/{filename}")
        for name in cfg.get("stale", {}).get(arch, {}):
            bundled = _match_pattern(name, wildcard(versions[name]), arch)
            if fnmatch.filter(published[arch], bundled):
                errors.append(
                    f"{name}: {arch} now has {bundled}; drop it from [tool.spyre-rpms.stale.{arch}]"
                )
    if errors:
        raise LockError("\n".join(errors))
    return {name: versions[name] for name in cfg["packages"]}


def verify(cfg, lock_text):
    """Offline: the lock is exactly what this tool renders for `cfg`."""
    data = tomllib.loads(lock_text)
    bundle = data.get("bundle", {})
    versions = {name: spec["version"] for name, spec in data.get("packages", {}).items()}
    errors = []
    if bundle.get("id") != cfg["bundle"]:
        errors.append(
            f"lock is from bundle {bundle.get('id')!r}, pyproject.toml names {cfg['bundle']!r}"
        )
    if list(versions) != cfg["packages"]:
        errors.append(
            f"lock packages {list(versions)} differ from pyproject.toml {cfg['packages']}"
        )
    errors += [
        f"{name} {version} has no `.<commit>_<build>` suffix"
        for name, version in versions.items()
        if not COMMIT_BUILD.search(version)
    ]
    errors += [
        f"{name} {versions.get(name)} differs from its base {base} {versions.get(base)}"
        for name, base in cfg.get("devel", {}).items()
        if versions.get(name) != versions.get(base)
    ]
    if not errors and lock_text != render(cfg, versions):
        errors.append("lock differs from what this tool renders for it (edited by hand?)")
    if errors:
        raise LockError("\n".join(errors))


def _commit(version):
    return version[: COMMIT_BUILD.search(version).start(3) - 1]


def wildcard(version):
    return f"{_commit(version)}_*"


def _count(commit):
    return int(STALE_COMMIT.search(commit).group(1))


def overrides(cfg, versions):
    """{arch: {name: version pattern}} for Power and Z, [tool.spyre-rpms.stale] applied.

    A stale pin holds back every package built from the same commit as the one it names,
    so -devel/-headers and siblings like ibm-senlib-dd2 stay in step with it.
    """
    out, errors = {}, []
    for arch in ARCHES:
        if arch == BUNDLE_ARCH:
            continue
        pins = {name: wildcard(v) for name, v in versions.items()}
        held = {}
        for stale_name, stale in cfg.get("stale", {}).get(arch, {}).items():
            bundled = _commit(versions[stale_name])
            if _count(stale) >= _count(bundled):
                errors.append(
                    f"[tool.spyre-rpms.stale.{arch}] {stale_name} = {stale!r} is not older "
                    f"than the bundle's {bundled}"
                )
                continue
            for name, version in versions.items():
                if _commit(version) != bundled:
                    continue
                if held.setdefault(name, stale) != stale:
                    errors.append(
                        f"[tool.spyre-rpms.stale.{arch}] holds {name} at both "
                        f"{held[name]} and {stale}"
                    )
                pins[name] = f"{stale}_*"
        out[arch] = pins
    if errors:
        raise LockError("\n".join(errors))
    return out


def render(cfg, versions):
    width = max(map(len, versions))

    def table(header, pins, marked=True):
        rows = []
        for name, v in pins.items():
            row = f"{name:<{width}} = {{ version = {json.dumps(v)} }}"
            if marked and v != wildcard(versions[name]):
                row += f"  # stale: bundle has {_commit(versions[name])}"
            rows.append(row)
        return "\n".join([header, *rows])

    arch_pins = overrides(cfg, versions)
    return f"""\
# Generated from the Spyre RPM bundle below by scripts/lock_spyre_rpms.py.
# Change [tool.spyre-rpms] in pyproject.toml and re-run it instead of editing.
#
# x86_64 is pinned to the bundle's exact builds. Power (ppc64le) and Z (s390x)
# pin the same commits with the build number wildcarded: builds differ across
# platforms. A row marked `stale` is held at an older commit by
# [tool.spyre-rpms.stale.<arch>] until that arch builds the bundled one.

version = 2

[defaults]
# Artifactory subdir (<repo>/<tree>/<arch>/); "" is the prod/base tree.
tree = ""

[bundle]
id = {json.dumps(cfg["bundle"])}

# x86_64
{table("[packages]", versions, marked=False)}

# Power
{table("[overrides.ppc64le]", arch_pins["ppc64le"])}

# Z
{table("[overrides.s390x]", arch_pins["s390x"])}
"""


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pyproject", default=REPO_ROOT / "pyproject.toml")
    p.add_argument("--lock", default=REPO_ROOT / "spyre-rpms.lock")
    p.add_argument(
        "--verify",
        action="store_true",
        help="check, without network, that --lock is what this tool generates from --pyproject",
    )
    args = p.parse_args()

    if args.verify:
        try:
            verify(load_config(args.pyproject), Path(args.lock).read_text())
        except LockError as e:
            sys.exit(f"{e}\nregenerate with `python3 scripts/lock_spyre_rpms.py`")
        print(f"{args.lock} matches {args.pyproject}")
        return

    client = Artifactory(_env("ARTIFACTORY_BASE_URL", "ARTIFACTORY_URL"), _env("ARTIFACTORY_TOKEN"))
    try:
        cfg = load_config(args.pyproject)
        versions = resolve(
            cfg,
            client,
            _env("ARTIFACTORY_BUNDLE_PATH"),
            _env("ARTIFACTORY_RPM_PATH", "ARTIFACTORY_RPM_REPO"),
        )
    except LockError as e:
        sys.exit(f"{e}\n{args.lock} left unchanged")
    Path(args.lock).write_text(render(cfg, versions))
    for name, version in versions.items():
        print(f"{name:<{max(map(len, versions))}} {version}")
    print(f"wrote {args.lock} from bundle {cfg['bundle']}")


if __name__ == "__main__":
    main()
