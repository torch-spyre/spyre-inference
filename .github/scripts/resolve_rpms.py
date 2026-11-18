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

"""Resolve pinned Spyre RPMs from Artifactory for spyre-rpms.lock.

The lock's `[packages]` pins the EXACT x86_64 build (the trailing `_<buildnum>`
is part of the pin — CI installs exactly that build). `[overrides.<arch>]` pins
the same commit on other arches but wildcards the build (`_*`), because we have
no CI to validate an exact build there; this script resolves the `_*` to the
newest matching build via an Artifactory AQL query.

Subcommands:
  resolve --arch ARCH   Print each resolved RPM's repo-relative path.
  validate              Assert every package resolves on ALL arches.
  names                 Print package names (no network).

Env: ARTIFACTORY_BASE_URL (or ARTIFACTORY_URL), ARTIFACTORY_RPM_PATH (or
ARTIFACTORY_RPM_REPO), ARTIFACTORY_TOKEN (resolve/validate only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request

ARCHES = ("x86_64", "ppc64le", "s390x")
DEFAULT_LOCK = "spyre-rpms.lock"


class _StripAuthOnCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    """Drop the Artifactory bearer token when a redirect leaves the origin host.

    The stdlib default replays every request header, including Authorization,
    onto the redirect target. A redirect to another host (a misconfigured
    reverse proxy, or one injected by a network attacker) would then hand the
    ARTIFACTORY_TOKEN to that host. requests strips credentials across hosts;
    urllib does not, so we do it here.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and _host(req.full_url) != _host(newurl):
            new.remove_header("Authorization")
        return new


def _host(url):
    return urllib.parse.urlsplit(url).netloc.lower()


def artifactory_open(url, token, data=None, method=None):
    """Open a credentialed Artifactory request over HTTPS only.

    Rejects a non-https base URL so the bearer token is never sent in the
    clear, and uses an opener that strips the token on a cross-host redirect.
    """
    if urllib.parse.urlsplit(url).scheme != "https":
        sys.exit(f"::error::Artifactory URL must use https, refusing to send token to {url!r}")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "text/plain"},
        method=method,
    )
    opener = urllib.request.build_opener(_StripAuthOnCrossHostRedirect)
    return opener.open(req)


# The lock can come from an untrusted fork PR and its fields are formatted into
# a credentialed AQL query, so they must not be able to escape it.
_NAME_RE = re.compile(r"\A[A-Za-z0-9._+-]+\Z")
_VERSION_RE = re.compile(r"\A[A-Za-z0-9._+*-]+\Z")
_TREE_RE = re.compile(r"\A[A-Za-z0-9._/-]*\Z")


def _check_field(kind, value, pattern):
    if not isinstance(value, str) or not pattern.match(value):
        sys.exit(f"::error::illegal {kind} in spyre-rpms.lock: {value!r}")
    return value


def _check_tree(value):
    _check_field("tree", value, _TREE_RE)
    if ".." in value.split("/"):
        sys.exit(f"::error::illegal tree in spyre-rpms.lock (path traversal): {value!r}")
    return value


def load_data(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def package_names(data):
    return list(data.get("packages", {}).keys())


def resolve_specs(data, arch):
    """Per-arch (name, version, tree), applying [overrides.<arch>] over [packages].

    x86_64 has no overrides, so it uses the exact build pinned in [packages];
    ppc64le/s390x pick up the wildcarded (`_*`) version from their override.
    """
    default_tree = data.get("defaults", {}).get("tree", "")
    overrides = data.get("overrides", {}).get(arch, {})
    specs = []
    for name, spec in data.get("packages", {}).items():
        ov = overrides.get(name, {})
        version = ov.get("version", spec["version"])
        tree = ov.get("tree", spec.get("tree", default_tree))
        _check_field("package name", name, _NAME_RE)
        _check_field("version", version, _VERSION_RE)
        _check_tree(tree)
        specs.append((name, version, tree))
    return specs


def _env(*names):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    sys.exit(f"::error::none of these env vars are set: {', '.join(names)}")


def _match_pattern(name, version, arch):
    # `version` already carries the exact `_<buildnum>` (x86_64) or a `_*`
    # wildcard (override arches); an exact pattern matches a single file.
    return f"{name}-{version}.el10.{arch}.rpm"


def _aql_newest(base_url, repo, token, tree, arch, name_pattern):
    """Return the newest matching filename in <repo>/<tree>/<arch>, or None."""
    path = f"{tree}/{arch}" if tree else arch
    query = (
        'items.find({"repo":"%s","path":"%s","name":{"$match":"%s"}})'
        '.sort({"$desc":["created"]}).limit(1)' % (repo, path, name_pattern)
    )
    with artifactory_open(
        f"{base_url}/artifactory/api/search/aql", token, data=query.encode(), method="POST"
    ) as resp:
        results = json.load(resp).get("results", [])
    return results[0]["name"] if results else None


def resolve_arch(specs, base_url, repo, token, arch):
    missing = []
    out = []
    for name, version, tree in specs:
        pat = _match_pattern(name, version, arch)
        fn = _aql_newest(base_url, repo, token, tree, arch, pat)
        if fn is None:
            missing.append(f"{name} @ {version} (tree={tree or 'prod'}, {arch})")
            continue
        relpath = f"{tree}/{arch}/{fn}" if tree else f"{arch}/{fn}"
        out.append((relpath, fn))
    return out, missing


def cmd_resolve(args):
    data = load_data(args.lock)
    base_url = _env("ARTIFACTORY_BASE_URL", "ARTIFACTORY_URL").rstrip("/")
    repo = _env("ARTIFACTORY_RPM_PATH", "ARTIFACTORY_RPM_REPO")
    token = _env("ARTIFACTORY_TOKEN")
    specs = resolve_specs(data, args.arch)
    resolved, missing = resolve_arch(specs, base_url, repo, token, args.arch)
    if missing:
        for m in missing:
            print(f"::error::could not resolve {m}", file=sys.stderr)
        sys.exit(1)
    for relpath, _ in resolved:
        print(relpath)


def cmd_validate(args):
    data = load_data(args.lock)
    base_url = _env("ARTIFACTORY_BASE_URL", "ARTIFACTORY_URL").rstrip("/")
    repo = _env("ARTIFACTORY_RPM_PATH", "ARTIFACTORY_RPM_REPO")
    token = _env("ARTIFACTORY_TOKEN")
    ok = True
    for name in package_names(data):
        found = {}
        detail = {}
        for arch in ARCHES:
            _, version, tree = next(s for s in resolve_specs(data, arch) if s[0] == name)
            pat = _match_pattern(name, version, arch)
            found[arch] = _aql_newest(base_url, repo, token, tree, arch, pat)
            detail[arch] = version
        if all(found.values()):
            print(f"OK   {name}")
        else:
            ok = False
            gaps = ", ".join(f"{a} ({detail[a]})" for a in ARCHES if not found[a])
            print(f"FAIL {name} — missing on: {gaps}")
    if not ok:
        print(
            "::error::a pinned build/commit is not published for every arch. "
            "For x86_64 pick an exact build present in prod; for ppc64le/s390x "
            "pin a commit present on that arch.",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_names(args):
    for name in package_names(load_data(args.lock)):
        print(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", default=DEFAULT_LOCK, help="path to spyre-rpms.lock")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("resolve", help="print repo-relative RPM paths for one arch")
    r.add_argument("--arch", required=True, choices=ARCHES)
    r.set_defaults(func=cmd_resolve)

    v = sub.add_parser("validate", help="assert every package resolves on all arches")
    v.set_defaults(func=cmd_validate)

    n = sub.add_parser("names", help="print package names (no network)")
    n.set_defaults(func=cmd_names)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
