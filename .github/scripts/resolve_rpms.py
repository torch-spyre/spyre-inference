#!/usr/bin/env python3
"""Resolve pinned Spyre RPMs from Artifactory for spyre-rpms.lock.

Each lock entry pins a commit (arch-independent); the exact build — the
trailing `_<buildnum>`, which diverges per arch — is resolved here via an
Artifactory AQL query for the newest build of that commit on the given arch.

Subcommands:
  resolve --arch ARCH   Print each resolved RPM's repo-relative path.
  validate              Assert every package resolves on BOTH x86_64 and s390x.
  names                 Print package names (no network).

Env: ARTIFACTORY_BASE_URL (or ARTIFACTORY_URL), ARTIFACTORY_RPM_PATH (or
ARTIFACTORY_RPM_REPO), ARTIFACTORY_TOKEN (resolve/validate only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
import urllib.request

ARCHES = ("x86_64", "s390x")
DEFAULT_LOCK = "spyre-rpms.lock"


def load_lock(path):
    with open(path, "rb") as f:
        data = tomllib.load(f)
    default_tree = data.get("defaults", {}).get("tree", "")
    pkgs = []
    for name, spec in data.get("packages", {}).items():
        pkgs.append((name, spec["version"], spec.get("tree", default_tree)))
    return pkgs


def _env(*names):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    sys.exit(f"::error::none of these env vars are set: {', '.join(names)}")


def _match_pattern(name, version, tree, arch):
    # base tree filenames carry a per-arch `_<buildnum>` before `.el10`; other
    # (dev) trees embed a build hash captured by a `*` already in `version`.
    infix = "_*" if tree == "" else ""
    return f"{name}-{version}{infix}.el10.{arch}.rpm"


def _aql_newest(base_url, repo, token, tree, arch, name_pattern):
    """Return the newest matching filename in <repo>/<tree>/<arch>, or None."""
    path = f"{tree}/{arch}" if tree else arch
    query = (
        'items.find({"repo":"%s","path":"%s","name":{"$match":"%s"}})'
        '.sort({"$desc":["created"]}).limit(1)' % (repo, path, name_pattern)
    )
    req = urllib.request.Request(
        f"{base_url}/artifactory/api/search/aql",
        data=query.encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "text/plain"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        results = json.load(resp).get("results", [])
    return results[0]["name"] if results else None


def resolve_arch(pkgs, base_url, repo, token, arch):
    missing = []
    out = []
    for name, version, tree in pkgs:
        pat = _match_pattern(name, version, tree, arch)
        fn = _aql_newest(base_url, repo, token, tree, arch, pat)
        if fn is None:
            missing.append(f"{name} @ {version} (tree={tree or 'base'}, {arch})")
            continue
        relpath = f"{tree}/{arch}/{fn}" if tree else f"{arch}/{fn}"
        out.append((relpath, fn))
    return out, missing


def cmd_resolve(args):
    pkgs = load_lock(args.lock)
    base_url = _env("ARTIFACTORY_BASE_URL", "ARTIFACTORY_URL").rstrip("/")
    repo = _env("ARTIFACTORY_RPM_PATH", "ARTIFACTORY_RPM_REPO")
    token = _env("ARTIFACTORY_TOKEN")
    resolved, missing = resolve_arch(pkgs, base_url, repo, token, args.arch)
    if missing:
        for m in missing:
            print(f"::error::could not resolve {m}", file=sys.stderr)
        sys.exit(1)
    for relpath, _ in resolved:
        print(relpath)


def cmd_validate(args):
    pkgs = load_lock(args.lock)
    base_url = _env("ARTIFACTORY_BASE_URL", "ARTIFACTORY_URL").rstrip("/")
    repo = _env("ARTIFACTORY_RPM_PATH", "ARTIFACTORY_RPM_REPO")
    token = _env("ARTIFACTORY_TOKEN")
    ok = True
    for name, version, tree in pkgs:
        row = f"{name} @ {version} (tree={tree or 'base'})"
        found = {}
        for arch in ARCHES:
            pat = _match_pattern(name, version, tree, arch)
            found[arch] = _aql_newest(base_url, repo, token, tree, arch, pat)
        if all(found.values()):
            print(f"OK   {row}")
        else:
            ok = False
            gaps = ", ".join(a for a in ARCHES if not found[a])
            print(f"FAIL {row} — missing on: {gaps}")
    if not ok:
        print(
            "::error::a pinned commit is not published for every arch; "
            "pick a commit present on both x86_64 and s390x.",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_names(args):
    for name, _, _ in load_lock(args.lock):
        print(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", default=DEFAULT_LOCK, help="path to spyre-rpms.lock")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("resolve", help="print repo-relative RPM paths for one arch")
    r.add_argument("--arch", required=True, choices=ARCHES)
    r.set_defaults(func=cmd_resolve)

    v = sub.add_parser("validate", help="assert every package resolves on both arches")
    v.set_defaults(func=cmd_validate)

    n = sub.add_parser("names", help="print package names (no network)")
    n.set_defaults(func=cmd_names)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
