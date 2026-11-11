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

"""Point a user-space Spyre RPM tree's hardcoded /opt/ibm/spyre paths at itself.

senlib bakes /opt/ibm/spyre into its binaries (boot sequence, configs, RPATH), so an
rpm2cpio tree still reads the image's files. Each occurrence becomes a same-length
/tmp/XYZ/spyre alias, and /tmp/XYZ links to the tree's opt/ibm.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

OLD = b"/opt/ibm/spyre"
ALIAS_DIR = "/tmp"
_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
# Compilers store path tails as 8-byte movabs immediates (libsenlib: "e/senlib"), so
# the alias keeps "/spyre" and any immediate still holding "opt/" is reported.
_FRAGMENTS = [OLD[i : i + 8] for i in range(4)]


def alias_for(root: str) -> str:
    digest = hashlib.sha256(os.path.realpath(root).encode()).digest()
    tag = "".join(_ALPHABET[b % len(_ALPHABET)] for b in digest[:3])
    alias = f"{ALIAS_DIR}/{tag}/spyre"
    assert len(alias) == len(OLD) and alias.encode()[8:] == OLD[8:], alias
    return alias


def _files(root: str):
    for dirpath, _, names in os.walk(root):
        for name in names:
            path = os.path.join(dirpath, name)
            if not os.path.islink(path) and os.path.isfile(path):
                yield path


def rewrite(root: str, alias: str) -> int:
    new = alias.encode()
    total = 0
    for path in _files(root):
        with open(path, "rb") as f:
            data = f.read()
        n = data.count(OLD)
        if n:
            st = os.stat(path)
            with open(path, "r+b") as f:
                f.write(data.replace(OLD, new))
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
            total += n
    return total


def leftovers(root: str) -> list[str]:
    bad = []
    for path in _files(root):
        with open(path, "rb") as f:
            data = f.read()
        if any(frag in data for frag in _FRAGMENTS):
            bad.append(os.path.relpath(path, root))
    return bad


def link(root: str, alias: str) -> None:
    target = os.path.realpath(os.path.join(root, "opt", "ibm"))
    link_path = os.path.dirname(alias)
    try:
        os.symlink(target, link_path)
    except FileExistsError:
        if os.path.realpath(link_path) != target:
            raise SystemExit(
                f"{link_path} exists and does not point at {target}; remove it "
                "or install into another prefix"
            ) from None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", help="directory the RPMs were unpacked into (holds opt/ibm/spyre)")
    args = ap.parse_args(argv)

    if not os.path.isdir(os.path.join(args.root, "opt", "ibm", "spyre")):
        print(f"{args.root} has no opt/ibm/spyre", file=sys.stderr)
        return 1
    alias = alias_for(args.root)
    n = rewrite(args.root, alias)
    bad = leftovers(args.root)
    if bad:
        print(
            f"{OLD.decode()} survives as a partial path in {len(bad)} file(s); "
            "a new build assembles it in a way this rewrite cannot reach:",
            file=sys.stderr,
        )
        for path in bad:
            print(f"  {path}", file=sys.stderr)
        return 1
    print(f"rewrote {n} occurrence(s) of {OLD.decode()} -> {alias}", file=sys.stderr)
    link(args.root, alias)
    print(alias)
    return 0


if __name__ == "__main__":
    sys.exit(main())
