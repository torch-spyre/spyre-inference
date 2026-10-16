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

"""Write a components.txt describing the Spyre RPMs actually extracted in CI.

torch-spyre keys its kernel cache on library versions read from
LIB_VERSION_FILE, so that file has to describe the RPMs CI installed rather than
the ones baked into the image.

Packages come from spyre-rpms.lock; versions come from the extracted RPM
filenames, not the lock, because the lock wildcards the build number on
ppc64le/s390x (`_*`).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from resolve_rpms import DEFAULT_LOCK, load_data, package_names  # noqa: E402


def _version_from_filename(filename: str, name: str, arch: str) -> str | None:
    m = re.fullmatch(rf"{re.escape(name)}-(.+)\.{re.escape(arch)}\.rpm", filename)
    return m.group(1) if m else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rpm-dir", required=True, help="directory holding the cached RPMs")
    p.add_argument("--arch", required=True, help="RPM arch suffix, e.g. x86_64")
    p.add_argument("--output", required=True, help="components.txt path to write")
    p.add_argument("--lock", default=DEFAULT_LOCK, help="path to spyre-rpms.lock")
    p.add_argument(
        "--merge",
        action="store_true",
        help="update only the components found in --rpm-dir, keeping the others",
    )
    args = p.parse_args()

    components = package_names(load_data(args.lock))
    if not components:
        sys.exit(f"::error::no [packages] entries in {args.lock!r}.")

    try:
        entries = os.listdir(args.rpm_dir)
    except FileNotFoundError:
        sys.exit(f"::error::RPM directory not found: {args.rpm_dir!r}")

    resolved = {}
    for name in components:
        # `-[0-9]` so ibm-deeptools does not also match ibm-deeptools-devel.
        for entry in sorted(entries):
            if not re.match(rf"{re.escape(name)}-[0-9]", entry):
                continue
            version = _version_from_filename(entry, name, args.arch)
            if version is not None:
                resolved[name] = version
                break

    if args.merge:
        existing = {}
        try:
            with open(args.output, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and ":" in line:
                        k, v = line.split(":", 1)
                        existing[k.strip()] = v.strip()
        except FileNotFoundError:
            sys.exit(
                f"::error::--merge given but {args.output!r} does not exist; "
                "run the baseline spyre-rpm-install step first."
            )
        if not resolved:
            print(
                f"No {args.arch} RPMs for tracked components in {args.rpm_dir!r}; "
                f"leaving {args.output} unchanged."
            )
            return
        existing.update(resolved)
        lines = [f"{n}:{existing[n]}" for n in components if n in existing]
    else:
        missing = [n for n in components if n not in resolved]
        if missing:
            sys.exit(
                f"::error::no extracted RPM found for {', '.join(missing)} "
                f"(arch {args.arch}) in {args.rpm_dir!r}; cannot write components.txt."
            )
        lines = [f"{n}:{resolved[n]}" for n in components]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.output}:")
    for line in lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
