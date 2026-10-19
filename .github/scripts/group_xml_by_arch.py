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

"""Symlinks JUnit XML into <dest>/<leg arch>/, flattened, for the per-arch v2 ingest."""

import argparse
import os
from pathlib import Path

DEFAULT_ARCH = "x86_64"
KNOWN_ARCHES = ("x86_64", "ppc64le", "s390x")


def canonical(arch: str) -> str:
    a = arch.strip().lower()
    return "x86_64" if a in ("amd64", "x86", "x86-64", "x86_64") else a


def leg_arch(xml_path: Path) -> str:
    # A name-derived sidecar first, then any leg-arch-*.txt beside the XML.
    sidecar = xml_path.with_name(xml_path.stem.replace("junit-", "leg-arch-") + ".txt")
    for candidate in (sidecar, *sorted(xml_path.parent.glob("leg-arch-*.txt"))):
        try:
            arch = candidate.read_text().strip()
        except OSError:
            continue
        if arch:
            return arch
    return ""


def group(src: Path, dest: Path) -> dict[str, int]:
    groups: dict[str, int] = {}
    for xml_path in sorted(src.rglob("*.xml")):
        arch = canonical(leg_arch(xml_path) or DEFAULT_ARCH)
        dest_dir = dest / arch
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Flattened with the relative path as filename: the action globs non-recursively.
        rel = xml_path.relative_to(src)
        (dest_dir / str(rel).replace(os.sep, "__")).symlink_to(xml_path.resolve())
        groups[arch] = groups.get(arch, 0) + 1
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", default="xml_artifacts", type=Path)
    parser.add_argument("--dest-dir", default="xml_by_arch", type=Path)
    args = parser.parse_args()

    groups = group(args.src_dir, args.dest_dir)
    print("Grouped XML files by leg arch:")
    for arch, n in sorted(groups.items()):
        print(f"  {arch}: {n} file(s)")

    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            for arch in KNOWN_ARCHES:
                present = "true" if groups.get(arch, 0) > 0 else "false"
                f.write(f"has_{arch}={present}\n")

    unknown = sorted(set(groups) - set(KNOWN_ARCHES))
    if unknown:
        print(
            f"::warning::XML grouped under unrecognised arch(es) {unknown}; v2 will not be "
            "written for them -- add to KNOWN_ARCHES in group_xml_by_arch.py and a matching "
            "ingest step in push-to-clickhouse.yaml."
        )


if __name__ == "__main__":
    main()
