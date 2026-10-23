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

"""Report Spyre libraries and packages this process loads from the image's /opt/ibm/spyre
instead of the locked tree at $SENTIENT_BASE_INSTALL_DIR. Runs one device op first so
lazily loaded libraries are mapped. Findings are warnings unless --strict.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

IMAGE_ROOT = "/opt/ibm/spyre"
SPYRE_LIB_MARKERS = ("/ibm/spyre/", "/spyre-libs/", "torch_spyre", "senlib", "flex")


def mapped_libraries(maps_text: str) -> list[str]:
    paths = set()
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue  # anonymous mapping
        path = fields[5].removesuffix(" (deleted)")
        if path.startswith("/") and ".so" in os.path.basename(path):
            paths.add(path)
    return sorted(paths)


def leaked(path: str, locked_root: str) -> bool:
    """True if `path` is in the image's Spyre tree rather than the locked one."""
    real = os.path.realpath(path)
    in_image = any(p == IMAGE_ROOT or p.startswith(IMAGE_ROOT + "/") for p in (path, real))
    in_lock = bool(locked_root) and any(p.startswith(locked_root + "/") for p in (path, real))
    return in_image and not in_lock


def module_origins(names) -> dict[str, str | None]:
    origins = {}
    for name in names:
        mod = sys.modules.get(name)
        if mod is not None:
            origins[name] = getattr(mod, "__file__", None)
            continue
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        origins[name] = spec.origin if spec else None
    return origins


def touch_device():
    import torch
    import torch_spyre  # noqa: F401

    x = torch.ones(64, 64, dtype=torch.float16, device="spyre")
    return (x + x).cpu()[0, 0].item()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strict", action="store_true", help="exit 1 if anything loads from the image")
    p.add_argument("--maps", help="read this maps file instead of probing the device (for tests)")
    args = p.parse_args()

    locked_root = os.path.realpath(os.environ.get("SENTIENT_BASE_INSTALL_DIR", ""))
    if locked_root == os.path.realpath(IMAGE_ROOT):
        locked_root = ""
    findings = []
    print(f"SENTIENT_BASE_INSTALL_DIR = {os.environ.get('SENTIENT_BASE_INSTALL_DIR', '<unset>')}")
    for var in ("SENLIB_INSTALL_DIR", "LIB_VERSION_FILE", "AIU_AUTOGEN_SENLIB_CONFIG_FILE"):
        value = os.environ.get(var)
        print(f"{var} = {value or '<unset>'}")
        if value and leaked(value, locked_root):
            findings.append(f"{var} points into the image: {value}")

    if args.maps:
        maps_path = args.maps
    else:
        # A mismatched stack fails here; what it mapped before failing is still worth reporting.
        try:
            print(f"device op result: {touch_device()}")
        except Exception as e:
            findings.append(f"device op failed; lazy libraries may be missing: {e!r}"[:500])
        maps_path = "/proc/self/maps"
    with open(maps_path) as f:
        maps_text = f.read()

    print("\n--- mapped Spyre libraries")
    for path in mapped_libraries(maps_text):
        if not any(m in path for m in SPYRE_LIB_MARKERS):
            continue
        bad = leaked(path, locked_root)
        print(f"{'IMAGE ' if bad else 'ok    '} {path}")
        if bad:
            findings.append(f"library loaded from the image: {path}")

    print("\n--- Python package origins")
    for name, origin in module_origins(["torch_spyre", "senlib"]).items():
        print(f"       {name}: {origin or '<not importable>'}")
        if origin and leaked(origin, locked_root):
            findings.append(f"Python package {name} resolves to the image: {origin}")

    if not locked_root:
        findings.append("SENTIENT_BASE_INSTALL_DIR is unset or the image tree; nothing to compare")

    print()
    level = "error" if args.strict else "warning"
    for f in findings:
        print(f"::{level}::{f}")
    print(f"{len(findings)} finding(s)")
    sys.exit(1 if args.strict and findings else 0)


if __name__ == "__main__":
    main()
