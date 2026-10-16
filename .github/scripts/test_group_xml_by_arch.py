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

"""group_xml_by_arch.py: arch is a run_id hash input, so a misgrouped file is misattributed."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import group_xml_by_arch as gxa  # noqa: E402

_GROUP_SCRIPT = pathlib.Path(gxa.__file__)


def _xml(path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<testsuite/>")
    return path


def test_canonical_folds_x86_aliases():
    assert {gxa.canonical(a) for a in ("amd64", " X86 ", "x86-64", "x86_64")} == {"x86_64"}
    assert gxa.canonical("PPC64LE\n") == "ppc64le"


def test_grouping_by_sidecar_with_default_and_flattened_names(tmp_path):
    src, dest = tmp_path / "xml_artifacts", tmp_path / "xml_by_arch"
    _xml(src / "junit-x" / "junit-x.xml")  # no sidecar -> default arch
    _xml(src / "junit-p" / "junit-p.xml")
    (src / "junit-p" / "leg-arch-p.txt").write_text("ppc64le\n")
    _xml(src / "junit-s" / "sub" / "junit-s.xml")  # non-name-derived sidecar beside it
    (src / "junit-s" / "sub" / "leg-arch-other.txt").write_text("s390x")

    assert gxa.group(src, dest) == {"x86_64": 1, "ppc64le": 1, "s390x": 1}
    assert [p.name for p in (dest / "s390x").iterdir()] == ["junit-s__sub__junit-s.xml"]
    link = dest / "ppc64le" / "junit-p__junit-p.xml"
    assert link.is_symlink() and link.resolve() == (src / "junit-p" / "junit-p.xml").resolve()


def test_outputs_and_unknown_arch_warning(tmp_path):
    src = tmp_path / "xml_artifacts"
    _xml(src / "junit-r" / "junit-r.xml")
    (src / "junit-r" / "leg-arch-r.txt").write_text("riscv64")
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(_GROUP_SCRIPT),
            "--src-dir",
            str(src),
            "--dest-dir",
            str(tmp_path / "xml_by_arch"),
        ],
        env={**os.environ, "GITHUB_OUTPUT": str(out)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.read_text().splitlines() == [
        "has_x86_64=false",
        "has_ppc64le=false",
        "has_s390x=false",
    ]
    assert "::warning::" in result.stdout and "riscv64" in result.stdout
