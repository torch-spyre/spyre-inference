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

"""CPU-only tests for the CI components.txt generator (no hardware needed).

The invariants: the versions describe the RPMs actually extracted, a package
never picks up a longer sibling's version (`-devel`), and `--merge` leaves
untouched components alone.

The cache-key tests feed the generated file to torch-spyre's own
`_get_dxp_version` and `code_hash` rather than re-deriving the key format here,
so they check the real contract: different RPMs, different key.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / ".github" / "scripts" / "write_components_file.py"

LOCK = """\
version = 2

[defaults]
tree = ""

[packages]
ibm-deeptools       = { version = "2.0.0-0.main.1+2429.561f418_335" }
ibm-deeptools-devel = { version = "2.0.0-0.main.1+2429.561f418_335" }
ibm-flex            = { version = "2.0.0-0.main.1+553.8581a91_384" }
"""

DEEPTOOLS_V = "2.0.0-0.main.1+2429.561f418_335.el10"
FLEX_V = "2.0.0-0.main.1+553.8581a91_384.el10"


def _run(tmp_path, rpms, *args, arch="x86_64", lock=LOCK):
    """Write a lock + fake RPM dir, run the script, return (proc, output text)."""
    lock_path = tmp_path / "spyre-rpms.lock"
    lock_path.write_text(lock)
    rpm_dir = tmp_path / "rpms"
    rpm_dir.mkdir(exist_ok=True)
    for name in rpms:
        (rpm_dir / name).touch()
    out = tmp_path / "out" / "components.txt"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--rpm-dir",
            str(rpm_dir),
            "--arch",
            arch,
            "--output",
            str(out),
            "--lock",
            str(lock_path),
            *args,
        ],
        capture_output=True,
        text=True,
    )
    return proc, (out.read_text() if out.exists() else None)


def _parse(text):
    return dict(line.split(":", 1) for line in text.strip().splitlines())


def _rpms(arch="x86_64"):
    return [
        f"ibm-deeptools-{DEEPTOOLS_V}.{arch}.rpm",
        f"ibm-deeptools-devel-{DEEPTOOLS_V}.{arch}.rpm",
        f"ibm-flex-{FLEX_V}.{arch}.rpm",
    ]


def test_versions_come_from_rpm_filenames(tmp_path):
    proc, text = _run(tmp_path, _rpms())
    assert proc.returncode == 0, proc.stderr
    assert _parse(text) == {
        "ibm-deeptools": DEEPTOOLS_V,
        "ibm-deeptools-devel": DEEPTOOLS_V,
        "ibm-flex": FLEX_V,
    }


def test_package_list_comes_from_the_lock(tmp_path):
    """A package absent from the lock is not written, even if its RPM is present."""
    lock = LOCK.replace(
        'ibm-deeptools-devel = { version = "2.0.0-0.main.1+2429.561f418_335" }\n', ""
    )
    proc, text = _run(tmp_path, _rpms(), lock=lock)
    assert proc.returncode == 0, proc.stderr
    assert "ibm-deeptools-devel" not in _parse(text)


def test_each_package_gets_its_own_version(tmp_path):
    """A package and its longer sibling resolve independently.

    `ibm-deeptools-devel` shares the `ibm-deeptools-` prefix, so a matcher that
    ignored the boundary would let one supply the other's version.
    """
    devel_v = "3.3.3-0.main.9+1.deadbee_1.el10"
    rpms = [
        f"ibm-deeptools-devel-{devel_v}.x86_64.rpm",
        f"ibm-deeptools-{DEEPTOOLS_V}.x86_64.rpm",
        f"ibm-flex-{FLEX_V}.x86_64.rpm",
    ]
    proc, text = _run(tmp_path, rpms)
    assert proc.returncode == 0, proc.stderr
    parsed = _parse(text)
    assert parsed["ibm-deeptools"] == DEEPTOOLS_V
    assert parsed["ibm-deeptools-devel"] == devel_v


def test_version_must_start_with_a_digit(tmp_path):
    """The `-[0-9]` anchor: only a version-looking suffix may supply a version.

    A stray sibling that sorts before the real RPM (any char below `0` after the
    prefix) must not be picked up. Without the anchor `ibm-deeptools` resolves to
    `-extra-1.0`, silently poisoning the cache key.
    """
    rpms = [
        "ibm-deeptools--extra-1.0.el10.x86_64.rpm",
        f"ibm-deeptools-{DEEPTOOLS_V}.x86_64.rpm",
        f"ibm-deeptools-devel-{DEEPTOOLS_V}.x86_64.rpm",
        f"ibm-flex-{FLEX_V}.x86_64.rpm",
    ]
    proc, text = _run(tmp_path, rpms)
    assert proc.returncode == 0, proc.stderr
    assert _parse(text)["ibm-deeptools"] == DEEPTOOLS_V


def test_missing_package_is_a_hard_error(tmp_path):
    """A lock package with no extracted RPM must fail, not write a partial file."""
    proc, text = _run(tmp_path, [f"ibm-flex-{FLEX_V}.x86_64.rpm"])
    assert proc.returncode != 0
    assert "ibm-deeptools" in proc.stderr
    assert text is None


def test_merge_updates_only_the_overridden_package(tmp_path):
    base_proc, _ = _run(tmp_path, _rpms())
    assert base_proc.returncode == 0, base_proc.stderr

    new_flex = "3.1.0-0.main.7+9999.abcdef1_777.el10"
    lock_path = tmp_path / "spyre-rpms.lock"
    override = tmp_path / "override"
    override.mkdir()
    (override / f"ibm-flex-{new_flex}.x86_64.rpm").touch()
    out = tmp_path / "out" / "components.txt"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--rpm-dir",
            str(override),
            "--arch",
            "x86_64",
            "--output",
            str(out),
            "--lock",
            str(lock_path),
            "--merge",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    parsed = _parse(out.read_text())
    assert parsed["ibm-flex"] == new_flex
    assert parsed["ibm-deeptools"] == DEEPTOOLS_V


def test_merge_without_a_baseline_fails(tmp_path):
    proc, _ = _run(tmp_path, _rpms(), "--merge")
    assert proc.returncode != 0
    assert "--merge" in proc.stderr


def test_merge_with_no_matching_arch_rpms_is_a_noop(tmp_path):
    base_proc, before = _run(tmp_path, _rpms())
    assert base_proc.returncode == 0, base_proc.stderr
    out = tmp_path / "out" / "components.txt"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--rpm-dir",
            str(tmp_path / "rpms"),
            "--arch",
            "s390x",
            "--output",
            str(out),
            "--lock",
            str(tmp_path / "spyre-rpms.lock"),
            "--merge",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.read_text() == before


def test_real_lock_parses(tmp_path):
    """The checked-in lock must be readable by the script's own loader."""
    sys.path.insert(0, str(SCRIPT.parent))
    from resolve_rpms import load_data, package_names

    names = package_names(load_data(str(SCRIPT.parent.parent.parent / "spyre-rpms.lock")))
    assert "ibm-deeptools" in names and "ibm-flex" in names


# The cache-key tests below drive torch-spyre directly; a venv older than the
# torch-spyre pin lacks this module. Guard those two tests only -- everything
# above exercises the script alone and must still run.
requires_kernel_cache = pytest.mark.skipif(
    importlib.util.find_spec("torch_spyre.execution.kernel_cache") is None,
    reason="installed torch-spyre predates execution.kernel_cache; run `uv sync`",
)


def _cache_key_for(components_path, monkeypatch):
    """torch-spyre's own cache key for a components.txt, kernel content fixed."""
    from torch._inductor.codecache import code_hash
    from torch_spyre.execution.kernel_cache import _get_dxp_version

    monkeypatch.setenv("LIB_VERSION_FILE", str(components_path))
    _get_dxp_version.cache_clear()  # lru_cache would pin the first file read
    return code_hash(b"identical-kernel-content", extra=_get_dxp_version())


@requires_kernel_cache
def test_two_rpm_sets_give_torch_spyre_two_cache_keys(tmp_path, monkeypatch):
    """Two different RPM sets must produce two different torch-spyre cache keys.

    This is the whole point of the script: an identical kernel compiled against
    different deeptools/flex builds must not collide in the cache. Rather than
    re-deriving the key format here, feed each generated file to torch-spyre's
    own `_get_dxp_version` and `code_hash` and require the results to differ.
    """
    old_flex = "2.0.0-0.main.1+553.8581a91_384.el10"
    new_flex = "3.1.0-0.main.7+9999.abcdef1_777.el10"

    keys = []
    for tag, flex_v in (("a", old_flex), ("b", new_flex)):
        case = tmp_path / tag
        case.mkdir()
        rpms = [
            f"ibm-deeptools-{DEEPTOOLS_V}.x86_64.rpm",
            f"ibm-deeptools-devel-{DEEPTOOLS_V}.x86_64.rpm",
            f"ibm-flex-{flex_v}.x86_64.rpm",
        ]
        proc, text = _run(case, rpms)
        assert proc.returncode == 0, proc.stderr
        assert flex_v in text

        # Kernel content is fixed, so only the components file can move the key.
        keys.append(_cache_key_for(case / "out" / "components.txt", monkeypatch))

    assert keys[0] != keys[1], (
        "same cache key for different ibm-flex builds: a stale kernel would be "
        "reused across compiler versions"
    )


@requires_kernel_cache
def test_same_rpms_give_torch_spyre_a_stable_cache_key(tmp_path, monkeypatch):
    """The converse: unchanged RPMs must not invalidate the cache."""
    keys = []
    for tag in ("a", "b"):
        case = tmp_path / tag
        case.mkdir()
        proc, _ = _run(case, _rpms())
        assert proc.returncode == 0, proc.stderr
        keys.append(_cache_key_for(case / "out" / "components.txt", monkeypatch))

    assert keys[0] == keys[1], "identical RPM sets produced different cache keys"
