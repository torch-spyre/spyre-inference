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

from __future__ import annotations

import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import relocate_spyre_libs as rl  # noqa: E402

LIB = (
    b"\x7fELF\x00/opt/ibm/spyre/senlib/lib\x00/opt/ibm/spyre/senlib/bin/data/boot_seq/init.txt\x00"
)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "libs"
    lib = root / "opt/ibm/spyre/senlib/lib"
    lib.mkdir(parents=True)
    (lib / "libsenlib.so").write_bytes(LIB)
    yield root
    link = os.path.dirname(rl.alias_for(str(root)))
    if os.path.islink(link):
        os.unlink(link)


def _alias(root):
    return rl.alias_for(str(root))


def test_alias_keeps_length_and_tail(tmp_path):
    alias = rl.alias_for(str(tmp_path))
    assert len(alias) == len(rl.OLD)
    assert alias.startswith("/tmp/") and alias.endswith("/spyre")


def test_alias_is_stable_per_root(tmp_path):
    assert rl.alias_for(str(tmp_path / "a")) == rl.alias_for(str(tmp_path / "a"))
    assert rl.alias_for(str(tmp_path / "a")) != rl.alias_for(str(tmp_path / "b"))


def test_rewrite_keeps_size_and_leaves_nothing(tree):
    so = tree / "opt/ibm/spyre/senlib/lib/libsenlib.so"
    alias = _alias(tree)
    assert rl.rewrite(str(tree), alias) == 2
    data = so.read_bytes()
    assert len(data) == len(LIB)
    assert rl.OLD not in data
    assert data.count(alias.encode()) == 2
    assert rl.leftovers(str(tree)) == []


def test_rewrite_is_idempotent(tree):
    alias = _alias(tree)
    rl.rewrite(str(tree), alias)
    assert rl.rewrite(str(tree), alias) == 0


def test_tail_immediate_survives(tree):
    so = tree / "opt/ibm/spyre/senlib/lib/libsenlib.so"
    so.write_bytes(LIB + b"\x48\xb9e/senlib")
    rl.rewrite(str(tree), _alias(tree))
    assert rl.leftovers(str(tree)) == []


def test_host_config_path_is_not_a_leftover(tree):
    so = tree / "opt/ibm/spyre/senlib/lib/libsenlib.so"
    so.write_bytes(LIB + b"/etc/ibm/spyre/topo.json\x00")
    rl.rewrite(str(tree), _alias(tree))
    assert rl.leftovers(str(tree)) == []


def test_head_immediate_is_reported(tree):
    so = tree / "opt/ibm/spyre/senlib/lib/libsenlib.so"
    so.write_bytes(LIB + b"\x48\xb8/opt/ibm\x48\xb9/spyre/s")
    rl.rewrite(str(tree), _alias(tree))
    assert rl.leftovers(str(tree)) == ["opt/ibm/spyre/senlib/lib/libsenlib.so"]


def test_link_points_at_opt_ibm(tree):
    alias = rl.alias_for(str(tree))
    rl.link(str(tree), alias)
    assert os.path.realpath(os.path.dirname(alias)) == str((tree / "opt/ibm").resolve())
    rl.link(str(tree), alias)


def test_link_refuses_a_foreign_target(tree, tmp_path):
    alias = rl.alias_for(str(tree))
    os.symlink(tmp_path, os.path.dirname(alias))
    with pytest.raises(SystemExit, match="does not point at"):
        rl.link(str(tree), alias)


def test_main_fails_without_tree(tmp_path):
    assert rl.main([str(tmp_path)]) == 1
