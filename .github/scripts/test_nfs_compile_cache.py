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

"""Logic tests for the NFS-backed compile cache helper: delta save, union restore,
scope isolation and key eviction, on real tars."""

from __future__ import annotations

import os
import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nfs_compile_cache as ncc  # noqa: E402

MAIN = {"event_name": "push", "ref": "refs/heads/main"}
MQ = {"event_name": "merge_group", "ref": "refs/heads/gh-readonly-queue/main/x"}


def _pr(n):
    return {"event_name": "pull_request", "ref": f"refs/pull/{n}/merge", "pr_number": n}


@pytest.fixture
def env(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text("lock")
    return {"nfs": tmp_path / "nfs", "lock": str(lock), "tmp": tmp_path}


def _flags(ctx):
    out = []
    for k, v in ctx.items():
        out += [f"--{k.replace('_', '-')}", str(v)]
    return out


def _restore(env, job, ctx):
    local = env["tmp"] / job
    out = env["tmp"] / f"{job}.out"
    argv = ["restore", "--nfs-root", str(env["nfs"]), "--local-dir", str(local)]
    argv += ["--arch", "x86_64", "--key-file", env["lock"], "--github-output", str(out)]
    argv += ["--manifest", str(env["tmp"] / f"{job}.manifest")]
    assert ncc.main(argv + _flags(ctx)) == 0
    key = dict(line.split("=", 1) for line in out.read_text().splitlines())["cache-key"]
    return local, key


def _save(env, job, ctx, key, *extra):
    argv = ["save", "--nfs-root", str(env["nfs"]), "--local-dir", str(env["tmp"] / job)]
    argv += ["--arch", "x86_64", "--key", key, "--name", job]
    argv += ["--manifest", str(env["tmp"] / f"{job}.manifest"), *extra]
    assert ncc.main(argv + _flags(ctx)) == 0


def _put(root, rel, text="x"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _job(env, job, ctx, files):
    local, key = _restore(env, job, ctx)
    for rel in files:
        _put(local, rel, rel)
    _save(env, job, ctx, key)
    return key


@pytest.mark.parametrize(
    "rel,shippable",
    [
        ("fxgraph/ab/abc/entry", True),
        ("aotautograd/abc/entry", True),
        ("ab/cabcdef.py", True),
        ("inductor-spyre-cache/k1/bundle.mlir", True),
        ("inductor-spyre-cache/k1.tmp.0f00/bundle.mlir", False),
        ("inductor-spyre-cache/failed/k1/bundle.mlir", False),
        ("fxgraph/ab/abc/.123.456.tmp", False),
        ("inductor-spyre/abcd_sdsc_x/bundle.mlir", False),
        ("locks/abc.lock", False),
        ("precompiled_headers/x.gch", False),
    ],
)
def test_shippable(rel, shippable):
    assert ncc._shippable(pathlib.PurePosixPath(rel)) is shippable


def test_shards_union_through_deltas(env):
    _job(env, "a", MAIN, ["fxgraph/a", "inductor-spyre-cache/ka/bundle.mlir"])
    _job(env, "b", MAIN, ["fxgraph/b", "locks/b.lock"])
    local, _ = _restore(env, "c", MAIN)
    assert ncc.list_files(local) == {
        "fxgraph/a",
        "fxgraph/b",
        "inductor-spyre-cache/ka/bundle.mlir",
    }


def test_save_ships_only_new_files(env):
    _job(env, "a", MAIN, ["fxgraph/a"])
    local, key = _restore(env, "b", MAIN)
    _put(local, "fxgraph/b")
    _save(env, "b", MAIN, key)
    key_dir = env["nfs"] / "main" / "x86_64" / key
    assert ncc.list_files(_unpack(env, key_dir / "delta-b.tar")) == {"fxgraph/b"}


def _unpack(env, tar):
    dest = env["tmp"] / f"unpack-{tar.name}"
    dest.mkdir()
    ncc._extract(tar, dest)
    return dest


def test_nothing_new_writes_nothing(env):
    key = _job(env, "a", MAIN, ["fxgraph/a"])
    _, key = _restore(env, "b", MAIN)
    _save(env, "b", MAIN, key)
    assert not (env["nfs"] / "main" / "x86_64" / key / "delta-b.tar").exists()


def test_pr_writes_own_scope_and_reads_main(env):
    _job(env, "main", MAIN, ["fxgraph/m"])
    _job(env, "pr", _pr(7), ["fxgraph/p"])
    assert ncc.list_files(_restore(env, "pr-again", _pr(7))[0]) == {"fxgraph/m", "fxgraph/p"}
    assert ncc.list_files(_restore(env, "other-pr", _pr(8))[0]) == {"fxgraph/m"}
    assert ncc.list_files(_restore(env, "main-again", MAIN)[0]) == {"fxgraph/m"}


def test_merge_queue_writes_main(env):
    _job(env, "mq", MQ, ["fxgraph/q"])
    assert ncc.list_files(_restore(env, "push", MAIN)[0]) == {"fxgraph/q"}
    assert ncc.list_files(_restore(env, "pr", _pr(7))[0]) == {"fxgraph/q"}


def test_merge_queue_for_another_branch_writes_nothing(env):
    other = {"event_name": "merge_group", "ref": "refs/heads/gh-readonly-queue/release/x"}
    _job(env, "mq", other, ["fxgraph/q"])
    assert ncc.list_files(_restore(env, "push", MAIN)[0]) == set()


def test_lockfile_change_starts_cold(env):
    _job(env, "a", MAIN, ["fxgraph/a"])
    pathlib.Path(env["lock"]).write_text("bumped")
    local, _ = _restore(env, "b", MAIN)
    assert ncc.list_files(local) == set()


def _cleanup(env, *extra):
    assert ncc.main(["cleanup", "--nfs-root", str(env["nfs"]), *extra]) == 0


def _age(key_dir, minutes):
    old = time.time() - minutes * 60
    for p in key_dir.iterdir():
        os.utime(p, (old, old))


def _key_for(env, i):
    pathlib.Path(env["lock"]).write_text(f"lock-{i}")
    local, key = _restore(env, f"j{i}", MAIN)
    _put(local, f"fxgraph/{i}")
    _save(env, f"j{i}", MAIN, key)
    return env["nfs"] / "main" / "x86_64" / key


def test_cleanup_keeps_newest_keys(env):
    dirs = [_key_for(env, i) for i in range(3)]
    for i, d in enumerate(dirs):
        _age(d, 10 - i)
    _cleanup(env, "--keep-keys", "2")
    assert [d.exists() for d in dirs] == [False, True, True]


def test_rerun_of_old_commit_does_not_evict_current_key(env):
    old, current = _key_for(env, 0), _key_for(env, 1)
    _age(old, 20)
    _age(current, 10)
    # A re-run of the old commit adds a fresh delta to the old key.
    pathlib.Path(env["lock"]).write_text("lock-0")
    local, key = _restore(env, "rerun", MAIN)
    _put(local, "fxgraph/rerun")
    _save(env, "rerun", MAIN, key)
    _cleanup(env, "--keep-keys", "1")
    assert current.exists() and not old.exists()


def test_save_never_evicts(env):
    for i in range(4):
        _key_for(env, i)
    assert len(list((env["nfs"] / "main" / "x86_64").iterdir())) == 4


def test_bad_delta_is_skipped_and_set_aside(env):
    key = _job(env, "a", MAIN, ["fxgraph/a"])
    key_dir = env["nfs"] / "main" / "x86_64" / key
    (key_dir / "delta-zz.tar").write_text("not a tar")
    _job(env, "b", MAIN, ["fxgraph/b"])
    local, _ = _restore(env, "c", MAIN)
    assert ncc.list_files(local) == {"fxgraph/a", "fxgraph/b"}
    assert not (key_dir / "delta-zz.tar").exists()
    assert (key_dir / ".bad-delta-zz.tar").exists()
    # Later jobs no longer see it, so they stay warm and write nothing new.
    _restore(env, "d", MAIN)
    assert sorted(p.name for p in key_dir.glob("delta-*.tar")) == ["delta-a.tar", "delta-b.tar"]


def test_stale_pr_scope_swept_by_cleanup(env):
    key = _job(env, "pr", _pr(7), ["fxgraph/p"])
    key_dir = env["nfs"] / "pr-7" / "x86_64" / key
    _age(key_dir, 15 * 24 * 60)
    _job(env, "main", MAIN, ["fxgraph/m"])
    assert (key_dir / "delta-pr.tar").exists()
    _cleanup(env)
    assert not key_dir.exists()
