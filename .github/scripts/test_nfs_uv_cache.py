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

"""Logic tests for the NFS-backed uv cache helper: key, scope rules, atomic
save, cross-scope isolation, and GC. The tar roundtrip runs on real files.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nfs_uv_cache as nuc  # noqa: E402


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def lockfiles(tmp_path):
    a = _write(tmp_path / "uv.lock", "lock-a\n")
    b = _write(tmp_path / "spyre-rpms.lock", "rpm-b\n")
    return [a, b]


def test_key_is_deterministic(lockfiles):
    assert nuc.compute_key(lockfiles) == nuc.compute_key(lockfiles)
    assert len(nuc.compute_key(lockfiles)) == nuc.KEY_LEN


def test_key_tracks_content_and_order(tmp_path):
    a = _write(tmp_path / "a.lock", "one")
    b = _write(tmp_path / "b.lock", "two")
    base = nuc.compute_key([a, b])
    _write(tmp_path / "a.lock", "one-changed")
    assert nuc.compute_key([a, b]) != base
    _write(tmp_path / "a.lock", "one")
    assert nuc.compute_key([b, a]) != base


def test_key_salt_busts_everything(lockfiles):
    base = nuc.compute_key(lockfiles)
    assert nuc.compute_key(lockfiles, salt="nfs-uv-cache-v2") != base


@pytest.mark.parametrize(
    "event,ref,pr,expected",
    [
        ("push", "refs/heads/main", "", "main"),
        ("push", "refs/heads/feature", "", None),
        ("pull_request", "refs/pull/7/merge", "7", "pr-7"),
        ("pull_request", "refs/pull/7/merge", "", None),
        ("merge_group", "refs/heads/gh-readonly-queue/main/x", "", None),
        ("workflow_dispatch", "refs/heads/main", "", None),
        ("schedule", "refs/heads/main", "", None),
    ],
)
def test_write_scope(event, ref, pr, expected):
    assert nuc.write_scope(event, ref, pr) == expected


def test_read_scopes_pr_prefers_own_then_main():
    assert nuc.read_scopes("pull_request", "refs/pull/9/merge", "9") == ["pr-9", "main"]


@pytest.mark.parametrize(
    "event,ref,pr",
    [
        ("push", "refs/heads/main", ""),
        ("merge_group", "refs/heads/gh-readonly-queue/main/x", ""),
        ("workflow_dispatch", "refs/heads/main", ""),
        ("pull_request", "refs/pull/9/merge", ""),  # no number -> main only
    ],
)
def test_read_scopes_non_pr_is_main_only(event, ref, pr):
    assert nuc.read_scopes(event, ref, pr) == ["main"]


def _args(cmd, nfs, local, keyfiles, arch="x86_64", **extra):
    argv = [cmd, "--nfs-root", str(nfs), "--local-dir", str(local), "--arch", arch]
    for kf in keyfiles:
        argv += ["--key-file", kf]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


def _populate(cache_dir):
    (cache_dir / "wheels").mkdir(parents=True, exist_ok=True)
    (cache_dir / "wheels" / "pkg.whl").write_bytes(b"PK\x03\x04 fake wheel")
    (cache_dir / "marker.txt").write_text("built")


def _push_main():
    return {"event_name": "push", "ref": "refs/heads/main"}


def _pr(n):
    return {"event_name": "pull_request", "ref": f"refs/pull/{n}/merge", "pr_number": n}


def test_roundtrip_ship_wipe_restore_main(tmp_path, lockfiles):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)

    assert nuc.main(_args("save", nfs, local, lockfiles, **_push_main())) == 0
    key = nuc.compute_key(lockfiles)
    assert (nfs / "main" / "x86_64" / f"{key}.tar").is_file()

    shutil.rmtree(local)
    gh_env = tmp_path / "env"
    gh_out = tmp_path / "out"
    assert (
        nuc.main(
            _args(
                "restore",
                nfs,
                local,
                lockfiles,
                github_env=gh_env,
                github_output=gh_out,
                **_push_main(),
            )
        )
        == 0
    )
    assert (local / "wheels" / "pkg.whl").read_bytes() == b"PK\x03\x04 fake wheel"
    assert f"UV_CACHE_DIR={local}" in gh_env.read_text()
    assert "cache-hit=true" in gh_out.read_text()
    assert f"cache-key={key}" in gh_out.read_text()


def test_pr_writes_isolated_scope_main_cannot_read(tmp_path):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    pr_lock = [_write(tmp_path / "uv.lock", "pr-only-lock")]
    _populate(local)
    assert nuc.main(_args("save", nfs, local, pr_lock, **_pr("42"))) == 0
    key = nuc.compute_key(pr_lock)
    assert (nfs / "pr-42" / "x86_64" / f"{key}.tar").is_file()
    assert not (nfs / "main").exists()

    shutil.rmtree(local)
    gh_out = tmp_path / "out"
    assert (
        nuc.main(_args("restore", nfs, local, pr_lock, github_output=gh_out, **_push_main())) == 0
    )
    assert "cache-hit=false" in gh_out.read_text()
    assert not (local / "marker.txt").exists()


def test_pr_reads_own_scope_as_hit(tmp_path):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    pr_lock = [_write(tmp_path / "uv.lock", "pr-lock")]
    _populate(local)
    assert nuc.main(_args("save", nfs, local, pr_lock, **_pr("42"))) == 0

    shutil.rmtree(local)
    gh_out = tmp_path / "out"
    assert nuc.main(_args("restore", nfs, local, pr_lock, github_output=gh_out, **_pr("42"))) == 0
    assert "cache-hit=true" in gh_out.read_text()
    assert (local / "marker.txt").exists()


def test_pr_falls_back_to_main_scope(tmp_path):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    lock = [_write(tmp_path / "uv.lock", "shared")]
    _populate(local)
    assert nuc.main(_args("save", nfs, local, lock, **_push_main())) == 0

    shutil.rmtree(local)
    gh_out = tmp_path / "out"
    assert nuc.main(_args("restore", nfs, local, lock, github_output=gh_out, **_pr("7"))) == 0
    assert "cache-hit=true" in gh_out.read_text()  # falls back to main
    assert (local / "marker.txt").exists()

    _populate(local)
    assert nuc.main(_args("save", nfs, local, lock, **_pr("7"))) == 0
    assert not (nfs / "pr-7").exists()


def test_save_skipped_in_readonly_context(tmp_path, lockfiles):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)
    assert (
        nuc.main(
            _args(
                "save",
                nfs,
                local,
                lockfiles,
                event_name="merge_group",
                ref="refs/heads/gh-readonly-queue/main/x",
            )
        )
        == 0
    )
    assert not (nfs / "main").exists()


def test_save_idempotent_when_key_present(tmp_path, lockfiles):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)
    assert nuc.main(_args("save", nfs, local, lockfiles, **_push_main())) == 0
    key = nuc.compute_key(lockfiles)
    tar = nfs / "main" / "x86_64" / f"{key}.tar"
    first = tar.stat().st_mtime_ns
    (local / "marker.txt").write_text("DIFFERENT")
    assert nuc.main(_args("save", nfs, local, lockfiles, **_push_main())) == 0
    assert tar.stat().st_mtime_ns == first


def test_corrupt_tar_restores_cold(tmp_path, lockfiles):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    key = nuc.compute_key(lockfiles)
    (nfs / "main" / "x86_64").mkdir(parents=True)
    (nfs / "main" / "x86_64" / f"{key}.tar").write_bytes(b"not a tar at all")
    (local / "stale").mkdir(parents=True)
    gh_out = tmp_path / "out"
    assert (
        nuc.main(_args("restore", nfs, local, lockfiles, github_output=gh_out, **_push_main())) == 0
    )
    assert not (local / "stale").exists()
    assert "cache-hit=false" in gh_out.read_text()


def test_gc_keeps_newest_and_protects_target(tmp_path):
    arch_dir = tmp_path / "main" / "x86_64"
    arch_dir.mkdir(parents=True)
    tars = []
    for i in range(5):
        t = arch_dir / f"k{i}.tar"
        t.write_bytes(b"x")
        os.utime(t, (time.time() + i, time.time() + i))
        tars.append(t)
    nuc._gc(arch_dir, keep=2, protect=tars[0])
    survivors = {p.name for p in arch_dir.iterdir()}
    assert survivors == {"k4.tar", "k3.tar", "k0.tar"}


def test_sweep_evicts_stale_pr_scopes_only(tmp_path):
    nfs = tmp_path / "nfs"
    old = nfs / "pr-1" / "x86_64"
    fresh = nfs / "pr-2" / "x86_64"
    main = nfs / "main" / "x86_64"
    for d in (old, fresh, main):
        d.mkdir(parents=True)
        (d / "k.tar").write_bytes(b"x")
    stale = time.time() - 30 * 86400
    os.utime(old / "k.tar", (stale, stale))
    os.utime(main / "k.tar", (stale, stale))  # main is never swept by ttl
    nuc._sweep_pr_scopes(nfs, ttl_days=14)
    assert not (old / "k.tar").exists()
    assert (fresh / "k.tar").exists()
    assert (main / "k.tar").exists()


def test_partial_restore_from_newest_tar(tmp_path):
    """A new key with no exact tar extracts the scope's newest tar: warm, not a hit."""
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)
    assert (
        nuc.main(_args("save", nfs, local, [_write(tmp_path / "uv.lock", "v1")], **_push_main()))
        == 0
    )

    shutil.rmtree(local)
    gh_out = tmp_path / "out"
    assert (
        nuc.main(
            _args(
                "restore",
                nfs,
                local,
                [_write(tmp_path / "uv.lock", "v2")],
                github_output=gh_out,
                **_push_main(),
            )
        )
        == 0
    )
    text = gh_out.read_text()
    assert "cache-hit=false" in text
    assert "restored-from=main/" in text
    assert (local / "marker.txt").exists()


@pytest.mark.parametrize("compress", ["gzip", "zstd"])
def test_compression_roundtrip(tmp_path, lockfiles, compress):
    if compress == "zstd" and shutil.which("zstd") is None:
        pytest.skip("zstd binary not installed")
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)
    assert nuc.main(_args("save", nfs, local, lockfiles, compress=compress, **_push_main())) == 0
    ext = nuc.COMPRESS[compress][1]
    assert (nfs / "main" / "x86_64" / f"{nuc.compute_key(lockfiles)}{ext}").is_file()

    shutil.rmtree(local)
    gh_out = tmp_path / "out"
    assert (
        nuc.main(_args("restore", nfs, local, lockfiles, github_output=gh_out, **_push_main())) == 0
    )
    assert "cache-hit=true" in gh_out.read_text()
    assert (local / "marker.txt").read_text() == "built"


def test_key_is_path_independent(tmp_path):
    """Byte-identical lockfiles at different absolute paths must hash the same."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ka = _write(tmp_path / "a" / "uv.lock", "same-bytes")
    kb = _write(tmp_path / "b" / "uv.lock", "same-bytes")
    assert nuc.compute_key([ka]) == nuc.compute_key([kb])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
