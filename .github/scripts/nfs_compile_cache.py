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

"""Share the Inductor + Spyre kernel compile cache across CI jobs via the NFS mount.

Unlike the uv cache, each shard compiles different graphs, so one tar per key
would keep only the first writer's. Every cache we ship is content-addressed and
write-once, so instead each job writes only the files it compiled as its own
delta tar, and restore extracts every delta under the key:

    <nfs-root>/<scope>/<arch>/<key>/delta-<run>-<attempt>-<slot>.tar

Key and scopes are nfs_uv_cache.py's, so a lockfile change starts a new key,
except that the merge queue also writes `main`: its commit is exactly what lands
there, and a lockfile bump otherwise compiles everything cold twice (queue, then
the push). Best-effort: on any failure the job compiles cold.
"""

import argparse
import contextlib
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nfs_uv_cache import (  # noqa: E402
    MAIN_SCOPE,
    _extract,
    _log,
    _reset_dir,
    _sweep_pr_scopes,
    _warn,
    _write_kv,
    compute_key,
    read_scopes,
    write_scope,
)

# Bump to start every scope from an empty cache.
KEY_SALT = "nfs-compile-cache-v1"

# Per-compile scratch, lock files and host-specific precompiled headers; none is
# a cache entry another job can use. Inductor's CPU kernels (*.so) do ship: they
# are built with -march=native and keyed on the vector ISA only (see CI.md).
EXCLUDED_TOP_DIRS = {"inductor-spyre", "locks", "precompiled_headers"}
FAILED_KERNELS = ("inductor-spyre-cache", "failed")


def _key_dir(nfs_root, scope, arch, key):
    return Path(nfs_root) / scope / arch / key


def _shippable(rel):
    parts = rel.parts
    if parts[0] in EXCLUDED_TOP_DIRS or parts[:2] == FAILED_KERNELS:
        return False
    # Uncommitted kernel dirs (<key>.tmp.<uuid>) and write_atomic temp files.
    return not any(".tmp" in part for part in parts)


def list_files(root):
    root = Path(root)
    out = set()
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            rel = (Path(dirpath) / name).relative_to(root)
            if _shippable(rel):
                out.add(rel.as_posix())
    return out


def _deltas(key_dir):
    return sorted(key_dir.glob("delta-*.tar")) if key_dir.is_dir() else []


def _merge_into(src, dest):
    """Move every file of src into dest; content-addressed, so an existing file is identical."""
    for dirpath, _, filenames in os.walk(src):
        rel = Path(dirpath).relative_to(src)
        (dest / rel).mkdir(parents=True, exist_ok=True)
        for name in filenames:
            os.replace(Path(dirpath) / name, dest / rel / name)


def _restore_delta(tar, local, scratch):
    """Extract one delta via scratch; a bad tar is renamed aside so no later job reads it."""
    _reset_dir(scratch)
    try:
        _extract(tar, scratch)
    except (subprocess.CalledProcessError, OSError) as exc:
        _warn(f"skipping unreadable compile cache delta {tar} ({type(exc).__name__})")
        with contextlib.suppress(OSError):
            tar.rename(tar.with_name(f".bad-{tar.name}"))
        return False
    _merge_into(scratch, local)
    return True


def cmd_restore(args):
    local = Path(args.local_dir)
    _reset_dir(local)
    scratch = local.with_name(local.name + ".restore")
    key, restored = "", 0
    try:
        key = compute_key(args.key_file, salt=KEY_SALT)
        for scope in read_scopes(args.event_name, args.ref, args.pr_number):
            for tar in _deltas(_key_dir(args.nfs_root, scope, args.arch, key)):
                restored += _restore_delta(tar, local, scratch)
    except OSError as exc:
        _warn(f"compile cache restore failed ({type(exc).__name__}: {exc}); starting cold")
        _reset_dir(local)
        restored = 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    files = list_files(local)
    Path(args.manifest).write_text("\n".join(sorted(files)), encoding="utf-8")
    _write_kv(args.github_output, {"cache-key": key})
    if restored:
        kernels = local / "inductor-spyre-cache"
        n_kernels = sum(1 for p in kernels.iterdir() if p.is_dir()) if kernels.is_dir() else 0
        _log(
            f"✅ compile cache: restored {len(files)} files ({n_kernels} Spyre kernels) "
            f"from {restored} deltas"
        )
    else:
        _log(f"🧊 compile cache MISS key {key} ({args.arch}); compiling cold")
    return 0


def _write_scope(event_name, ref, pr_number):
    if event_name == "merge_group" and ref.startswith("refs/heads/gh-readonly-queue/main/"):
        return MAIN_SCOPE
    return write_scope(event_name, ref, pr_number)


def cmd_save(args):
    scope = _write_scope(args.event_name, args.ref, args.pr_number)
    if scope is None:
        _log(f"read-only context (event={args.event_name}); skipping compile cache save")
        return 0
    if not args.key:
        _log("restore computed no cache key; skipping compile cache save")
        return 0
    local = Path(args.local_dir)
    try:
        before = set(Path(args.manifest).read_text(encoding="utf-8").splitlines())
        new = sorted(list_files(local) - before)
        new_kernels = {p.split("/")[1] for p in new if p.startswith("inductor-spyre-cache/")}
        _log(f"compile cache: this job added {len(new)} files ({len(new_kernels)} Spyre kernels)")
        if not new:
            return 0
        key_dir = _key_dir(args.nfs_root, scope, args.arch, args.key)
        key_dir.mkdir(parents=True, exist_ok=True)
        target = key_dir / f"delta-{args.name}.tar"
        # Temp beside the target, so the rename is atomic and readers never see a partial tar.
        tmp = key_dir / f".delta-{args.name}.{os.getpid()}.tmp"
        listing = key_dir / f".delta-{args.name}.{os.getpid()}.list"
        try:
            listing.write_bytes(b"\0".join(os.fsencode(p) for p in new) + b"\0")
            subprocess.run(
                ["tar", "-cf", str(tmp), "-C", str(local), "--null", "-T", str(listing)],
                check=True,
            )
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
            listing.unlink(missing_ok=True)
        size = target.stat().st_size / (1024 * 1024)
        _log(f"✅ saved compile cache delta {target.relative_to(args.nfs_root)} ({size:.0f}MB)")
    except (subprocess.CalledProcessError, OSError) as exc:
        _warn(f"compile cache save failed ({type(exc).__name__}: {exc}); continuing")
    return 0


def _key_birth(key_dir):
    """When a key's first delta landed; a re-run of an old commit must not make its key look new."""
    times = []
    for delta in key_dir.glob("delta-*.tar"):
        with contextlib.suppress(FileNotFoundError):
            times.append(delta.stat().st_mtime)
    return min(times, default=0.0)


def cmd_cleanup(args):
    """Nightly: drop keys older than the newest `keep` per scope/arch, and stale PR scopes."""
    root = Path(args.nfs_root)
    start = time.monotonic()
    with contextlib.suppress(OSError):
        _sweep_pr_scopes(root, args.pr_ttl_days)
    for arch_dir in sorted(p for p in root.glob("*/*") if p.is_dir()):
        keys = sorted((p for p in arch_dir.iterdir() if p.is_dir()), key=_key_birth, reverse=True)
        for stale in keys[args.keep_keys :]:
            t0 = time.monotonic()
            shutil.rmtree(stale, ignore_errors=True)
            rel = stale.relative_to(root)
            _log(f"🧹 evicted compile cache key {rel} in {time.monotonic() - t0:.0f}s")
        # A PR scope swept down to nothing leaves empty dirs behind.
        with contextlib.suppress(OSError):
            for key_dir in arch_dir.iterdir():
                if key_dir.is_dir() and not any(key_dir.iterdir()):
                    key_dir.rmdir()
    _log(f"compile cache cleanup done in {time.monotonic() - start:.0f}s")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--nfs-root", required=True, help="e.g. $STORAGE_1_DIR/.cache/compile")
        p.add_argument("--local-dir", required=True, help="the job's TORCHINDUCTOR_CACHE_DIR")
        p.add_argument("--manifest", required=True, help="files present after the restore")
        p.add_argument("--arch", default=platform.machine())
        p.add_argument("--event-name", default=os.getenv("GITHUB_EVENT_NAME", ""))
        p.add_argument("--ref", default=os.getenv("GITHUB_REF", ""))
        p.add_argument("--pr-number", default="")

    r = sub.add_parser("restore")
    add_common(r)
    r.add_argument("--key-file", action="append", required=True, dest="key_file")
    r.add_argument("--github-output", default=os.getenv("GITHUB_OUTPUT", ""))
    r.set_defaults(func=cmd_restore)

    s = sub.add_parser("save")
    add_common(s)
    s.add_argument("--key", required=True, help="the restore's cache-key output")
    s.add_argument("--name", required=True, help="unique delta id, e.g. <run>-<attempt>-<slot>")
    s.set_defaults(func=cmd_save)

    c = sub.add_parser("cleanup")
    c.add_argument("--nfs-root", required=True)
    c.add_argument("--keep-keys", type=int, default=2, help="keys to retain per scope/arch")
    c.add_argument(
        "--pr-ttl-days", type=int, default=14, help="evict PR-scope deltas older than this"
    )
    c.set_defaults(func=cmd_cleanup)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
