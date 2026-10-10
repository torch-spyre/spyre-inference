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

"""Ship the uv cache to and from the shared NFS mount, actions/cache style.

uv only ever sees a plain local cache dir, so `uv sync` never takes a lock on
NFS -- pointing UV_CACHE_DIR straight at NFS serializes every pod on uv's
per-path build locks, which NFS hands between waiters only every ~20-30s. NFS is
a dumb shelf of one tar per key at <nfs-root>/<scope>/<arch>/<key>.tar.

Scopes mirror GitHub's cache service so a PR cannot poison main: push-to-main
writes the "main" scope every job reads; PR #N writes an isolated "pr-<N>" it
alone reads (falling back to main). N is GitHub-assigned, so a fork cannot forge
it to reach main or a sibling. A job writes only when the key is absent from
every scope it reads, so an ordinary PR writes nothing; the atomic rename keeps
concurrent writers of one key safe. A flat mount has no server-side ACL, so this
is the honest-path topology, not an enforced boundary -- the gate on an
untrusted fork is whatever approves fork PRs onto these runners. Restore and
save are best-effort: on failure uv starts cold, never a red job.
"""

import argparse
import contextlib
import hashlib
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Bump to invalidate every cached tar at once after a format change.
KEY_SALT = "nfs-uv-cache-v1"
KEY_LEN = 16

# Default is plain `.tar`: the cache is mostly already-compressed wheels, and it
# needs no compressor binary on the runner.
TAR_SUFFIXES = (".tar.zst", ".tar.gz", ".tar")

COMPRESS = {
    "none": (None, ".tar"),
    "zstd": ("zstd -1 -T0", ".tar.zst"),
    "gzip": ("gzip", ".tar.gz"),
}


def _log(msg):
    print(msg, flush=True)


def _warn(msg):
    print(f"::warning::{msg}", flush=True)


def uv_version(uv):
    if not uv:
        return ""
    try:
        out = subprocess.run([uv, "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        _warn(f"could not run `{uv} --version` ({type(exc).__name__}); key ignores the uv version")
        return ""
    parts = out.split()
    return parts[1] if len(parts) > 1 else out.strip()


def compute_key(key_files, salt=KEY_SALT, uv_version=""):
    """Content-address the cache on the lockfiles' basenames and bytes.

    Only the basename, never the absolute path, feeds the digest -- runners with
    different GITHUB_WORKSPACE must agree on the key for byte-identical lockfiles,
    or a PR never hits the shared main scope. The uv version is keyed too: a release
    can move every bucket (0.13.0: wheels-v6 -> v7), leaving an exact hit uv can't read.
    """
    digest = hashlib.sha256(salt.encode())
    if uv_version:
        digest.update(b"\0uv\0" + uv_version.encode())
    for path in key_files:
        digest.update(b"\0")
        digest.update(os.fsencode(os.path.basename(path)))
        digest.update(b"\0")
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:KEY_LEN]


MAIN_SCOPE = "main"


def read_scopes(event_name, ref, pr_number):
    """Scopes this job may read, highest priority first (a PR falls back to main)."""
    if event_name == "pull_request" and pr_number:
        return [f"pr-{pr_number}", MAIN_SCOPE]
    return [MAIN_SCOPE]


def write_scope(event_name, ref, pr_number):
    """The one scope this job may write, or None in a read-only context."""
    if event_name == "push" and ref == "refs/heads/main":
        return MAIN_SCOPE
    if event_name == "pull_request" and pr_number:
        return f"pr-{pr_number}"
    return None


def _reset_dir(path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _is_empty(path):
    return not any(path.iterdir())


def _scope_dir(nfs_root, scope, arch):
    return Path(nfs_root) / scope / arch


def _find_exact(arch_dir, key):
    for suffix in TAR_SUFFIXES:
        candidate = arch_dir / f"{key}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _newest_tar(arch_dir):
    if not arch_dir.is_dir():
        return None
    tars = [p for p in arch_dir.iterdir() if p.is_file() and p.name.endswith(TAR_SUFFIXES)]
    if not tars:
        return None
    return max(tars, key=lambda p: p.stat().st_mtime)


def _extract(tar_path, dest):
    # tar preserves uv's internal hardlinks and avoids a per-file metadata storm
    # over NFS; GNU tar auto-detects compression on extract.
    subprocess.run(
        ["tar", "-xf", str(tar_path), "-C", str(dest)],
        check=True,
    )


def _create_tar(src, tar_path, compress_prog):
    cmd = ["tar"]
    if compress_prog:
        cmd += ["--use-compress-program", compress_prog]
    cmd += ["-cf", str(tar_path), "-C", str(src), "."]
    subprocess.run(cmd, check=True)


def _write_kv(path, pairs):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in pairs.items():
            handle.write(f"{key}={value}\n")


def cmd_restore(args):
    scopes = read_scopes(args.event_name, args.ref, args.pr_number)
    local = Path(args.local_dir)

    key = ""
    hit = False
    restored_from = ""
    try:
        key = compute_key(args.key_file, uv_version=args.uv_version)
        _reset_dir(local)
        # Prefer an exact-key tar in any readable scope (priority order); else
        # the newest tar there, so a lockfile change still starts warm.
        chosen = None
        for scope in scopes:
            exact = _find_exact(_scope_dir(args.nfs_root, scope, args.arch), key)
            if exact is not None:
                chosen, hit = (scope, exact), True
                break
        if chosen is None:
            for scope in scopes:
                cand = _newest_tar(_scope_dir(args.nfs_root, scope, args.arch))
                if cand is not None:
                    chosen = (scope, cand)
                    break
        if chosen is not None:
            scope, tar = chosen
            try:
                _extract(tar, local)
                restored_from = f"{scope}/{args.arch}/{tar.name}"
            except (subprocess.CalledProcessError, OSError) as exc:
                # A half-extracted cache is worse than none: wipe and go cold.
                _warn(f"uv cache restore failed ({type(exc).__name__}); starting cold")
                _reset_dir(local)
                restored_from = ""
                hit = False
    except OSError as exc:
        _warn(f"uv cache restore errored ({type(exc).__name__}); starting cold")

    # Always hand uv a usable local cache dir, hit or miss.
    local.mkdir(parents=True, exist_ok=True)
    _write_kv(args.github_env, {"UV_CACHE_DIR": str(local)})
    _write_kv(
        args.github_output,
        {"cache-hit": "true" if hit else "false", "cache-key": key, "restored-from": restored_from},
    )

    scope_list = "+".join(scopes)
    if hit:
        _log(f"✅ uv cache HIT {restored_from} -> {local}")
    elif restored_from:
        _log(f"♻️ uv cache partial: restored {restored_from} (key {key}); uv fills the diff")
    else:
        _log(f"🧊 uv cache MISS key {key} in [{scope_list}] ({args.arch}); starting cold")
    return 0


def _gc(arch_dir, keep, protect):
    # A skip-save on a fallback-scope hit reaches here with the (unwritten,
    # nonexistent) write-scope dir, so missing is a no-op, not a warning.
    if keep <= 0 or not arch_dir.is_dir():
        return
    tars = sorted(
        (p for p in arch_dir.iterdir() if p.is_file() and p.name.endswith(TAR_SUFFIXES)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in tars[keep:]:
        if stale == protect:
            continue
        try:
            stale.unlink()
            _log(f"🧹 evicted old cache {stale.name}")
        except OSError:
            pass


def _sweep_pr_scopes(nfs_root, ttl_days):
    """Evict PR-scope tars older than ttl_days; main is bounded by _gc instead."""
    if ttl_days <= 0:
        return
    cutoff = time.time() - ttl_days * 86400
    root = Path(nfs_root)
    if not root.is_dir():
        return
    for scope_dir in root.glob("pr-*"):
        if not scope_dir.is_dir():
            continue
        for tar in scope_dir.rglob("*"):
            try:
                if (
                    tar.is_file()
                    and tar.name.endswith(TAR_SUFFIXES)
                    and tar.stat().st_mtime < cutoff
                ):
                    tar.unlink()
                    _log(f"🧹 evicted stale PR cache {tar.relative_to(root)}")
            except OSError:
                pass


def cmd_save(args):
    # Sweep stale PR scopes on every invocation (even read-only contexts), so a
    # merged PR's cache is reclaimed by whatever job runs next.
    with contextlib.suppress(OSError):
        _sweep_pr_scopes(args.nfs_root, args.pr_ttl_days)

    scope = write_scope(args.event_name, args.ref, args.pr_number)
    if scope is None:
        _log(f"read-only context (event={args.event_name} ref={args.ref}); skipping uv cache save")
        return 0

    prog, ext = COMPRESS[args.compress]
    local = Path(args.local_dir)

    try:
        key = compute_key(args.key_file, uv_version=args.uv_version)
        arch_dir = _scope_dir(args.nfs_root, scope, args.arch)
        target = arch_dir / f"{key}{ext}"
        # Skip if the key is already readable here, so an ordinary PR (key in
        # main) writes nothing; only a new key populates its own scope.
        for readable in read_scopes(args.event_name, args.ref, args.pr_number):
            found = _find_exact(_scope_dir(args.nfs_root, readable, args.arch), key)
            if found is not None:
                _log(f"uv cache key {key} already in {readable}/{args.arch}; skipping save")
                _gc(arch_dir, args.keep, target)
                return 0
        if not local.is_dir() or _is_empty(local):
            _log(f"no local uv cache at {local}; nothing to save")
            return 0

        # After a uv upgrade the restored tar holds the old buckets too; a plain prune (not
        # --ci, which drops downloaded wheels) removes them.
        if args.uv:
            try:
                subprocess.run(
                    [args.uv, "cache", "prune"],
                    env={**os.environ, "UV_CACHE_DIR": str(local)},
                    check=True,
                )
            except (subprocess.CalledProcessError, OSError) as exc:
                _warn(f"uv cache prune failed ({type(exc).__name__}); saving unpruned")

        arch_dir.mkdir(parents=True, exist_ok=True)
        # Temp lives in the target dir so the rename is a same-filesystem,
        # atomic rename(2) -- readers never see a half-written tar.
        tmp = arch_dir / f".{key}.{os.getpid()}.{random.randrange(1 << 30):x}.tmp"
        try:
            _create_tar(local, tmp, prog)
            # Re-check under the gun: a concurrent writer may have won while we
            # tarred. Equal content, so skipping ours (not overwriting) is fine.
            if _find_exact(arch_dir, key) is not None:
                _log(
                    f"uv cache {scope}/{args.arch} key {key} appeared during save; discarding ours"
                )
            else:
                os.replace(tmp, target)
                size = target.stat().st_size / (1024 * 1024)
                _log(f"✅ saved uv cache {scope}/{args.arch}/{target.name} ({size:.0f}MB)")
        finally:
            if tmp.exists():
                tmp.unlink()
        _gc(arch_dir, args.keep, target)
    except (subprocess.CalledProcessError, OSError) as exc:
        # Never fail CI on a cache write (NFS full, perms, race): warn and move on.
        _warn(f"uv cache save failed ({type(exc).__name__}: {exc}); continuing")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--nfs-root", required=True, help="e.g. $STORAGE_1_DIR/.cache/uv")
        p.add_argument("--local-dir", required=True, help="local UV_CACHE_DIR to populate/ship")
        # Matches `uname -m` (x86_64, ppc64le, s390x); wheels are arch-specific.
        p.add_argument("--arch", default=platform.machine())
        p.add_argument("--key-file", action="append", required=True, dest="key_file")
        p.add_argument("--event-name", default=os.getenv("GITHUB_EVENT_NAME", ""))
        p.add_argument("--ref", default=os.getenv("GITHUB_REF", ""))
        p.add_argument("--pr-number", default="")
        p.add_argument("--uv", default="", help="uv binary: keys on its version, prunes on save")

    r = sub.add_parser("restore")
    add_common(r)
    r.add_argument("--github-env", default=os.getenv("GITHUB_ENV", ""))
    r.add_argument("--github-output", default=os.getenv("GITHUB_OUTPUT", ""))
    r.set_defaults(func=cmd_restore)

    s = sub.add_parser("save")
    add_common(s)
    s.add_argument("--compress", choices=sorted(COMPRESS), default="none")
    s.add_argument("--keep", type=int, default=5, help="tars to retain per scope/arch")
    s.add_argument(
        "--pr-ttl-days", type=int, default=14, help="evict PR-scope tars older than this"
    )
    s.set_defaults(func=cmd_save)

    args = parser.parse_args(argv)
    args.uv_version = uv_version(args.uv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
