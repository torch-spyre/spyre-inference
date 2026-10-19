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

"""Put the vLLM benchmark trace files on this host, then print their env vars.

The serve benchmarks replay recorded traces. Until now every host was expected
to have them pre-mounted under /models, which only the x86_64 benchmark hosts
do, so the s390x and ppc64le perf lanes could not run a serve config at all.
This fetches them from the artifact store into a cache dir instead, so any
arch can run.

Cache-first by design: a file whose SHA256 already matches ARTIFACTS is left
alone and never re-downloaded, so a warm host does no network I/O. Only a
missing or corrupt file is fetched. The digests below are the integrity gate,
so a truncated transfer or a silently-replaced artifact fails here rather than
skewing a benchmark.

The cache dir is usually shared and writable by many jobs at once (on x86 it is
a ReadWriteMany PVC mounted into every runner pod), so a fetch takes an
exclusive lock per file and re-checks the cache after acquiring it. Cross-pod
deduplication depends on NFS locking (see ``file_lock``). Downloads land on a
`.part` sibling and are renamed into place only after digest verification.

Eval the output to set the vars the configs reference:

    eval "$(python3 .github/scripts/fetch_bench_datasets.py)"

Subcommands:
  fetch (default)   Ensure each dataset is cached, print `export VAR=path`.
                    An artifact it cannot get is a loud warning, not an error:
                    most benchmark configs replay no trace and stay valid
                    without one, and a config that does need one fails by name
                    in run_vllm_benchmarks.py, which checks per config. Only
                    artifacts that are actually present get exported.
  env               Print `export VAR=path` only, fetching nothing.
  verify            Check the cache and exit non-zero if anything is missing or
                    corrupt, reporting which. Exports only the usable ones.

A SPYRE_*_DATASET already set to an existing file is honoured as-is and neither
re-fetched nor overridden, so a host with its own copy keeps using it.

Env:
  SPYRE_BENCH_DATA_DIR   cache dir (default ~/.cache/spyre/artifacts). CI points
                         this at the shared PVC the runners mount.
  ARTIFACTORY_BASE_URL   artifact store base, e.g. https://<host>
  ARTIFACTORY_DATA_PATH  repo-relative ROOT that every artifact path below hangs
                         off, e.g. <repo>/spyredata/components. Each ARTIFACTS
                         entry adds its own path under this root, so a new
                         dataset or artifact needs a table entry, not a new
                         secret.
  ARTIFACTORY_TOKEN      bearer token (only needed when something must be fetched)
"""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from contextlib import contextmanager, suppress
from pathlib import Path

# Artifacts this repo pulls from the store, keyed by the env var that names the
# local copy. `path` is relative to ARTIFACTORY_DATA_PATH and `sha256` is the
# integrity gate; see the module docstring. Keeping the path per entry means a
# new dataset or artifact anywhere under the root is a line here rather than
# another secret.
#
# All thirteen files in the trace prefix are listed, not only the two that a
# benchmark config selects today (aiops and cics). A fork PR gets no secrets and
# so cannot fetch a missing artifact for itself, which would make every new
# trace-replaying config wait on a config change PLUS a cache dispatch. Mirroring
# the whole prefix costs one transfer on a shared volume and removes that wait.
# The digests match .github/cache_config/hf_models_and_datasets.yaml, which the
# writer dispatch uses to populate that cache; keep the two in step.
#
# A consumer that needs only a subset (a Jenkins node with its own local cache and
# credentials of its own, say) can fetch per artifact rather than the whole table.
_TRACES = "spyre-inference/vllm-bench-data/converted_traces/2025.11.03_e2ee1b0_reordered"

ARTIFACTS = {
    "SPYRE_AIOPS_DATASET": {
        "path": f"{_TRACES}/aiops_results_2025.11.03_e2ee1b0_correct_order.jsonl",
        "sha256": "468cf059f2ec3b14108efc09baffabf5c5a440172a25660673d5a8fa029d0637",
    },
    "SPYRE_ALL_SEQUENCES_DATASET": {
        "path": f"{_TRACES}/all_sequences_2025.11.03_e2ee1b0_correct_order.jsonl",
        "sha256": "6df5d4de50a28b6ed7ad1d0b8794417899d18548627529a526bef93967906e3c",
    },
    "SPYRE_ALL_SEQUENCES_TRUNCATED_16K_DATASET": {
        "path": f"{_TRACES}/all_sequences_2025.11.03_e2ee1b0_correct_order_truncated_16k.jsonl",
        "sha256": "03b9e0d4b4432e0094f84b98fa89e6bfb2b402d35fef84421558d77ea36f5197",
    },
    "SPYRE_ALL_SEQUENCES_TRUNCATED_1K_DATASET": {
        "path": f"{_TRACES}/all_sequences_2025.11.03_e2ee1b0_correct_order_truncated_1k.jsonl",
        "sha256": "6ad8b08d2a600a24993a9570100e14e8d40ffe1308802ee12698f0d7b37c0b90",
    },
    "SPYRE_ALL_SEQUENCES_TRUNCATED_2K_DATASET": {
        "path": f"{_TRACES}/all_sequences_2025.11.03_e2ee1b0_correct_order_truncated_2k.jsonl",
        "sha256": "9b28b894978bd940ba860187d932de86de382534f04e1b284b61af5877418adb",
    },
    "SPYRE_ALL_SEQUENCES_TRUNCATED_32K_DATASET": {
        "path": f"{_TRACES}/all_sequences_2025.11.03_e2ee1b0_correct_order_truncated_32k.jsonl",
        "sha256": "a9d6a8a6cf849a60b56dbc8bd1b1eb9b95babb6a2d6a7747878e167869637b5d",
    },
    "SPYRE_ALL_SEQUENCES_TRUNCATED_4K_DATASET": {
        "path": f"{_TRACES}/all_sequences_2025.11.03_e2ee1b0_correct_order_truncated_4k.jsonl",
        "sha256": "8d76a859a67677a091290dad395d03acf88719688128530d7a943c4f23b1acff",
    },
    "SPYRE_ALL_SEQUENCES_TRUNCATED_8K_DATASET": {
        "path": f"{_TRACES}/all_sequences_2025.11.03_e2ee1b0_correct_order_truncated_8k.jsonl",
        "sha256": "bd6579a7dbce7c80c3908e3150455eb13af940ee900c354539e7f6491512fce7",
    },
    "SPYRE_CICS_DATASET": {
        "path": f"{_TRACES}/cics_results_2025.11.03_e2ee1b0_correct_order.jsonl",
        "sha256": "e0efa895b4a22b601748c7dda9e8a2db146773fb8735fc45ee2920c042b40fb7",
    },
    "SPYRE_DB2_DATASET": {
        "path": f"{_TRACES}/db2_results_2025.11.03_e2ee1b0_correct_order.jsonl",
        "sha256": "b219891bee31b62d82f83ddc4b478bf75ac9039762b22b1afac5fc437cb2bac1",
    },
    "SPYRE_DB2_TRUNCATED_32K_DATASET": {
        "path": f"{_TRACES}/db2_results_2025.11.03_e2ee1b0_correct_order_truncated_32k.jsonl",
        "sha256": "9c0584930498e268b0af52adfc411c71d34a9065c16742381281ef578d911283",
    },
    "SPYRE_IMS_DATASET": {
        "path": f"{_TRACES}/ims_results_2025.11.03_e2ee1b0_correct_order.jsonl",
        "sha256": "49a284e3f10027bb430f168a4fba7310d7dcb3147565ffed1bb87d1fc6fe9c18",
    },
    "SPYRE_TLS_DATASET": {
        "path": f"{_TRACES}/tls_results_2025.11.03_e2ee1b0_correct_order.jsonl",
        "sha256": "9b1cc4d35e3626d1a6bbe0f998caa039b70d694241f25835a2083e1854e14028",
    },
}

DEFAULT_CACHE_DIR = "~/.cache/spyre/artifacts"


def cache_dir() -> Path:
    return Path(os.environ.get("SPYRE_BENCH_DATA_DIR") or DEFAULT_CACHE_DIR).expanduser()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # Traces run to hundreds of MB, so stream rather than read() them.
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_note(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.sha256")


def _remembered_digest(path: Path, stat: os.stat_result) -> str | None:
    """The digest recorded for this exact (size, mtime_ns), or None.

    A trace is hundreds of MB on a shared network volume, so re-reading it on
    every warm run costs more than the fetch it is guarding. The note is only a
    shortcut: any disagreement about size or mtime, or any unreadable note, falls
    through to a real hash rather than trusting the record.
    """
    try:
        note = json.loads(_digest_note(path).read_text())
        if note["size"] == stat.st_size and note["mtime_ns"] == stat.st_mtime_ns:
            return str(note["sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None


def _remember_digest(path: Path, stat: os.stat_result, sha: str) -> None:
    """Record `sha` for this (size, mtime_ns). Best effort: a read-only cache is fine."""
    # The note is an optimisation, so losing it costs a re-hash and nothing else.
    with suppress(OSError):
        _digest_note(path).write_text(
            json.dumps({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha})
        )


def cache_state(path: Path, expected_sha: str) -> str:
    """Classify the cached copy: "ok", "absent" or "corrupt".

    Callers need the reason, not just a boolean: `fetch` re-downloads either way,
    but `verify` has to say whether a file is missing or present-and-wrong.
    """
    if not path.is_file():
        return "absent"
    stat = path.stat()
    actual = _remembered_digest(path, stat)
    if actual is None:
        actual = sha256_of(path)
        _remember_digest(path, stat, actual)
    if actual == expected_sha:
        return "ok"
    print(
        f"{path.name}: cached copy has sha256 {actual}, expected {expected_sha}",
        file=sys.stderr,
    )
    return "corrupt"


class FetchUnavailable(Exception):
    """A fetch cannot be attempted or did not succeed.

    Raised rather than exiting so `fetch` can report the artifact it could not
    get and still leave the run to decide. Only some benchmark configs replay a
    trace: the rest must stay runnable on a host with no store access, e.g. a
    fork pull request, where GitHub withholds every secret.
    """


def _require(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise FetchUnavailable(
            f"{var} is not set, so a missing artifact cannot be fetched. "
            "Set it or pre-populate the cache."
        )
    return value


@contextmanager
def file_lock(path: Path):
    """Hold an exclusive lock for one dataset, so parallel jobs fetch it once.

    The lock file is a separate `.lock` sibling, never the dataset itself: locking
    the dataset would mean opening it for write and truncating a good cached copy.

    The lock is an optimisation, not a correctness requirement. It is verified to
    deduplicate concurrent fetches within one host; across pods the CI cache is an
    NFS volume, where `flock` depends on the server lock manager and has not been
    tested here. If it does not hold, two pods both download and both verify the
    digest before an atomic rename, so the loser wastes bandwidth and the cache
    still ends up correct. Nothing downstream reads a partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def download(rel_path: str, dest: Path, expected_sha: str) -> None:
    """Fetch one artifact into `dest`, verifying the digest before it is published.

    `rel_path` is relative to ARTIFACTORY_DATA_PATH. Downloads to a `.part`
    sibling and renames only after the digest matches, so an interrupted run can
    never leave a short file that later looks cached.
    """
    base = _require("ARTIFACTORY_BASE_URL").rstrip("/")
    root = _require("ARTIFACTORY_DATA_PATH").strip("/")
    token = _require("ARTIFACTORY_TOKEN")
    url = f"{base}/artifactory/{root}/{rel_path.lstrip('/')}"

    # PID-suffixed so two processes can never write the same temp file.
    partial = dest.with_suffix(f"{dest.suffix}.{os.getpid()}.part")
    print(f"Fetching {rel_path} ...", file=sys.stderr)
    # The token goes in on stdin as a curl config file, never in argv: argv is
    # world-readable through /proc/<pid>/cmdline on a shared runner.
    returncode = subprocess.run(
        [
            "curl",
            "-fSL",
            "--no-progress-meter",
            "--retry",
            "3",
            "--retry-delay",
            "5",
            "-o",
            str(partial),
            "-K",
            "-",
            url,
        ],
        input=f'header = "Authorization: Bearer {token}"\n',
        text=True,
        check=False,
    ).returncode
    if returncode != 0:
        partial.unlink(missing_ok=True)
        raise FetchUnavailable(f"{rel_path}: download failed (curl exit {returncode})")

    actual = sha256_of(partial)
    if actual != expected_sha:
        partial.unlink(missing_ok=True)
        raise FetchUnavailable(
            f"{rel_path}: sha256 {actual} does not match expected {expected_sha}"
        )
    partial.replace(dest)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="fetch", choices=["fetch", "env", "verify"])
    args = parser.parse_args()

    destination = cache_dir()
    cache_dir_error = None
    if args.command == "fetch":
        try:
            destination.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # An unusable cache root is reported per artifact below, like any
            # other fetch failure, rather than killing a run whose configs may
            # need no trace at all.
            cache_dir_error = exc

    missing = []
    corrupt = []
    unavailable = []
    for var, spec in ARTIFACTS.items():
        # An operator who already has the trace somewhere else wins: honour a
        # pre-set var when it points at a real file, rather than overriding it
        # with a cache path and fetching a second copy.
        preset = os.environ.get(var)
        if preset and Path(preset).is_file():
            print(f"export {var}={preset}")
            continue

        # Mirror the store's layout under the cache dir so two artifacts that
        # share a basename cannot collide.
        path = destination / spec["path"]
        # `env` only reports where the files belong, so it never hashes or fetches.
        if args.command == "env":
            print(f"export {var}={path}")
            continue

        state = cache_state(path, spec["sha256"])
        if state != "ok":
            if args.command != "fetch":
                (missing if state == "absent" else corrupt).append(path)
                # Do not export a path `verify` just judged unusable.
                continue
            if cache_dir_error is not None:
                unavailable.append((var, cache_dir_error))
                continue
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with file_lock(path):
                    # Another job may have finished the fetch while we waited.
                    if cache_state(path, spec["sha256"]) != "ok":
                        download(spec["path"], path, spec["sha256"])
            except (FetchUnavailable, OSError) as exc:
                # OSError too, not just FetchUnavailable: a read-only or full
                # cache dir, a lock file that cannot be created, or a failed
                # rename must not take down the configs that need no trace.
                unavailable.append((var, exc))
                # Export nothing for an artifact that is not there. A path
                # that does not exist would make the failure name the path
                # instead of the fetch that could not get it.
                continue
        print(f"export {var}={path}")

    if unavailable:
        # Loud, but NOT fatal. Most benchmark configs replay no trace and stay
        # valid without these, so failing here would take down work that does
        # not depend on them. A config that DOES need one still fails, by name,
        # in run_vllm_benchmarks.py, which checks per config: that is what keeps
        # a serve job from reporting success while measuring nothing.
        banner = "=" * 72
        print(banner, file=sys.stderr)
        print("WARNING: could not fetch benchmark artifacts.", file=sys.stderr)
        for var, exc in unavailable:
            print(f"  {var}: {exc}", file=sys.stderr)
        print(
            "Configs that replay one of these traces will fail by name. Configs "
            "that do not need a trace are unaffected and still run.",
            file=sys.stderr,
        )
        print(banner, file=sys.stderr)

    if missing or corrupt:
        for path in missing:
            print(f"missing from cache: {path}", file=sys.stderr)
        for path in corrupt:
            print(f"corrupt in cache (sha256 mismatch): {path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
