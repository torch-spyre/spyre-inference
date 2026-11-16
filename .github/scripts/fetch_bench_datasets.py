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

"""Put the vLLM benchmark datasets on this host, then print their env vars.

The datasets are defined once, as engine `Asset` + `Dataset` + `Store` resources under benchmarks/
(assets-datasets.yaml, datasets.yaml, stores.yaml): store path, sha256, whether the file is an
archive, and the variable the benchmark configs read (`consumer.env`, `consumer.media_env`). This
module resolves exactly those definitions, from the datasets namespace of the artifact store only.

Cache-first: a file whose SHA256 already matches is left alone, so a warm host does no network
I/O. A missing or corrupt file is fetched to a `.part` sibling, verified, and renamed into place,
so a truncated transfer or a replaced artifact fails here and never reaches a benchmark. The cache
dir is usually shared by many jobs (a ReadWriteMany volume in CI), so a fetch takes an exclusive
per-file lock and re-checks after acquiring it. Archives are extracted without ever leaving the
extraction directory (absolute, `..` and link members are refused).

Eval the output to set the vars the configs reference:

    eval "$(python3 .github/scripts/fetch_bench_datasets.py)"

Subcommands:
  fetch (default)  Ensure each dataset is cached, print `export VAR=path`. A dataset it cannot get
                   is a loud warning, not an error: most configs replay no trace. A config that
                   needs a missing file fails by name in run_vllm_benchmarks.py.
  env              Print `export VAR=path` only, fetching nothing.
  verify           Check the cache and exit non-zero if anything is missing or corrupt. Writes
                   nothing; `--materialize` also extracts archives.
  status           One line per dataset: its variable, its state (cached, absent, corrupt, preset)
                   and the benchmark cases that read it. Fetches nothing.

A variable already set to an existing file is honoured as-is and neither re-fetched nor overridden.

Env:
  SPYRE_BENCH_DATA_DIR       cache dir (default ~/.cache/spyre/artifacts); CI points it at the
                             shared volume
  SPYRE_DATASET_DEFS         directory holding the definitions (default: benchmarks/ of this repo)
  ARTIFACTORY_BASE_URL       artifact store base, e.g. https://<host>
  ARTIFACTORY_DATASETS_PATH  repo-relative root of the datasets namespace; no fallback to another
                             root
  ARTIFACTORY_TOKEN          bearer token, only needed when something must be fetched (the CI
                             credential for the datasets namespace)
"""

from __future__ import annotations

import fcntl
import hashlib
import http.client
import json
import os
import shlex
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from argparse import ArgumentParser
from contextlib import contextmanager, suppress
from pathlib import Path

import yaml

DOWNLOAD_TIMEOUT_S = 120
DEFAULT_CACHE_DIR = "~/.cache/spyre/artifacts"
_DEFAULT_DEFS = Path(__file__).resolve().parents[2] / "benchmarks"
_COMPLETE = ".complete"


class FetchUnavailable(Exception):
    """A fetch or extraction cannot be done; the run decides what that costs.

    Messages name the dataset and a short reason, never a URL, a token or a header value.
    """


def definitions_dir() -> Path:
    return Path(os.environ.get("SPYRE_DATASET_DEFS") or _DEFAULT_DEFS)


def _documents(defs: Path) -> list[dict]:
    docs = []
    for path in sorted(defs.rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        docs += [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]
    return docs


def load_definitions(defs: Path | None = None) -> tuple[dict, dict, dict]:
    """(ARTIFACTS, ARCHIVES, store) from the Asset, Dataset and Store resources.

    ARTIFACTS and ARCHIVES map the consumer variable to {"path", "sha256", "name"}; an archive adds
    "media_var" and "extract". `path` is relative to the datasets root. `store` holds the names of
    the environment variables the Store declares, and whether it allows a plain-http endpoint.
    """
    where = defs or definitions_dir()
    docs = _documents(where)
    stores = {d["metadata"]["name"]: d["spec"] for d in docs if d.get("kind") == "Store"}
    used = {d["spec"].get("asset") for d in docs if d.get("kind") == "Dataset"}
    artifacts, archives, store_name = {}, {}, None
    for doc in docs:
        if doc.get("kind") != "Asset":
            continue
        ann = doc["metadata"].get("annotations") or {}
        var = ann.get("consumer.env")
        if not var:
            continue
        name = doc["metadata"]["name"]
        if name not in used:
            raise RuntimeError(f"Asset {name} is not read by any Dataset")
        spec = doc["spec"]
        (entry,) = spec["integrity"]["files"]
        if store_name not in (None, spec["source"]["store"]):
            raise RuntimeError("dataset Assets must share one Store")
        store_name = spec["source"]["store"]
        out = {
            "path": f"{spec['source']['path'].strip('/')}/{entry['path']}",
            "sha256": str(entry["sha256"]),
            "name": name,
        }
        if ann.get("archive"):
            out["media_var"] = ann["consumer.media_env"]
            out["extract"] = ann["extract"]
            archives[var] = out
        else:
            artifacts[var] = out
    if not artifacts and not archives:
        raise RuntimeError(f"{where} defines no datasets")
    store = stores[store_name]
    names = {
        "base": store["endpoint"]["env"],
        "root": store["root"]["env"],
        "token": store["auth"]["refs"][0]["env"],
        "allow_insecure": bool(store.get("allow_insecure")),
    }
    return artifacts, archives, names


ARTIFACTS, ARCHIVES, STORE = load_definitions()


def cache_dir() -> Path:
    return Path(os.environ.get("SPYRE_BENCH_DATA_DIR") or DEFAULT_CACHE_DIR).expanduser()


def extract_dir(cache: Path, spec: dict) -> Path:
    return Path(str(cache / spec["path"]).removesuffix(".tar.gz"))


def archive_defaults(cache: Path | None = None) -> dict[str, str]:
    cache = cache or cache_dir()
    defaults = {}
    for var, spec in ARCHIVES.items():
        out = extract_dir(cache, spec)
        defaults[var] = str(out / "dataset.jsonl")
        defaults[spec["media_var"]] = str(out)
    return defaults


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

    A trace is hundreds of MB on a shared network volume, so re-reading it on every warm run costs
    more than the fetch it guards. The note is only a shortcut: any disagreement about size or
    mtime, or any unreadable note, falls through to a real hash rather than trusting the record.
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
    note = json.dumps({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha})
    with suppress(OSError):
        _digest_note(path).write_text(note)


def cache_state(path: Path, expected_sha: str, remember: bool = True) -> str:
    """Classify the cached copy: "ok", "absent" or "corrupt".

    `remember=False` keeps the call from writing a digest note, for read-only commands.
    """
    if not path.is_file():
        return "absent"
    stat = path.stat()
    actual = _remembered_digest(path, stat)
    if actual is None:
        actual = sha256_of(path)
        if remember:
            _remember_digest(path, stat, actual)
    if actual == expected_sha:
        return "ok"
    print(f"{path.name}: cached copy has sha256 {actual}, expected {expected_sha}", file=sys.stderr)
    return "corrupt"


@contextmanager
def file_lock(path: Path):
    """Hold an exclusive lock for one dataset, so parallel jobs fetch it once.

    The lock file is a separate `.lock` sibling, never the dataset itself: locking the dataset
    would mean opening it for write and truncating a good cached copy. The lock is an
    optimisation, not a correctness requirement: across pods the cache is an NFS volume where
    `flock` depends on the server lock manager. If it does not hold, two pods both download and
    both verify the digest before an atomic rename, so the cache still ends up correct.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# --- extraction ---------------------------------------------------------------------------------


def safe_extract(archive: Path, out: Path) -> None:
    """Extract `archive` into `out`, refusing a member that lands outside it or is a link/device."""
    root = out.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (out / member.name).resolve()
            if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise FetchUnavailable(f"{archive.name}: unsafe member {member.name!r}")
        try:
            tar.extractall(out, filter="data")
        except TypeError:  # an interpreter without extraction filters; members were checked above
            tar.extractall(out)


def _extraction_complete(out: Path, archive_sha: str) -> bool:
    """True when `out` was published by a finished extraction of this exact archive."""
    try:
        marker = json.loads((out / _COMPLETE).read_text())
    except (OSError, ValueError):
        return False
    return marker.get("archive_sha256") == archive_sha and (out / "dataset.jsonl").is_file()


def _build_extraction(archive: Path, spec: dict, final: Path, work: Path) -> None:
    """Extract into `work`, write the absolute-path dataset file and the completion marker."""
    work.mkdir(parents=True)
    safe_extract(archive, work)
    source = work / "prompts.jsonl"
    if not source.is_file():
        raise FetchUnavailable(f"{archive.name}: extraction is missing prompts.jsonl")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    for row in rows:
        row["image_files"] = [str(final / p) for p in row["image_files"]]
        for image in row["image_files"]:
            if not (work / Path(image).relative_to(final)).is_file():
                raise FetchUnavailable(f"{archive.name}: image {Path(image).name} is missing")
    with (work / "dataset.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (work / _COMPLETE).write_text(json.dumps({"archive_sha256": spec["sha256"]}))


def materialize(archive: Path, spec: dict) -> tuple[Path, Path]:
    """Return (dataset_jsonl, media_dir) of a complete extraction, making it if needed.

    The extraction is built in a sibling temp dir and renamed into place, so a killed or
    out-of-space run leaves no half-written directory that a later run could accept. A directory
    without a matching completion marker is rebuilt.
    """
    out = extract_dir(cache_dir(), spec)
    if _extraction_complete(out, spec["sha256"]):
        return out / "dataset.jsonl", out
    with file_lock(archive):
        if _extraction_complete(out, spec["sha256"]):
            return out / "dataset.jsonl", out
        work = out.parent / f".{out.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp"
        try:
            _build_extraction(archive, spec, out, work)
            if out.exists():
                shutil.rmtree(out)
            work.rename(out)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return out / "dataset.jsonl", out


# --- download -----------------------------------------------------------------------------------


def _require(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise FetchUnavailable(
            f"{var} is not set, so a missing artifact cannot be fetched. "
            "Set it or pre-populate the cache."
        )
    return value


def _clean(var: str) -> str:
    """The variable's value, stripped; a control character is refused without echoing the value."""
    value = _require(var).strip()
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise FetchUnavailable(f"{var} contains a control character")
    return value


class _SameHostRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but never carry the token to another host or over plain http."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != "https":
            return None
        if target.netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


def _open_url(url: str, token: str):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    opener = urllib.request.build_opener(_SameHostRedirect)
    return opener.open(request, timeout=DOWNLOAD_TIMEOUT_S)


_RETRIES = 3
_RETRY_DELAY_S = 5


def _fetch_once(url, token, partial, opener):
    with opener(url, token) as response, partial.open("wb") as handle:
        while chunk := response.read(4 * 1024 * 1024):
            handle.write(chunk)


def download(rel_path, dest, expected_sha, opener=_open_url, sleep=time.sleep):
    """Fetch one dataset file into `dest`, verifying the digest before it is published.

    `rel_path` is relative to the datasets root. The file is written to a unique `.part` sibling
    and renamed only after the digest matches, so an interrupted run never leaves a short file
    that later looks cached. The token travels in a header in memory, never in argv, and no error
    text echoes the URL or any header value. Transient failures (connection errors, 429, 5xx) are
    retried; 401, 403 and 404 are not. A plain-http endpoint is refused unless the Store allows it.
    """
    base = _clean(STORE["base"]).rstrip("/")
    root = _clean(STORE["root"]).strip("/")
    token = _clean(STORE["token"])
    if not base.startswith("https://") and not STORE["allow_insecure"]:
        raise FetchUnavailable(f"{STORE['base']} is not an https endpoint")
    url = f"{base}/artifactory/{root}/{rel_path.lstrip('/')}"
    partial = dest.with_suffix(f"{dest.suffix}.{os.getpid()}-{uuid.uuid4().hex[:8]}.part")
    print(f"Fetching {rel_path} ...", file=sys.stderr)
    for attempt in range(1, _RETRIES + 1):
        try:
            _fetch_once(url, token, partial, opener)
            break
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            reason = f"HTTP {exc.code}"
            transient = exc.code == 429 or exc.code >= 500
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
            partial.unlink(missing_ok=True)
            reason = type(exc).__name__
            transient = not isinstance(exc, ValueError)
        if not transient or attempt == _RETRIES:
            raise FetchUnavailable(f"{rel_path}: download failed ({reason})") from None
        sleep(_RETRY_DELAY_S)
    actual = sha256_of(partial)
    if actual != expected_sha:
        partial.unlink(missing_ok=True)
        raise FetchUnavailable(
            f"{rel_path}: sha256 {actual} does not match expected {expected_sha}"
        )
    partial.replace(dest)


def ensure_cached(spec, destination, force=False, opener=_open_url, sleep=time.sleep) -> str:
    """Make one dataset file present and verified under `destination`.

    Returns "cached" or "downloaded".
    """
    path = destination / spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        if not force and cache_state(path, spec["sha256"]) == "ok":
            return "cached"
        download(spec["path"], path, spec["sha256"], opener=opener, sleep=sleep)
    return "downloaded"


# --- commands -----------------------------------------------------------------------------------


def cases_reading(asset: str) -> list[str]:
    """Names of the benchmark cases whose inputs read a Dataset built on `asset`."""
    docs = _documents(definitions_dir())
    sets = {
        d["metadata"]["name"]
        for d in docs
        if d.get("kind") == "Dataset" and d["spec"].get("asset") == asset
    }
    found = []
    for doc in docs:
        if doc.get("kind") != "Benchmark":
            continue
        for variant in doc["spec"].get("variants") or []:
            inputs = (variant.get("workload") or {}).get("inputs") or []
            if any(i.get("dataset") in sets for i in inputs if isinstance(i, dict)):
                found.append(variant["name"])
    return found


def status() -> None:
    destination = cache_dir()
    for var, spec in {**ARTIFACTS, **ARCHIVES}.items():
        preset = os.environ.get(var)
        if preset and Path(preset).is_file():
            state = "preset"
        else:
            found = cache_state(destination / spec["path"], spec["sha256"], remember=False)
            state = "cached" if found == "ok" else found
        readers = cases_reading(spec["name"])
        suffix = f": {', '.join(readers)}" if readers else ""
        print(f"{var}\t{state}\t{len(readers)} case(s){suffix}")
    print("A selected case whose file is absent fails the run by name (run_vllm_benchmarks.py).")


def _export(var: str, value) -> None:
    print(f"export {var}={shlex.quote(str(value))}")


def _present(command, cache_dir_error, path, spec, name, missing, corrupt, unavailable):
    """Make `path` a verified copy of `spec`. False when it is not usable (and why is recorded)."""
    state = cache_state(path, spec["sha256"], remember=command == "fetch")
    if state == "ok":
        return True
    if command != "fetch":
        (missing if state == "absent" else corrupt).append(path)
        return False
    if cache_dir_error is not None:
        unavailable.append((name, cache_dir_error))
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(path):
            # Another job may have finished the fetch while we waited.
            if cache_state(path, spec["sha256"]) != "ok":
                download(spec["path"], path, spec["sha256"])
    except (FetchUnavailable, OSError, ValueError) as exc:
        unavailable.append((name, exc))
        return False
    return True


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", default="fetch", choices=["fetch", "env", "verify", "status"]
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="with verify: also extract archives (verify and env otherwise never write)",
    )
    args = parser.parse_args()

    if args.command == "status":
        status()
        return

    destination = cache_dir()
    cache_dir_error = None
    if args.command == "fetch":
        try:
            destination.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # An unusable cache root is reported per dataset below, like any other fetch failure,
            # rather than killing a run whose configs may need no trace at all.
            cache_dir_error = exc

    missing, corrupt, unavailable = [], [], []
    for var, spec in ARTIFACTS.items():
        # An operator who already has the trace somewhere else wins.
        preset = os.environ.get(var)
        if preset and Path(preset).is_file():
            _export(var, preset)
            continue
        path = destination / spec["path"]
        # `env` only reports where the files belong, so it never hashes or fetches.
        if args.command == "env":
            _export(var, path)
            continue
        if _present(args.command, cache_dir_error, path, spec, var, missing, corrupt, unavailable):
            _export(var, path)

    for var, spec in ARCHIVES.items():
        media = spec["media_var"]
        preset, preset_media = os.environ.get(var), os.environ.get(media)
        if preset and Path(preset).is_file() and preset_media and Path(preset_media).is_dir():
            _export(var, preset)
            _export(media, preset_media)
            continue
        archive = destination / spec["path"]
        out = extract_dir(destination, spec)
        if args.command == "env":
            _export(var, out / "dataset.jsonl")
            _export(media, out)
            continue
        if not _present(
            args.command, cache_dir_error, archive, spec, var, missing, corrupt, unavailable
        ):
            continue
        if args.command == "verify" and not args.materialize:
            if _extraction_complete(out, spec["sha256"]):
                _export(var, out / "dataset.jsonl")
                _export(media, out)
            continue
        try:
            dataset, out = materialize(archive, spec)
        except (FetchUnavailable, OSError, ValueError, tarfile.TarError) as exc:
            unavailable.append((var, exc))
            continue
        _export(var, dataset)
        _export(media, out)

    if unavailable:
        # Loud, but NOT fatal. Most benchmark configs replay no trace and stay valid without these.
        # A config that DOES need one fails by name in run_vllm_benchmarks.py, which checks per
        # config: that is what keeps a serve job from reporting success while measuring nothing.
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
