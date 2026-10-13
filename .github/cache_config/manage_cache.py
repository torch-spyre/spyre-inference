# /// script
# dependencies = [
#   "huggingface-hub",
#   "pyyaml",
# ]
# ///

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

import argparse
import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public",
        action="store_true",
        help="Cache public_models",
    )
    parser.add_argument(
        "--datasets",
        action="store_true",
        help="Cache artifactory_datasets",
    )
    args = parser.parse_args()

    config_file = os.getenv(
        "CACHE_CONFIG_FILE_PATH", ".github/cache_config/hf_models_and_datasets.yaml"
    )
    if not os.path.exists(config_file):
        print(f"❌ Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    with open(config_file, encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Error parsing {config_file}: {e}")
            sys.exit(1)

    if args.datasets:
        _run_artifactory_datasets(config, config_file)
    elif args.public:
        _run_public(config, config_file)
    else:
        _run_gated(config, config_file)


def _parse_entry(entry):
    if isinstance(entry, str):
        return entry, None
    return entry["repo"], entry.get("revision")


# Duplicate pytorch dumps. Kept out of the cache when a safetensors copy exists.
# A repo whose only weights are ``pytorch_model.bin`` (CLIP ViT-B/32) cannot use
# this ignore: the offline jobs then open a snapshot that has no weights at all.
_DUPLICATE_WEIGHTS = ["*.pt", "*.pth", "*.bin"]


def _has_pytorch_weights(snapshot: str) -> bool:
    """True when ``snapshot`` contains safetensors or a ``pytorch_model`` bin."""
    root = Path(snapshot)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
            return True
        if name.startswith("pytorch_model") and name.endswith(".bin"):
            return True
    return False


def _run_public(config, config_file):
    """Cache public_models entries."""
    models = config.get("public_models", [])
    if not models:
        print(f"⚠️ Warning: No public_models defined in {config_file}.")
        return

    # Imported inside the function, and after the guards above, so `--datasets`
    # runs without huggingface_hub installed and an empty config still exits 0.
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    print(f"📋 Found {len(models)} public model(s) to cache:", models)
    failed_models = []
    for entry in models:
        repo_id, revision = _parse_entry(entry)
        print(f"\n🚀 Processing: {repo_id}...")
        try:
            cached = snapshot_download(
                repo_id,
                revision=revision,
                local_files_only=True,
                ignore_patterns=_DUPLICATE_WEIGHTS,
            )
        except LocalEntryNotFoundError:
            cached = None
        if cached is not None and _has_pytorch_weights(cached):
            print(f"✅ {repo_id}: already cached")
            continue
        try:
            snapshot = snapshot_download(
                repo_id,
                revision=revision,
                local_files_only=False,
                ignore_patterns=_DUPLICATE_WEIGHTS,
            )
            # Prefer safetensors. Fall back to the bin only when that left no weights,
            # so models that ship both do not also store the duplicate ``.bin``.
            if not _has_pytorch_weights(snapshot):
                snapshot = snapshot_download(
                    repo_id,
                    revision=revision,
                    local_files_only=False,
                    ignore_patterns=["*.pt", "*.pth"],
                )
            if not _has_pytorch_weights(snapshot):
                raise OSError(f"{repo_id} snapshot has no pytorch weights")
            print(f"✅ {repo_id}: downloaded and cached")
        except Exception as e:
            print(f"⚠️ Warning: failed to cache {repo_id}: {e}")
            failed_models.append(repo_id)
    if failed_models:
        print(f"\n⚠️ Completed with warnings. Failed to cache: {failed_models}")
    else:
        print("\n🎉 All public models successfully cached!")


def _run_gated(config, config_file):
    """Cache gated_models entries."""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN secret is not available or empty.")
        sys.exit(1)
    print("the HF_TOKEN is non-empty, length:", len(token))
    force = os.getenv("FORCE_DOWNLOAD") == "true"
    models = config.get("gated_models", [])
    if not models:
        print(f"⚠️ Warning: No gated_models defined in {config_file}.")
        sys.exit(0)
    from huggingface_hub import snapshot_download

    print(f"📋 Found {len(models)} model(s) to cache:", models)
    failed_models = []
    for entry in models:
        repo_id, revision = _parse_entry(entry)
        print(f"\n🚀 Processing: {repo_id}...")
        try:
            # snapshot_download automatically reads and uses the HF_HOME env var
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                token=token,
                force_download=force,
                ignore_patterns=["*.pt", "*.pth", "*.bin"],
            )
            print(f"✅ Success: {repo_id} cache verified!")
        except Exception as e:
            print(f"❌ Failed to download {repo_id}: {e}")
            failed_models.append(repo_id)
    if failed_models:
        print(f"\n❌ Pipeline completed with errors. Failed models: {failed_models}")
        sys.exit(1)
    print("\n🎉 All models successfully processed and cached!")


# Per socket operation, so it bounds a stall rather than the whole transfer.
DOWNLOAD_TIMEOUT_S = 120

# urllib leaves `HTTPError.reason` empty on some Artifactory responses, so carry a
# short hint per code. Text only, no URL or token.
HTTP_HINTS = {
    401: "Unauthorized: check ARTIFACTORY_TOKEN",
    403: "Forbidden: token lacks read access to ARTIFACTORY_DATA_PATH",
    404: "Not Found: check ARTIFACTORY_DATA_PATH and the subdir",
}


def _sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_env(name):
    value = os.getenv(name)
    if not value:
        print(f"❌ Error: {name} is not available or empty.")
        sys.exit(1)
    return value


def _run_artifactory_datasets(config, config_file):
    """Cache artifactory_datasets entries into the shared runner volume.

    These files live in Artifactory, not on HuggingFace, so they need a token.
    The benchmark job that reads them runs on pull_request from forks, where
    GitHub withholds every secret, so it can only ever read the cache. This
    dispatch is the writer.
    """
    spec = config.get("artifactory_datasets") or {}
    files = spec.get("files", [])
    if not files:
        print(f"⚠️ Warning: No artifactory_datasets defined in {config_file}.")
        return

    dest_dir = os.getenv("SPYRE_BENCH_DATA_DIR")
    if not dest_dir:
        storage = os.getenv("STORAGE_1_DIR")
        if not storage:
            print("❌ Error: set SPYRE_BENCH_DATA_DIR or STORAGE_1_DIR.")
            sys.exit(1)
        dest_dir = os.path.join(storage, ".cache", "artifacts")

    base_url = _require_env("ARTIFACTORY_BASE_URL").rstrip("/")
    token = _require_env("ARTIFACTORY_TOKEN")
    root = _require_env("ARTIFACTORY_DATA_PATH").strip("/")

    subdir = spec.get("subdir", "").strip("/")
    # Mirror the Artifactory layout under the cache root so the path a benchmark
    # computes from the same subdir resolves without a second mapping.
    target_dir = Path(dest_dir) / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    force = os.getenv("FORCE_DOWNLOAD") == "true"

    print(f"📋 Found {len(files)} dataset(s) to cache into {target_dir}")
    failed = []
    for entry in files:
        name = entry["name"]
        # str(): YAML reads an all-digit digest as an int, which would never
        # equal the hexdigest and would re-download forever.
        expected = str(entry["sha256"])
        target = target_dir / name
        print(f"\n🚀 Processing: {name}...")

        if target.is_file() and not force:
            actual = _sha256_of(target)
            if actual == expected:
                print(f"✅ {name}: already cached")
                continue
            print(f"♻️ {name}: sha256 {actual} != {expected}, re-downloading")

        url = f"{base_url}/artifactory/{root}/{subdir}/{name}"
        # A partial file under the real name would look like a valid cache entry
        # to the benchmark, so download beside it and rename only after the
        # digest matches. The pid keeps two concurrent writers off one temp file.
        partial = target.with_name(f"{name}.{os.getpid()}.part")
        try:
            request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
            # Per socket operation, not for the whole transfer, so the 98MB
            # largest file is fine; without it a stalled connection would hang
            # until the job timeout hours later.
            with (
                urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response,
                open(partial, "wb") as handle,
            ):
                shutil.copyfileobj(response, handle)
            actual = _sha256_of(partial)
            if actual != expected:
                partial.unlink(missing_ok=True)
                raise ValueError(f"sha256 {actual}, expected {expected}")
            os.replace(partial, target)
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"✅ {name}: downloaded and verified ({size_mb:.1f}MB)")
        except (OSError, ValueError, urllib.error.URLError) as exc:
            partial.unlink(missing_ok=True)
            # Never echo the exception text: the URL carries the internal host and
            # a TLS or proxy error quotes it back ("hostname '<host>' doesn't
            # match ..."). Report the HTTP code where there is one, since
            # Artifactory answers a bad token with a 401 whose `reason` is empty,
            # and otherwise just the exception type plus errno.
            if isinstance(exc, urllib.error.HTTPError):
                reason = f"HTTP {exc.code} {exc.reason or HTTP_HINTS.get(exc.code, '')}".strip()
            else:
                errno = getattr(exc, "errno", None) or getattr(
                    getattr(exc, "reason", None), "errno", None
                )
                reason = type(exc).__name__ + (f" (errno {errno})" if errno else "")
            print(f"❌ Failed to download {name}: {reason}")
            failed.append(name)

    if failed:
        print(f"\n❌ Pipeline completed with errors. Failed datasets: {failed}")
        sys.exit(1)
    print("\n🎉 All datasets successfully processed and cached!")


if __name__ == "__main__":
    main()
