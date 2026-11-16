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
import os
import sys
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
        _run_artifactory_datasets()
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


def _run_artifactory_datasets():
    """Cache the benchmark datasets into the shared runner volume.

    The datasets are the Asset definitions under benchmarks/, resolved by
    .github/scripts/fetch_bench_datasets.py.
    They need a token, and the benchmark job that reads them runs on pull_request from forks, where
    GitHub withholds
    every secret, so it can only read the cache. This dispatch is the writer.
    """
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "fetch_bench_datasets.py"
    spec = importlib.util.spec_from_file_location("fetch_bench_datasets", script)
    fetcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetcher)

    dest_dir = os.getenv("SPYRE_BENCH_DATA_DIR")
    if not dest_dir:
        storage = os.getenv("STORAGE_1_DIR")
        if not storage:
            print("❌ Error: set SPYRE_BENCH_DATA_DIR or STORAGE_1_DIR.")
            sys.exit(1)
        dest_dir = os.path.join(storage, ".cache", "artifacts")

    force = os.getenv("FORCE_DOWNLOAD") == "true"
    datasets = {**fetcher.ARTIFACTS, **fetcher.ARCHIVES}
    print(f"📋 Found {len(datasets)} dataset(s) to cache under {Path(dest_dir)}")
    failed = []
    for var, entry in datasets.items():
        print(f"\n🚀 Processing: {var}...")
        try:
            outcome = fetcher.ensure_cached(entry, Path(dest_dir), force)
            print(f"✅ {var}: {outcome} and verified")
        except (fetcher.FetchUnavailable, OSError) as exc:
            # FetchUnavailable messages carry the dataset path and an HTTP code, never the store
            # URL.
            print(f"❌ Failed to cache {var}: {exc}")
            failed.append(var)

    if failed:
        print(f"\n❌ Pipeline completed with errors. Failed datasets: {failed}")
        sys.exit(1)
    print("\n🎉 All datasets successfully processed and cached!")


if __name__ == "__main__":
    main()
