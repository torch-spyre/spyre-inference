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
import yaml
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public",
        action="store_true",
        help="Cache public_models",
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

    if args.public:
        _run_public(config, config_file)
    else:
        _run_gated(config, config_file)


def _run_public(config, config_file):
    """Cache public_models entries."""
    models = config.get("public_models", [])
    if not models:
        print(f"⚠️ Warning: No public_models defined in {config_file}.")
        return
    print(f"📋 Found {len(models)} public model(s) to cache:", models)
    failed_models = []
    for repo_id in models:
        print(f"\n🚀 Processing: {repo_id}...")
        try:
            snapshot_download(repo_id, local_files_only=True, ignore_patterns=["*.pt", "*.bin"])
            print(f"✅ {repo_id}: already cached")
            continue
        except LocalEntryNotFoundError:
            pass
        try:
            snapshot_download(repo_id, local_files_only=False, ignore_patterns=["*.pt", "*.bin"])
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
    print(f"📋 Found {len(models)} model(s) to cache:", models)
    failed_models = []
    for repo_id in models:
        print(f"\n🚀 Processing: {repo_id}...")
        try:
            # snapshot_download automatically reads and uses the HF_HOME env var
            snapshot_download(
                repo_id=repo_id,
                token=token,
                force_download=force,
                ignore_patterns=["*.pt", "*.bin"],
            )
            print(f"✅ Success: {repo_id} cache verified!")
        except Exception as e:
            print(f"❌ Failed to download {repo_id}: {e}")
            failed_models.append(repo_id)
    if failed_models:
        print(f"\n❌ Pipeline completed with errors. Failed models: {failed_models}")
        sys.exit(1)
    print("\n🎉 All models successfully processed and cached!")


if __name__ == "__main__":
    main()
