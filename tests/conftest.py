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

from functools import cache
from pathlib import Path

import pytest
import yaml
from spyre_testing_plugin.tags import model_from_params, result_tags

from spyre_inference import envs
from spyre_inference.v1.worker import compile_guard

_CACHE_CONFIG = (
    Path(__file__).resolve().parent.parent
    / ".github"
    / "cache_config"
    / "hf_models_and_datasets.yaml"
)


def _has_hf_token() -> bool:
    import huggingface_hub

    return huggingface_hub.get_token() is not None


@cache
def _gated_models() -> set[str]:
    if not _CACHE_CONFIG.exists():
        return set()
    try:
        raw = yaml.safe_load(_CACHE_CONFIG.read_text()) or {}
        entries = raw.get("gated_models", [])
        return {
            entry if isinstance(entry, str) else entry["repo"]
            for entry in entries
            if isinstance(entry, str) or (isinstance(entry, dict) and "repo" in entry)
        }
    except Exception:
        return set()


def _is_gated_model(model: str) -> bool:
    return model in _gated_models()


@cache
def _is_model_available(model: str) -> bool:
    if not _is_gated_model(model):
        return True
    if _has_hf_token():
        return True
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model, local_files_only=True)
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(items):
    """Skip batched decode until it composes with the default tiled page walk.

    Also stamps each item's JUnit tags here, not in a fixture: pytest skips a
    function-scoped autouse fixture entirely for a marked skip, a true skipif,
    xfail(run=False), or when an earlier fixture already failed, so a fixture-based
    tagger silently drops tags on exactly the cases that most need them. Collection
    time has no such gap, same as the upstream-test tagging in pytest_plugin.py.
    """
    skip = pytest.mark.skip(
        reason="will be re-enabled once batched decode is working with for_each_tile"
    )
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        for name, value in result_tags(params):
            item.user_properties.append((name, value))

        # Match only the test id/class/parameter portion, not the file path: the
        # global skip must not exclude the card-free entry-local CPU tests by filename.
        nodeid = item.nodeid.partition("::")[2].lower()
        if (
            "batched_decode" in nodeid
            or "batcheddecode" in nodeid
            or "enable_batched_decode" in item.fixturenames
            or "batched_decode_calls" in item.fixturenames
        ):
            item.add_marker(skip)

        model = model_from_params(params)
        if model and not _is_model_available(model):
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"Access to gated model '{model}' requires an authorized "
                        "Hugging Face token or local cache."
                    )
                )
            )


@pytest.fixture(autouse=True)
def _clear_env_cache():
    """envs.py caches each SPYRE_* value on first read; drop the cache around
    every test so monkeypatched vars take effect and don't leak between tests."""
    envs.clear_env_cache()
    yield
    envs.clear_env_cache()


@pytest.fixture(autouse=True)
def _disarm_compile_guard():
    """The compile guard installs a process-wide Dynamo callback, so an armed guard
    leaking out of a test would raise inside an unrelated one.

    Only the arming is undone, never the registry: `watch` runs at import for the
    attention kernels and once per memoized kernel elsewhere (`_compiled_kernels`,
    `layer.spyre_moe_regions`, `self.spyre_compiled_kernel`), and those memos outlive
    the test, so a registration dropped here never comes back and a later
    warmup-coverage test would pass with its kernel silently unwatched. Clearing the
    reported-once set is enough for isolation: it is what makes one violation log
    once, so leaving it would mute an expected report in the next test.
    """
    compile_guard.disarm()
    yield
    compile_guard.disarm()
    compile_guard.clear_reported()
