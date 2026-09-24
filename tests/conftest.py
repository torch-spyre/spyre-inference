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

import pytest
from spyre_testing_plugin.tags import result_tags

from spyre_inference import envs
from spyre_inference.v1.worker import compile_guard


def pytest_collection_modifyitems(items):
    """Skip batched decode until it composes with the default tiled page walk."""
    skip = pytest.mark.skip(
        reason="will be re-enabled once batched decode is working with for_each_tile"
    )
    for item in items:
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


@pytest.fixture(autouse=True)
def _emit_result_tags(request, record_property):
    """Autouse: stamp each local test's `model__`/`testtype__` JUnit tags (see
    spyre_testing_plugin.tags). Upstream tests are tagged in the plugin's
    collection hook instead."""
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    for name, value in result_tags(params):
        record_property(name, value)
