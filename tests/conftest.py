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

import math

import pytest
from spyre_testing_plugin.tags import result_tags

from spyre_inference import envs


@pytest.fixture(autouse=True)
def _clear_env_cache():
    """envs.py caches each SPYRE_* value on first read; drop the cache around
    every test so monkeypatched vars take effect and don't leak between tests."""
    envs.clear_env_cache()
    yield
    envs.clear_env_cache()


@pytest.fixture(autouse=True)
def _emit_result_tags(request, record_property):
    """Autouse: stamp each local test's `model__`/`testtype__` JUnit tags (see
    spyre_testing_plugin.tags). Upstream tests are tagged in the plugin's
    collection hook instead."""
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    for name, value in result_tags(params):
        record_property(name, value)


@pytest.fixture(scope="session")
def hf_runner():
    """Upstream's ``HfRunner``, resolved lazily so the tree is only cloned when a test
    that needs it is collected."""
    from spyre_testing_plugin.upstream import ensure_upstream_tests_importable

    ensure_upstream_tests_importable()
    from tests.conftest import HfRunner

    class CpuHfRunner(HfRunner):
        def get_default_device(self):
            return "cpu"

    return CpuHfRunner


@pytest.fixture
def example_prompts() -> list[str]:
    """Upstream's ``example_prompts``, read with ``readlines()`` as its ``_read_prompts``
    does -- trailing newlines included, since those are the strings its model tests send."""
    from spyre_testing_plugin.upstream import ensure_upstream_tests_importable

    tests_dir = ensure_upstream_tests_importable()
    with open(tests_dir / "prompts" / "example.txt") as f:
        return f.readlines()


@pytest.fixture
def hf_embeddings(hf_runner):
    """Live CPU HF embeddings for `(model, revision, prompts)`.

    `is_sentence_transformer=True` applies the checkpoint's own pooling and normalization,
    matching what the vLLM side runs when no `pooler_config` overrides it. Prompts must
    arrive stripped: sentence-transformers strips its inputs, so otherwise the two sides
    tokenize different text.
    """

    def _embed(model: str, revision: str, prompts: list[str]) -> list[list[float]]:
        with hf_runner(model, revision=revision, is_sentence_transformer=True) as hf_model:
            return hf_model.encode(prompts)

    return _embed


@pytest.fixture
def assert_embeddings_close():
    """Upstream's `check_embeddings_close`, plus a finiteness check it omits."""
    from spyre_testing_plugin.upstream import ensure_upstream_tests_importable

    ensure_upstream_tests_importable()
    from tests.models.utils import check_embeddings_close

    def _assert(label: str, embeddings, refs, tol: float = 1e-2) -> None:
        for embedding in embeddings:
            assert all(math.isfinite(x) for x in embedding), f"{label}: non-finite embedding value"
        check_embeddings_close(
            embeddings_0_lst=refs,
            embeddings_1_lst=embeddings,
            name_0=f"hf ({label})",
            name_1=f"spyre ({label})",
            tol=tol,
        )

    return _assert


@pytest.fixture(scope="session")
def vllm_runner():
    """Upstream's ``VllmRunner``, from the pinned vLLM ``tests/`` tree."""
    from spyre_testing_plugin.upstream import ensure_upstream_tests_importable

    ensure_upstream_tests_importable()
    from tests.conftest import VllmRunner

    return VllmRunner
