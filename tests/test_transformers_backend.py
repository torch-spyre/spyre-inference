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

"""Tests for the HuggingFace Transformers backend (model_impl='transformers').

TODO: Delete this file once https://github.com/torch-spyre/spyre-inference/issues/324
is resolved and re-enable the upstream tests in upstream_tests.yaml.
"""

from __future__ import annotations

import pytest


@pytest.mark.uses_subprocess
@pytest.mark.parametrize(
    "model",
    [
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
)
def test_transformers_generate(model: str) -> None:
    """Verify model_impl='transformers' loads and generates non-empty output."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        dtype="float16",
        enforce_eager=False,
        max_model_len=128,
        max_num_seqs=2,
        model_impl="transformers",
    )
    model_config = llm.llm_engine.model_config
    assert model_config.using_transformers_backend()

    sp = SamplingParams(max_tokens=8, temperature=0.0)
    outputs = llm.generate(["Hello, world!"], sp)
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0
