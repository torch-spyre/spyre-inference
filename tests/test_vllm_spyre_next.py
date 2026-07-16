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

from vllm import LLM, RequestOutput, SamplingParams
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.config import AttentionConfig

import pytest


@pytest.mark.uses_subprocess
def test_basic_model_load():
    model = LLM(
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        max_model_len=128,
        max_num_seqs=2,
        attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
    )

    sampling_params = SamplingParams(max_tokens=5)
    output: list[RequestOutput] = model.generate(
        prompts="Hello World", sampling_params=sampling_params
    )

    assert len(output[0].outputs[0].text) > 0


@pytest.mark.uses_subprocess
def test_long_context_model_load():
    """Verify that user-specified large max_model_len values are honored, and
    that long contexts don't crash."""
    model = LLM(
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        max_model_len=131072,
        max_num_seqs=8,
        attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
    )

    sampling_params = SamplingParams(max_tokens=32)
    output: list[RequestOutput] = model.generate(
        prompts="Hello World", sampling_params=sampling_params
    )

    assert len(output[0].outputs[0].text) > 0
