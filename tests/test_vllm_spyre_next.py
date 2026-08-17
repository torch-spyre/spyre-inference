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
@pytest.mark.skip(
    reason=(
        "The dense paged KV cache does not fit Spyre's per-core tensor span limit "
        "at long context. max_model_len=131072 allocates one "
        "[8192, 8, 128, 128] fp16 tensor per layer (1024 MB), and the backend "
        "rejects it with 'per-core tensor span 1024.000 MB ... exceeds hardware "
        "limit of 256.00 MB' — work division cannot split the coordinates "
        "further. The previous list-of-one-tensor-per-page layout stayed under "
        "the limit because each page was its own small allocation, but it could "
        "not express an indirect page gather. Unskip once the cache is chunked "
        "into <=256 MB tensors, or once multi-core indirect access lands "
        "(torch-spyre#2725, torch-spyre#3499)."
    )
)
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
