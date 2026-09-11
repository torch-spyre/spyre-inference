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
from vllm import LLM, RequestOutput, SamplingParams
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@pytest.mark.uses_subprocess
def test_basic_model_load():
    model = LLM(
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        max_model_len=128,
        max_num_seqs=2,
        enforce_eager=True,
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
        enforce_eager=True,
        attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
    )

    sampling_params = SamplingParams(max_tokens=32)
    output: list[RequestOutput] = model.generate(
        prompts="Hello World", sampling_params=sampling_params
    )

    assert len(output[0].outputs[0].text) > 0


# On-device counterpart to the CPU pad-shim unit tests: qwrt/Swedish0.1M is the
# smallest public model that trips BOTH pads at once — head_dim 16 -> 128 (QK-norm)
# and intermediate_size 160 -> 192 (SwiGLU) — and must still decode correctly.
# Its tokenizer stub is broken (returns [] for everything); the model is byte-level
# (vocab 256), so the prompt is fed as raw UTF-8 byte ids, not through the tokenizer.
_PADDED_PROMPT_TOKEN_IDS = list("Sverige är ett land i norra Europa".encode())

# Greedy continuation from transformers CPU (fp32, unpadded) on the same byte-id
# prompt, kept only up to the first near-tie.
_PADDED_REFERENCE_TOKEN_IDS = [32, 111, 99, 104, 32]


@pytest.mark.uses_subprocess
def test_padded_head_dim_and_intermediate_size_generate() -> None:
    """Loading the model fires both pads; greedy decode matches the unpadded
    reference token ids."""
    llm = LLM(
        model="qwrt/Swedish0.1M",
        dtype="float16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=1,
    )

    # Both padding passes must have run during check_and_update_config.
    hf_config = llm.llm_engine.model_config.hf_config
    assert hf_config.head_dim == 128
    assert hf_config._spyre_orig_head_dim == 16
    assert hf_config.intermediate_size == 192
    assert hf_config._spyre_orig_intermediate_size == 160

    sp = SamplingParams(temperature=0.0, max_tokens=len(_PADDED_REFERENCE_TOKEN_IDS))
    outputs = llm.generate({"prompt_token_ids": _PADDED_PROMPT_TOKEN_IDS}, sp, use_tqdm=False)

    assert list(outputs[0].outputs[0].token_ids) == _PADDED_REFERENCE_TOKEN_IDS
