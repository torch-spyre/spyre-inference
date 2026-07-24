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

import os

from vllm import LLM, RequestOutput, SamplingParams
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.config import AttentionConfig

import pytest


def _spyre_device_count() -> int:
    """Number of visible Spyre cards, or 0 if unavailable.

    Reads AIU_WORLD_SIZE (set by the Spyre runtime environment) instead of
    touching the Spyre runtime, so `uses_subprocess` tests don't import
    torch_spyre in the main pytest process.
    """
    try:
        return int(os.environ.get("AIU_WORLD_SIZE", "0"))
    except ValueError:
        return 0


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


@pytest.mark.uses_subprocess
@pytest.mark.skipif(_spyre_device_count() < 1, reason="needs a Spyre card")
def test_batched_decode_does_not_exhaust_compile_cache():
    """Regression: batched decode must not exhaust the dynamo compile cache.

    The KV-cache scatter once used ``torch.ops.spyre.overwrite``, which compiles
    one binary per unique slot offset. Those recompiles overflow dynamo's
    ``accumulated_recompile_limit`` (256), after which the eager fallback
    re-dispatches into itself and recurses (``RecursionError``), killing the
    engine mid-batch — originally hit via ``vllm bench serve`` with 10 prompts.

    This workload is strictly heavier (12 varied ~128-token prompts, 32 output
    tokens), so it crossed the limit pre-fix and must now run to completion.
    """
    model = LLM(
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        max_model_len=2048,
        max_num_seqs=4,
        attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
    )

    # Keep these long and varied: the crash needs ~256 cumulative recompiles,
    # overwrite alone caps near 128, so the rest must come from batch-shape
    # variety. Shorter prompts (validated at ~16-60 tokens) do NOT cross the
    # limit and stop catching the regression.
    prompts = [("word " * (120 + i * 5)).strip() for i in range(12)]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
    output: list[RequestOutput] = model.generate(prompts=prompts, sampling_params=sampling_params)

    assert len(output) == len(prompts)
    assert all(len(o.outputs[0].text) > 0 for o in output)
