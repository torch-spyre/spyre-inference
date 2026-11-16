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

Stands in for upstream's ``tests/models/transformers/test_backend.py``, which is
disabled in ``upstream_tests.yaml``: it compares against an HF CPU reference over 32
tokens of logprobs, which fp16 on Spyre is unlikely to satisfy. The native Spyre path
is the reference here instead.
"""

from __future__ import annotations

import pytest
from spyre_testing_plugin.greedy_checks import Trace, assert_greedy_runs_agree, logprob_trace

# Wide top-1/top-2 margin at every step, so no near-tie decides the comparison.
PROMPTS = [
    "Count to ten: one, two, three",
    "1, 2, 3, 4, 5, 6,",
]

# The paths run different module code and round differently, so a near-tie may flip; a real
# break puts the other path's token outside the top-k at once.
MAX_TOKENS = 8
NUM_LOGPROBS = 5
LOGPROB_TOLERANCE = 0.1


def _generate_greedy(model: str, model_impl: str) -> Trace:
    from vllm import LLM, SamplingParams
    from vllm.distributed import cleanup_dist_env_and_memory

    llm = LLM(
        model=model,
        dtype="float16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=2,
        model_impl=model_impl,
    )
    assert llm.llm_engine.model_config.using_transformers_backend() == (
        model_impl == "transformers"
    )
    outputs = llm.generate(
        PROMPTS, SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, logprobs=NUM_LOGPROBS)
    )
    trace = logprob_trace(outputs)

    del llm
    cleanup_dist_env_and_memory()
    return trace


@pytest.mark.uses_subprocess
@pytest.mark.parametrize(
    "model",
    [
        # head_dim=64 takes the padded branch. The unpadded one (micro-g3.3, 128) is
        # covered by upstream's test_backend.py::test_models in upstream_tests.yaml.
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
)
def test_transformers_matches_native(model: str) -> None:
    """The Transformers backend must generate what the native Spyre path does.

    Content, not just non-empty output: a broken RoPE or a norm falling back to an
    unsupported fp32 promotion still yields fluent text, just unrelated to the prompt.
    """
    transformers = _generate_greedy(model, "transformers")
    native = _generate_greedy(model, "vllm")
    assert_greedy_runs_agree(
        native, transformers, LOGPROB_TOLERANCE, names=("native", "transformers")
    )
