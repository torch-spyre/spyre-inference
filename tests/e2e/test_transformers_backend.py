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

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]

# The two paths are not bit-identical: they run different module code (HF's vs vLLM's)
# and round their rotation caches differently, so greedy sampling eventually tie-breaks
# apart. The failure mode being guarded against diverges from the first token or two.
MAX_TOKENS = 8


def _generate_greedy(model: str, model_impl: str) -> list[list[int]]:
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
    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0))
    token_ids = [list(o.outputs[0].token_ids) for o in outputs]

    del llm
    cleanup_dist_env_and_memory()
    return token_ids


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
    transformers_ids = _generate_greedy(model, "transformers")
    native_ids = _generate_greedy(model, "vllm")

    assert transformers_ids == native_ids
