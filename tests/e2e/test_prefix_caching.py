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

"""Prefix-caching (APC) end-to-end tests for the Spyre backend."""

import pytest

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = pytest.mark.prefix_caching

_MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"

# A shared prefix that spans at least two full Spyre KV-cache blocks
# (block_size=128).  The module-level assertion below verifies this.
_SHARED_PREFIX = (
    "You are a knowledgeable assistant specialised in world geography. "
    "Answer every question clearly and concisely. "
    "Here are some facts you should keep in mind: "
    "France is in Western Europe and its capital is Paris. "
    "Germany is in Central Europe and its capital is Berlin. "
    "Japan is in East Asia and its capital is Tokyo. "
    "Australia is a continent and a country; its capital is Canberra. "
    "Brazil is the largest country in South America; its capital is Brasília. "
    "Canada is the second largest country in the world; its capital is Ottawa. "
    "China is the most populous country in Asia; its capital is Beijing. "
    "India is the second most populous country in the world; its capital is New Delhi. "
    "The United States of America spans North America; its capital is Washington D.C. "
    "Russia is the largest country in the world by area; its capital is Moscow. "
    "The Nile is the longest river in Africa, flowing through eleven countries "
    "before emptying into the Mediterranean Sea near Alexandria, Egypt. "
    "Mount Everest, located in the Himalayas on the Nepal-Tibet border, is the "
    "highest point on Earth at 8 848 metres above sea level. "
    "The Amazon rainforest covers more than five million square kilometres across "
    "nine South American countries and produces roughly twenty percent of the "
    "world's oxygen. "
    "The Sahara is the largest hot desert on Earth, stretching across much of "
    "northern Africa from the Atlantic coast to the Red Sea. "
    "Now answer the following question: "
)

_PROMPTS = [
    _SHARED_PREFIX + "What is the capital of France?",
    _SHARED_PREFIX + "What is the capital of Germany?",
]

# Verify at import time that the prefix comfortably covers two KV-cache blocks
# (2 × 128 = 256).  Fails immediately with a clear message if the prefix is
# ever shortened below the threshold.
_n_prefix_tokens = len(AutoTokenizer.from_pretrained(_MODEL).encode(_SHARED_PREFIX))
assert _n_prefix_tokens > 256, (
    f"_SHARED_PREFIX tokenises to only {_n_prefix_tokens} tokens, which does "
    "not comfortably cover two Spyre KV-cache blocks (2×128=256).  "
    "Lengthen the prefix so it produces more than 256 tokens."
)


def _make_llm(*, enable_prefix_caching: bool) -> LLM:
    return LLM(
        model=_MODEL,
        enforce_eager=True,
        dtype="float16",
        max_model_len=512,
        max_num_seqs=4,
        enable_prefix_caching=enable_prefix_caching,
        attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
    )


@pytest.mark.uses_subprocess
def test_prefix_caching_output_matches_no_caching() -> None:
    """Prefix-caching must not change the generated tokens.

    Runs the same prompts on two separate engines — APC enabled (cold) and
    APC disabled — then asserts every prompt produces the same token sequence.
    """
    sp = SamplingParams(temperature=0.0, max_tokens=16)

    llm_apc = _make_llm(enable_prefix_caching=True)
    apc_outputs = llm_apc.generate(_PROMPTS, sp, use_tqdm=False)
    del llm_apc

    llm_plain = _make_llm(enable_prefix_caching=False)
    plain_outputs = llm_plain.generate(_PROMPTS, sp, use_tqdm=False)
    del llm_plain

    assert len(apc_outputs) == len(plain_outputs) == len(_PROMPTS)
    for i, (apc, plain) in enumerate(zip(apc_outputs, plain_outputs)):
        apc_ids = list(apc.outputs[0].token_ids)
        plain_ids = list(plain.outputs[0].token_ids)
        assert apc_ids == plain_ids, (
            f"Prompt {i}: token mismatch with vs without prefix caching.\n"
            f"  with APC : {apc_ids}\n"
            f"  without  : {plain_ids}"
        )


@pytest.mark.uses_subprocess
def test_prefix_cache_hit_reported_after_warmup() -> None:
    """After a warmup pass the engine must report a cache hit.

    ``RequestOutput.num_cached_tokens`` reflects how many prefix tokens were
    served from the block pool rather than recomputed.  At least one prompt in
    the batch must report a non-zero value after the shared prefix is cached.
    """
    sp = SamplingParams(temperature=0.0, max_tokens=16)

    llm = _make_llm(enable_prefix_caching=True)
    # Prime the cache with the shared prefix.
    llm.generate(_PROMPTS[0], sp, use_tqdm=False)
    outputs = llm.generate(_PROMPTS, sp, use_tqdm=False)
    del llm

    cached_counts = [o.num_cached_tokens for o in outputs]
    assert any(c is not None and c > 0 for c in cached_counts), (
        "Expected at least one prompt to report a prefix-cache hit after "
        f"warmup, but got num_cached_tokens={cached_counts}"
    )


@pytest.mark.uses_subprocess
def test_prefix_caching_warm_output_matches_cold() -> None:
    """A warm cache hit must produce the same tokens as a cold run.

    Both passes use the same engine: the cold run fires first (no prior
    generate on this engine), then a warmup call primes the prefix cache
    before the warm run.  The two output sequences must be bit-identical.
    """
    sp = SamplingParams(temperature=0.0, max_tokens=16)

    llm = _make_llm(enable_prefix_caching=True)
    cold_outputs = llm.generate(_PROMPTS, sp, use_tqdm=False)
    # Prime the cache, then re-run the same prompts.
    llm.generate(_PROMPTS[0], sp, use_tqdm=False)
    warm_outputs = llm.generate(_PROMPTS, sp, use_tqdm=False)
    del llm

    assert len(cold_outputs) == len(warm_outputs) == len(_PROMPTS)
    for i, (cold, warm) in enumerate(zip(cold_outputs, warm_outputs)):
        cold_ids = list(cold.outputs[0].token_ids)
        warm_ids = list(warm.outputs[0].token_ids)
        assert cold_ids == warm_ids, (
            f"Prompt {i}: cold vs warm token mismatch with prefix caching.\n"
            f"  cold (no warmup) : {cold_ids}\n"
            f"  warm (after hit) : {warm_ids}"
        )
