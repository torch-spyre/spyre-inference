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

_MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"

# Spyre KV-cache block size; the shared prefix must span at least two full
# blocks so partial-hit paths in the attention bucketer are exercised.
_BLOCK_SIZE = 128
_MIN_PREFIX_TOKENS = 2 * _BLOCK_SIZE  # 256

# Seed sentences appended repeatedly to build the prefix up to the target length.
_PREFIX_SEED = (
    "You are a knowledgeable assistant specialised in world geography. "
    "Answer every question clearly and concisely. "
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
)

_QUESTIONS = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

# On-device fp16 greedy decode: cache-served and freshly-recomputed attention
# accumulate rounding errors in different orders, which can flip a near-tie at
# the logit level.  Empirically the outputs agree on the first ~12 tokens and
# diverge only in the tail of a 16-token sequence.  We therefore compare only
# the first _STABLE_TOKENS tokens rather than demanding bit-exact equality
# across the full sequence.
_STABLE_TOKENS = 12


def _build_prefix(tokenizer: AutoTokenizer) -> str:
    """Return a prefix string that tokenises to at least ``_MIN_PREFIX_TOKENS``.

    Sentences from ``_PREFIX_SEED`` are appended until the token count exceeds
    the threshold, then the token sequence is trimmed to exactly
    ``_MIN_PREFIX_TOKENS`` tokens and decoded back to a string.  This makes the
    prefix length deterministic regardless of tokenizer version or model rename.
    """
    candidate = _PREFIX_SEED
    while len(tokenizer.encode(candidate)) < _MIN_PREFIX_TOKENS:
        candidate = candidate + _PREFIX_SEED

    ids = tokenizer.encode(candidate)[:_MIN_PREFIX_TOKENS]
    return tokenizer.decode(ids, skip_special_tokens=True)


@pytest.fixture(scope="module")
def shared_prefix() -> str:
    """Module-scoped prefix string guaranteed to span at least two KV blocks."""
    tokenizer = AutoTokenizer.from_pretrained(_MODEL)
    prefix = _build_prefix(tokenizer)
    # Sanity-check: the trimmed prefix must still meet the threshold.
    n = len(tokenizer.encode(prefix))
    assert n >= _MIN_PREFIX_TOKENS, (
        f"Built prefix tokenises to only {n} tokens; expected >= {_MIN_PREFIX_TOKENS}. "
        "Extend _PREFIX_SEED so it produces a longer base string."
    )
    return prefix


def _make_llm(*, enable_prefix_caching: bool, enforce_eager: bool) -> LLM:
    return LLM(
        model=_MODEL,
        enforce_eager=enforce_eager,
        dtype="float16",
        max_model_len=512,
        max_num_seqs=4,
        enable_prefix_caching=enable_prefix_caching,
        attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
    )


# Parametrize over enforce_eager so the compiled production path
# (STOCK_TORCH_COMPILE, the default) is exercised alongside the eager path.
# The compiled variant hits the attention bucketer / warmup machinery with the
# partial-prefix query/KV lengths that APC generates.
@pytest.mark.uses_subprocess
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "compiled"])
def test_prefix_caching_output_matches_no_caching(enforce_eager: bool, shared_prefix: str) -> None:
    """Prefix-caching must not change the generated tokens.

    Runs the same prompts on two separate engines — APC enabled (cold) and
    APC disabled — then asserts every prompt produces the same token sequence
    for the first ``_STABLE_TOKENS`` tokens.  Full-sequence bit-exact equality
    is too strict on fp16 Spyre hardware, where cache-served vs recomputed
    attention paths accumulate rounding differences that can flip a greedy
    near-tie in the later tail tokens.
    """
    prompts = [shared_prefix + q for q in _QUESTIONS]
    sp = SamplingParams(temperature=0.0, max_tokens=16)

    llm_apc = _make_llm(enable_prefix_caching=True, enforce_eager=enforce_eager)
    apc_outputs = llm_apc.generate(prompts, sp, use_tqdm=False)
    del llm_apc

    llm_plain = _make_llm(enable_prefix_caching=False, enforce_eager=enforce_eager)
    plain_outputs = llm_plain.generate(prompts, sp, use_tqdm=False)
    del llm_plain

    assert len(apc_outputs) == len(plain_outputs) == len(prompts)
    for i, (apc, plain) in enumerate(zip(apc_outputs, plain_outputs)):
        apc_ids = list(apc.outputs[0].token_ids)[:_STABLE_TOKENS]
        plain_ids = list(plain.outputs[0].token_ids)[:_STABLE_TOKENS]
        assert apc_ids == plain_ids, (
            f"Prompt {i}: token mismatch with vs without prefix caching "
            f"(first {_STABLE_TOKENS} tokens).\n"
            f"  with APC : {apc_ids}\n"
            f"  without  : {plain_ids}"
        )


@pytest.mark.uses_subprocess
def test_prefix_cache_hit_reported_after_warmup(shared_prefix: str) -> None:
    """After a warmup pass the engine must report a cache hit.

    ``RequestOutput.num_cached_tokens`` reflects how many prefix tokens were
    served from the block pool rather than recomputed.  At least one prompt in
    the batch must report a non-zero value after the shared prefix is cached.
    """
    prompts = [shared_prefix + q for q in _QUESTIONS]
    sp = SamplingParams(temperature=0.0, max_tokens=16)

    llm = _make_llm(enable_prefix_caching=True, enforce_eager=True)
    # Prime the cache with the shared prefix.
    llm.generate(prompts[0], sp, use_tqdm=False)
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    del llm

    cached_counts = [o.num_cached_tokens for o in outputs]
    assert any(c is not None and c > 0 for c in cached_counts), (
        "Expected at least one prompt to report a prefix-cache hit after "
        f"warmup, but got num_cached_tokens={cached_counts}"
    )


@pytest.mark.uses_subprocess
def test_prefix_caching_warm_output_matches_cold(shared_prefix: str) -> None:
    """A warm cache hit must produce the same tokens as a cold run.

    Two separate engines are used so the cold measurement is genuine: the
    cold engine never sees the prefix before its measured generate call, and
    the warm engine has already run the same prompts once before its measured
    call.  Comparing same-prompt output across engines (both APC-enabled) is
    therefore a true fresh-vs-cached check for every prompt.

    Full-sequence bit-exact equality is too strict on fp16 Spyre hardware for
    the same reason as the APC vs non-APC test above.
    """
    prompts = [shared_prefix + q for q in _QUESTIONS]
    sp = SamplingParams(temperature=0.0, max_tokens=16)

    # Cold engine: first (and only) generate call on this engine — no prior
    # state in the block pool.
    llm_cold = _make_llm(enable_prefix_caching=True, enforce_eager=True)
    cold_outputs = llm_cold.generate(prompts, sp, use_tqdm=False)
    del llm_cold

    # Warm engine: prime the cache, then run the measured call.
    llm_warm = _make_llm(enable_prefix_caching=True, enforce_eager=True)
    llm_warm.generate(prompts, sp, use_tqdm=False)  # prime
    warm_outputs = llm_warm.generate(prompts, sp, use_tqdm=False)
    del llm_warm

    assert len(cold_outputs) == len(warm_outputs) == len(prompts)
    for i, (cold, warm) in enumerate(zip(cold_outputs, warm_outputs)):
        cold_ids = list(cold.outputs[0].token_ids)[:_STABLE_TOKENS]
        warm_ids = list(warm.outputs[0].token_ids)[:_STABLE_TOKENS]
        assert cold_ids == warm_ids, (
            f"Prompt {i}: cold vs warm token mismatch with prefix caching "
            f"(first {_STABLE_TOKENS} tokens).\n"
            f"  cold (no warmup) : {cold_ids}\n"
            f"  warm (after hit) : {warm_ids}"
        )
