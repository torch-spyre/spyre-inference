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
# the logit level.  We therefore compare only the first _STABLE_TOKENS tokens,
# and accept a divergence there only when it is such a tie: the token each
# engine picked must be within _TIE_TOLERANCE (in logprob) of the other
# engine's best candidate.  Measured APC-vs-plain logprob noise on this
# hardware is up to ~0.06, with or without --load-format spyre.
_STABLE_TOKENS = 12
_TIE_TOLERANCE = 0.1


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


def _within_tie(logprobs: dict, token_id: int) -> bool:
    """Whether ``token_id`` is within ``_TIE_TOLERANCE`` of the best candidate."""
    if token_id not in logprobs:
        return False
    best = max(lp.logprob for lp in logprobs.values())
    return best - logprobs[token_id].logprob <= _TIE_TOLERANCE


def _assert_agrees_with_baseline(label: str, i: int, got, plain) -> None:
    """``got`` must match ``plain`` over the first ``_STABLE_TOKENS`` tokens, except that
    a divergence on a near-tie (both engines' picks within ``_TIE_TOLERANCE`` of each
    other's best candidate) is accepted. Nothing after it is compared: the contexts differ."""
    got_out, plain_out = got.outputs[0], plain.outputs[0]
    got_ids = list(got_out.token_ids)[:_STABLE_TOKENS]
    plain_ids = list(plain_out.token_ids)[:_STABLE_TOKENS]
    if got_ids == plain_ids:
        return
    k = next((j for j, (g, p) in enumerate(zip(got_ids, plain_ids)) if g != p), None)
    is_tie = k is not None and (
        _within_tie(plain_out.logprobs[k], got_ids[k])
        and _within_tie(got_out.logprobs[k], plain_ids[k])
    )
    assert is_tie, (
        f"Prompt {i}: {label} output does not match non-APC baseline "
        f"(first {_STABLE_TOKENS} tokens, first divergence at token {k}).\n"
        f"  {label:<9}: {got_ids}\n"
        f"  plain    : {plain_ids}"
    )


def _fake_request_output(ids, logprobs_per_position):
    """A minimal RequestOutput stand-in: token ids plus {token_id: logprob} per position."""
    from types import SimpleNamespace

    out = SimpleNamespace(
        token_ids=ids,
        logprobs=[
            {t: SimpleNamespace(logprob=v) for t, v in position.items()}
            for position in logprobs_per_position
        ],
    )
    return SimpleNamespace(outputs=[out])


def test_baseline_check_accepts_identical_outputs() -> None:
    lp = [{1: -0.1}, {2: -0.1}]
    _assert_agrees_with_baseline(
        "cold APC", 0, _fake_request_output([1, 2], lp), _fake_request_output([1, 2], lp)
    )


def test_baseline_check_accepts_a_divergence_on_a_near_tie() -> None:
    """The logged case: APC top-2 equal (-1.807), plain's differ by 0.047."""
    got = _fake_request_output([1, 3], [{1: -0.1}, {3: -1.807, 4: -1.807}])
    plain = _fake_request_output([1, 4], [{1: -0.1}, {4: -1.782, 3: -1.829}])
    _assert_agrees_with_baseline("cold APC", 0, got, plain)


def test_baseline_check_rejects_a_divergence_with_a_clear_preference() -> None:
    got = _fake_request_output([1, 3], [{1: -0.1}, {3: -0.2, 4: -3.0}])
    plain = _fake_request_output([1, 4], [{1: -0.1}, {4: -0.2, 3: -3.0}])
    with pytest.raises(AssertionError, match="first divergence at token 1"):
        _assert_agrees_with_baseline("cold APC", 0, got, plain)


def test_baseline_check_rejects_a_pick_outside_the_other_engines_candidates() -> None:
    got = _fake_request_output([1, 3], [{1: -0.1}, {3: -1.0}])
    plain = _fake_request_output([1, 4], [{1: -0.1}, {4: -1.0}])  # 3 not in plain's top-k
    with pytest.raises(AssertionError):
        _assert_agrees_with_baseline("cold APC", 0, got, plain)


def test_baseline_check_rejects_a_length_mismatch() -> None:
    got = _fake_request_output([1, 2], [{1: -0.1}, {2: -0.1}])
    plain = _fake_request_output([1, 2, 3], [{1: -0.1}, {2: -0.1}, {3: -0.1}])
    with pytest.raises(AssertionError):
        _assert_agrees_with_baseline("cold APC", 0, got, plain)


# Parametrize over enforce_eager so the compiled production path
# (STOCK_TORCH_COMPILE, the default) is exercised alongside the eager path.
# The compiled variant hits the attention bucketer / warmup machinery with the
# partial-prefix query/KV lengths that APC generates.
@pytest.mark.uses_subprocess
@pytest.mark.parametrize(
    "enforce_eager",
    [
        pytest.param(True, id="eager"),
        pytest.param(False, id="compiled", marks=pytest.mark.e2e_smoke),
    ],
)
def test_prefix_caching(enforce_eager: bool, shared_prefix: str) -> None:
    """Prefix-caching must report cache hits and match non-APC token generation.

    Per mode (eager / compiled), runs prompts on:
    1. An APC-enabled engine twice:
       - Cold pass: primes the KV cache with the shared prefix.
       - Warm pass: serves the shared prefix from cache and asserts
         ``num_cached_tokens >= _BLOCK_SIZE`` for every prompt.
    2. A non-APC baseline engine.

    Asserts that both cold and warm APC outputs agree with the non-APC baseline
    for the first ``_STABLE_TOKENS`` tokens, tolerating a divergence on an fp16
    near-tie (see ``_assert_agrees_with_baseline``). Bit-exact equality is too
    strict on fp16 Spyre hardware, where cache-served vs recomputed attention
    paths accumulate rounding differences.
    """
    prompts = [shared_prefix + q for q in _QUESTIONS]
    sp = SamplingParams(temperature=0.0, max_tokens=16, logprobs=5)

    llm_apc = _make_llm(enable_prefix_caching=True, enforce_eager=enforce_eager)
    cold_outputs = llm_apc.generate(prompts, sp, use_tqdm=False)
    warm_outputs = llm_apc.generate(prompts, sp, use_tqdm=False)
    del llm_apc

    for i, warm in enumerate(warm_outputs):
        assert warm.num_cached_tokens is not None and warm.num_cached_tokens >= _BLOCK_SIZE, (
            f"Prompt {i}: expected num_cached_tokens >= {_BLOCK_SIZE} on warm pass, "
            f"got {warm.num_cached_tokens}"
        )

    llm_plain = _make_llm(enable_prefix_caching=False, enforce_eager=enforce_eager)
    plain_outputs = llm_plain.generate(prompts, sp, use_tqdm=False)
    del llm_plain

    assert len(cold_outputs) == len(warm_outputs) == len(plain_outputs) == len(prompts)
    for i, (cold, warm, plain) in enumerate(zip(cold_outputs, warm_outputs, plain_outputs)):
        _assert_agrees_with_baseline("cold APC", i, cold, plain)
        _assert_agrees_with_baseline("warm APC", i, warm, plain)
