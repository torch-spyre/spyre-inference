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

"""Prefix-caching (APC) tests for the Spyre backend.

Three properties are verified end-to-end:

1. **Correctness** — prompts with a shared prefix produce identical output
   tokens whether the KV cache for that prefix was already computed (hit) or
   not (cold).

2. **Cache hit reported** — after a warm-up pass the engine reports
   ``num_cached_tokens > 0`` for a repeated prefix, confirming the Spyre
   paged-KV backend actually reuses blocks rather than recomputing them.

3. **Output consistency without prefix caching** — as a control, disabling
   APC still yields valid output (non-empty), though no cache hit is expected.
"""

from __future__ import annotations

import gc

import pytest

# ---------------------------------------------------------------------------
# Test fixtures and shared data
# ---------------------------------------------------------------------------

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"

# A moderately long shared prefix — long enough to fill at least one block
# (Spyre block size is 64, so ~70+ tokens of prefix text is sufficient).
_SHARED_PREFIX = (
    "You are a knowledgeable assistant specialised in world geography. "
    "Answer every question clearly and concisely. "
    "Here are some facts you should keep in mind: "
    "France is in Western Europe and its capital is Paris. "
    "Germany is in Central Europe and its capital is Berlin. "
    "Japan is in East Asia and its capital is Tokyo. "
    "Australia is a continent and a country; its capital is Canberra. "
    "Brazil is the largest country in South America; its capital is Brasília. "
    "Now answer the following question: "
)

_QUESTIONS = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

_PROMPTS = [_SHARED_PREFIX + q for q in _QUESTIONS]

_SAMPLING_KWARGS = dict(temperature=0.0, max_tokens=8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(*, enable_prefix_caching: bool, warmup: bool) -> list:
    """Run a two-prompt batch and return raw ``RequestOutput`` objects.

    When *warmup* is ``True``, runs a prior generate call with the first
    prompt to populate the prefix cache before the measured batch.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        dtype="float16",
        max_model_len=256,
        max_num_seqs=4,
        enable_prefix_caching=enable_prefix_caching,
    )

    sp = SamplingParams(**_SAMPLING_KWARGS)

    if warmup:
        # Prime the cache with the shared prefix by running one prompt first.
        llm.generate(_PROMPTS[0], sp, use_tqdm=False)

    outputs = llm.generate(_PROMPTS, sp, use_tqdm=False)

    del llm
    gc.collect()
    return outputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.prefix_caching
@pytest.mark.uses_subprocess
def test_prefix_caching_output_matches_no_caching() -> None:
    """Prefix-caching must not change the generated tokens.

    Runs the same prompts with APC enabled (cold — no warmup) and disabled,
    then checks that every prompt yields the same output token sequence.
    This guards against the Spyre block-reuse path producing different KV
    contents than a fresh computation.
    """
    cached_outputs = _run(enable_prefix_caching=True, warmup=False)
    plain_outputs = _run(enable_prefix_caching=False, warmup=False)

    assert len(cached_outputs) == len(plain_outputs) == len(_PROMPTS)
    for i, (cached, plain) in enumerate(zip(cached_outputs, plain_outputs)):
        cached_ids = list(cached.outputs[0].token_ids)
        plain_ids = list(plain.outputs[0].token_ids)
        assert cached_ids == plain_ids, (
            f"Prompt {i}: token mismatch with vs without prefix caching.\n"
            f"  with APC : {cached_ids}\n"
            f"  without  : {plain_ids}"
        )


@pytest.mark.prefix_caching
@pytest.mark.uses_subprocess
def test_prefix_cache_hit_reported_after_warmup() -> None:
    """After a warmup pass the engine must report a cache hit.

    ``RequestOutput.num_cached_tokens`` is set by vLLM's scheduler and
    reflects how many prefix tokens were served from the block pool rather
    than recomputed.  At least one prompt in the batch must report a
    non-zero value when the shared prefix has been previously cached.
    """
    outputs = _run(enable_prefix_caching=True, warmup=True)

    cached_counts = [o.num_cached_tokens for o in outputs]
    assert any(c > 0 for c in cached_counts), (
        "Expected at least one prompt to report a prefix-cache hit after "
        f"warmup, but got num_cached_tokens={cached_counts}"
    )


@pytest.mark.prefix_caching
@pytest.mark.uses_subprocess
def test_prefix_caching_warm_output_matches_cold() -> None:
    """A warm cache hit must produce the same tokens as a cold run.

    After warmup the scheduler skips recomputing the shared prefix blocks
    and restores them from the KV cache.  The resulting token sequence must
    be bit-identical to a cold run with APC enabled.
    """
    cold_outputs = _run(enable_prefix_caching=True, warmup=False)
    warm_outputs = _run(enable_prefix_caching=True, warmup=True)

    assert len(cold_outputs) == len(warm_outputs) == len(_PROMPTS)
    for i, (cold, warm) in enumerate(zip(cold_outputs, warm_outputs)):
        cold_ids = list(cold.outputs[0].token_ids)
        warm_ids = list(warm.outputs[0].token_ids)
        assert cold_ids == warm_ids, (
            f"Prompt {i}: cold vs warm token mismatch with prefix caching.\n"
            f"  cold (no warmup) : {cold_ids}\n"
            f"  warm (after hit) : {warm_ids}"
        )
