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

"""A greedy request must answer the same way however the KV cache was used before it.

The paged kernel gathers whole KV pages, so a sequence whose length is not a multiple of
``block_size`` also reads the unused slots of its last block, and vLLM hands those blocks
to later requests. Masked slots must not reach the output; on Spyre they do
(torch-spyre#4517: fp16 ``exp()`` saturates at ``2**-24`` instead of underflowing to zero,
so an additively masked position keeps a softmax weight), which makes one request's
logprobs depend on the requests before it.

So: run a short request, dirty its blocks with a longer one, run the short request again,
and require the two logprob traces to be bit-identical.

Every test here therefore xfails today, and the fix belongs in the backend rather than in
a plugin-side workaround. ``tests/probes/test_fp16_exp_underflow_probe.py`` is the
strict-xfail signal for the arithmetic itself; when it XPASSes, drop the xfails here.

The second test covers the other masked-slot kind: a padded block column names whatever
page its block-table row last held, and the whole column is masked. That needs a block
count off the bucket lattice, which the test above cannot express -- at
``_MAX_MODEL_LEN = 256`` the lattice is ``[1, 2]``, so padded == real at every prompt
length.

The third runs a whole batch, which is what the batched decode kernel needs: both of the
above decode one sequence at a time, and that kernel reads the batch through a block table
and a padded-row scheme of its own.
"""

from __future__ import annotations

import pytest

# enforce_eager=False builds a subprocess EngineCore, so uses_subprocess runs this before
# any in-process test claims the Spyre device.
pytestmark = pytest.mark.uses_subprocess

# Not strict: the leak is always present, but whether it moves a logprob depends on what
# the dirtying request happened to leave in the masked slots, so an individual case can
# pass without the backend being fixed. The strict signal is the probe.
_XFAIL_REASON = (
    "torch-spyre#4517: the device's fp16 exp() saturates at 2**-24 instead of underflowing "
    "to zero, so masked KV slots reach the attention output and a request's logprobs depend "
    "on the requests before it. Fix is in the backend; see "
    "tests/probes/test_fp16_exp_underflow_probe.py."
)

_MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
# Short, so most of its single 128-slot KV block stays masked.
_PROBE_PROMPT = "What are IBMs main businesses?"
# Long, so its KV lands on the slots the probe request leaves masked. Both prompts have to
# fit the one prefill bucket below.
_DIRTY_PROMPT = (
    "Describe in as much detail as you can manage the history of the city of Zurich, "
    "its two rivers, its universities and museums, the industries based there, and the "
    "reasons that travellers from abroad most often give for wanting to visit it."
)
_MAX_TOKENS = 4
_NUM_LOGPROBS = 20
_MAX_MODEL_LEN = 256
_PREFILL_BUCKET = 128


def _logprobs(engine, prompt) -> list[dict[int, float]]:
    """Per-step ``{token id: logprob}`` for a greedy continuation of a str or TokensPrompt."""
    from vllm import SamplingParams

    output = engine.generate(
        prompt,
        SamplingParams(
            temperature=0.0,
            max_tokens=_MAX_TOKENS,
            logprobs=_NUM_LOGPROBS,
            ignore_eos=True,  # a fixed-length trace, so the comparison covers every step
        ),
        use_tqdm=False,
    )[0].outputs[0]
    assert output.logprobs is not None
    return [{tid: lp.logprob for tid, lp in step.items()} for step in output.logprobs]


@pytest.mark.xfail(strict=False, reason=_XFAIL_REASON)
def test_logprobs_do_not_depend_on_earlier_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same greedy request twice, with a different request in between."""
    from vllm import LLM

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    engine = LLM(
        model=_MODEL,
        enforce_eager=False,
        max_model_len=_MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=_PREFILL_BUCKET,
        compilation_config={"compile_sizes": [_PREFILL_BUCKET, 1]},
    )

    first = _logprobs(engine, _PROBE_PROMPT)
    # Hands the probe request's blocks to a different tenant, whose KV then sits in the
    # slots the probe request's last block leaves past its end.
    _logprobs(engine, _DIRTY_PROMPT)
    second = _logprobs(engine, _PROBE_PROMPT)

    diverged = next(
        (i for i, (a, b) in enumerate(zip(first, second)) if a != b),
        None,
    )
    assert diverged is None, (
        f"logprobs at step {diverged} changed after an unrelated request ran in "
        f"between: a stale KV tail is leaking into paged attention.\n"
        f"  first={first[diverged]}\n  second={second[diverged]}"
    )


# 1024, not 512: the leak is always present, but its value only changes often enough to
# surface as pass-to-pass divergence at the larger prefill bucket.
_PAD_MAX_MODEL_LEN = 1024
_BLOCK_SIZE = 128
# 3 blocks, not a bucket, so build() pads to 4; stays 3 for every generated token too.
_PAD_PROMPT_LEN = 300


def _block_counts() -> tuple[int, int]:
    """``(real, padded)`` block counts for ``_PAD_PROMPT_LEN``."""
    import bisect

    from spyre_inference.v1.attention.spyre_attn_bucketer import _powers_of_two_up_to

    buckets = sorted(
        {
            (kv + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            for kv in _powers_of_two_up_to(_PAD_MAX_MODEL_LEN, start=_BLOCK_SIZE)
        }
    )
    real = (_PAD_PROMPT_LEN + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    return real, buckets[bisect.bisect_left(buckets, real)]


@pytest.mark.xfail(strict=False, reason=_XFAIL_REASON)
def test_logprobs_do_not_depend_on_the_padded_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request whose block count is padded up must still answer identically."""
    from vllm import LLM
    from vllm.inputs import TokensPrompt

    real, padded = _block_counts()
    # Without this the test would pass while covering nothing if the lattice changed.
    assert padded > real, (
        f"_PAD_PROMPT_LEN={_PAD_PROMPT_LEN} gives real=={padded}==padded blocks, so this "
        f"test no longer exercises a padded block; pick a length off the bucket lattice."
    )

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    engine = LLM(
        model=_MODEL,
        enforce_eager=False,
        max_model_len=_PAD_MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=_PAD_MAX_MODEL_LEN,
        compilation_config={"compile_sizes": [_PAD_MAX_MODEL_LEN, 1]},
    )

    # Exact token ids: a tokenizer would not let the test pin the block count.
    def ids(n: int, seed: int) -> list[int]:
        return [1000 + ((i + 1) * (7919 + 13 * seed)) % 3000 for i in range(n)]

    probe = TokensPrompt(prompt_token_ids=ids(_PAD_PROMPT_LEN, 1))
    # Fills every block the probe's padded count reaches.
    dirty = TokensPrompt(prompt_token_ids=ids(_PAD_MAX_MODEL_LEN - _MAX_TOKENS, 2))

    traces = []
    for _ in range(5):
        _logprobs(engine, dirty)
        traces.append(_logprobs(engine, probe))

    for i, trace in enumerate(traces[1:], start=2):
        diverged = next(
            (j for j, (a, b) in enumerate(zip(traces[0], trace)) if a != b),
            None,
        )
        assert diverged is None, (
            f"pass {i} diverged from pass 1 at step {diverged} for a request with "
            f"real={real} padded={padded} blocks: the {padded - real} padded block(s) "
            f"are reaching KV that their all--inf mask should have excluded.\n"
            f"  pass1={traces[0][diverged]}\n  pass{i}={trace[diverged]}"
        )


# The batched decode kernel serves batches of at least `_MIN_BATCHED_SEQS` sequences, so
# four concurrent requests are the smallest batch that reaches it. Prefills are serialised
# (SPYRE_MAX_NUM_PARTIAL_PREFILLS defaults to 1), so the batch only fills up once every
# prompt has prefilled -- hence more tokens than the solo tests, to leave steps where all
# four decode together.
_BATCH_NUM_SEQS = 4
_BATCH_MAX_TOKENS = 8
# Short, for the same reason as `_PROBE_PROMPT`, and distinct so a leak between slots
# carries a neighbour's KV rather than a copy of this slot's own.
_BATCH_PROBE_PROMPTS = [
    "What are IBMs main businesses?",
    "Name three rivers in Europe.",
    "Why is the sky blue?",
    "Who wrote the Odyssey?",
]


def _batch_logprobs(engine, prompts: list[str]) -> list[list[dict[int, float]]]:
    """One per-step logprob trace per prompt, for a batch in a single `generate` call."""
    from vllm import SamplingParams

    outputs = engine.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            max_tokens=_BATCH_MAX_TOKENS,
            logprobs=_NUM_LOGPROBS,
            ignore_eos=True,
        ),
        use_tqdm=False,
    )
    assert len(outputs) == len(prompts)
    traces = []
    for output in outputs:
        steps = output.outputs[0].logprobs
        assert steps is not None
        traces.append([{tid: lp.logprob for tid, lp in step.items()} for step in steps])
    return traces


@pytest.mark.xfail(strict=False, reason=_XFAIL_REASON)
def test_logprobs_do_not_depend_on_earlier_requests_in_a_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same batch twice, with other requests in between, while decode is batched.

    Batched decode reads the whole batch through one padded gather, off a block table of
    its own (`chunk_page_ids_cpu`, not the per-seq `page_index_tables_cpu`), with the block
    axis padded up to a whole chunk and the row axis up to the num_seqs lattice. Which
    masked slots a row reaches is therefore not the per-seq loop's answer.

    The batch is identical in both passes, so the schedule repeats and with it the chunked
    reduction order (which is not comparable across batch sizes -- see
    test_spyre_attn_batched_decode_correctness); only the physical blocks differ. Prompts
    stay under one block, so nothing is prefix-cacheable and every pass really re-prefills.
    """
    from vllm import LLM

    assert len(_BATCH_PROBE_PROMPTS) == _BATCH_NUM_SEQS

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    # Pinned on, so a default flip cannot silently move this test onto the per-seq loop.
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")

    engine = LLM(
        model=_MODEL,
        enforce_eager=False,
        max_model_len=_MAX_MODEL_LEN,
        max_num_seqs=_BATCH_NUM_SEQS,
        max_num_batched_tokens=_PREFILL_BUCKET,
        # Every reachable body token count: a prefill step, a full decode batch, and one
        # sequence decoding alone. Shorter batches pad up onto `_BATCH_NUM_SEQS`.
        compilation_config={"compile_sizes": [_PREFILL_BUCKET, _BATCH_NUM_SEQS, 1]},
    )

    first = _batch_logprobs(engine, _BATCH_PROBE_PROMPTS)
    # Hands the probe batch's blocks to other tenants, whose KV then sits in the slots the
    # probe requests' last blocks leave past their end. Identical prompts are enough: the
    # comparison is pass-to-pass, and in the first pass those blocks held something else.
    _batch_logprobs(engine, [_DIRTY_PROMPT] * _BATCH_NUM_SEQS)
    second = _batch_logprobs(engine, _BATCH_PROBE_PROMPTS)

    for slot, (before, after) in enumerate(zip(first, second)):
        diverged = next(
            (i for i, (a, b) in enumerate(zip(before, after)) if a != b),
            None,
        )
        assert diverged is None, (
            f"batch slot {slot} diverged at step {diverged} after an unrelated batch ran "
            f"in between: a stale KV tail is leaking into batched decode.\n"
            f"  first={before[diverged]}\n  second={after[diverged]}"
        )
