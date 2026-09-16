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
(torch-spyre#4517), which makes one request's logprobs depend on the requests before it.

So: run a short request, dirty its blocks with a longer one, run the short request again,
and require the two logprob traces to be bit-identical.
``SpyreAttentionImpl._clear_masked_kv_slots`` is what makes that hold. When
``tests/probes/test_fp16_exp_underflow_probe.py`` flips to XPASS the backend no longer needs
the workaround, and this test guards its removal.

The second test covers the padded block columns: ``build()`` rounds a sequence's block
count onto the recorder's buckets and gathers the padded columns too. That needs a block
count off the bucket lattice, which the test above cannot express -- at
``_MAX_MODEL_LEN = 256`` the lattice is ``[1, 2]``, so padded == real at every prompt
length.
"""

from __future__ import annotations

import pytest

# enforce_eager=False builds a subprocess EngineCore, so uses_subprocess runs this before
# any in-process test claims the Spyre device.
pytestmark = pytest.mark.uses_subprocess

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
# surface as pass-to-pass divergence at the larger prefill bucket. The deterministic
# detector is test_spyre_attn.py::test_padded_blocks_do_not_read_a_stale_page.
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
            f"are gathering a slot that is not held at zero.\n"
            f"  pass1={traces[0][diverged]}\n  pass{i}={trace[diverged]}"
        )
