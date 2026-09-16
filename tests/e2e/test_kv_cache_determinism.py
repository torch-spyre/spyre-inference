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

The paged kernel gathers whole KV pages, so a sequence also reads the unused slots of its
last block, and vLLM hands those blocks to later requests. On Spyre fp16 ``exp()``
saturates at ``2**-24`` instead of underflowing to zero, so those masked slots keep a
softmax weight and reach the output (torch-spyre#4517). Each test therefore runs a request,
dirties its blocks with another, and reruns it -- and xfails until the backend is fixed.
See ``tests/probes/test_fp16_exp_underflow_probe.py``.
"""

from __future__ import annotations

import contextlib
import gc
import os
from collections.abc import Iterator

import pytest
from spyre_testing_plugin.vfio_reaper import wait_until_card_free

# enforce_eager=False spawns an EngineCore subprocess, which cannot claim the Spyre card if
# an in-process test already has.
pytestmark = pytest.mark.uses_subprocess

# Not strict: the leak is always present, but whether it moves a given logprob depends on
# what the dirtying request left in the masked slots. The probe is the strict signal.
_XFAIL_REASON = (
    "torch-spyre#4517: masked KV slots reach the attention output, so a request's logprobs "
    "depend on the requests before it. The fix belongs in the backend; see "
    "tests/probes/test_fp16_exp_underflow_probe.py."
)

_MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
# Short, so most of its single 128-slot KV block stays masked.
_PROBE_PROMPT = "What are IBMs main businesses?"
# Long, so its KV lands on the slots the probe leaves masked; both must fit _PREFILL_BUCKET.
_DIRTY_PROMPT = (
    "Describe in as much detail as you can manage the history of the city of Zurich, "
    "its two rivers, its universities and museums, the industries based there, and the "
    "reasons that travellers from abroad most often give for wanting to visit it."
)
_MAX_TOKENS = 4
_NUM_LOGPROBS = 20
_MAX_MODEL_LEN = 256
_PREFILL_BUCKET = 128


@contextlib.contextmanager
def _engine(**kwargs) -> Iterator:
    """An `LLM` that is shut down before the test's assertion leaves this frame.

    These tests are expected to fail, and a raised assertion keeps its frame -- and so the
    `LLM` in it -- alive past teardown, where the plugin SIGKILLs the worker still holding
    the card. That kill leaves the VFIO release in flight, and the next in-process card test
    fails with ``DeviceOpenFail ... "Device or resource busy"``. Shutting down in `finally`
    runs before the assertion propagates, so the worker exits on its own and there is
    nothing to reap.
    """
    from vllm import LLM

    engine = LLM(**kwargs)
    try:
        yield engine
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=60)
        del engine
        gc.collect()
        wait_until_card_free(exclude_pids={os.getpid()}, timeout=60)


def _batch_logprobs(
    engine, prompts: list, max_tokens: int = _MAX_TOKENS
) -> list[list[dict[int, float]]]:
    """One per-step ``{token id: logprob}`` trace per prompt, from a single `generate`."""
    from vllm import SamplingParams

    outputs = engine.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=_NUM_LOGPROBS,
            ignore_eos=True,  # fixed-length trace, so the comparison covers every step
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


def _logprobs(engine, prompt) -> list[dict[int, float]]:
    """`_batch_logprobs` for one prompt."""
    return _batch_logprobs(engine, [prompt])[0]


def _assert_fits_prefill(prompts: list[str]) -> None:
    """Past _PREFILL_BUCKET a prompt chunks, which is not what these tests measure."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_MODEL)
    for prompt in prompts:
        num_tokens = len(tokenizer(prompt).input_ids)
        assert num_tokens <= _PREFILL_BUCKET, (
            f"prompt is {num_tokens} tokens, over _PREFILL_BUCKET={_PREFILL_BUCKET}: {prompt!r}"
        )


@pytest.mark.xfail(strict=False, reason=_XFAIL_REASON)
def test_logprobs_do_not_depend_on_earlier_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same greedy request twice, with a different request in between."""
    _assert_fits_prefill([_PROBE_PROMPT, _DIRTY_PROMPT])
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    with _engine(
        model=_MODEL,
        enforce_eager=False,
        max_model_len=_MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=_PREFILL_BUCKET,
        compilation_config={"compile_sizes": [_PREFILL_BUCKET, 1]},
        # Off: a prefix-cache hit would reuse the probe's own KV instead of recomputing it.
        enable_prefix_caching=False,
    ) as engine:
        first = _logprobs(engine, _PROBE_PROMPT)
        # Discarded: it exists to leave another tenant's KV in the slots the probe request's
        # last block keeps past its end.
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


# 1024, not 512: the masked value changes often enough to surface as pass-to-pass
# divergence only at the larger bucket.
_PAD_MAX_MODEL_LEN = 1024
_BLOCK_SIZE = 128
_PAD_PROMPT_LEN = 300


def _block_counts() -> tuple[int, int]:
    """``(real, padded)`` block counts for ``_PAD_PROMPT_LEN``, from the bucketer itself."""
    from unittest.mock import MagicMock

    from spyre_inference.v1.attention.spyre_attn_bucketer import SpyreAttnBucketer

    config = MagicMock()
    config.cache_config.block_size = _BLOCK_SIZE
    config.model_config.max_model_len = _PAD_MAX_MODEL_LEN
    config.scheduler_config.max_num_batched_tokens = _PAD_MAX_MODEL_LEN
    config.scheduler_config.max_num_seqs = 1
    real = (_PAD_PROMPT_LEN + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    padded = SpyreAttnBucketer(config).find_blocks_bucket(real)
    assert padded is not None, f"real={real} blocks is above every bucket"
    return real, padded


@pytest.mark.xfail(strict=False, reason=_XFAIL_REASON)
def test_logprobs_do_not_depend_on_the_padded_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request whose block count is padded up must still answer identically."""
    from vllm.inputs import TokensPrompt

    real, padded = _block_counts()
    assert padded > real, (
        f"_PAD_PROMPT_LEN={_PAD_PROMPT_LEN} gives real=={real}==padded blocks, so this "
        f"test no longer exercises a padded block; pick a length off the bucket lattice."
    )

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    # Exact token ids: a tokenizer would not let the test pin the block count.
    def ids(n: int, seed: int) -> list[int]:
        return [1000 + ((i + 1) * (7919 + 13 * seed)) % 3000 for i in range(n)]

    probe = TokensPrompt(prompt_token_ids=ids(_PAD_PROMPT_LEN, 1))
    # Fills every block the probe's padded count reaches.
    dirty = TokensPrompt(prompt_token_ids=ids(_PAD_MAX_MODEL_LEN - _MAX_TOKENS, 2))

    with _engine(
        model=_MODEL,
        enforce_eager=False,
        max_model_len=_PAD_MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=_PAD_MAX_MODEL_LEN,
        compilation_config={"compile_sizes": [_PAD_MAX_MODEL_LEN, 1]},
        # Off: a prefix-cache hit would reuse the probe's own KV instead of recomputing it.
        enable_prefix_caching=False,
    ) as engine:
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


# _MIN_BATCHED_SEQS is 4, and prefills serialise (SPYRE_MAX_NUM_PARTIAL_PREFILLS defaults to
# 1), so the batch only fills up once every prompt has prefilled -- hence more tokens here.
_BATCH_NUM_SEQS = 4
_BATCH_MAX_TOKENS = 8
# Short like `_PROBE_PROMPT`, and distinct so a leak carries a neighbour's KV rather than a
# copy of the slot's own.
_BATCH_PROBE_PROMPTS = [
    "What are IBMs main businesses?",
    "Name three rivers in Europe.",
    "Why is the sky blue?",
    "Who wrote the Odyssey?",
]


@pytest.mark.xfail(strict=False, reason=_XFAIL_REASON)
def test_logprobs_do_not_depend_on_earlier_requests_in_a_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same batch twice, with other requests in between, while decode is batched."""
    from spyre_inference.v1.attention.spyre_attn_bucketer import _MIN_BATCHED_SEQS

    assert len(_BATCH_PROBE_PROMPTS) == _BATCH_NUM_SEQS
    # Below the threshold the batch takes the per-seq loop, and this tests nothing batched.
    assert _BATCH_NUM_SEQS >= _MIN_BATCHED_SEQS
    _assert_fits_prefill([*_BATCH_PROBE_PROMPTS, _DIRTY_PROMPT])

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    # Pinned on, so a default flip cannot silently move this test onto the per-seq loop.
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")

    with _engine(
        model=_MODEL,
        enforce_eager=False,
        max_model_len=_MAX_MODEL_LEN,
        max_num_seqs=_BATCH_NUM_SEQS,
        max_num_batched_tokens=_PREFILL_BUCKET,
        # A prefill step, a full decode batch, and one sequence decoding alone.
        compilation_config={"compile_sizes": [_PREFILL_BUCKET, _BATCH_NUM_SEQS, 1]},
        # Off: a prefix-cache hit would reuse the probe's own KV instead of recomputing it.
        enable_prefix_caching=False,
    ) as engine:
        first = _batch_logprobs(engine, _BATCH_PROBE_PROMPTS, _BATCH_MAX_TOKENS)
        # Discarded: it rehomes the probe batch's blocks onto other tenants. Same batch size
        # in both passes, since the chunked reduction order is not comparable across sizes.
        _batch_logprobs(engine, [_DIRTY_PROMPT] * _BATCH_NUM_SEQS, _BATCH_MAX_TOKENS)
        second = _batch_logprobs(engine, _BATCH_PROBE_PROMPTS, _BATCH_MAX_TOKENS)

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
