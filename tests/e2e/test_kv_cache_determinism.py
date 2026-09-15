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
``SpyreAttentionImpl._clear_new_kv_block_tails`` is what makes that hold. When
``tests/probes/test_masked_kv_slot_probe.py`` flips to XPASS the backend no longer needs
the workaround, and this test guards its removal.
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


def _logprobs(engine, prompt: str) -> list[dict[int, float]]:
    """Per-step ``{token id: logprob}`` for a greedy continuation of `prompt`."""
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
