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

"""``--kv-cache-dtype fp8`` through the engine: greedy decoding tracks the fp16 cache."""

from __future__ import annotations

import gc
import os

import pytest
from spyre_testing_plugin.vfio_reaper import wait_until_card_free

# enforce_eager=False spawns an EngineCore subprocess, which cannot claim the Spyre card if
# an in-process test already has.
pytestmark = pytest.mark.uses_subprocess

_MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
_PROMPTS = [
    "What are IBMs main businesses?",
    "Describe the city of Zurich, its two rivers and its universities.",
]
_MAX_TOKENS = 8
_NUM_LOGPROBS = 5
_MAX_MODEL_LEN = 256


def _greedy(kv_cache_dtype: str) -> list[list[dict[int, float]]]:
    """Per-prompt, per-step ``{token id: logprob}`` from one engine, shut down after."""
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=_MODEL,
        enforce_eager=False,
        max_model_len=_MAX_MODEL_LEN,
        max_num_seqs=len(_PROMPTS),
        max_num_batched_tokens=_MAX_MODEL_LEN,
        kv_cache_dtype=kv_cache_dtype,
    )
    try:
        outputs = engine.generate(
            _PROMPTS,
            SamplingParams(
                temperature=0.0, max_tokens=_MAX_TOKENS, logprobs=_NUM_LOGPROBS, ignore_eos=True
            ),
            use_tqdm=False,
        )
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=60)
        del engine
        gc.collect()
        wait_until_card_free(exclude_pids={os.getpid()}, timeout=60)
    traces = []
    for output in outputs:
        steps = output.outputs[0].logprobs
        assert steps is not None
        traces.append([{tid: lp.logprob for tid, lp in step.items()} for step in steps])
    return traces


def test_fp8_kv_cache_tracks_the_fp16_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first token matches, and every fp16 pick stays in the fp8 run's top tokens.

    Not exact: e4m3 rounds K and V, so a near-tie can flip and the two runs then diverge,
    but a broken store or read lands outside the top tokens within a step or two.
    """
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    ref = _greedy("auto")
    fp8 = _greedy("fp8")
    # e4m3 rounds every stored K and V, so bit-identical logprobs mean the fp8 cache
    # never reached the attention impl and the checks below would pass vacuously.
    assert fp8 != ref, "fp8 run matches the fp16 cache exactly; was the fp8 cache used?"

    for prompt, ref_steps, fp8_steps in zip(_PROMPTS, ref, fp8):
        ref_first = max(ref_steps[0], key=ref_steps[0].get)
        fp8_first = max(fp8_steps[0], key=fp8_steps[0].get)
        assert fp8_first == ref_first, f"{prompt!r}: first token {fp8_first} != {ref_first}"
        for step, (ref_lp, fp8_lp) in enumerate(zip(ref_steps, fp8_steps)):
            ref_pick = max(ref_lp, key=ref_lp.get)
            assert ref_pick in fp8_lp, f"{prompt!r} step {step}: {ref_pick} not in {fp8_lp}"
            if max(fp8_lp, key=fp8_lp.get) != ref_pick:
                # Past a flip the two runs condition on different tokens.
                break
