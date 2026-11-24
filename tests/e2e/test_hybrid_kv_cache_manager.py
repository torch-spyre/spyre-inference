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

"""An engine run with vLLM's hybrid KV cache manager on answers as one with it off.

The manager is opt-in (``--no-disable-hybrid-kv-cache-manager``) on the V2 runner. Remove
this test once the flag is dropped or the manager defaults to on, when every hybrid-model
e2e test covers it.
"""

from __future__ import annotations

import gc
import os

import pytest
from spyre_testing_plugin.vfio_reaper import wait_until_card_free

from spyre_inference.models.gemma4 import GEMMA4_TEXT_BACKBONE_OVERRIDE

# enforce_eager=False spawns an EngineCore subprocess, which cannot claim the Spyre card if
# an in-process test already has.
pytestmark = pytest.mark.uses_subprocess

# Sliding and full attention layers, so the manager overlays the groups' pages.
_MODEL = "joerunde/gemma-4-moe-nano"
_MAX_MODEL_LEN = 2048
_CHUNK = 512
_MAX_TOKENS = 16
_NUM_LOGPROBS = 5
# Past the 1024-token window, so the sliding groups free the blocks it has left.
_LONG_PROMPT_LEN = 1500
_SHORT_PROMPT_LEN = 200
# Both runs read the same pages through the same kernels, so the logprobs agree exactly on
# a card; the slack only absorbs a compiled plan differing between the two engines.
_LOGPROB_ATOL = 0.05


def _ids(n: int, seed: int) -> list[int]:
    return [1000 + ((i + 1) * (7919 + 13 * seed)) % 3000 for i in range(n)]


def _run(disable_hybrid_kv_cache_manager: bool) -> tuple[int, list[list[dict[int, float]]]]:
    """``(num_gpu_blocks, one per-step {token id: logprob} trace per prompt)``."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=_MODEL,
        hf_overrides=GEMMA4_TEXT_BACKBONE_OVERRIDE,
        enforce_eager=False,
        max_model_len=_MAX_MODEL_LEN,
        max_num_seqs=2,
        max_num_batched_tokens=_CHUNK,
        # Off: a prefix-cache hit in the second run would skip the prefill under test.
        enable_prefix_caching=False,
        disable_hybrid_kv_cache_manager=disable_hybrid_kv_cache_manager,
    )
    try:
        num_gpu_blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
        outputs = llm.generate(
            [
                TokensPrompt(prompt_token_ids=_ids(_LONG_PROMPT_LEN, 1)),
                TokensPrompt(prompt_token_ids=_ids(_SHORT_PROMPT_LEN, 2)),
            ],
            SamplingParams(
                temperature=0.0, max_tokens=_MAX_TOKENS, logprobs=_NUM_LOGPROBS, ignore_eos=True
            ),
            use_tqdm=False,
        )
        traces = []
        for output in outputs:
            steps = output.outputs[0].logprobs
            assert steps is not None
            traces.append([{t: lp.logprob for t, lp in step.items()} for step in steps])
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=60)
        del llm
        gc.collect()
        freed = wait_until_card_free(exclude_pids={os.getpid()}, timeout=60)
    assert freed, "Spyre devices were not released after LLM shutdown"
    return num_gpu_blocks, traces


def test_hybrid_kv_cache_manager_matches_it_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the pages each group views, the kernel block split and the sliding groups'
    block freeing, which a vLLM bump changing placement or kernel block selection breaks."""
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    off_blocks, off = _run(disable_hybrid_kv_cache_manager=True)
    on_blocks, on = _run(disable_hybrid_kv_cache_manager=False)

    # The same memory buys more manager blocks once groups share pages; equal counts
    # would mean the flag did not reach the manager.
    assert on_blocks > off_blocks
    for p, (trace_off, trace_on) in enumerate(zip(off, on, strict=True)):
        for step, (a, b) in enumerate(zip(trace_off, trace_on, strict=True)):
            assert max(a, key=a.get) == max(b, key=b.get), (
                f"prompt {p} step {step}: greedy token differs\n  off={a}\n  on={b}"
            )
            for token in a.keys() & b.keys():
                assert abs(a[token] - b[token]) <= _LOGPROB_ATOL, (
                    f"prompt {p} step {step} token {token}: logprob {b[token]} with the "
                    f"manager on, {a[token]} off"
                )
