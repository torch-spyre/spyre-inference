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
from spyre_testing_plugin.greedy_checks import Trace, assert_greedy_runs_agree, logprob_trace
from spyre_testing_plugin.vfio_reaper import wait_until_card_free

from spyre_inference.models.gemma4 import GEMMA4_TEXT_BACKBONE_OVERRIDE

# enforce_eager=False spawns an EngineCore subprocess, which cannot claim the Spyre card if
# an in-process test already has.
pytestmark = pytest.mark.uses_subprocess

# Sliding and full attention layers, so the manager overlays the groups' pages.
_MODEL = "joerunde/gemma-4-moe-nano"
_MAX_MODEL_LEN = 2048
_CHUNK = 512
# Both prompts cross a block boundary while decoding, so blocks are allocated after the
# sliding groups have freed the ones that left the 1024-token window.
_MAX_TOKENS = 64
_NUM_LOGPROBS = 5
_LONG_PROMPT_LEN = 1500
_SHORT_PROMPT_LEN = 200
# Barely more blocks than both requests hold at once: the blocks allocated while decoding
# then come back from the free queue, so a block one group freed is reused by another.
_HYBRID_NUM_GPU_BLOCKS = 82
_LOGPROB_TOLERANCE = 0.1


def _ids(n: int, seed: int) -> list[int]:
    return [1000 + ((i + 1) * (7919 + 13 * seed)) % 3000 for i in range(n)]


def _num_kv_cache_groups(worker) -> int:
    return len(worker.model_runner.kv_cache_config.kv_cache_groups)


def _run(disable_hybrid_kv_cache_manager: bool) -> tuple[int, Trace]:
    """``(KV cache groups, the run's greedy logprob trace)``."""
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
        num_gpu_blocks_override=None if disable_hybrid_kv_cache_manager else _HYBRID_NUM_GPU_BLOCKS,
    )
    try:
        (num_groups,) = llm.collective_rpc(_num_kv_cache_groups)
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
        trace = logprob_trace(outputs)
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=60)
        del llm
        gc.collect()
        freed = wait_until_card_free(exclude_pids={os.getpid()}, timeout=60)
    assert freed, "Spyre devices were not released after LLM shutdown"
    return num_groups, trace


# Two compiled engine inits (warmup + attention recording, ~860s each on x86_64 CI pods in
# run 45572834994) exceed the 1800s default on their own before generate runs at all.
@pytest.mark.timeout(3600)
def test_hybrid_kv_cache_manager_matches_it_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the pages each group views, the kernel block split and the sliding groups'
    block freeing and reuse, which a vLLM bump changing placement or kernel block selection
    breaks."""
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    # collective_rpc ships _num_kv_cache_groups to the engine process with cloudpickle.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    off_groups, off = _run(disable_hybrid_kv_cache_manager=True)
    on_groups, on = _run(disable_hybrid_kv_cache_manager=False)

    # One group per sliding layer and one for the full-attention layer once the manager is
    # on; one group would mean the flag did not reach it.
    assert (off_groups, on_groups) == (1, 6)
    assert_greedy_runs_agree(off, on, _LOGPROB_TOLERANCE, names=("manager off", "manager on"))
