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

"""End-to-end tests for continuous batching on Spyre.

Verifies that iteration-level continuous batching (waiting-queue admissions and dynamic slot
recycling) produces deterministic outputs identical to isolated single-request execution.
"""

from __future__ import annotations

import contextlib
import gc
import os
from collections.abc import Iterator

import pytest
from spyre_testing_plugin.vfio_reaper import wait_until_card_free

# enforce_eager=False spawns an EngineCore subprocess, requiring uses_subprocess
# so it does not collide with in-process VFIO card holders.
pytestmark = pytest.mark.uses_subprocess

_MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
_MAX_MODEL_LEN = 256
_PREFILL_BUCKET = 128
_BATCH_NUM_SEQS = 2

_PROMPTS = [
    "What are IBMs main businesses?",
    "Name three rivers in Europe.",
    "Why is the sky blue?",
    "Who wrote the Odyssey?",
]


@contextlib.contextmanager
def _engine(**kwargs) -> Iterator:
    """An `LLM` that is cleanly shut down upon exit to avoid leaking VFIO device state."""
    from vllm import LLM

    engine = LLM(**kwargs)
    try:
        yield engine
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=60)
        del engine
        gc.collect()
        wait_until_card_free(exclude_pids={os.getpid()}, timeout=60)


def test_continuous_batching_queue_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that requests waiting in queue are continuously admitted as active slots free up."""
    from vllm import SamplingParams

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    # Pinned on, so a default flip cannot silently move this test onto the per-seq loop.
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")

    # Staggered max_tokens: prompt 0 finishes first (6 tokens), admitting prompt 2 from queue
    # while prompt 1 is still generating (10 tokens).
    max_tokens = [6, 10, 6, 10]

    with _engine(
        model=_MODEL,
        enforce_eager=False,
        max_model_len=_MAX_MODEL_LEN,
        max_num_seqs=_BATCH_NUM_SEQS,
        max_num_batched_tokens=_PREFILL_BUCKET,
        # Pre-compile the prefill bucket, the 2-wide decode batch, and 1-wide decode
        compilation_config={"compile_sizes": [_PREFILL_BUCKET, _BATCH_NUM_SEQS, 1]},
        enable_prefix_caching=False,
    ) as engine:
        # 1. Isolated ground truth execution
        isolated_outputs = []
        for prompt, max_tok in zip(_PROMPTS, max_tokens):
            sp = SamplingParams(temperature=0.0, max_tokens=max_tok, ignore_eos=True)
            res = engine.generate([prompt], sp, use_tqdm=False)
            isolated_outputs.append(res[0].outputs[0].token_ids)

        # 2. Continuous batched execution with queue admission
        sampling_params_list = [
            SamplingParams(temperature=0.0, max_tokens=max_tok, ignore_eos=True)
            for max_tok in max_tokens
        ]
        batched_res = engine.generate(_PROMPTS, sampling_params_list, use_tqdm=False)
        batched_outputs = [r.outputs[0].token_ids for r in batched_res]

    # 3. Assert exact token-for-token determinism
    for i, (expected, actual) in enumerate(zip(isolated_outputs, batched_outputs)):
        assert expected == actual, (
            f"Prompt {i} diverged during queue admission: {expected} != {actual}"
        )
