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

"""TP=4 distributed tests"""

from __future__ import annotations

import gc
import os

import pytest
from spyre_testing_plugin.vfio_reaper import wait_until_card_free


def _generate(model: str, tp: int) -> list[list[int]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        dtype="float16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=2,
    )
    try:
        outs = llm.generate(
            ["Hello, world!", "The capital of France is"],
            SamplingParams(max_tokens=8, temperature=0.0),
        )
        result = [list(o.outputs[0].token_ids) for o in outs]
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=60)
        del llm
        gc.collect()
        freed = wait_until_card_free(exclude_pids={os.getpid()}, timeout=60)
    # Outside the finally so a generate failure doesn't mask this check.
    assert freed, "Spyre devices were not released after LLM shutdown"
    return result


def _assert_matches_tp1(tp1: list[list[int]], tp4: list[list[int]]) -> None:
    """Each TP=4 sequence must share a >=2-token prefix with its TP=1 twin.

    Later divergence is expected: fp16 reduction order differs across shards.
    """

    def prefix_len(a: list[int], b: list[int]) -> int:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return i
        return min(len(a), len(b))

    for i, (a, b) in enumerate(zip(tp1, tp4)):
        n = prefix_len(a, b)
        assert n >= 2, (
            f"prompt {i}: tp1 and tp4 diverged at token {n} "
            f"(expected >=2 matching tokens). tp1={a} tp4={b}"
        )


@pytest.mark.uses_subprocess
@pytest.mark.distributed_tp4
def test_tp4_llm_construction() -> None:
    """Construct `vllm.LLM(tensor_parallel_size=4)` end-to-end.

    Goes through the real `MultiprocExecutor` worker-spawn path that
    `vllm serve --tensor-parallel-size 4` uses.
    """
    from vllm import LLM

    LLM(
        model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        tensor_parallel_size=4,
        dtype="float16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=2,
    )


@pytest.mark.uses_subprocess
@pytest.mark.distributed_tp4
def test_tp4_llm_generate_matches_tp1() -> None:
    """TP=1 vs TP=4 greedy-decode prefix-match test on ibm-ai-platform/micro-g3.3-8b-instruct-1b.

    Runs identical prompts at TP=1 and TP=4 with `temperature=0` and
    asserts the first 2 output tokens match per prompt. Later divergence
    is expected from float16 reduction-order differences across shards.
    """
    model = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
    _assert_matches_tp1(
        _generate(model, tp=1),
        _generate(model, tp=4),
    )
