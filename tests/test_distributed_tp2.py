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

"""TP=2 distributed tests"""

from __future__ import annotations

import gc
import os

import pytest


def _spyre_device_count() -> int:
    """Return the number of visible Spyre cards, or 0 if unavailable.

    Reads AIU_WORLD_SIZE (set by the Spyre runtime environment when
    cards are visible) instead of touching the Spyre runtime, so
    `uses_subprocess` tests don't import torch_spyre in the main
    pytest process.
    """
    try:
        return int(os.environ.get("AIU_WORLD_SIZE", "0"))
    except ValueError:
        return 0


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
def test_tp2_tensor_model_parallel_all_reduce(run_tp_probe) -> None:
    """End-to-end TP=2 `tensor_model_parallel_all_reduce` on real Spyre cards.

    Spawns one subprocess per rank, each running through vllm's real
    `init_worker_distributed_environment` against a real `VllmConfig`,
    then verifies SpyreCommunicator.all_reduce returns numerically correct
    results on a 1-D probe and a (seq, hidden) slab. This is the cheapest
    surface that exercises the full collective stack — keep it as a fast
    canary so an all_reduce regression doesn't surface only as a degenerate
    end-to-end greedy-decode in `test_tp2_llm_generate_matches_tp1`.
    """
    run_tp_probe("tp_all_reduce", world_size=2)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
def test_tp2_vocab_parallel_embedding(run_tp_probe) -> None:
    """End-to-end TP=2 SpyreVocabParallelEmbedding forward on real Spyre cards.

    Spawns one subprocess per rank, brings up spyreccl through vllm's real
    init path, constructs the OOT VocabParallelEmbedding, and asserts each
    rank's all-reduced output matches the full-vocab F.embedding reference.
    """
    run_tp_probe("vocab_parallel_embedding", world_size=2)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
@pytest.mark.parametrize(
    "probe",
    [
        "merged_column_parallel_linear",
        "qkv_parallel_linear",
        "row_parallel_linear",
    ],
)
def test_tp_linear_layers(run_tp_probe, probe: str) -> None:
    """End-to-end TP=2 test of a Spyre linear layer on Spyre cards.

    Spawns one subprocess per rank, running through vllm's real
    `init_worker_distributed_environment` against a real `VllmConfig`,
    then verifies the layer returns numerically correct results on TP=2
    with Spyre communication. Parametrized over the three layer types:
    SpyreMergedColumnParallelLinear (output sharding), SpyreQKVParallelLinear
    (Q/K/V sharding), and SpyreRowParallelLinear (input sharding).
    """
    run_tp_probe(probe, world_size=2)


@pytest.mark.spyre
@pytest.mark.uses_subprocess
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
def test_tp2_llm_construction() -> None:
    """Construct `vllm.LLM(tensor_parallel_size=2)` end-to-end.

    Goes through the real `MultiprocExecutor` worker-spawn path that
    `vllm serve --tensor-parallel-size 2` uses.
    """
    from vllm import LLM

    LLM(
        model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        tensor_parallel_size=2,
        dtype="float16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=2,
    )


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
def test_tp2_llm_generate_matches_tp1() -> None:
    """TP=1 vs TP=2 greedy-decode prefix-match test on ibm-ai-platform/micro-g3.3-8b-instruct-1b.

    Runs identical prompts at TP=1 and TP=2 with `temperature=0` and
    asserts the first 2 output tokens match per prompt. Later divergence
    is expected from float16 reduction-order differences between the
    TP=1 and TP=2 paths.
    """
    from vllm import LLM, SamplingParams

    prompts = ["Hello, world!", "The capital of France is"]
    sp = SamplingParams(max_tokens=8, temperature=0.0)

    def run(tp: int) -> list[list[int]]:
        llm = LLM(
            model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
            tensor_parallel_size=tp,
            dtype="float16",
            enforce_eager=True,
            max_model_len=128,
            max_num_seqs=2,
        )
        outs = llm.generate(prompts, sp)
        result = [list(o.outputs[0].token_ids) for o in outs]
        # vllm doesn't expose an explicit LLM.shutdown(); rely on GC +
        # child-process reaping. Revisit if this flakes.
        del llm
        gc.collect()
        return result

    def _matching_prefix_len(a: list[int], b: list[int]) -> int:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return i
        return min(len(a), len(b))

    tp1 = run(tp=1)
    tp2 = run(tp=2)
    for i, (a, b) in enumerate(zip(tp1, tp2)):
        n = _matching_prefix_len(a, b)
        assert n >= 2, (
            f"prompt {i}: tp1 and tp2 diverged at token {n} "
            f"(expected >=2 matching tokens). tp1={a} tp2={b}"
        )
