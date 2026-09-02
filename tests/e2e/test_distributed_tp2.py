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

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_device_count


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
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


def _generate(
    tp: int,
    enforce_eager: bool,
    compilation_config: dict | None = None,
) -> list[list[int]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        tensor_parallel_size=tp,
        dtype="float16",
        enforce_eager=enforce_eager,
        max_model_len=128,
        max_num_seqs=2,
        **({"compilation_config": compilation_config} if compilation_config is not None else {}),
    )
    outs = llm.generate(
        ["Hello, world!", "The capital of France is"],
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    result = [list(o.outputs[0].token_ids) for o in outs]
    # vllm has no explicit LLM.shutdown(); rely on GC + child-process reaping.
    del llm
    gc.collect()
    return result


def _assert_matches_tp1(tp1: list[list[int]], tp2: list[list[int]]) -> None:
    """Each TP=2 sequence must share a >=2-token prefix with its TP=1 twin.

    Later divergence is expected: fp16 reduction order differs between the paths.
    """

    def prefix_len(a: list[int], b: list[int]) -> int:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return i
        return min(len(a), len(b))

    for i, (a, b) in enumerate(zip(tp1, tp2)):
        n = prefix_len(a, b)
        assert n >= 2, (
            f"prompt {i}: tp1 and tp2 diverged at token {n} "
            f"(expected >=2 matching tokens). tp1={a} tp2={b}"
        )


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
def test_tp2_llm_generate_matches_tp1() -> None:
    """TP=1 vs TP=2 greedy-decode prefix match, eager."""
    _assert_matches_tp1(_generate(tp=1, enforce_eager=True), _generate(tp=2, enforce_eager=True))


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
def test_tp2_compiled_llm_generate_matches_tp1() -> None:
    """TP=1 vs TP=2 greedy-decode prefix match, compiled: the in-graph reduction.

    compile_sizes capped at 5 buckets; full 35-bucket warmup exceeds the
    pytest-timeout (1800 s) on cold-cache CI.
    """
    _cc = {"compile_sizes": [1, 2, 4, 8, 16]}
    _assert_matches_tp1(
        _generate(tp=1, enforce_eager=False, compilation_config=_cc),
        _generate(tp=2, enforce_eager=False, compilation_config=_cc),
    )
