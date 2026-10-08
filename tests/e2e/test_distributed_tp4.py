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

import importlib.util
import pathlib

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_device_count

_helpers_path = pathlib.Path(__file__).parent / "_helpers.py"
_spec = importlib.util.spec_from_file_location("_e2e_helpers", _helpers_path)
_helpers = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_helpers)  # type: ignore[union-attr]
_generate = _helpers.generate


def _assert_matches_tp1(tp1: list[list[int]], tp4: list[list[int]]) -> None:
    """Each TP=4 sequence must share a >=2-token prefix with its TP=1 twin.

    Later divergence is expected: fp16 reduction order differs across shards.
    """
    assert len(tp1) == len(tp4) == 2, (
        f"prompt count mismatch: expected 2 sequences each, "
        f"tp1 returned {len(tp1)}, tp4 returned {len(tp4)}"
    )

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
@pytest.mark.skipif(
    spyre_device_count() < 4,
    reason="needs >=4 Spyre cards; skipping TP=4 distributed test",
)
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
@pytest.mark.skipif(
    spyre_device_count() < 4,
    reason="needs >=4 Spyre cards; skipping TP=4 distributed test",
)
def test_tp4_llm_generate_matches_tp1() -> None:
    """TP=1 vs TP=4 greedy-decode prefix-match test on ibm-ai-platform/micro-g3.3-8b-instruct-1b.

    Runs identical prompts at TP=1 and TP=4 with `temperature=0` and
    asserts the first 2 output tokens match per prompt. Later divergence
    is expected from float16 reduction-order differences across shards.
    """
    model = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
    _assert_matches_tp1(
        _generate(model, tp=1, enforce_eager=True),
        _generate(model, tp=4, enforce_eager=True),
    )
