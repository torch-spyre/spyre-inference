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

import importlib.util
import pathlib

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_device_count

from spyre_inference.models.gemma4 import GEMMA4_TEXT_BACKBONE_OVERRIDE

_helpers_path = pathlib.Path(__file__).parent / "_helpers.py"
_spec = importlib.util.spec_from_file_location("_e2e_helpers", _helpers_path)
_helpers = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_helpers)  # type: ignore[union-attr]
_generate = _helpers.generate


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
    model = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
    _assert_matches_tp1(
        _generate(model, tp=1, enforce_eager=True),
        _generate(model, tp=2, enforce_eager=True),
    )


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
@pytest.mark.parametrize(
    "model,hf_overrides",
    [
        ("ibm-ai-platform/micro-g3.3-8b-instruct-1b", None),
        # gemma-4 vision checkpoints resolve the multimodal architecture, so this row
        # pins the text-only backbone -- the decoder is what TP splits anyway, and the
        # tower's weights and warmup would be paid for nothing.
        ("google/gemma-4-26B-A4B", GEMMA4_TEXT_BACKBONE_OVERRIDE),
    ],
    ids=["micro-g3.3", "gemma-4-26B-A4B-text"],
)
def test_tp2_compiled_llm_generate_matches_tp1(model: str, hf_overrides) -> None:
    """TP=1 vs TP=2 greedy-decode prefix match, compiled: the in-graph reduction.

    compile_sizes is pinned to the reachable token counts: 1 (one sequence
    decoding alone), 2 (both decoding), and 16 as the prefill/scheduler cap.
    """
    _cc = {"compile_sizes": [1, 2, 16]}
    _assert_matches_tp1(
        _generate(
            model, tp=1, enforce_eager=False, compilation_config=_cc, hf_overrides=hf_overrides
        ),
        _generate(
            model, tp=2, enforce_eager=False, compilation_config=_cc, hf_overrides=hf_overrides
        ),
    )
