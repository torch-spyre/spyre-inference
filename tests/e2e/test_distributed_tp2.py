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
from spyre_testing_plugin.pytest_plugin import spyre_device_count
from spyre_testing_plugin.tags import record_serving_capability
from spyre_testing_plugin.vfio_reaper import wait_until_card_free

from spyre_inference.models.gemma4 import GEMMA4_TEXT_BACKBONE_OVERRIDE

MAX_MODEL_LEN = 128
MAX_NUM_SEQS = 2


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
def test_tp2_llm_construction(record_property) -> None:
    """Construct `vllm.LLM(tensor_parallel_size=2)` end-to-end.

    Goes through the real `MultiprocExecutor` worker-spawn path that
    `vllm serve --tensor-parallel-size 2` uses.
    """
    from vllm import LLM

    model = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
    record_serving_capability(
        record_property,
        model,
        "served",
        2,
        MAX_MODEL_LEN,
        MAX_NUM_SEQS,
        task="generate",
        enforce_eager=True,
    )
    LLM(
        model=model,
        tensor_parallel_size=2,
        dtype="float16",
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
    )


def _generate(
    model: str,
    tp: int,
    enforce_eager: bool,
    compilation_config: dict | None = None,
    hf_overrides=None,
) -> list[list[int]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        dtype="float16",
        enforce_eager=enforce_eager,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        **({"compilation_config": compilation_config} if compilation_config is not None else {}),
        **({"hf_overrides": hf_overrides} if hf_overrides is not None else {}),
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
    # Outside the finally, where a generate failure would mask this check's own.
    assert freed, "Spyre devices were not released after LLM shutdown"
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
def test_tp2_compiled_llm_generate_matches_tp1(model: str, hf_overrides, record_property) -> None:
    """TP=1 vs TP=2 greedy-decode prefix match, compiled: the in-graph reduction.

    compile_sizes is pinned to the reachable token counts: 1 (one sequence
    decoding alone), 2 (both decoding), and 16 as the prefill/scheduler cap.
    Its served TP=2 verdict also requires TP=1 to run and match.
    """
    _cc = {"compile_sizes": [1, 2, 16]}
    record_serving_capability(
        record_property,
        model,
        "served",
        2,
        MAX_MODEL_LEN,
        MAX_NUM_SEQS,
        task="generate",
        enforce_eager=False,
        text_only=hf_overrides is not None,
    )
    _assert_matches_tp1(
        _generate(
            model, tp=1, enforce_eager=False, compilation_config=_cc, hf_overrides=hf_overrides
        ),
        _generate(
            model, tp=2, enforce_eager=False, compilation_config=_cc, hf_overrides=hf_overrides
        ),
    )


MULTIMODAL_MODEL = "mistralai/Ministral-3-14B-Instruct-2512-BF16"
# Short enough to keep both warmups (TP=1, then TP=2) inside one test's budget: the 176x176
# image and the prompt fit, and attention records only the buckets up to it.
MULTIMODAL_MAX_MODEL_LEN = 256


def _image_conversation():
    """One deterministic 176x176 image: an 11x11 patch grid, coprime with the 64 stick."""
    import base64
    import io

    from PIL import Image

    image = Image.new("RGB", (176, 176))
    pixels = image.load()
    for y in range(176):
        for x in range(176):
            pixels[x, y] = (x * 7 % 256, y * 5 % 256, (x + y) * 3 % 256)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": "Describe this image in one short sentence."},
            ],
        }
    ]


def _generate_image(tp: int) -> list[list[int]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MULTIMODAL_MODEL,
        config_format="mistral",
        tensor_parallel_size=tp,
        dtype="float16",
        max_model_len=MULTIMODAL_MAX_MODEL_LEN,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1},
        # One decoding sequence, and a prefill of the whole prompt.
        compilation_config={"compile_sizes": [1, MULTIMODAL_MAX_MODEL_LEN]},
    )
    try:
        outs = llm.chat([_image_conversation()], SamplingParams(max_tokens=8, temperature=0.0))
        result = [list(o.outputs[0].token_ids) for o in outs]
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=60)
        del llm
        gc.collect()
        freed = wait_until_card_free(exclude_pids={os.getpid()}, timeout=60)
    assert freed, "Spyre devices were not released after LLM shutdown"
    return result


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.multimodal
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
@pytest.mark.timeout(3600)
def test_tp2_compiled_image_prompt_matches_tp1(monkeypatch) -> None:
    """TP=1 vs TP=2 greedy prefix match on one image prompt, compiled: the vision blocks
    and projection run their TP all_reduces inside the compiled graphs, and serving must
    not compile anything."""
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    monkeypatch.setenv("SPYRE_COMPILE_GUARD", "error")
    _assert_matches_tp1(_generate_image(tp=1), _generate_image(tp=2))
