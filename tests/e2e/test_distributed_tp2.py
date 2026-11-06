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
NUM_LOGPROBS = 5
# Largest gap, in nats, between TP=1's and TP=2's logprob of the same token at a step where
# both runs condition on the same prefix. The 6-layer nano model's fp16 gaps run larger.
LOGPROB_TOLERANCE = {"joerunde/gemma-4-moe-nano": 1.0}
DEFAULT_LOGPROB_TOLERANCE = 0.1


def _generate(
    model: str,
    tp: int,
    enforce_eager: bool,
    compilation_config: dict | None = None,
    hf_overrides=None,
    kv_cache_dtype: str = "auto",
) -> tuple[list[list[int]], list[list[dict[int, float]]]]:
    """Per-prompt token ids, and per-step ``{token id: logprob}`` for the top tokens."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        dtype="float16",
        enforce_eager=enforce_eager,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        kv_cache_dtype=kv_cache_dtype,
        **({"compilation_config": compilation_config} if compilation_config is not None else {}),
        **({"hf_overrides": hf_overrides} if hf_overrides is not None else {}),
    )
    try:
        outs = llm.generate(
            ["Hello, world!", "The capital of France is"],
            SamplingParams(max_tokens=8, temperature=0.0, logprobs=NUM_LOGPROBS),
        )
        tokens = [list(o.outputs[0].token_ids) for o in outs]
        logprobs = [
            [{tid: lp.logprob for tid, lp in step.items()} for step in o.outputs[0].logprobs]
            for o in outs
        ]
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=60)
        del llm
        gc.collect()
        freed = wait_until_card_free(exclude_pids={os.getpid()}, timeout=60)
    # Outside the finally, where a generate failure would mask this check's own.
    assert freed, "Spyre devices were not released after LLM shutdown"
    return tokens, logprobs


def _assert_matches_tp1(
    tp1: list[list[int]], tp2: list[list[int]], names: tuple[str, str] = ("tp1", "tp2")
) -> None:
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
            f"prompt {i}: {names[0]} and {names[1]} diverged at token {n} "
            f"(expected >=2 matching tokens). {names[0]}={a} {names[1]}={b}"
        )


def _assert_logprobs_match_tp1(
    model: str,
    tp1: tuple[list[list[int]], list[list[dict[int, float]]]],
    tp2: tuple[list[list[int]], list[list[dict[int, float]]]],
) -> None:
    """The token prefix check, then at every step up to the first divergence TP=1's token is
    among TP=2's top tokens with a logprob within the model's tolerance."""
    tolerance = LOGPROB_TOLERANCE.get(model, DEFAULT_LOGPROB_TOLERANCE)
    _assert_matches_tp1(tp1[0], tp2[0])
    for i, (tokens, ref_steps, steps) in enumerate(zip(tp1[0], tp1[1], tp2[1])):
        for step, (pick, ref_lp, lp) in enumerate(zip(tokens, ref_steps, steps)):
            assert pick in lp, f"prompt {i} step {step}: tp1 token {pick} not in tp2 top {lp}"
            gap = abs(ref_lp[pick] - lp[pick])
            assert gap <= tolerance, (
                f"prompt {i} step {step}: token {pick} logprob tp1 {ref_lp[pick]:.3f} "
                f"vs tp2 {lp[pick]:.3f} (gap {gap:.3f} > {tolerance})"
            )
            if max(lp, key=lp.get) != pick:
                # Past a flip the two runs condition on different tokens.
                break


@pytest.mark.e2e_smoke
@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
def test_tp2_llm_generate_matches_tp1(record_property) -> None:
    """TP=1 vs TP=2 greedy-decode prefix and logprob match, eager."""
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
    _assert_logprobs_match_tp1(
        model,
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
        pytest.param(
            "ibm-ai-platform/micro-g3.3-8b-instruct-1b", None, marks=pytest.mark.e2e_smoke
        ),
        # gemma-4 vision checkpoints resolve the multimodal architecture, so this row
        # pins the text-only backbone -- the decoder is what TP splits anyway, and the
        # tower's weights and warmup would be paid for nothing. The nano model keeps
        # the 26B-A4B layout (MoE, sliding + full attention) at 6 layers.
        ("joerunde/gemma-4-moe-nano", GEMMA4_TEXT_BACKBONE_OVERRIDE),
    ],
    ids=["micro-g3.3", "gemma-4-moe-nano-text"],
)
def test_tp2_compiled_llm_generate_matches_tp1(model: str, hf_overrides, record_property) -> None:
    """TP=1 vs TP=2 greedy-decode prefix and logprob match, compiled: the in-graph reduction.

    compile_sizes is pinned to the reachable token counts: 1 (one sequence
    decoding alone), 2 (both decoding), and 64 as the prefill/scheduler cap.
    64 is one fp16 stick, so the prefill pads onto a stick-aligned bucket and
    the MoE is traced into the block graph with its all-reduce.
    Its served TP=2 verdict also requires TP=1 to run and match.
    """
    _cc = {"compile_sizes": [1, 2, 64]}
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
    _assert_logprobs_match_tp1(
        model,
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


@pytest.mark.e2e_smoke
@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
def test_tp2_fp8_kv_cache_matches_fp16_cache() -> None:
    """TP=2 with a float8 KV cache against the fp16 cache, compiled.

    Each rank holds half the KV heads, which changes the head-major page shape and the
    per-head store index list the single-card fp8 test covers.
    """
    model = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
    _cc = {"compile_sizes": [1, 2, 16]}
    fp16, _ = _generate(model, tp=2, enforce_eager=False, compilation_config=_cc)
    fp8, _ = _generate(
        model, tp=2, enforce_eager=False, compilation_config=_cc, kv_cache_dtype="fp8"
    )
    _assert_matches_tp1(fp16, fp8, names=("fp16 cache", "fp8 cache"))
