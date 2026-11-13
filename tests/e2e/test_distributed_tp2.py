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
# Max logprob gap (nats) per token while TP=1 and TP=2 share a prefix; the 6-layer nano runs wider.
LOGPROB_TOLERANCE = {"joerunde/gemma-4-moe-nano": 0.7}
DEFAULT_LOGPROB_TOLERANCE = 0.1


def _generate(
    model: str,
    tp: int,
    enforce_eager: bool,
    compilation_config: dict | None = None,
    hf_overrides=None,
    kv_cache_dtype: str = "auto",
) -> tuple[list[list[int]], list[list[dict[int, float]]]]:
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
            ["Hello, world!", "Count to ten: one, two, three"],
            SamplingParams(max_tokens=8, temperature=0.0, logprobs=NUM_LOGPROBS, ignore_eos=True),
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

    assert len(tp1) == len(tp2), f"prompt count: {names[0]} {len(tp1)} vs {names[1]} {len(tp2)}"
    for i, (a, b) in enumerate(zip(tp1, tp2)):
        n = prefix_len(a, b)
        assert n >= 2, (
            f"prompt {i}: {names[0]} and {names[1]} diverged at token {n} "
            f"(expected >=2 matching tokens). {names[0]}={a} {names[1]}={b}"
        )


def _assert_top_logprobs_close(
    where: str, ref: dict[int, float], lp: dict[int, float], tolerance: float
) -> None:
    """Every token in either top-k agrees within the tolerance.

    A token outside one run's top-k sits at or below that run's k-th logprob, so it
    fails only when its other logprob clears that bound by more than the tolerance.
    """
    for a, b, names in ((ref, lp, ("tp1", "tp2")), (lp, ref, ("tp2", "tp1"))):
        floor = min(b.values())
        for tid, value in a.items():
            gap = abs(value - b[tid]) if tid in b else value - floor
            assert gap <= tolerance, (
                f"{where}: token {tid} logprob {names[0]} {value:.3f} vs {names[1]} "
                f"{b.get(tid, f'<= {floor:.3f}')} (gap {gap:.3f} > {tolerance})"
            )


def _assert_logprobs_match_tp1(
    model: str,
    tp1: tuple[list[list[int]], list[list[dict[int, float]]]],
    tp2: tuple[list[list[int]], list[list[dict[int, float]]]],
) -> None:
    """Up to the first divergence, both runs' top tokens agree within the tolerance.

    A divergence must be a near tie: each run's winner is in the other's top tokens and
    beats it by no more than the tolerance in both runs.
    """
    tolerance = LOGPROB_TOLERANCE.get(model, DEFAULT_LOGPROB_TOLERANCE)
    for name, (tokens, logprobs) in (("tp1", tp1), ("tp2", tp2)):
        for i, (seq, steps) in enumerate(zip(tokens, logprobs, strict=True)):
            assert len(seq) == len(steps), (
                f"prompt {i}: {name} has {len(seq)} tokens but {len(steps)} logprob rows"
            )
    for i, (ref_tokens, tokens, ref_steps, steps) in enumerate(
        zip(tp1[0], tp2[0], tp1[1], tp2[1], strict=True)
    ):
        for step, (ref_pick, pick, ref_lp, lp) in enumerate(
            zip(ref_tokens, tokens, ref_steps, steps)
        ):
            where = f"prompt {i} step {step}"
            assert ref_pick in lp, f"{where}: tp1 token {ref_pick} not in tp2 top {lp}"
            assert pick in ref_lp, f"{where}: tp2 token {pick} not in tp1 top {ref_lp}"
            _assert_top_logprobs_close(where, ref_lp, lp, tolerance)
            if pick != ref_pick:
                margins = (ref_lp[ref_pick] - ref_lp[pick], lp[pick] - lp[ref_pick])
                assert max(margins) <= tolerance, (
                    f"{where}: tp1 picked {ref_pick}, tp2 picked {pick}; not a near tie "
                    f"(margins tp1 {margins[0]:.3f}, tp2 {margins[1]:.3f} > {tolerance})"
                )
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
    "model,hf_overrides,prefill",
    [
        pytest.param(
            "ibm-ai-platform/micro-g3.3-8b-instruct-1b", None, 64, marks=pytest.mark.e2e_smoke
        ),
        # gemma-4 vision checkpoints resolve the multimodal architecture, so these rows
        # pin the text-only backbone -- the decoder is what TP splits anyway, and the
        # tower's weights and warmup would be paid for nothing. The nano model keeps
        # the 26B-A4B layout (MoE, sliding + full attention) at 6 layers. Its MoE loop runs
        # one expert per trip on the 64-token bucket and grouped on the 128-token one.
        ("joerunde/gemma-4-moe-nano", GEMMA4_TEXT_BACKBONE_OVERRIDE, 64),
        ("joerunde/gemma-4-moe-nano", GEMMA4_TEXT_BACKBONE_OVERRIDE, 128),
    ],
    ids=["micro-g3.3", "gemma-4-moe-nano-text", "gemma-4-moe-nano-text-grouped"],
)
def test_tp2_compiled_llm_generate_matches_tp1(
    model: str, hf_overrides, prefill: int, record_property
) -> None:
    """TP=1 vs TP=2 greedy-decode prefix and logprob match, compiled: the in-graph reduction.

    compile_sizes is pinned to the reachable token counts: 1 (one sequence
    decoding alone), 2 (both decoding), and ``prefill`` as the prefill/scheduler cap.
    64 is one fp16 stick, so the prefill pads onto a stick-aligned bucket and
    the MoE is traced into the block graph with its all-reduce. The second gemma-4
    row uses 128, where its MoE loop runs grouped by default.
    Its served TP=2 verdict also requires TP=1 to run and match.
    """
    _cc = {"compile_sizes": [1, 2, prefill]}
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
