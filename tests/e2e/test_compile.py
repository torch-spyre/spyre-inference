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

"""Torch.compile tests"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from spyre_inference.models.gemma4 import GEMMA4_TEXT_BACKBONE_OVERRIDE

# enforce_eager=False builds a subprocess EngineCore, so uses_subprocess runs these
# before any in-process test initializes the Spyre device (a subprocess cannot open
# the VFIO device once the main pytest process holds it).
pytestmark = pytest.mark.uses_subprocess

_POOLING_MODEL = "ibm-granite/granite-embedding-125m-english"
_POOLING_REFS = Path(__file__).parent.parent / "data" / "encoder_embed_refs.json"
_COSINE_MIN = 0.99


@pytest.mark.parametrize(
    "model_ref_output",
    [
        pytest.param(
            (
                "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
                "\n\nIBMs main businesses are the companies that provide the services of the",
                None,
            ),
            marks=pytest.mark.e2e_smoke,
        ),
        # FP8 checkpoint of the same model: QFP8WT weights and FP8 linears traced into the
        # block graph.
        pytest.param(
            (
                "ibm-ai-platform/micro-g3.3-8b-instruct-1b-FP8",
                "\n\nIBMs are a set of standards that are used to measure the performance",
                None,
            ),
            marks=[pytest.mark.fp8, pytest.mark.e2e_smoke],
        ),
        (
            "google/gemma-3-1b-it",
            "\n\nIBM's main businesses are:\n\n*   **Consulting:** Providing",
            None,
        ),
        # Both gemma-4 checkpoints carry a vision_config, so an unpinned run resolves
        # Gemma4ForConditionalGeneration and builds the tower. This test is about the
        # compiled decoder -- pin the backbone rather than pay for the tower; the vision
        # path is covered by tests/models/test_gemma4_mm.py.
        (
            "google/gemma-4-31B",
            "\n\nWhat are the main businesses of IBM?\n\nWhat are the main businesses of",
            GEMMA4_TEXT_BACKBONE_OVERRIDE,
        ),
        # The 26B-A4B layout (MoE, sliding + full attention) at 6 layers, real weights;
        # the full 26B-A4B is compiled by the quality gate (test_model_quality.py).
        pytest.param(
            (
                "joerunde/gemma-4-moe-nano",
                " street and methodically knocking people's hats off then I account it high time",
                GEMMA4_TEXT_BACKBONE_OVERRIDE,
            ),
            marks=pytest.mark.e2e_smoke,
        ),
    ],
    ids=[
        "micro-g3.3",
        "micro-g3.3-fp8",
        "gemma-3-1b-it",
        "gemma-4-31B-text",
        "gemma-4-moe-nano-text",
    ],
)
def test_basic_llm_inference(model_ref_output, monkeypatch: pytest.MonkeyPatch) -> None:
    """Construct `vllm.LLM(enforce_eager=False)` end-to-end under ``SPYRE_COMPILE_GUARD=error``.

    No compilation_config is passed: the platform defaults a non-eager run to
    STOCK_TORCH_COMPILE (one transformer block at a time + attention kernel).

    The guard is what exercises its own integration rather than its mechanics -- the
    worker arming it after ``warming_up_model()``, and the ``watch`` registrations in the
    attention backends, the model runner, the MoE regions and the FP8 linear path. If
    warmup really compiles everything a request needs, arming changes nothing.
    """
    monkeypatch.setenv("SPYRE_COMPILE_GUARD", "error")
    model, ref_output, hf_overrides = model_ref_output
    _assert_compiled_output(model, ref_output, monkeypatch, hf_overrides=hf_overrides)


def test_whole_model_granularity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole-model graph still produces the same tokens."""
    monkeypatch.setenv("SPYRE_COMPILE_GRANULARITY", "model")
    _assert_compiled_output(
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        "\n\nIBMs main businesses are the companies that provide the services of the",
        monkeypatch,
    )


@pytest.mark.e2e_smoke
def test_compiled_pooling_encoder_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compiled pooling pads to ``(B, L)`` and matches cached HF refs.

    Two prompts at ``max_num_seqs=2`` / ``max_model_len=64`` warmup body ``T``
    and attention ``(1, 64)`` / ``(2, 64)``. Runtime 1D-pads the body; SDPA
    gathers onto ``(2, 64)``.
    """
    from vllm import LLM

    refs = json.loads(_POOLING_REFS.read_text())[_POOLING_MODEL]
    prompts = refs["prompts"]
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    engine = LLM(
        model=_POOLING_MODEL,
        runner="pooling",
        enforce_eager=False,
        max_model_len=64,
        max_num_seqs=2,
    )
    outputs = engine.embed(prompts)
    assert len(outputs) == len(prompts)
    for out, ref_emb in zip(outputs, refs["embeddings"]):
        emb = out.outputs.embedding
        assert len(emb) == len(ref_emb)
        assert all(math.isfinite(x) for x in emb)
        sim = F.cosine_similarity(
            torch.tensor(emb, dtype=torch.float32),
            torch.tensor(ref_emb, dtype=torch.float32),
            dim=0,
        ).item()
        assert sim >= _COSINE_MIN, f"cosine {sim:.4f} < {_COSINE_MIN}"


@pytest.mark.e2e_smoke
def test_transformers_backend_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compile the Transformers backend and check against a known reference.

    Guards the Spyre-safe attention forward (transpose + contiguous + reshape)
    against the silent data corruption that the unfused view chain caused.
    """
    _assert_compiled_output(
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        "\n\nIBMs main businesses are the companies that provide the services of the",
        monkeypatch,
        model_impl="transformers",
    )


def test_v2_warmup_covers_every_compile_a_request_needs() -> None:
    """The guard test on the V2 runner, plus a prompt-logprobs request: its chunk is the
    one lm_head width wider than any sampled batch, which warmup once left cold.

    In a fresh interpreter: the Triton gate binds the runner choice at vLLM import, which
    this process has long since done."""
    node = f"{__file__}::test_compile_guard_with_prompt_logprobs"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            node,
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "not upstream",
        ],
        env={**os.environ, "SPYRE_MODEL_RUNNER": "v2", "_SPYRE_V2_CHILD": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    # A skipped or deselected child also exits 0.
    assert "1 passed" in result.stdout, result.stdout[-4000:]


@pytest.mark.skipif(os.environ.get("_SPYRE_V2_CHILD") != "1", reason="run by the V2 test above")
def test_compile_guard_with_prompt_logprobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPYRE_COMPILE_GUARD", "error")
    _assert_compiled_output(
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        "\n\nIBMs main businesses are the companies that provide the services of the",
        monkeypatch,
        prompt_logprobs=1,
    )


def _assert_compiled_output(
    model: str,
    ref_output: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    hf_overrides=None,
    model_impl: str | None = None,
    prompt_logprobs: int | None = None,
) -> None:
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    prompt = "What are IBMs main businesses?"

    kwargs: dict = dict(
        model=model,
        enforce_eager=False,
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=8,
        compilation_config=CompilationConfig(compile_sizes=[1, 8]),
        **({"hf_overrides": hf_overrides} if hf_overrides is not None else {}),
    )
    if model_impl is not None:
        kwargs["model_impl"] = model_impl

    engine = LLM(**kwargs)

    output = engine.generate(
        prompt,
        SamplingParams(temperature=0.0, max_tokens=16, prompt_logprobs=prompt_logprobs),
        use_tqdm=False,
    )

    assert prompt == output[0].prompt, "Model output contained wrong prompt!"
    assert ref_output == output[0].outputs[0].text, "Model produced wrong output!"
    if prompt_logprobs is not None:
        assert output[0].prompt_logprobs, "No prompt logprobs returned!"
