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
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.compile

_POOLING_MODEL = "ibm-granite/granite-embedding-125m-english"
_POOLING_REFS = Path(__file__).parent.parent / "data" / "encoder_embed_refs.json"
_COSINE_MIN = 0.99


@pytest.mark.parametrize(
    "model_ref_output",
    [
        (
            "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
            "\n\nIBMs main businesses are the companies that provide the services of the",
        ),
        (
            "google/gemma-3-1b-it",
            "\n\nIBM's main businesses are:\n\n*   **Consulting:** Providing",
        ),
        (
            "google/gemma-4-31B",
            "\n\nWhat are the main businesses of IBM?\n\nWhat are the main businesses of",
        ),
    ],
)
def test_basic_llm_inference(model_ref_output, monkeypatch: pytest.MonkeyPatch) -> None:
    """Construct `vllm.LLM(enforce_eager=False)` end-to-end.

    No compilation_config is passed: the platform defaults a non-eager run to
    STOCK_TORCH_COMPILE (one transformer block at a time + attention kernel).
    """
    model, ref_output = model_ref_output
    _assert_compiled_output(model, ref_output, monkeypatch)


def test_whole_model_granularity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole-model graph still produces the same tokens."""
    monkeypatch.setenv("SPYRE_COMPILE_GRANULARITY", "model")
    _assert_compiled_output(
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        "\n\nIBMs main businesses are the companies that provide the services of the",
        monkeypatch,
    )


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


def _assert_compiled_output(
    model: str,
    ref_output: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_impl: str | None = None,
) -> None:
    from vllm import LLM, SamplingParams

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    prompt = "What are IBMs main businesses?"

    kwargs: dict = dict(
        model=model,
        enforce_eager=False,
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=8,
    )
    if model_impl is not None:
        kwargs["model_impl"] = model_impl

    engine = LLM(**kwargs)

    output = engine.generate(
        prompt,
        SamplingParams(temperature=0.0, max_tokens=16),
        use_tqdm=False,
    )

    assert prompt == output[0].prompt, "Model output contained wrong prompt!"
    assert ref_output == output[0].outputs[0].text, "Model produced wrong output!"
