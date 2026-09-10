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

import pytest

# enforce_eager=False builds a subprocess EngineCore, so uses_subprocess runs these
# before any in-process test initializes the Spyre device (a subprocess cannot open
# the VFIO device once the main pytest process holds it).
pytestmark = pytest.mark.uses_subprocess

_POOLING_MODEL = "ibm-granite/granite-embedding-125m-english"
_POOLING_REVISION = "4ab61ffd423be45cd932b21a7c696063d82bf45f"
_POOLING_PROMPTS = ["Hello world.", "The quick brown fox jumps over the lazy dog."]


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
        (
            "google/gemma-4-26B-A4B",
            "\n\nWhat is the difference between a product and a service?\n\nWhat is the",
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


def test_compiled_pooling_encoder_buckets(
    hf_embeddings, assert_embeddings_close, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compiled pooling pads to ``(B, L)`` and matches live HF.

    Two prompts at ``max_num_seqs=2`` / ``max_model_len=64`` warmup body ``T``
    and attention ``(1, 64)`` / ``(2, 64)``. Runtime 1D-pads the body; SDPA
    gathers onto ``(2, 64)``.
    """
    from vllm import LLM

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    hf_embs = hf_embeddings(_POOLING_MODEL, _POOLING_REVISION, _POOLING_PROMPTS)

    engine = LLM(
        model=_POOLING_MODEL,
        runner="pooling",
        enforce_eager=False,
        max_model_len=64,
        max_num_seqs=2,
    )
    outputs = engine.embed(_POOLING_PROMPTS)
    assert len(outputs) == len(_POOLING_PROMPTS)
    assert_embeddings_close(_POOLING_MODEL, [out.outputs.embedding for out in outputs], hf_embs)


def _assert_compiled_output(model: str, ref_output: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    prompt = "What are IBMs main businesses?"

    engine = LLM(
        model=model,
        enforce_eager=False,
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=8,
        compilation_config=CompilationConfig(compile_sizes=[1, 8]),
    )

    output = engine.generate(
        prompt,
        SamplingParams(temperature=0.0, max_tokens=16),
        use_tqdm=False,
    )

    assert prompt == output[0].prompt, "Model output contained wrong prompt!"
    assert ref_output == output[0].outputs[0].text, "Model produced wrong output!"
