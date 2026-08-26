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

pytestmark = pytest.mark.compile


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


def _assert_compiled_output(model: str, ref_output: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm import LLM, SamplingParams

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    prompt = "What are IBMs main businesses?"

    engine = LLM(
        model=model,
        enforce_eager=False,
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=8,
    )

    output = engine.generate(
        prompt,
        SamplingParams(temperature=0.0, max_tokens=16),
        use_tqdm=False,
    )

    assert prompt == output[0].prompt, "Model output contained wrong prompt!"
    assert ref_output == output[0].outputs[0].text, "Model produced wrong output!"
