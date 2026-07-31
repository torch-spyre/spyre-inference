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


@pytest.mark.parametrize(
    "model_ref_output",
    [
        (
            "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
            "\n\nA list of Identified Benefits under Debt Management – Count",
        ),
        pytest.param(
            ("google/gemma-3-1b-it", "\n\nIBM's main"),
            marks=pytest.mark.skip(reason="Gemma3 currently doesn't work with torch.compile"),
        ),
    ],
)
def test_basic_llm_inference(model_ref_output, monkeypatch: pytest.MonkeyPatch) -> None:
    """Construct `vllm.LLM(enforce_eager=False)` end-to-end."""
    from vllm import LLM

    prompt = "What are IBMs main businesses?"

    model, ref_output = model_ref_output

    engine = LLM(
        model=model,
        enforce_eager=False,
        compilation_config={"mode": "STOCK_TORCH_COMPILE"},
        max_model_len=128,
        max_num_seqs=2,
    )

    output = engine.generate(prompt, use_tqdm=False)

    assert prompt == output[0].prompt, "Model output contained wrong prompt!"
    assert ref_output == output[0].outputs[0].text, "Model produced wrong output!"
