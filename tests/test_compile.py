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

import pytest

def test_basic_llm_inference() -> None:
    """Construct `vllm.LLM(enforce_eager=False)` end-to-end.
    """
    from vllm import LLM
    
    prompt = "What are IBMs main businesses?"
    reference_output = "\n\nA list of Identified Benefits under thisones main businesses"

    engine = LLM(
        model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        dtype="float16",
        enforce_eager=False,
        max_model_len=128,
        max_num_seqs=2,
    )
    
    output = engine.generate(prompt, use_tqdm=False)
    
    assert prompt == output[0].prompt, "Model output contained wrong prompt!"
    assert reference_output == output[0].outputs[0].text, "Model produced wrong output!"