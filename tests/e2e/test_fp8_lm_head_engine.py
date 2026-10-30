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

"""The FP8 LM head through a real engine: warmup, logits row padding, compile guard.

Every cached FP8 checkpoint lists ``lm_head`` in its ``ignore``, so no other engine
test reaches the FP8 head. This one forces the gate on, in-process so the patch reaches
the model. Like ``test_transformers_matches_native``, it is ordered with the
``uses_subprocess`` tests, ahead of the session ``tp_group`` that would already hold the
TP group the engine initializes, and tears the engine's distributed state down after.
"""

import pytest


@pytest.mark.fp8
@pytest.mark.uses_subprocess
def test_fp8_lm_head_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig
    from vllm.distributed import cleanup_dist_env_and_memory

    from spyre_inference.custom_ops import parallel_lm_head

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    monkeypatch.setattr(parallel_lm_head, "checkpoint_quantizes_head", lambda *_: True)
    loads = []
    load = parallel_lm_head.SpyreFp8LMHeadMethod.process_weights_after_loading

    def counting_load(self, layer):
        loads.append(layer)
        load(self, layer)

    monkeypatch.setattr(
        parallel_lm_head.SpyreFp8LMHeadMethod, "process_weights_after_loading", counting_load
    )

    prompt = "What are IBMs main businesses?"
    engine = None
    try:
        engine = LLM(
            model="ibm-ai-platform/micro-g3.3-8b-instruct-1b-FP8",
            enforce_eager=False,
            max_model_len=128,
            max_num_seqs=2,
            max_num_batched_tokens=8,
            compilation_config=CompilationConfig(compile_sizes=[1, 8]),
        )
        output = engine.generate(
            prompt, SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False
        )
    finally:
        # A half-built engine still holds the TP group the next test would initialize.
        del engine
        cleanup_dist_env_and_memory()

    assert loads, "the head stayed FP16"
    # The FP16-head reference: FP8 logits keep the same greedy tokens on this prompt.
    expected = "\n\nIBMs are a set of standards that are used to measure the performance"
    assert output[0].outputs[0].text == expected
