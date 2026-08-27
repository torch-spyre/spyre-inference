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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Upstream RMSNorm.forward_native upcasts fp16->fp32; torch-spyre rejects the "
        "resulting mixed-EA broadcast at the S=64 prefill ('Multi-arg pointwise with "
        "mixed EA'). SpyreRMSNorm (fp16, no upcast) works around it. When this passes, "
        "torch-spyre supports native RMSNorm and the custom op can be removed."
    ),
)
def test_native_rmsnorm_prefill_s64_lowers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream native RMSNorm must lower at the S=64 prefill.

    Whole-model (block-graph) phenomenon: a lone torch.compile of forward_native does
    not reproduce it, and S=1 decode does not either -- only the S=64 prefill compile.
    """
    from vllm import LLM, SamplingParams

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    # register_oot renames the op to its class name, so disabling the Spyre op needs
    # -RMSNorm/-TPAwareRMSNorm (the in-tree -rms_norm is a no-op here).
    engine = LLM(
        model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        enforce_eager=False,
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        compilation_config={
            "custom_ops": ["all", "-RMSNorm", "-TPAwareRMSNorm"],
            "compile_sizes": [64, 1],
        },
    )

    # The mixed-EA rejection fires during engine construction (prefill warmup compile),
    # before generate; generation just confirms the graph runs.
    output = engine.generate(
        {"prompt_token_ids": list(range(1, 65))},
        SamplingParams(temperature=0.0, max_tokens=4),
        use_tqdm=False,
    )

    assert output[0].outputs[0].text, "S=64 prefill graph did not produce output"
