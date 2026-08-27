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
        "Upstream vLLM RMSNorm (forward_native) upcasts fp16->fp32, which torch-spyre "
        "cannot yet lower at the S=64 prefill: the upcast lands in a staggered EA while "
        "the mean/rsqrt reduction stays STANDARD, and the broadcast is rejected as "
        "'Multi-arg pointwise with mixed EA'. The Spyre custom op (SpyreRMSNorm."
        "forward_oot, fp16, no upcast) works around it. This test disables the custom "
        "op (-RMSNorm/-TPAwareRMSNorm) to track the upstream path -- when it starts "
        "passing, torch-spyre supports native RMSNorm and the custom SpyreRMSNorm op "
        "can be removed."
    ),
)
def test_native_rmsnorm_prefill_s64_lowers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream (native) RMSNorm must lower at the S=64 prefill.

    Disables the Spyre custom RMSNorm op so the model dispatches to vLLM's upstream
    ``RMSNorm.forward_native``, which upcasts fp16->fp32. With ``compile_native``
    defaulting to False, the disabled op returns the *eager* ``forward_native``, which
    the enclosing STOCK_TORCH_COMPILE block graph then captures. On device the upcast
    lands in a staggered (``FP32_TO_DL16``) element arrangement while the mean/rsqrt
    reduction stays ``STANDARD``; broadcasting the STANDARD ``[64, 1]`` factor against
    the staggered ``[64, 4096]`` tensor is a multi-arg pointwise with mixed EA, which
    the Spyre backend rejects at compile time::

        Unsupported: Spyre backend does not support: Multi-arg pointwise with mixed EA:
        STANDARD input buf26 must broadcast (device stick dimension size 1) to be
        compatible with a staggered EA. Its stick maps to host dim 0 of size 64

    Same failure as the colleague's benchmark (same ``buf26``), with the custom op off::

        vllm bench latency --model ibm-ai-platform/micro-g3.3-8b-instruct-1b \\
          --input-len 64 --output-len 64 --batch-size 1 --max-model-len 128 \\
          -cc.compile_sizes='[64,1]' -cc.custom_ops='["-RMSNorm","-TPAwareRMSNorm"]'

    It is a whole-model (block-graph) fusion phenomenon that depends on the real EA of
    the tensor feeding the norm: a lone ``torch.compile(RMSNorm.forward_native)`` does
    NOT reproduce it (it miscompiles to NaN instead), which is why isolated unit tests
    miss it. S=1 (decode) does not reproduce it; only the S=64 prefill compile does.
    """
    from vllm import LLM, SamplingParams

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    # Disable the Spyre custom op so dispatch falls back to upstream forward_native
    # (the "main way"). NOTE: ``register_oot`` renames the op to its *class* name, so
    # SpyreRMSNorm registers as ``RMSNorm`` and SpyreTPAwareRMSNorm as ``TPAwareRMSNorm``
    # -- ``-rms_norm`` (the in-tree CustomOp name) is a no-op here. ``all`` keeps every
    # other custom op on. A single-chunk 64-token prefill (max_num_batched_tokens >= 64)
    # at compile_sizes [64, 1] mirrors the failing `--input-len 64 -cc.compile_sizes='[64,1]'`.
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

    # Engine construction already warms up (compiles) the S=64 prefill graph -- the
    # mixed-EA rejection fires there, before generate. Generation confirms it runs.
    output = engine.generate(
        {"prompt_token_ids": list(range(1, 65))},
        SamplingParams(temperature=0.0, max_tokens=4),
        use_tqdm=False,
    )

    assert output[0].outputs[0].text, "S=64 prefill graph did not produce output"
