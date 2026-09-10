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

"""Output-quality gate for the product decoder models: compiled Spyre output vs. live HF.

Modelled on ``tests/models/language/generation/test_hybrid.py::test_models`` from upstream
vLLM, down to using its own ``check_logprobs_close`` as the comparison.
"""

from __future__ import annotations

import functools

import pytest
from vllm import SamplingParams

pytestmark = [pytest.mark.model_quality, pytest.mark.uses_subprocess]

DECODER_MODELS = [
    "ibm-granite/granite-3.3-8b-instruct",
    "ibm-granite/granite-4.1-8b",
    "google/gemma-4-31B",
    "google/gemma-4-26B-A4B",
    "meta-llama/Llama-3.1-8B-Instruct",
]

MODEL_REVISIONS = {
    "ibm-granite/granite-3.3-8b-instruct": "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
    "ibm-granite/granite-4.1-8b": "1504002f650e656a0a3789d99574df12e3e94ed0",
    "google/gemma-4-31B": "5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89",
    "google/gemma-4-26B-A4B": "24548b62aa021d562695c04aaf7758a1ea47990b",
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
}

FP8_DECODER_MODELS = [
    "ibm-granite/granite-3.3-8b-instruct-FP8",
    "ibm-granite/granite-4.1-8b-fp8",
]
FP8_REVISIONS = {
    "ibm-granite/granite-3.3-8b-instruct-FP8": "4b5990b8d402a75febe0086abbf1e490af494e3d",
    "ibm-granite/granite-4.1-8b-fp8": "070021b3608433b6107a00733d561c9779b9937e",
}
FP8_MAX_TOKENS = 8

MAX_TOKENS = 16
NUM_LOGPROBS = 5
HF_DTYPE = "float32"

MAX_MODEL_LEN = 256
MAX_NUM_SEQS = 3
# Passing compile_sizes skips the buckets platform.py would derive, and platform.py clamps
# max_num_batched_tokens down to the largest one, so this is the top of COMPILE_SIZES.
MAX_NUM_BATCHED_TOKENS = 64
COMPILE_SIZES = [MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS]
# Stated rather than inherited: VllmRunner always passes a block size, so platform.py never
# applies the Spyre default, and VllmRunner's 16 would align up to 64 and resize the KV cache.
BLOCK_SIZE = 128
# Slack for the prompt-fit guard below: it counts with a bare `tokenizer(prompt)` while the
# engine tokenizes through vLLM, which can differ by a special token or two.
PROMPT_TOKEN_MARGIN = 8


@functools.cache
def _check_logprobs_close():
    """Upstream's comparison, resolved on first use rather than at import: the tree is a git
    clone, and every CI job imports this module during collection even when ``-m`` deselects it.
    """
    from spyre_testing_plugin.upstream import ensure_upstream_tests_importable

    ensure_upstream_tests_importable()
    from tests.models.utils import check_logprobs_close

    return check_logprobs_close


@pytest.mark.parametrize("model", DECODER_MODELS)
def test_decoder_model_output(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    """Compiled Spyre output matches live HF for ``model``."""
    revision = MODEL_REVISIONS[model]
    prompts = example_prompts

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    tokenizer = _pinned_tokenizer(model, revision)
    _assert_prompts_fit_prefill_bucket(tokenizer, model, prompts)

    # fp32 explicitly: HfRunner's "auto" resolves to CpuPlatform's first supported dtype,
    # bfloat16, whose mantissa is shorter than the fp16 device path this is adjudicating.
    with hf_runner(model, dtype=HF_DTYPE, revision=revision, processor=tokenizer) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(prompts, MAX_TOKENS, NUM_LOGPROBS)

    with vllm_runner(
        model,
        revision=revision,
        tokenizer_revision=revision,
        enforce_eager=False,
        trust_remote_code=False,
        enable_chunked_prefill=None,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        block_size=BLOCK_SIZE,
        compilation_config={"compile_sizes": COMPILE_SIZES},
    ) as spyre_model:
        spyre_outputs = spyre_model.generate_greedy_logprobs(prompts, MAX_TOKENS, NUM_LOGPROBS)

    check_logprobs_close = _check_logprobs_close()
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=spyre_outputs,
        name_0="hf",
        name_1="spyre",
    )


@pytest.mark.parametrize("model", FP8_DECODER_MODELS)
def test_fp8_decoder_model_smoke(
    vllm_runner,
    example_prompts,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    """A compiled FP8 checkpoint loads and decodes.

    No HF comparison: transformers does not dequantize compressed-tensors on CPU, so there is
    nothing to compare against. The unquantized siblings gate the numerics.
    """
    revision = FP8_REVISIONS[model]
    prompts = example_prompts

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    _assert_prompts_fit_prefill_bucket(_pinned_tokenizer(model, revision), model, prompts)

    with vllm_runner(
        model,
        revision=revision,
        tokenizer_revision=revision,
        enforce_eager=False,
        trust_remote_code=False,
        enable_chunked_prefill=None,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        block_size=BLOCK_SIZE,
        compilation_config={"compile_sizes": COMPILE_SIZES},
    ) as spyre_model:
        # generate_w_logprobs rather than generate_greedy, which prepends the prompt ids
        # to every completion and would defeat the token count below.
        outputs = spyre_model.generate_w_logprobs(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=FP8_MAX_TOKENS, ignore_eos=True),
        )

    for prompt, (token_ids, text, _) in zip(prompts, outputs, strict=True):
        print(f"\n{model}  prompt: {prompt!r}\n    Spyre: {text!r}")
        # Token count only: this case must not assume the tokens decode to non-empty text.
        assert len(token_ids) == FP8_MAX_TOKENS, (
            f"{model}: generated {len(token_ids)} of {FP8_MAX_TOKENS} tokens"
        )


def _pinned_tokenizer(model: str, revision: str):
    """The model's tokenizer at `revision`, also passed to `HfRunner` as its `processor`.

    `HfRunner` forwards `revision` only to the weights, so its own tokenizer would come from
    main; passing this one also skips its unconditional `AutoProcessor` load, which for the
    multimodal checkpoints pulls an image processor needing torchvision -- absent from the
    Spyre torch build -- for a text-only comparison.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, revision=revision)


def _assert_prompts_fit_prefill_bucket(tokenizer, model: str, prompts: list[str]) -> None:
    """Fail loudly if a prompt outgrew the largest compiled prefill bucket.

    Past the largest bucket ``SpyreShapeBucketer.find_bucket`` returns None and the shape runs
    unpadded, so an over-long prompt is a silent Dynamo recompile inside generate().
    """
    limit = MAX_NUM_BATCHED_TOKENS - PROMPT_TOKEN_MARGIN
    for prompt in prompts:
        num_tokens = len(tokenizer(prompt).input_ids)
        assert num_tokens <= limit, (
            f"{model}: prompt is {num_tokens} tokens, over the {limit}-token bound this "
            f"guard holds ({PROMPT_TOKEN_MARGIN} below the largest compiled bucket, "
            f"{MAX_NUM_BATCHED_TOKENS}) -- past the bucket it would recompile at generate() "
            f"time. Shorten it, or raise MAX_NUM_BATCHED_TOKENS here: {prompt!r}"
        )
