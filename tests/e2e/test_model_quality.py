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

"""Output-quality gate for the product decoder models: compiled Spyre output against a
cached CPU HF reference, comparing token ids and per-token probabilities.

Prompts and references: ``python tests/data/generate_decoder_output_refs.py``
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
from vllm import LLM, RequestOutput, SamplingParams

pytestmark = [pytest.mark.model_quality, pytest.mark.uses_subprocess]

DECODER_MODELS = [
    "ibm-granite/granite-3.3-8b-instruct",
    "ibm-granite/granite-4.1-8b",
    "google/gemma-4-31B",
    "meta-llama/Llama-3.1-8B-Instruct",
]

# Weight-only FP8 (compressed-tensors) checkpoints, each mapped to the unquantized sibling
# whose reference entry it borrows prompts from -- the smoke test below compares no
# output, it only needs prompts that fit a compiled prefill bucket. Both run end to end on
# Spyre today, so the sibling entries are what gate the numerics.
FP8_DECODER_MODELS = {
    "ibm-granite/granite-3.3-8b-instruct-FP8": "ibm-granite/granite-3.3-8b-instruct",
    "ibm-granite/granite-4.1-8b-fp8": "ibm-granite/granite-4.1-8b",
}
FP8_REVISIONS = {
    "ibm-granite/granite-3.3-8b-instruct-FP8": "4b5990b8d402a75febe0086abbf1e490af494e3d",
    "ibm-granite/granite-4.1-8b-fp8": "070021b3608433b6107a00733d561c9779b9937e",
}
# Short: nothing is compared, so the run only has to prove decode advances at all.
FP8_MAX_TOKENS = 8

# fp16 on device reorders accumulation against the fp32 reference, so probabilities are
# compared with a tolerance. Same default as sendnn-inference's TEST_ABS_TOL.
ABS_TOL = float(os.environ.get("SPYRE_TEST_ABS_TOL", "0.08"))
# ABS_TOL alone is not a uniform bound: it holds p=0.999 to 8% but lets p=0.08 land
# anywhere in [0, 0.16], a 2x relative error, so the gate is loosest exactly where the
# reference is least certain. Below the ABS_TOL/REL_TOL crossover the bound goes
# relative, holding a low-confidence token to the same *fraction* instead of the same
# margin. At the defaults the crossover is p=0.16, so nothing above it changes.
REL_TOL = float(os.environ.get("SPYRE_TEST_REL_TOL", "0.5"))

# Enough of the distribution that HF's greedy token is present even when Spyre picks a
# different one -- `_compare_against_hf` needs p(HF token) under *Spyre* to tell a
# near-tie from two distributions that disagree. 20 is vLLM's default `max_logprobs`.
NUM_LOGPROBS = 20

MAX_MODEL_LEN = 256
MAX_NUM_SEQS = 3
# Caps the compiled buckets (platform.py) and so warmup; every prompt fits one bucket.
MAX_NUM_BATCHED_TOKENS = 64
COMPILE_SIZES = [MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS]

_REF_PATH = Path(__file__).parent.parent / "data" / "decoder_output_refs.json"
_REFERENCES: dict = json.loads(_REF_PATH.read_text()) if _REF_PATH.exists() else {}


@pytest.mark.parametrize("model", DECODER_MODELS)
def test_decoder_model_output(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compiled Spyre output matches the cached HF reference for `model`."""
    ref = _REFERENCES.get(model)
    assert ref is not None, (
        f"No HF reference for {model} in {_REF_PATH.name}; regenerate with "
        f"`python tests/data/generate_decoder_output_refs.py --models {model}`"
    )

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    prompts = [result["prompt"] for result in ref["results"]]
    max_tokens = ref["max_tokens"]
    revision = ref["revision"]

    _assert_prompts_fit_prefill_bucket(model, revision, prompts)

    engine = LLM(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        enforce_eager=False,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        compilation_config={"compile_sizes": COMPILE_SIZES},
    )

    outputs = engine.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=NUM_LOGPROBS,  # sampled token plus enough to locate HF's
            ignore_eos=True,  # the reference is a fixed-length run with EOS disabled
        ),
        use_tqdm=False,
    )

    assert [output.prompt for output in outputs] == prompts, "Model output contained wrong prompt!"
    matched = [
        _compare_against_hf(model, hf_result, output)
        for hf_result, output in zip(ref["results"], outputs)
    ]
    # A prompt that diverges early verifies only the steps before the split, so a green
    # case is not automatically a well-covered one. Printed (PYTEST_ARGS carries -s) so
    # the coverage a run actually achieved is visible without having to fail first.
    per_prompt = ", ".join(f"{n}/{max_tokens}" for n in matched)
    print(
        f"\n{model}: matched {sum(matched)}/{len(prompts) * max_tokens} reference steps "
        f"({per_prompt} per prompt). Prompts that stop after a step or two diverged on a "
        f"near-tie and gate little -- see MODEL_PROMPTS in "
        f"tests/data/generate_decoder_output_refs.py."
    )


@pytest.mark.parametrize("model", FP8_DECODER_MODELS)
def test_fp8_decoder_model_smoke(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A compiled FP8 checkpoint loads and decodes.

    Load-and-decode only, with no reference comparison: a reference for a
    compressed-tensors checkpoint means dequantizing it on CPU first, which the generator
    does not do. So this holds the FP8 weight load and the ``aten._scaled_mm`` kernel
    (``custom_ops/fp8_linear_kernel.py``) to running at all rather than to a numerical
    bound -- `test_decoder_model_output` gates the unquantized siblings' output.
    """
    base = FP8_DECODER_MODELS[model]
    base_ref = _REFERENCES.get(base)
    assert base_ref is not None, (
        f"No HF reference for {base} in {_REF_PATH.name}, and {model} borrows its prompts; "
        f"regenerate with `python tests/data/generate_decoder_output_refs.py --models {base}`"
    )

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    prompts = [result["prompt"] for result in base_ref["results"]]
    revision = FP8_REVISIONS[model]

    _assert_prompts_fit_prefill_bucket(model, revision, prompts)

    engine = LLM(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        enforce_eager=False,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        compilation_config={"compile_sizes": COMPILE_SIZES},
    )

    outputs = engine.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=FP8_MAX_TOKENS, ignore_eos=True),
        use_tqdm=False,
    )

    assert [output.prompt for output in outputs] == prompts, "Model output contained wrong prompt!"
    for output in outputs:
        completion = output.outputs[0]
        print(f"\n{model}  prompt: {output.prompt!r}\n    Spyre: {completion.text!r}")
        assert len(completion.token_ids) == FP8_MAX_TOKENS, (
            f"{model}: generated {len(completion.token_ids)} of {FP8_MAX_TOKENS} tokens"
        )
        assert completion.text.strip(), f"{model}: empty completion for {output.prompt!r}"


def _assert_prompts_fit_prefill_bucket(model: str, revision: str, prompts: list[str]) -> None:
    """Fail loudly if a prompt outgrew the largest compiled prefill bucket.

    Nothing else does: `next_bucket` stick-aligns past the end of the ladder
    (spyre_shape_bucketer.py), so an over-long prompt is not an error but an uncompiled
    shape -- it recompiles inside generate(), and the raised
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS lets that grind for hours instead of failing. Run
    before the engine is built so an edited prompt costs seconds, not a warmup.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    for prompt in prompts:
        num_tokens = len(tokenizer(prompt).input_ids)
        assert num_tokens <= MAX_NUM_BATCHED_TOKENS, (
            f"{model}: prompt is {num_tokens} tokens, past the largest compiled bucket "
            f"({MAX_NUM_BATCHED_TOKENS}) -- it would recompile at generate() time. Shorten "
            f"it, or raise MAX_NUM_BATCHED_TOKENS here and in the generator: {prompt!r}"
        )


def _prob_tol(reference_prob: float) -> float:
    """Tolerance for one probability comparison, tightening as the reference gets small.

    ``min`` and not ``max``: this only ever tightens ABS_TOL, never loosens it, so the
    bound is the stricter of "within ABS_TOL" and "within REL_TOL of the reference".
    """
    return min(ABS_TOL, REL_TOL * reference_prob)


def _compare_against_hf(model: str, hf_result: dict[str, Any], output: RequestOutput) -> int:
    completion = output.outputs[0]
    token_ids = list(completion.token_ids)
    logprobs = [completion.logprobs[i][t].logprob for i, t in enumerate(token_ids)]

    print(f"\n{model}  prompt: {hf_result['prompt']!r}")
    print(f"    HF:    {hf_result['text']!r}")
    print(f"    Spyre: {completion.text!r}")

    assert len(token_ids) == len(hf_result["token_ids"]), (
        f"{model}: generated {len(token_ids)} tokens, reference has {len(hf_result['token_ids'])}"
    )

    for step, (hf_id, hf_logprob, token_id, logprob) in enumerate(
        zip(hf_result["token_ids"], hf_result["logprobs"], token_ids, logprobs, strict=True)
    ):
        hf_prob, prob = math.exp(hf_logprob), math.exp(logprob)
        tol = _prob_tol(hf_prob)
        detail = (
            f"step {step}: token {token_id} ({completion.logprobs[step][token_id].decoded_token!r},"
            f" p={prob:.4f}) vs HF {hf_id} ({hf_result['tokens'][step]!r}, p={hf_prob:.4f})"
        )

        if hf_id != token_id:
            # Greedy paths only diverge legitimately on a near-tie. Judge that on the HF
            # token in *both* distributions, never on the two sampled tokens' own
            # probabilities: those agree whenever the models are equally confident, so
            # HF at p=0.9 on one token and Spyre at p=0.9 on another -- a total
            # disagreement -- would read as a tie. Past this step the prefixes differ,
            # so no later token is comparable either way.
            spyre_hf = completion.logprobs[step].get(hf_id)
            assert spyre_hf is not None, (
                f"{model}: wrong token and HF's token is outside Spyre's top "
                f"{NUM_LOGPROBS}, so the distributions disagree outright, {detail}"
            )
            spyre_hf_prob = math.exp(spyre_hf.logprob)
            assert abs(spyre_hf_prob - hf_prob) <= tol, (
                f"{model}: wrong token and p(HF token) differs by more than {tol:.4f} "
                f"(Spyre {spyre_hf_prob:.4f} vs HF {hf_prob:.4f}), {detail}"
            )
            # A tie also means Spyre itself ranks the two level. Without this, a flat HF
            # distribution (its own argmax at p=0.1) would excuse Spyre being confidently
            # elsewhere at p=0.85, since p(HF token) still matches at 0.1 in both.
            # Bound is doubled: HF picked its token, so it led there
            # (p_hf(spyre token) <= hf_prob), and each of the two may drift by `tol` in
            # the opposite direction, which is what flipped the argmax in the first place.
            tie_tol = 2 * tol
            assert abs(prob - spyre_hf_prob) <= tie_tol, (
                f"{model}: wrong token, and Spyre puts it {prob - spyre_hf_prob:.4f} > "
                f"{tie_tol:.4f} above HF's token (p={spyre_hf_prob:.4f}), so this is not "
                f"a near-tie, {detail}"
            )
            print(
                f"    diverged on a near-tie at {detail}; p(HF token) on Spyre "
                f"{spyre_hf_prob:.4f}; not comparing further"
            )
            return step

        assert abs(hf_prob - prob) <= tol, (
            f"{model}: probability differs by more than {tol:.4f}, {detail}"
        )

    return len(token_ids)
