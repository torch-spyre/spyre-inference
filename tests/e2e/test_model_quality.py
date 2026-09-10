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

"""Compiled decoder output vs cached CPU HF references: token ids and probabilities.

References: ``python tests/data/generate_decoder_output_refs.py``
"""

from __future__ import annotations

import json
import math
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from vllm import LLM, RequestOutput, SamplingParams

pytestmark = [pytest.mark.model_quality, pytest.mark.uses_subprocess]

DECODER_MODELS = [
    "ibm-granite/granite-3.3-8b-instruct",
    "ibm-granite/granite-4.1-8b",
    "google/gemma-4-31B",
    "google/gemma-4-26B-A4B",
    "meta-llama/Llama-3.1-8B-Instruct",
]

# Maps to the unquantized sibling whose prompts the smoke case borrows.
FP8_DECODER_MODELS = {
    "ibm-granite/granite-3.3-8b-instruct-FP8": "ibm-granite/granite-3.3-8b-instruct",
    "ibm-granite/granite-4.1-8b-fp8": "ibm-granite/granite-4.1-8b",
}
FP8_REVISIONS = {
    "ibm-granite/granite-3.3-8b-instruct-FP8": "4b5990b8d402a75febe0086abbf1e490af494e3d",
    "ibm-granite/granite-4.1-8b-fp8": "070021b3608433b6107a00733d561c9779b9937e",
}
FP8_MAX_TOKENS = 8

# fp16 on device reorders accumulation against the fp32 reference, so probabilities need a
# tolerance. The mean is the bound that holds quality; ABS_TOL only catches gross breakage,
# loose because a reference near p=0.5 measured 0.115 apart across two CI pods with every
# token still exact (PR #723).
MEAN_ABS_TOL = float(os.environ.get("SPYRE_TEST_MEAN_ABS_TOL", "0.03"))
ABS_TOL = float(os.environ.get("SPYRE_TEST_ABS_TOL", "0.20"))
# Low-probability steps need a relative bound; a flat one permits an arbitrary ratio.
REL_TOL = float(os.environ.get("SPYRE_TEST_REL_TOL", "0.5"))
TIE_ABS_TOL = float(os.environ.get("SPYRE_TEST_TIE_ABS_TOL", "0.08"))

# vLLM's `max_logprobs` ceiling. HF's token must be in Spyre's top-k on a disagreement.
NUM_LOGPROBS = 20

# A near-tie split ends the comparison, so coverage is a fact about the prompts, not about
# quality: fp16 drift alone decides the argmax where the reference is a coin flip
# (granite-4.1 opens one prompt on p=0.4961). Hence reported, not asserted.
COVERAGE_WARN_FRACTION = float(os.environ.get("SPYRE_TEST_COVERAGE_WARN_FRACTION", "0.5"))

MAX_MODEL_LEN = 256
MAX_NUM_SEQS = 3
# platform.py clamps max_num_batched_tokens to max(compile_sizes), so this is its top entry.
MAX_NUM_BATCHED_TOKENS = 64
COMPILE_SIZES = [MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS]
# The guard counts with a bare `tokenizer()`; vLLM can add a special token or two.
PROMPT_TOKEN_MARGIN = 8

_REF_PATH = Path(__file__).parent.parent / "data" / "decoder_output_refs.json"
_REFERENCES: dict = json.loads(_REF_PATH.read_text()) if _REF_PATH.exists() else {}


@pytest.mark.parametrize("model", DECODER_MODELS)
def test_decoder_model_output(
    model: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, str], None],
) -> None:
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
            logprobs=NUM_LOGPROBS,
            ignore_eos=True,  # the reference is a fixed-length run with EOS disabled
        ),
        use_tqdm=False,
    )

    assert [output.prompt for output in outputs] == prompts, "Model output contained wrong prompt!"
    matched = [
        _compare_against_hf(model, hf_result, output)
        for hf_result, output in zip(ref["results"], outputs)
    ]
    _report_reference_coverage(model, matched, max_tokens, record_property)


@pytest.mark.parametrize("model", FP8_DECODER_MODELS)
def test_fp8_decoder_model_smoke(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A compiled FP8 checkpoint loads and decodes; no reference (no CPU dequant)."""
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
        # Token count only: this case must not assume the tokens decode to non-empty text.
        assert len(completion.token_ids) == FP8_MAX_TOKENS, (
            f"{model}: generated {len(completion.token_ids)} of {FP8_MAX_TOKENS} tokens"
        )


class LowReferenceCoverage(UserWarning):
    """A decoder case compared fewer than ``COVERAGE_WARN_FRACTION`` of the reference steps."""


def _report_reference_coverage(
    model: str,
    matched: list[int],
    max_tokens: int,
    record_property: Callable[[str, str], None],
) -> None:
    """Warn, never fail, on a case the prompts cut short; zero compared is the exception."""
    total = len(matched) * max_tokens
    compared = sum(matched)
    per_prompt = ", ".join(f"{n}/{max_tokens}" for n in matched)
    # `key__value` is the JUnit tag convention the ClickHouse ingest reads (plugin tags.py).
    record_property("tag", f"refcoverage__{compared}/{total}")
    print(f"\n{model}: compared {compared}/{total} reference steps ({per_prompt} per prompt).")

    assert compared > 0, (
        f"{model}: every prompt diverged on its first step, so not one probability was "
        f"compared and this case asserted nothing about output quality. The near-tie bounds "
        f"held at each split, so this is the fixture, not a regression: replace the prompts "
        f"in MODEL_PROMPTS (tests/data/generate_decoder_output_refs.py) with ones whose "
        f"reference does not open on a coin flip, and regenerate."
    )
    if compared < math.ceil(COVERAGE_WARN_FRACTION * total):
        warnings.warn(
            f"{model}: compared only {compared}/{total} reference steps ({per_prompt} per "
            f"prompt) -- the rest was cut off by a near-tie split, so this case gates less "
            f"than it looks like it does. The quality bounds passed on what it did compare. "
            f"The fix is more confident prompts in MODEL_PROMPTS "
            f"(tests/data/generate_decoder_output_refs.py), not a looser tolerance.",
            LowReferenceCoverage,
            stacklevel=2,
        )


def _assert_prompts_fit_prefill_bucket(model: str, revision: str, prompts: list[str]) -> None:
    """Past the largest bucket `find_bucket` returns None and the shape recompiles in
    generate(), so an over-long prompt fails here instead."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    limit = MAX_NUM_BATCHED_TOKENS - PROMPT_TOKEN_MARGIN
    for prompt in prompts:
        num_tokens = len(tokenizer(prompt).input_ids)
        assert num_tokens <= limit, (
            f"{model}: prompt is {num_tokens} tokens, over the {limit}-token bound this "
            f"guard holds ({PROMPT_TOKEN_MARGIN} below the largest compiled bucket, "
            f"{MAX_NUM_BATCHED_TOKENS}) -- past the bucket it would recompile at generate() "
            f"time. Shorten it, or raise MAX_NUM_BATCHED_TOKENS here and in the generator: "
            f"{prompt!r}"
        )


def _prob_tol(reference_prob: float) -> float:
    """Per-step cap when Spyre and HF picked the same token."""
    return min(ABS_TOL, REL_TOL * reference_prob)


def _tie_tol(reference_prob: float) -> float:
    """Bound for accepting a token disagreement as a near-tie rather than a regression."""
    return min(TIE_ABS_TOL, REL_TOL * reference_prob)


def _assert_mean_prob_error(model: str, prompt: str, diffs: list[float]) -> None:
    if not diffs:
        return
    mean = sum(diffs) / len(diffs)
    print(f"    prob error over {len(diffs)} compared steps: mean={mean:.4f} max={max(diffs):.4f}")
    assert mean <= MEAN_ABS_TOL, (
        f"{model}: mean probability error {mean:.4f} over {len(diffs)} steps exceeds "
        f"{MEAN_ABS_TOL:.4f} for prompt {prompt!r} -- the distribution drifted as a whole, "
        f"which no single-step bound catches. A regression, not a tolerance to raise."
    )


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

    diffs: list[float] = []
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
            # Judge the tie on HF's token: p(sampled) agrees for any two equally sure models.
            spyre_hf = completion.logprobs[step].get(hf_id)
            assert spyre_hf is not None, (
                f"{model}: wrong token and HF's token is outside Spyre's top "
                f"{NUM_LOGPROBS}, so the distributions disagree outright, {detail}"
            )
            spyre_hf_prob = math.exp(spyre_hf.logprob)
            ref_tol = _tie_tol(hf_prob)
            assert abs(spyre_hf_prob - hf_prob) <= ref_tol, (
                f"{model}: wrong token and p(HF token) differs by more than {ref_tol:.4f} "
                f"(Spyre {spyre_hf_prob:.4f} vs HF {hf_prob:.4f}), {detail}"
            )
            # A tie means Spyre ranks the two level too. Doubled: both sides may drift.
            tie_tol = 2 * ref_tol
            assert abs(prob - spyre_hf_prob) <= tie_tol, (
                f"{model}: wrong token, and Spyre puts it {prob - spyre_hf_prob:.4f} > "
                f"{tie_tol:.4f} above HF's token (p={spyre_hf_prob:.4f}), so this is not "
                f"a near-tie, {detail}"
            )
            print(
                f"    diverged on a near-tie at {detail}; p(HF token) on Spyre "
                f"{spyre_hf_prob:.4f}; not comparing further"
            )
            _assert_mean_prob_error(model, hf_result["prompt"], diffs)
            return step

        assert abs(hf_prob - prob) <= tol, (
            f"{model}: probability differs by more than {tol:.4f}, {detail}"
        )
        diffs.append(abs(hf_prob - prob))

    _assert_mean_prob_error(model, hf_result["prompt"], diffs)
    return len(token_ids)
