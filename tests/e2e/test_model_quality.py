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

# fp16 on device reorders accumulation against the fp32 reference, so probabilities are
# compared with a tolerance. Same default as sendnn-inference's TEST_ABS_TOL.
ABS_TOL = float(os.environ.get("SPYRE_TEST_ABS_TOL", "0.08"))

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
    for hf_result, output in zip(ref["results"], outputs):
        _compare_against_hf(model, hf_result, output)


def _compare_against_hf(model: str, hf_result: dict[str, Any], output: RequestOutput) -> None:
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
        zip(hf_result["token_ids"], hf_result["logprobs"], token_ids, logprobs)
    ):
        hf_prob, prob = math.exp(hf_logprob), math.exp(logprob)
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
            assert math.isclose(spyre_hf_prob, hf_prob, abs_tol=ABS_TOL), (
                f"{model}: wrong token and p(HF token) differs by more than {ABS_TOL} "
                f"(Spyre {spyre_hf_prob:.4f} vs HF {hf_prob:.4f}), {detail}"
            )
            # A tie also means Spyre itself ranks the two level. Without this, a flat HF
            # distribution (its own argmax at p=0.1) would excuse Spyre being confidently
            # elsewhere at p=0.85, since p(HF token) still matches at 0.1 in both.
            # Bound is 2*ABS_TOL, not ABS_TOL: HF picked its token, so it led there
            # (p_hf(spyre token) <= hf_prob), and each of the two may drift by ABS_TOL in
            # the opposite direction, which is what flipped the argmax in the first place.
            tie_tol = 2 * ABS_TOL
            assert math.isclose(prob, spyre_hf_prob, abs_tol=tie_tol), (
                f"{model}: wrong token, and Spyre puts it {prob - spyre_hf_prob:.4f} > "
                f"{tie_tol} above HF's token (p={spyre_hf_prob:.4f}), so this is not a "
                f"near-tie, {detail}"
            )
            print(
                f"    diverged on a near-tie at {detail}; p(HF token) on Spyre "
                f"{spyre_hf_prob:.4f}; not comparing further"
            )
            return

        assert math.isclose(hf_prob, prob, abs_tol=ABS_TOL), (
            f"{model}: probability differs by more than {ABS_TOL}, {detail}"
        )
