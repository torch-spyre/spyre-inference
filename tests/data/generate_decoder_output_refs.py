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

"""Write CPU HF greedy references for tests/e2e/test_model_quality.py.

The models are too large to run through transformers in CI, so the references are
generated here and checked in. Each run merges into the existing file.

    python tests/data/generate_decoder_output_refs.py
    python tests/data/generate_decoder_output_refs.py --models ibm-granite/granite-4.1-8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model ids must match tests/e2e/test_model_quality.py.
DECODER_MODELS = [
    "ibm-granite/granite-3.3-8b-instruct",
    "ibm-granite/granite-4.1-8b",
    "google/gemma-4-31B",
]

MODEL_REVISIONS = {
    "ibm-granite/granite-3.3-8b-instruct": "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
    "ibm-granite/granite-4.1-8b": "1504002f650e656a0a3789d99574df12e3e94ed0",
    "google/gemma-4-31B": "5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89",
}

# Must stay under MAX_NUM_BATCHED_TOKENS (test_model_quality.py) so each prefill lands
# in a single compiled bucket.
_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"
)
PROMPTS = [
    _TEMPLATE.format("Provide a list of instructions for preparing chicken soup."),
    _TEMPLATE.format("What are the main businesses of IBM?"),
    _TEMPLATE.format("Convert char to string in Java."),
]

# gemma-4 drifts from HF as the prompt grows, because torch-spyre runs RMSNorm in fp16:
# on the prompts above its first-token probability is 0.65 against HF's 0.84 and the
# continuation diverges, while short prompts match token for token. Neither the reference
# dtype (fp16 CPU HF agrees with fp32 to <0.002) nor torch.compile (eager deviates just
# as far) is involved, so drop this entry once torch-spyre normalises in fp32.
MODEL_PROMPTS = {
    "google/gemma-4-31B": [
        "What are IBMs main businesses?",
        "The capital of France is",
        "Q: What is the largest planet in our solar system?\nA:",
    ],
}

MAX_TOKENS = 16
_ROUND = 6

OUT_PATH = Path(__file__).parent / "decoder_output_refs.json"


def generate_reference(model_id: str, revision: str, dtype: torch.dtype) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    model.eval()
    # The test runs with ignore_eos=True, so the reference needs all MAX_TOKENS steps.
    model.generation_config.eos_token_id = None

    results = []
    for prompt in MODEL_PROMPTS.get(model_id, PROMPTS):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=MAX_TOKENS,
                return_dict_in_generate=True,
                output_scores=True,
            )
        # normalize_logits gives logprobs over the vocabulary, matching what vLLM reports.
        logprobs = model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )[0]
        new_token_ids = output.sequences[0, input_ids.shape[1] :]

        results.append(
            {
                "prompt": prompt,
                "text": tokenizer.decode(new_token_ids),
                "token_ids": [int(t) for t in new_token_ids],
                "tokens": [tokenizer.decode(t) for t in new_token_ids],
                "logprobs": [round(float(lp), _ROUND) for lp in logprobs],
            }
        )
        print(f"  {prompt!r}\n    -> {results[-1]['text']!r}", flush=True)

    return {
        "revision": revision,
        "max_tokens": MAX_TOKENS,
        "dtype": str(dtype).removeprefix("torch."),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DECODER_MODELS)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    data = json.loads(args.out.read_text()) if args.out.exists() else {}
    for model_id in args.models:
        print(f"Generating {model_id} ...", flush=True)
        data[model_id] = generate_reference(
            model_id, MODEL_REVISIONS[model_id], getattr(torch, args.dtype)
        )
        # Written per model: each one takes minutes and is easy to interrupt.
        args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
