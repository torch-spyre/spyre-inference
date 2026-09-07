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

"""Write CPU HF cross-encoder scores to rerank_score_refs.json for the reranker gates.

Each run merges into the existing file.

    python tests/data/generate_rerank_score_refs.py
    python tests/data/generate_rerank_score_refs.py --models BAAI/bge-reranker-large
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Model ids must match tests/e2e/test_encoder_models.py.
RERANKER_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-large",
]

MODEL_REVISIONS = {
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "BAAI/bge-reranker-large": "55611d7bca2a7133960a6d3b71e083071bbfc312",
}

# One query against documents a reranker should separate widely, listed most relevant
# first. The spread is what lets the test gate the *ranking* and not only the individual
# scores, so neighbouring documents have to stay far enough apart that fp16 drift cannot
# swap them. What has to clear that is the gap against the drift the two scores can absorb
# between them -- min(7e-3, 13% * p) each, the worst measured in test_encoder_models.py --
# and not a ratio of scores or of odds: odds explode near p=1, where drift is absolute, and
# collapse onto the score ratio near p=0, where it is relative, so either ratio rates the
# wrong pair as the risky one. The tightest pair below clears its combined drift by 2.8x
# (the top two, 0.99997 vs 0.96147: a gap of 0.039 against 0.014 of drift); every other
# pair clears it by 6x or more.
QUERY = "What is the capital of France?"
DOCUMENTS = [
    "The capital of France is Paris.",
    "Paris is the largest city in France by population.",
    "The Eiffel Tower stands on the Champ de Mars in Paris.",
    "France is a country in Western Europe with about 68 million inhabitants.",
    "Berlin is the capital of Germany.",
    "The IBM Spyre accelerator runs AI inference workloads.",
]

# bge-reranker-large saturates: it scores every Paris-adjacent document above 0.9994, so on
# the list above its top five land within 5e-4 of each other and their order is decided by
# fp16 noise rather than by relevance -- the same drift bound, failed. Give it documents
# spread across the range it actually resolves; the list below clears that bound by 4.4x at
# its tightest (0.00187 vs 0.00052, a gap of 1.4e-3 against 3.1e-4 of drift), so it is the
# safer of the two despite sitting lower. Drop this entry only for a model that separates
# the shared list.
MODEL_DOCUMENTS = {
    "BAAI/bge-reranker-large": [
        "The capital of France is Paris.",
        "France is a country in Western Europe with about 68 million inhabitants.",
        "France moved its seat of government several times in its history.",
        "Berlin is the capital of Germany.",
        "The IBM Spyre accelerator runs AI inference workloads.",
    ],
}

# Scores span ~1e-5 to ~1 and the test bounds small ones *relatively*, so a fixed number
# of decimals has to keep several significant digits at the bottom of that range.
_ROUND = 8

OUT_PATH = Path(__file__).parent / "rerank_score_refs.json"


def generate_reference(model_id: str, revision: str) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, dtype=torch.float32
    )
    model.eval()

    documents = MODEL_DOCUMENTS.get(model_id, DOCUMENTS)
    scores = []
    for document in documents:
        # The same tokenizer call vLLM's cross-encoder io_processor makes, one pair at a
        # time so no padding reaches the reference.
        inputs = tokenizer(text=QUERY, text_pair=document, return_tensors="pt")
        with torch.inference_mode():
            logit = model(**inputs).logits.reshape(-1)
        assert logit.numel() == 1, f"{model_id}: expected num_labels=1, got {logit.numel()}"
        # vLLM's PoolerClassify sigmoids a single-label head, so the reference is a
        # probability and the test's tolerance is an absolute bound on [0, 1].
        scores.append(round(float(torch.sigmoid(logit)[0]), _ROUND))
        print(f"  {document!r}\n    -> {scores[-1]:.6f}", flush=True)

    return {
        "revision": revision,
        "query": QUERY,
        "documents": documents,
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=RERANKER_MODELS)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    data = json.loads(args.out.read_text()) if args.out.exists() else {}
    for model_id in args.models:
        print(f"Scoring {model_id} ...", flush=True)
        data[model_id] = generate_reference(model_id, MODEL_REVISIONS[model_id])
        args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
