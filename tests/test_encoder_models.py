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

"""Spyre product embed tests vs cached HF refs in encoder_embed_refs.json.

Regenerate: ``python tests/data/generate_encoder_embed_refs.py``
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from vllm import LLM

EMBEDDING_MODELS = [
    "ibm-granite/granite-embedding-125m-english",
    "ibm-granite/granite-embedding-278m-multilingual",
    "intfloat/multilingual-e5-large",
    "sentence-transformers/all-roberta-large-v1",
]

# Match upstream check_embeddings_close(tol=1e-2).
COSINE_MIN = 0.99

_REF_PATH = Path(__file__).parent / "data" / "encoder_embed_refs.json"
_REFERENCES: dict = json.loads(_REF_PATH.read_text()) if _REF_PATH.exists() else {}


def _cosine(a: list[float], b: list[float]) -> float:
    return F.cosine_similarity(
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
        dim=0,
    ).item()


@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model", EMBEDDING_MODELS)
def test_encoder_embed_models(model: str) -> None:
    """Spyre embeddings match cached HF references within cosine tolerance."""
    ref = _REFERENCES.get(model)
    if ref is None:
        pytest.skip(f"No HF ref for {model}; run tests/data/generate_encoder_embed_refs.py")

    prompts = ref["prompts"]
    llm = LLM(
        model=model,
        runner="pooling",
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=True,
    )
    outputs = llm.embed(prompts)
    assert len(outputs) == len(prompts)

    for prompt, out, ref_emb in zip(prompts, outputs, ref["embeddings"]):
        emb = out.outputs.embedding
        assert len(emb) == len(ref_emb), (
            f"{model}: dim mismatch {len(emb)} vs cached {len(ref_emb)}"
        )
        assert all(math.isfinite(x) for x in emb)
        sim = _cosine(emb, ref_emb)
        assert sim >= COSINE_MIN, (
            f"{model}: cosine {sim:.4f} < {COSINE_MIN} vs cached HF reference for prompt {prompt!r}"
        )
