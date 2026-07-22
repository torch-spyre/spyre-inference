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

"""Smoke tests for product encoder embedding models on Spyre.

Checks ``LLM.embed()`` returns a finite vector. Upstream ``test_embedding.py``
covers HF cosine for BGE / MiniLM.
"""

from __future__ import annotations

import math

import pytest
from vllm import LLM

# Product embedding models (static batching / encoder-only pooling).
EMBEDDING_MODELS = [
    "ibm-granite/granite-embedding-125m-english",
    "ibm-granite/granite-embedding-278m-multilingual",
    "intfloat/multilingual-e5-large",
    "sentence-transformers/all-roberta-large-v1",
]


@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model", EMBEDDING_MODELS)
def test_encoder_embed_models(model: str) -> None:
    """Load model and return one finite embedding vector."""
    llm = LLM(
        model=model,
        runner="pooling",
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=True,
    )
    outputs = llm.embed(["Hello world."])
    assert len(outputs) == 1
    emb = outputs[0].outputs.embedding
    assert len(emb) > 0
    assert all(math.isfinite(x) for x in emb)
