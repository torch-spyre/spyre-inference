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

"""Spyre product embed tests vs cached HF refs and reranker smoke tests.

Regenerate embed refs: ``python tests/data/generate_encoder_embed_refs.py``
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from vllm import LLM
from vllm.config import PoolerConfig

EMBEDDING_MODELS = [
    "ibm-granite/granite-embedding-125m-english",
    "ibm-granite/granite-embedding-278m-multilingual",
    "intfloat/multilingual-e5-large",
    "sentence-transformers/all-roberta-large-v1",
]

# MEAN product models. The default embed e2e uses max_num_seqs=1; batched MEAN
# is ``test_encoder_embed_mean_multi_seq``.
MEAN_POOLING_MODELS = [
    "intfloat/multilingual-e5-large",
    "sentence-transformers/all-roberta-large-v1",
]

# None of the product encoder models above ship with LAST pooling (CLS or MEAN).
# Force LAST on a small CLS model so SpyreLastPool is covered end-to-end.
LAST_POOLING_MODEL = "ibm-granite/granite-embedding-125m-english"
LAST_POOLING_PROMPTS = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
]

# Cross-encoder reranker smoke (classify / score path). One model is enough —
# both BGE variants share XLMRobertaForSequenceClassification.
RERANKER_MODELS = [
    "BAAI/bge-reranker-v2-m3",
]

# Token classification: the model applies its own classifier after casting to
# head_dtype. prepare_token_head_for_spyre casts the classifier to fp16 so it
# runs on Spyre instead of detouring through SpyreCpuClassifier.
TOKEN_CLASSIFY_MODEL = "dslim/bert-base-NER"
TOKEN_CLASSIFY_PROMPTS = [
    "My name is Wolfgang and I live in Berlin",
    "George Washington went to Washington",
]

# Match upstream check_embeddings_close(tol=1e-2).
COSINE_MIN = 0.99

_REF_PATH = Path(__file__).parent.parent / "data" / "encoder_embed_refs.json"
_REFERENCES: dict = json.loads(_REF_PATH.read_text()) if _REF_PATH.exists() else {}


def _cosine(a: list[float], b: list[float]) -> float:
    return F.cosine_similarity(
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
        dim=0,
    ).item()


def _hf_last_token_embeddings(model: str, prompts: list[str]) -> list[list[float]]:
    """CPU HF last-nonpad-token + L2 (matches vLLM LastPool + normalize)."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    hf = AutoModel.from_pretrained(model)
    hf.eval()
    with torch.inference_mode():
        enc = tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        hs = hf(**enc).last_hidden_state
        idx = enc["attention_mask"].sum(dim=1) - 1
        emb = hs[torch.arange(hs.size(0)), idx]
        emb = F.normalize(emb.float(), p=2, dim=-1)
    return emb.tolist()


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


@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model", MEAN_POOLING_MODELS)
def test_encoder_embed_mean_multi_seq(model: str) -> None:
    """MEAN with ``max_num_seqs=2`` so two requests share one packed ``[T, H]``.

    The default embed e2e is ``max_num_seqs=1`` and never hits two sequences
    in one packed ``[T, H]`` copy.
    """
    ref = _REFERENCES.get(model)
    if ref is None:
        pytest.skip(f"No HF ref for {model}; run tests/data/generate_encoder_embed_refs.py")

    prompts = ref["prompts"]
    llm = LLM(
        model=model,
        runner="pooling",
        max_model_len=64,
        max_num_seqs=2,
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
            f"{model} batched MEAN: cosine {sim:.4f} < {COSINE_MIN} "
            f"vs cached HF reference for prompt {prompt!r}"
        )


@pytest.mark.uses_subprocess
def test_encoder_embed_last_pooling() -> None:
    """SpyreLastPool path: force LAST on granite-125m and match HF last-token.

    Product encoder models in ``EMBEDDING_MODELS`` are CLS or MEAN only; this
    override exercises the LAST gather + normalize path that
    ``configure_pooling_for_spyre`` patches to ``SpyreLastPool``.
    """
    prompts = LAST_POOLING_PROMPTS
    ref_embs = _hf_last_token_embeddings(LAST_POOLING_MODEL, prompts)

    llm = LLM(
        model=LAST_POOLING_MODEL,
        runner="pooling",
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=True,
        pooler_config=PoolerConfig(seq_pooling_type="LAST"),
    )
    outputs = llm.embed(prompts)
    assert len(outputs) == len(prompts)

    for prompt, out, ref_emb in zip(prompts, outputs, ref_embs):
        emb = out.outputs.embedding
        assert len(emb) == len(ref_emb), (
            f"LAST {LAST_POOLING_MODEL}: dim mismatch {len(emb)} vs HF {len(ref_emb)}"
        )
        assert all(math.isfinite(x) for x in emb)
        sim = _cosine(emb, ref_emb)
        assert sim >= COSINE_MIN, (
            f"LAST {LAST_POOLING_MODEL}: cosine {sim:.4f} < {COSINE_MIN} "
            f"vs HF last-token for prompt {prompt!r}"
        )


@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model", RERANKER_MODELS)
def test_encoder_rerank_models(model: str) -> None:
    """Load reranker and return one finite score via LLM.score()."""
    llm = LLM(
        model=model,
        runner="pooling",
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=True,
    )
    scores = llm.score("What is Spyre?", "An IBM AI accelerator.")
    assert len(scores) == 1
    assert math.isfinite(scores[0].outputs.score)


@pytest.mark.uses_subprocess
def test_encoder_token_classify() -> None:
    """Per-token scores match softmax(HF fp32 logits) and agree on every label."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKEN_CLASSIFY_MODEL)
    hf = AutoModelForTokenClassification.from_pretrained(TOKEN_CLASSIFY_MODEL, dtype=torch.float32)
    hf.eval()
    with torch.inference_mode():
        refs = [
            hf(**tok(p, return_tensors="pt")).logits[0].float().softmax(-1)
            for p in TOKEN_CLASSIFY_PROMPTS
        ]

    llm = LLM(
        model=TOKEN_CLASSIFY_MODEL,
        runner="pooling",
        max_model_len=64,
        max_num_seqs=2,
        enforce_eager=True,
    )
    outputs = llm.encode(TOKEN_CLASSIFY_PROMPTS, pooling_task="token_classify")
    assert len(outputs) == len(TOKEN_CLASSIFY_PROMPTS)

    for prompt, out, ref in zip(TOKEN_CLASSIFY_PROMPTS, outputs, refs):
        got = torch.as_tensor(out.outputs.data).float()
        assert got.shape == ref.shape, f"{prompt!r}: {tuple(got.shape)} vs {tuple(ref.shape)}"
        assert torch.equal(got.argmax(-1), ref.argmax(-1)), (
            f"{prompt!r}: labels {got.argmax(-1).tolist()} vs HF {ref.argmax(-1).tolist()}"
        )
        assert (got - ref).abs().max().item() < 1e-2, f"{prompt!r}: scores drifted from HF"
