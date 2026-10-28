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

"""Spyre product encoder tests vs cached HF refs: embeddings, reranker scores, labels.

Regenerate: ``generate_encoder_embed_refs.py``, ``generate_rerank_score_refs.py``
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from types import ModuleType

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
    # head_size=32 (below one stick): the models the encoder attention host
    # round-trip / bert_head_pad load-time padding exist for.
    "ibm-granite/granite-embedding-30m-english",
    "sentence-transformers/all-MiniLM-L6-v2",
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

# Cross-encoder rerankers (classify / score path). The BGE variants share
# XLMRobertaForSequenceClassification but not their weights or position table.
RERANKER_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-large",
]

# Pairs of 513-1024 tokens at the default-derived max_model_len (2048) batch onto the
# (1024, 2) rectangle, whose fused-QKV attention needs torch-spyre#5069 (#4893).
LONG_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LONG_RERANK_QUERY = "What is the performance of the Spyre accelerator for reranking tasks?"
LONG_RERANK_DOCUMENTS = [
    ("the spyre accelerator delivers fast reranking throughput on ibm hardware " * 45).strip(),
    (
        "reranking vllm spyre inference benchmark token test document paragraph context " * 45
    ).strip(),
]

# Token classification applies its own classifier after a head_dtype cast.
# Same path as sequence-classify: fp16 x @ Wᵀ and bias on Spyre.
TOKEN_CLASSIFY_MODEL = "dslim/bert-base-NER"
TOKEN_CLASSIFY_PROMPTS = [
    "My name is Wolfgang and I live in Berlin",
    "George Washington went to Washington",
]

# Match upstream check_embeddings_close(tol=1e-2).
COSINE_MIN = 0.99

# Sigmoid probabilities. An absolute bound of 0.03 permits an arbitrary relative
# error on a near-zero score, so the relative bound applies there. fp16 still
# moves a ~1e-5 probability by about that much, so the relative bound has a
# floor; below it the document is already not relevant.
SCORE_ABS_TOL = float(os.environ.get("SPYRE_TEST_SCORE_ABS_TOL", "0.03"))
SCORE_REL_TOL = float(os.environ.get("SPYRE_TEST_SCORE_REL_TOL", "0.5"))
SCORE_REL_FLOOR = float(os.environ.get("SPYRE_TEST_SCORE_REL_FLOOR", "2e-5"))

_REF_PATH = Path(__file__).parent.parent / "data" / "encoder_embed_refs.json"
_REFERENCES: dict = json.loads(_REF_PATH.read_text()) if _REF_PATH.exists() else {}

_RERANK_REF_PATH = Path(__file__).parent.parent / "data" / "rerank_score_refs.json"
_RERANK_REFERENCES: dict = (
    json.loads(_RERANK_REF_PATH.read_text()) if _RERANK_REF_PATH.exists() else {}
)
_RERANK_GENERATOR_PATH = _RERANK_REF_PATH.with_name("generate_rerank_score_refs.py")


def _rerank_generator() -> ModuleType:
    """The reference generator, loaded by path: ``tests`` is a namespace package, so
    importing it as ``tests.data...`` breaks whenever another ``tests`` package wins."""
    spec = importlib.util.spec_from_file_location(
        "generate_rerank_score_refs", _RERANK_GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None, _RERANK_GENERATOR_PATH
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cosine(a: list[float], b: list[float]) -> float:
    return F.cosine_similarity(
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
        dim=0,
    ).item()


def _hf_last_token_embeddings(model: str, revision: str, prompts: list[str]) -> list[list[float]]:
    """CPU HF last-nonpad-token + L2 (matches vLLM LastPool + normalize)."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    hf = AutoModel.from_pretrained(model, revision=revision)
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
    _assert_embeddings_match_refs(model, enforce_eager=True)


@pytest.mark.model_quality
@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model", EMBEDDING_MODELS)
def test_encoder_embed_models_compiled(model: str) -> None:
    """Same models and references, compiled rather than eager."""
    _assert_embeddings_match_refs(model, enforce_eager=False)


def _assert_embeddings_match_refs(model: str, enforce_eager: bool) -> None:
    ref = _REFERENCES.get(model)
    if ref is None:
        pytest.skip(f"No HF ref for {model}; run tests/data/generate_encoder_embed_refs.py")

    prompts = ref["prompts"]
    llm = LLM(
        model=model,
        revision=ref["revision"],
        tokenizer_revision=ref["revision"],
        runner="pooling",
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=enforce_eager,
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
        revision=ref["revision"],
        tokenizer_revision=ref["revision"],
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
    # Both sides are computed here, so the pin buys reproducibility, not a valid comparison.
    ref = _REFERENCES.get(LAST_POOLING_MODEL)
    if ref is None:
        pytest.skip(
            f"No HF ref for {LAST_POOLING_MODEL}; run tests/data/generate_encoder_embed_refs.py"
        )

    revision = ref["revision"]
    prompts = LAST_POOLING_PROMPTS
    ref_embs = _hf_last_token_embeddings(LAST_POOLING_MODEL, revision, prompts)

    llm = LLM(
        model=LAST_POOLING_MODEL,
        revision=revision,
        tokenizer_revision=revision,
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
    """Spyre reranker scores match the cached HF references within tolerance."""
    _assert_rerank_scores_match_refs(model, enforce_eager=True)


@pytest.mark.model_quality
@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model", RERANKER_MODELS)
def test_encoder_rerank_models_compiled(model: str) -> None:
    """Same models and references, compiled rather than eager."""
    _assert_rerank_scores_match_refs(model, enforce_eager=False)


def _assert_rerank_scores_match_refs(model: str, enforce_eager: bool) -> None:
    """Classifier GEMM runs on Spyre in fp16 (no fp32 matmul, torch-spyre#1794)."""
    ref = _RERANK_REFERENCES.get(model)
    if ref is None:
        pytest.skip(f"No HF ref for {model}; run tests/data/generate_rerank_score_refs.py")

    documents = ref["documents"]
    ref_scores = ref["scores"]
    llm = LLM(
        model=model,
        revision=ref["revision"],
        tokenizer_revision=ref["revision"],
        runner="pooling",
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=enforce_eager,
    )
    outputs = llm.score(ref["query"], documents)
    assert len(outputs) == len(documents)
    _assert_scores_match(model, documents, [out.outputs.score for out in outputs], ref_scores)


@pytest.mark.model_quality
@pytest.mark.uses_subprocess
def test_encoder_rerank_default_max_model_len_compiled() -> None:
    """Warmup at the default max_model_len survives, and a (1024, 2) batch matches HF."""
    ref = _RERANK_REFERENCES.get(LONG_RERANK_MODEL)
    if ref is None:
        pytest.skip(
            f"No HF ref for {LONG_RERANK_MODEL}; run tests/data/generate_rerank_score_refs.py"
        )
    revision = ref["revision"]

    ref_scores, token_counts = _rerank_generator().score_pairs(
        LONG_RERANK_MODEL, revision, LONG_RERANK_QUERY, LONG_RERANK_DOCUMENTS
    )
    for count in token_counts:
        assert 512 < count <= 1024, f"pair of {count} tokens no longer pads to L=1024"

    llm = LLM(
        model=LONG_RERANK_MODEL,
        revision=revision,
        tokenizer_revision=revision,
        runner="pooling",
        enforce_eager=False,
    )
    assert llm.llm_engine.model_config.max_model_len == 2048
    outputs = llm.score(LONG_RERANK_QUERY, LONG_RERANK_DOCUMENTS)
    assert len(outputs) == len(LONG_RERANK_DOCUMENTS)
    _assert_scores_match(
        LONG_RERANK_MODEL,
        LONG_RERANK_DOCUMENTS,
        [out.outputs.score for out in outputs],
        ref_scores,
    )


def _assert_scores_match(
    model: str, documents: list[str], scores: list[float], ref_scores: list[float]
) -> None:
    assert all(math.isfinite(s) for s in scores), f"{model}: non-finite score in {scores}"

    # Checked apart from the per-score bound: a pair can swap with both inside tolerance,
    # and all scores can drift one direction without reordering.
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ref_order = sorted(range(len(ref_scores)), key=lambda i: ref_scores[i], reverse=True)
    assert order == ref_order, (
        f"{model}: ranked documents {order} vs cached HF {ref_order}; "
        f"scores {scores} vs {ref_scores}"
    )

    for document, score, ref_score in zip(documents, scores, ref_scores, strict=True):
        tol = min(SCORE_ABS_TOL, max(SCORE_REL_TOL * ref_score, SCORE_REL_FLOOR))
        assert abs(score - ref_score) <= tol, (
            f"{model}: score {score:.6f} vs HF {ref_score:.6f} (tol {tol:.6f}) "
            f"for {document[:80]!r}"
        )


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
