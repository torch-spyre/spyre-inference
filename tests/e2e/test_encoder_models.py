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

"""Spyre product encoder tests vs live CPU HF: embeddings, reranker scores, labels.

References are computed in-run through upstream's ``HfRunner`` and compared with its
``check_embeddings_close``, the way vLLM's ``tests/models/language/pooling/`` tests do.
"""

from __future__ import annotations

import math
import os

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

# Cross-encoder rerankers (classify / score path). The BGE variants share
# XLMRobertaForSequenceClassification but not their weights or position table.
RERANKER_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-large",
]

# Token classification: the model applies its own classifier after casting to
# head_dtype. prepare_token_head_for_spyre casts the classifier to fp16 so it
# runs on Spyre instead of detouring through SpyreCpuClassifier.
TOKEN_CLASSIFY_MODEL = "dslim/bert-base-NER"
TOKEN_CLASSIFY_PROMPTS = [
    "My name is Wolfgang and I live in Berlin",
    "George Washington went to Washington",
]

# Upstream check_embeddings_close's tolerance: it asserts cosine >= 1 - tol.
EMBEDDING_TOL = 1e-2

# Local bounds because upstream has no reranker helper, and its cross-encoder test's inlined
# ones only hold with both sides at the same precision. Here fp16 on the card runs against an
# fp32 CPU reference, where a mid-range sigmoid score drifts by ~1e-2. Most scores sit just
# above zero, where an absolute bound admits any relative error, so the stricter one applies.
SCORE_ABS_TOL = float(os.environ.get("SPYRE_TEST_SCORE_ABS_TOL", "0.03"))
SCORE_REL_TOL = float(os.environ.get("SPYRE_TEST_SCORE_REL_TOL", "0.5"))

# Every gated model is pinned, matching .github/cache_config/hf_models_and_datasets.yaml.
MODEL_REVISIONS = {
    "ibm-granite/granite-embedding-125m-english": "4ab61ffd423be45cd932b21a7c696063d82bf45f",
    "ibm-granite/granite-embedding-278m-multilingual": "a9cb5338491faf32b73dd17b714a31821c021bbf",
    "intfloat/multilingual-e5-large": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
    "sentence-transformers/all-roberta-large-v1": "cf74d8acd4f198de950bf004b262e6accfed5d2c",
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "BAAI/bge-reranker-large": "55611d7bca2a7133960a6d3b71e083071bbfc312",
}

RERANK_QUERY = "What is the capital of France?"
# A relevant document, near-misses, and two irrelevant ones, so the scores have to spread.
RERANK_DOCUMENTS = [
    "The capital of France is Paris.",
    "Paris is the largest city in France by population.",
    "France is a country in Western Europe with about 68 million inhabitants.",
    "Berlin is the capital of Germany.",
    "The IBM Spyre accelerator runs AI inference workloads.",
]


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
def test_encoder_embed_models(
    hf_embeddings, assert_embeddings_close, example_prompts, model: str
) -> None:
    """Spyre embeddings match live HF within cosine tolerance."""
    _assert_embeddings_match_hf(
        hf_embeddings, assert_embeddings_close, example_prompts, model, enforce_eager=True
    )


@pytest.mark.model_quality
@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model", EMBEDDING_MODELS)
def test_encoder_embed_models_compiled(
    hf_embeddings, assert_embeddings_close, example_prompts, model: str
) -> None:
    """Same models and comparison, compiled rather than eager."""
    _assert_embeddings_match_hf(
        hf_embeddings, assert_embeddings_close, example_prompts, model, enforce_eager=False
    )


def _assert_embeddings_match_hf(
    hf_embeddings,
    assert_embeddings_close,
    prompts: list[str],
    model: str,
    enforce_eager: bool,
    max_num_seqs: int = 1,
) -> None:
    revision = MODEL_REVISIONS[model]
    # sentence-transformers strips its inputs, so the vLLM side must send the same text.
    prompts = [prompt.strip() for prompt in prompts]
    hf_embs = hf_embeddings(model, revision, prompts)

    llm = LLM(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        runner="pooling",
        max_model_len=64,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
    )
    outputs = llm.embed(prompts)
    assert len(outputs) == len(prompts)

    assert_embeddings_close(
        model, [out.outputs.embedding for out in outputs], hf_embs, EMBEDDING_TOL
    )


@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model", MEAN_POOLING_MODELS)
def test_encoder_embed_mean_multi_seq(
    hf_embeddings, assert_embeddings_close, example_prompts, model: str
) -> None:
    """MEAN with ``max_num_seqs=2`` so two requests share one packed ``[T, H]``.

    The default embed e2e is ``max_num_seqs=1`` and never hits two sequences
    in one packed ``[T, H]`` copy.
    """
    _assert_embeddings_match_hf(
        hf_embeddings,
        assert_embeddings_close,
        example_prompts,
        model,
        enforce_eager=True,
        max_num_seqs=2,
    )


@pytest.mark.uses_subprocess
def test_encoder_embed_last_pooling(assert_embeddings_close) -> None:
    """SpyreLastPool path: force LAST on granite-125m and match HF last-token.

    Product encoder models in ``EMBEDDING_MODELS`` are CLS or MEAN only; this
    override exercises the LAST gather + normalize path that
    ``configure_pooling_for_spyre`` patches to ``SpyreLastPool``.
    """
    revision = MODEL_REVISIONS[LAST_POOLING_MODEL]
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

    assert_embeddings_close(
        f"LAST {LAST_POOLING_MODEL}",
        [out.outputs.embedding for out in outputs],
        ref_embs,
        EMBEDDING_TOL,
    )


@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model", RERANKER_MODELS)
def test_encoder_rerank_models(hf_runner, model: str) -> None:
    """Spyre reranker scores match live HF within tolerance."""
    _assert_rerank_scores_match_hf(hf_runner, model, enforce_eager=True)


@pytest.mark.model_quality
@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model", RERANKER_MODELS)
def test_encoder_rerank_models_compiled(hf_runner, model: str) -> None:
    """Same models and comparison, compiled rather than eager."""
    _assert_rerank_scores_match_hf(hf_runner, model, enforce_eager=False)


def _assert_rerank_scores_match_hf(hf_runner, model: str, enforce_eager: bool) -> None:
    """Only the encoder body runs on Spyre: the fp32 classifier head has no FP32 batchmatmul
    (torch-spyre#1794), so the pooling tail stays on CPU even when compiled."""
    revision = MODEL_REVISIONS[model]

    # fp32 reference: the point of comparison is the fp16 device path against ground truth.
    pairs = [[RERANK_QUERY, document] for document in RERANK_DOCUMENTS]
    with hf_runner(model, revision=revision, dtype="float32", is_cross_encoder=True) as hf_model:
        hf_scores = hf_model.predict(pairs).tolist()

    llm = LLM(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        runner="pooling",
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=enforce_eager,
    )
    outputs = llm.score(RERANK_QUERY, RERANK_DOCUMENTS)
    assert len(outputs) == len(RERANK_DOCUMENTS)

    scores = [out.outputs.score for out in outputs]
    assert all(math.isfinite(s) for s in scores), f"{model}: non-finite score in {scores}"

    # Checked apart from the per-score bound: a pair can swap with both inside tolerance,
    # and all scores can drift one direction without reordering.
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hf_order = sorted(range(len(hf_scores)), key=lambda i: hf_scores[i], reverse=True)
    assert order == hf_order, (
        f"{model}: ranked documents {order} vs HF {hf_order}; scores {scores} vs {hf_scores}"
    )

    for document, score, hf_score in zip(RERANK_DOCUMENTS, scores, hf_scores, strict=True):
        tol = min(SCORE_ABS_TOL, SCORE_REL_TOL * hf_score)
        assert abs(score - hf_score) <= tol, (
            f"{model}: score {score:.6f} vs HF {hf_score:.6f} (tol {tol:.6f}) for {document!r}"
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
