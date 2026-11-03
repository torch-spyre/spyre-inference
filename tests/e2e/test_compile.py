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

"""Torch.compile tests"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from spyre_inference.models.gemma4 import GEMMA4_TEXT_BACKBONE_OVERRIDE

# enforce_eager=False builds a subprocess EngineCore, so uses_subprocess runs these
# before any in-process test initializes the Spyre device (a subprocess cannot open
# the VFIO device once the main pytest process holds it).
pytestmark = pytest.mark.uses_subprocess

_POOLING_MODEL = "ibm-granite/granite-embedding-125m-english"
_TRANSFORMERS_ENCODER_MODEL = "ibm-granite/granite-embedding-278m-multilingual"
_POOLING_REFS = Path(__file__).parent.parent / "data" / "encoder_embed_refs.json"
_COSINE_MIN = 0.99
# RoBERTa-family models (incl. XLM-R) read their position embedding at pos + 2. A second
# +2 (the double offset) still scores 0.992-0.998 on these prompts, so the bar has to sit
# above that.
_TRANSFORMERS_COSINE_MIN = 0.999
_TRANSFORMERS_NER_MODEL = "dslim/bert-base-NER"
_TRANSFORMERS_NER_PROMPTS = [
    "My name is Wolfgang and I live in Berlin",
    "George Washington went to Washington",
]


@pytest.mark.parametrize(
    "model_ref_output",
    [
        (
            "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
            "\n\nIBMs main businesses are the companies that provide the services of the",
            None,
        ),
        (
            "google/gemma-3-1b-it",
            "\n\nIBM's main businesses are:\n\n*   **Consulting:** Providing",
            None,
        ),
        # Both gemma-4 checkpoints carry a vision_config, so an unpinned run resolves
        # Gemma4ForConditionalGeneration and builds the tower. This test is about the
        # compiled decoder -- pin the backbone rather than pay for the tower; the vision
        # path is covered by tests/multimodal/test_gemma4_vision.py.
        (
            "google/gemma-4-31B",
            "\n\nWhat are the main businesses of IBM?\n\nWhat are the main businesses of",
            GEMMA4_TEXT_BACKBONE_OVERRIDE,
        ),
        (
            "google/gemma-4-26B-A4B",
            "\n\nWhat is the difference between a product and a service?\n\nWhat is the",
            GEMMA4_TEXT_BACKBONE_OVERRIDE,
        ),
    ],
    ids=["micro-g3.3", "gemma-3-1b-it", "gemma-4-31B-text", "gemma-4-26B-A4B-text"],
)
def test_basic_llm_inference(model_ref_output, monkeypatch: pytest.MonkeyPatch) -> None:
    """Construct `vllm.LLM(enforce_eager=False)` end-to-end.

    No compilation_config is passed: the platform defaults a non-eager run to
    STOCK_TORCH_COMPILE (one transformer block at a time + attention kernel).
    """
    model, ref_output, hf_overrides = model_ref_output
    _assert_compiled_output(model, ref_output, monkeypatch, hf_overrides=hf_overrides)


def test_whole_model_granularity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole-model graph still produces the same tokens."""
    monkeypatch.setenv("SPYRE_COMPILE_GRANULARITY", "model")
    _assert_compiled_output(
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        "\n\nIBMs main businesses are the companies that provide the services of the",
        monkeypatch,
    )


def test_compiled_pooling_encoder_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compiled pooling pads to ``(B, L)`` and matches cached HF refs.

    Two prompts at ``max_num_seqs=2`` / ``max_model_len=64`` warmup body ``T``
    and attention ``(1, 64)`` / ``(2, 64)``. Runtime 1D-pads the body; SDPA
    gathers onto ``(2, 64)``.
    """
    from vllm import LLM

    refs = json.loads(_POOLING_REFS.read_text())[_POOLING_MODEL]
    prompts = refs["prompts"]
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    engine = LLM(
        model=_POOLING_MODEL,
        runner="pooling",
        enforce_eager=False,
        max_model_len=64,
        max_num_seqs=2,
    )
    outputs = engine.embed(prompts)
    assert len(outputs) == len(prompts)
    for out, ref_emb in zip(outputs, refs["embeddings"]):
        emb = out.outputs.embedding
        assert len(emb) == len(ref_emb)
        assert all(math.isfinite(x) for x in emb)
        sim = F.cosine_similarity(
            torch.tensor(emb, dtype=torch.float32),
            torch.tensor(ref_emb, dtype=torch.float32),
            dim=0,
        ).item()
        assert sim >= _COSINE_MIN, f"cosine {sim:.4f} < {_COSINE_MIN}"


def test_transformers_backend_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compile the Transformers backend and check against a known reference.

    Guards the Spyre-safe attention forward (transpose + contiguous + reshape)
    against the silent data corruption that the unfused view chain caused.
    """
    _assert_compiled_output(
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        "\n\nIBMs main businesses are the companies that provide the services of the",
        monkeypatch,
        model_impl="transformers",
    )


def test_warmup_covers_every_compile_a_request_needs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve with ``SPYRE_COMPILE_GUARD=error``: a post-warmup compile kills the engine.

    The only test that exercises the guard's integration rather than its mechanics --
    the worker arming it after ``warming_up_model()``, and the ``watch`` registrations
    in the attention backends, the model runner, the MoE regions and the FP8 linear
    path. The unit tests hand-register their own functions on CPU, so a wrong callable
    registered there, or a kernel nobody registered at all, passes them.

    Uses the same model and shapes as ``test_basic_llm_inference``, so a failure here
    is a coverage gap rather than a new workload: if warmup really compiles everything
    a request needs, arming changes nothing.
    """
    monkeypatch.setenv("SPYRE_COMPILE_GUARD", "error")
    _assert_compiled_output(
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        "\n\nIBMs main businesses are the companies that provide the services of the",
        monkeypatch,
    )


def test_transformers_backend_encoder_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compiled ``model_impl="transformers"`` encoder embedding matches cached HF refs.

    ``SpyreTransformersEmbeddingModel`` is compiled per transformer block (the
    default ``SPYRE_COMPILE_GRANULARITY``). Verifies that the XLM-RoBERTa gather-free
    embedding patch and the single RoBERTa position offset survive compilation.
    """
    from vllm import LLM

    refs = json.loads(_POOLING_REFS.read_text())[_TRANSFORMERS_ENCODER_MODEL]
    prompts = refs["prompts"]
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    engine = LLM(
        model=_TRANSFORMERS_ENCODER_MODEL,
        runner="pooling",
        model_impl="transformers",
        enforce_eager=False,
        max_model_len=64,
        max_num_seqs=2,
    )
    outputs = engine.embed(prompts)
    assert len(outputs) == len(prompts)
    for out, ref_emb in zip(outputs, refs["embeddings"]):
        emb = out.outputs.embedding
        assert len(emb) == len(ref_emb)
        assert all(math.isfinite(x) for x in emb)
        sim = F.cosine_similarity(
            torch.tensor(emb, dtype=torch.float32),
            torch.tensor(ref_emb, dtype=torch.float32),
            dim=0,
        ).item()
        assert sim >= _TRANSFORMERS_COSINE_MIN, f"cosine {sim:.5f} < {_TRANSFORMERS_COSINE_MIN}"


def test_transformers_backend_token_classify_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compiled ``model_impl="transformers"`` BERT NER matches HF on every token's label.

    Covers ``SpyreTransformersForSequenceClassification`` (BERT, no position offset) on the
    ``token_classify`` task, where upstream's ``ClassifierWithReshape`` returns
    ``[tokens, 1, labels]``.
    """
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(_TRANSFORMERS_NER_MODEL)
    hf = AutoModelForTokenClassification.from_pretrained(
        _TRANSFORMERS_NER_MODEL, dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        refs = [
            hf(**tok(p, return_tensors="pt")).logits[0].softmax(-1)
            for p in _TRANSFORMERS_NER_PROMPTS
        ]

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    engine = LLM(
        model=_TRANSFORMERS_NER_MODEL,
        runner="pooling",
        model_impl="transformers",
        enforce_eager=False,
        max_model_len=128,
        max_num_seqs=1,
    )
    assert engine.llm_engine.model_config.using_transformers_backend()
    outputs = engine.encode(_TRANSFORMERS_NER_PROMPTS, pooling_task="token_classify")
    assert len(outputs) == len(_TRANSFORMERS_NER_PROMPTS)
    for prompt, out, ref in zip(_TRANSFORMERS_NER_PROMPTS, outputs, refs):
        got = torch.as_tensor(out.outputs.data).float().reshape(ref.shape)
        assert torch.isfinite(got).all()
        assert torch.equal(got.argmax(-1), ref.argmax(-1)), (
            f"{prompt!r}: labels {got.argmax(-1).tolist()} vs HF {ref.argmax(-1).tolist()}"
        )
        assert (got - ref).abs().max().item() < 1e-2, f"{prompt!r}: scores drifted from HF"


def _assert_compiled_output(
    model: str,
    ref_output: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    hf_overrides=None,
    model_impl: str | None = None,
) -> None:
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    prompt = "What are IBMs main businesses?"

    kwargs: dict = dict(
        model=model,
        enforce_eager=False,
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=8,
        compilation_config=CompilationConfig(compile_sizes=[1, 8]),
        **({"hf_overrides": hf_overrides} if hf_overrides is not None else {}),
    )
    if model_impl is not None:
        kwargs["model_impl"] = model_impl

    engine = LLM(**kwargs)

    output = engine.generate(
        prompt,
        SamplingParams(temperature=0.0, max_tokens=16),
        use_tqdm=False,
    )

    assert prompt == output[0].prompt, "Model output contained wrong prompt!"
    assert ref_output == output[0].outputs[0].text, "Model produced wrong output!"
