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

"""Compiled adapter embeddings agree with HF and remain request-independent."""

import gc

import pytest
import torch
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available
from transformers import AutoModel, AutoTokenizer
from vllm import LLM
from vllm.config import PoolerConfig


@pytest.mark.model_quality
@pytest.mark.uses_subprocess
@pytest.mark.parametrize(
    "model,pooling,max_model_len",
    [
        ("Qwen/Qwen3-Embedding-0.6B", "LAST", 128),
        ("sentence-transformers/all-MiniLM-L6-v2", "MEAN", 128),
        ("ibm-granite/granite-embedding-30m-english", "CLS", 128),
        ("sentence-transformers/multi-qa-distilbert-cos-v1", "MEAN", 128),
        ("sentence-transformers/all-mpnet-base-v2", "MEAN", 256),
    ],
)
def test_hf_adapters_embeddings(model, pooling, max_model_len):
    pytest.importorskip("hf_adapters.execution")
    if not spyre_available():
        pytest.skip("Spyre hardware is required")
    prompts = [
        "Hi.",
        "The capital of France is Paris.",
        "Sentence embeddings support semantic search across documents. "
        * (18 if max_model_len == 256 else 9),
    ]
    tokenizer = AutoTokenizer.from_pretrained(model)
    reference = AutoModel.from_pretrained(model, dtype=torch.float32).eval()
    refs, tokens = [], []
    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            tokens.append(inputs["input_ids"][0].tolist())
            h = reference(**inputs).last_hidden_state[0]
            pooled = h[-1] if pooling == "LAST" else h[0] if pooling == "CLS" else h.mean(0)
            refs.append(F.normalize(pooled, dim=-1))
    del reference
    gc.collect()
    llm = LLM(
        model=model,
        model_impl="transformers",
        runner="pooling",
        max_model_len=max_model_len,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        pooler_config=PoolerConfig(seq_pooling_type=pooling, use_activation=True),
        additional_config={"spyre_model_preparation": "hf_adapters"},
    )
    singles = [torch.tensor(llm.embed([prompt])[0].outputs.embedding) for prompt in prompts]
    for order in ([0, 1, 2], [0, 1, 2], [2, 0, 1], [0, 1, 2]):
        outputs = llm.embed([prompts[i] for i in order])
        for i, out in zip(order, outputs):
            actual = torch.tensor(out.outputs.embedding)
            assert out.prompt_token_ids == tokens[i]
            assert torch.isfinite(actual).all()
            assert F.cosine_similarity(actual, refs[i], dim=0) >= 0.99
            assert F.cosine_similarity(actual, singles[i], dim=0) >= 0.9999
            torch.testing.assert_close(actual.norm(), torch.tensor(1.0), atol=0.01, rtol=0)
