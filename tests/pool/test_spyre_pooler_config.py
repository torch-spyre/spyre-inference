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

"""Cheap unit tests for ``configure_pooling_for_spyre`` patching.

No Spyre hardware: builds minimal ``SequencePooler`` / ``DispatchPooler`` /
``TokenPooler`` graphs and checks CLS/LAST/MEAN/AllPool become Spyre forms.
Stick-aligned BERT/RoBERTa-scale FP32 heads use staggered-K upcast GEMM;
other FP32 linears still stay on CPU.

Host MEAN crop lives in ``tests/pool/test_spyre_mean_pool.py``. Destagger
of a device fp32 sum is ``test_spyre_fp32_reduce_d2h_with_destagger``
(xfail). FP32 heads are ``test_spyre_fp32_linear_for_pooling_heads``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.seqwise.heads import EmbeddingPoolerHead
from vllm.model_executor.layers.pooler.seqwise.methods import CLSPool, LastPool, MeanPool
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.model_executor.layers.pooler.special import DispatchPooler
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool, StepPool
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler

from spyre_inference.v1.pool.spyre_pooler import (
    SpyreAllPool,
    SpyreCLSPool,
    SpyreCpuClassifier,
    SpyreEmbeddingPoolerHead,
    SpyreLastPool,
    SpyreMeanPool,
    SpyreNormalize,
    configure_pooling_for_spyre,
    patch_pooler_for_spyre,
    run_pooling_tail_on_cpu,
)

_SPYRE = torch.device("cpu")  # configure only needs a device label for logging


def _embed_pooler(pooling) -> SequencePooler:
    return SequencePooler(
        pooling=pooling,
        head=EmbeddingPoolerHead(activation=PoolerNormalize()),
    )


def _model_with_pooler(pooler: nn.Module) -> nn.Module:
    model = nn.Module()
    model.pooler = pooler
    return model


def test_configure_pooling_patches_cls_to_spyre_cls_pool():
    model = _model_with_pooler(_embed_pooler(CLSPool()))
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    assert isinstance(model.pooler.pooling, SpyreCLSPool)
    assert isinstance(model.pooler.head, SpyreEmbeddingPoolerHead)
    assert isinstance(model.pooler.head.activation, SpyreNormalize)


def test_configure_pooling_patches_last_to_spyre_last_pool():
    model = _model_with_pooler(_embed_pooler(LastPool()))
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    assert isinstance(model.pooler.pooling, SpyreLastPool)
    assert isinstance(model.pooler.head, SpyreEmbeddingPoolerHead)


def test_configure_pooling_patches_mean_to_spyre_mean_pool():
    model = _model_with_pooler(_embed_pooler(MeanPool()))
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    assert isinstance(model.pooler.pooling, SpyreMeanPool)
    assert isinstance(model.pooler.head, SpyreEmbeddingPoolerHead)


def test_configure_pooling_dispatch_patches_embed_mean():
    pooler = DispatchPooler({"embed": _embed_pooler(MeanPool())})
    model = _model_with_pooler(pooler)
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    embed = model.pooler.poolers_by_task["embed"]
    assert isinstance(embed.pooling, SpyreMeanPool)


def test_configure_pooling_dispatch_patches_embed_cls():
    """DispatchPooler (real embed models) must still install SpyreCLSPool."""
    pooler = DispatchPooler({"embed": _embed_pooler(CLSPool())})
    model = _model_with_pooler(pooler)
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    embed = model.pooler.poolers_by_task["embed"]
    assert isinstance(embed.pooling, SpyreCLSPool)


def test_configure_pooling_dispatch_patches_embed_last():
    pooler = DispatchPooler({"embed": _embed_pooler(LastPool())})
    model = _model_with_pooler(pooler)
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    embed = model.pooler.poolers_by_task["embed"]
    assert isinstance(embed.pooling, SpyreLastPool)


def test_configure_pooling_fp32_classifier_falls_back_to_cpu():
    model = _model_with_pooler(_embed_pooler(CLSPool()))
    model.classifier = nn.Linear(8, 2)  # K not stick-aligned; upcast GEMM refuses it
    assert configure_pooling_for_spyre(model, _SPYRE) is False
    # CLS is swapped first; the FP32 linear still forces CPU.
    assert isinstance(model.pooler.pooling, SpyreCLSPool)


def test_configure_pooling_upcast_classifier_stays_on_device():
    """Stick-aligned BERT-scale classifier uses staggered-K GEMM (#868)."""
    from spyre_inference.v1.pool.spyre_upcast_linear import SpyreUpcastLinear

    model = _model_with_pooler(_embed_pooler(CLSPool()))
    model.classifier = nn.Linear(64, 3)
    model.head_dtype = torch.float32
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    assert isinstance(model.classifier, SpyreUpcastLinear)
    assert model.head_dtype == torch.float16
    assert model.classifier.weight.dtype == torch.float16


def test_configure_pooling_roberta_style_head_is_upcast():
    from spyre_inference.v1.pool.spyre_upcast_linear import SpyreUpcastLinear

    class RobertaStyleHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 2)

    model = _model_with_pooler(_embed_pooler(CLSPool()))
    model.classifier = RobertaStyleHead()
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    assert isinstance(model.classifier.dense, SpyreUpcastLinear)
    assert isinstance(model.classifier.out_proj, SpyreUpcastLinear)


def test_configure_pooling_bert_inner_pooler_is_upcast():
    from spyre_inference.v1.pool.spyre_upcast_linear import SpyreUpcastLinear

    class InnerPooler(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense = nn.Linear(64, 64)

    class Bert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pooler = InnerPooler()

    model = _model_with_pooler(_embed_pooler(CLSPool()))
    model.bert = Bert()
    model.classifier = nn.Linear(64, 2)
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    assert isinstance(model.classifier, SpyreUpcastLinear)
    assert isinstance(model.bert.pooler.dense, SpyreUpcastLinear)


def test_configure_pooling_wide_head_falls_back_to_cpu():
    """N too large and not square: do not materialize [M, N, K] (vocab heads)."""
    model = _model_with_pooler(_embed_pooler(CLSPool()))
    model.classifier = nn.Linear(64, 300)
    assert configure_pooling_for_spyre(model, _SPYRE) is False


def test_upcast_linear_matches_fp32_f_linear():
    from spyre_inference.v1.pool.spyre_upcast_linear import SpyreUpcastLinear

    torch.manual_seed(0)
    linear = nn.Linear(64, 3, dtype=torch.float32)
    wrapped = SpyreUpcastLinear.from_linear(linear, torch.device("cpu"))
    x = torch.randn(4, 64, dtype=torch.float16)
    got = wrapped(x).float()
    ref = torch.nn.functional.linear(x.float(), linear.weight.float(), linear.bias.float())
    torch.testing.assert_close(got, ref, atol=2e-3, rtol=2e-3)

    x3 = torch.randn(2, 5, 64, dtype=torch.float16)
    got3 = wrapped(x3).float()
    ref3 = torch.nn.functional.linear(x3.float(), linear.weight.float(), linear.bias.float())
    torch.testing.assert_close(got3, ref3, atol=2e-3, rtol=2e-3)


def test_upcast_linear_stacked_head_matches_cpu():
    """Roberta dense then out_proj: first layer D2Hs, second must follow."""
    from spyre_inference.v1.pool.spyre_upcast_linear import SpyreUpcastLinear

    torch.manual_seed(0)
    dense = nn.Linear(64, 64, dtype=torch.float32)
    out_proj = nn.Linear(64, 1, dtype=torch.float32)
    w_dense = SpyreUpcastLinear.from_linear(dense, torch.device("cpu"))
    w_out = SpyreUpcastLinear.from_linear(out_proj, torch.device("cpu"))
    x = torch.randn(4, 64, dtype=torch.float16)
    got = w_out(torch.tanh(w_dense(x))).float()
    ref = out_proj(torch.tanh(dense(x.float())))
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


def test_configure_pooling_no_pooler_returns_false():
    assert configure_pooling_for_spyre(nn.Module(), _SPYRE) is False


def test_model_applied_classifier_is_wrapped_for_cpu() -> None:
    """A classifier the pooler does not own is applied by the model: wrap it."""
    model = nn.Module()
    model.classifier = nn.Linear(4, 2, dtype=torch.float32)
    model.head_dtype = torch.float32
    pooler = SequencePooler(pooling=MeanPool(), head=None)

    run_pooling_tail_on_cpu(model, pooler)

    assert isinstance(model.classifier, SpyreCpuClassifier)
    assert model.head_dtype == torch.float16


def test_pooler_owned_classifier_is_not_wrapped() -> None:
    """A reranker head owns the classifier, so moving it to CPU is enough."""
    classifier = nn.Linear(4, 2, dtype=torch.float32)
    model = nn.Module()
    model.classifier = classifier
    pooler = SequencePooler(pooling=MeanPool(), head=None)
    pooler.head = nn.Module()
    pooler.head.classifier = classifier

    run_pooling_tail_on_cpu(model, pooler)

    assert model.classifier is classifier


def _token_pooler(cls) -> TokenPooler:
    """``AllPool.__init__`` reads the vLLM config; bypass it for a unit test."""
    pooling = cls.__new__(cls)
    nn.Module.__init__(pooling)
    pooling.enable_chunked_prefill = False
    return TokenPooler(pooling=pooling, head=None)


def test_token_pooler_all_pool_is_patched():
    pooler = _token_pooler(AllPool)
    num_patched, unsupported = patch_pooler_for_spyre(pooler)
    assert (num_patched, unsupported) == (1, [])
    assert isinstance(pooler.pooling, SpyreAllPool)


def test_token_pooler_step_pool_is_unsupported():
    """StepPool subclasses AllPool but indexes by step tag; keep it on CPU."""
    pooler = _token_pooler(StepPool)
    num_patched, unsupported = patch_pooler_for_spyre(pooler)
    assert (num_patched, unsupported) == (0, ["StepPool"])


def test_spyre_all_pool_matches_torch_split():
    counts = [3, 1, 4]
    hidden_states = torch.arange(sum(counts) * 9, dtype=torch.float16).reshape(-1, 9)

    class _Meta:
        def get_pooling_cursor(self):
            return type("C", (), {"num_scheduled_tokens_cpu": torch.tensor(counts)})()

    got = SpyreAllPool(enable_chunked_prefill=False)(hidden_states, _Meta())
    for chunk, expected in zip(got, torch.split(hidden_states, counts)):
        assert torch.equal(chunk, expected)
