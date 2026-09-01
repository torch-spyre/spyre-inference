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

No Spyre hardware: builds minimal ``SequencePooler`` / ``DispatchPooler`` graphs
and checks CLS/LAST/MEAN become Spyre forms. FP32 linear heads stay on CPU.

Numeric MEAN tests here are host arithmetic (fp32 correctness, fp16
accumulator, empty-batch dtype). Packed Spyre ``convert`` lives in
``tests/pool/test_spyre_mean_pool.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.seqwise.heads import (
    ClassifierPoolerHead,
    EmbeddingPoolerHead,
)
from vllm.model_executor.layers.pooler.seqwise.methods import CLSPool, LastPool, MeanPool
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.model_executor.layers.pooler.special import DispatchPooler

from spyre_inference.v1.pool.spyre_pooler import (
    SpyreCLSPool,
    SpyreEmbeddingPoolerHead,
    SpyreLastPool,
    SpyreMeanPool,
    SpyreNormalize,
    configure_pooling_for_spyre,
    mean_pooler_owns_packed_hidden_states,
)

_SPYRE = torch.device("cpu")  # configure only needs a device label for logging


def _embed_pooler(pooling) -> SequencePooler:
    return SequencePooler(
        pooling=pooling,
        head=EmbeddingPoolerHead(activation=PoolerNormalize()),
    )


def _classify_pooler(pooling) -> SequencePooler:
    return SequencePooler(pooling=pooling, head=ClassifierPoolerHead())


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
    model.classifier = nn.Linear(8, 2)  # float32 linear still not on Spyre
    assert configure_pooling_for_spyre(model, _SPYRE) is False
    # CLS is swapped first; the FP32 linear still forces CPU.
    assert isinstance(model.pooler.pooling, SpyreCLSPool)


def test_configure_pooling_no_pooler_returns_false():
    assert configure_pooling_for_spyre(nn.Module(), _SPYRE) is False


def test_mean_pooler_owns_packed_hidden_states():
    """Runner crop is MEAN's job; CLS/LAST and mixed dispatch keep it."""
    assert mean_pooler_owns_packed_hidden_states(None) is False
    assert mean_pooler_owns_packed_hidden_states(_embed_pooler(MeanPool())) is True
    assert mean_pooler_owns_packed_hidden_states(_embed_pooler(SpyreMeanPool())) is True
    assert mean_pooler_owns_packed_hidden_states(_embed_pooler(CLSPool())) is False
    assert mean_pooler_owns_packed_hidden_states(_embed_pooler(LastPool())) is False
    assert (
        mean_pooler_owns_packed_hidden_states(DispatchPooler({"embed": _embed_pooler(MeanPool())}))
        is True
    )
    assert (
        mean_pooler_owns_packed_hidden_states(
            DispatchPooler(
                {
                    "embed": _embed_pooler(MeanPool()),
                    # EmbeddingPoolerHead only supports embed; classify+CLS
                    # is the valid mixed dispatch DispatchPooler will accept.
                    "classify": _classify_pooler(CLSPool()),
                }
            )
        )
        is False
    )


def test_spyre_mean_pool_matches_per_seq_mean():
    """Host arithmetic: varlen [3, 2] tokens vs a CPU mean per sequence.

    Does not exercise Spyre ``convert``. See
    ``tests/pool/test_spyre_mean_pool.py``.
    """
    from spyre_inference.v1.pool.spyre_pooler import SpyreMeanPool

    hidden = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        dtype=torch.float32,
    )
    lens = torch.tensor([3, 2], dtype=torch.int64)

    class _Cursor:
        prompt_lens_cpu = lens

        def is_partial_prefill(self) -> bool:
            return False

    class _Meta:
        def get_pooling_cursor(self):
            return _Cursor()

    out = SpyreMeanPool().forward(hidden, _Meta())
    expected = torch.stack(
        [
            hidden[:3].mean(0),
            hidden[3:].mean(0),
        ]
    )
    torch.testing.assert_close(out, expected)


def test_spyre_mean_pool_empty_batch_is_float32():
    """Upstream MeanPool returns float32 for an empty batch, not activation dtype."""
    from spyre_inference.v1.pool.spyre_pooler import SpyreMeanPool

    hidden = torch.empty((0, 4), dtype=torch.float16)

    class _Cursor:
        prompt_lens_cpu = torch.tensor([], dtype=torch.int64)

        def is_partial_prefill(self) -> bool:
            return False

    class _Meta:
        def get_pooling_cursor(self):
            return _Cursor()

    out = SpyreMeanPool().forward(hidden, _Meta())
    assert out.dtype == torch.float32
    assert out.shape == (0, 4)


def test_spyre_mean_pool_accumulates_in_float32():
    """fp16 hidden states must not accumulate in fp16.

    ``2048 + 1`` is a tie in fp16 and rounds back to ``2048``, so an fp16 sum
    of one large row plus many small ones drops the small ones entirely.
    """
    from spyre_inference.v1.pool.spyre_pooler import SpyreMeanPool

    num_small = 512
    hidden = torch.cat(
        [
            torch.full((1, 2), 2048.0, dtype=torch.float16),
            torch.ones((num_small, 2), dtype=torch.float16),
        ]
    )
    lens = torch.tensor([num_small + 1], dtype=torch.int64)

    class _Cursor:
        prompt_lens_cpu = lens

        def is_partial_prefill(self) -> bool:
            return False

    class _Meta:
        def get_pooling_cursor(self):
            return _Cursor()

    out = SpyreMeanPool().forward(hidden, _Meta())

    assert out.dtype == torch.float32
    expected = hidden.to(torch.float32).mean(0, keepdim=True)
    torch.testing.assert_close(out, expected)
    # An fp16 accumulator would land near 2048/513 instead of 2560/513.
    assert out[0, 0].item() > 4.9


def test_spyre_mean_pool_skewed_lengths_do_not_pad_to_max():
    """A long seq plus short ones must not allocate ``num_seqs × max_len``."""
    from spyre_inference.v1.pool.spyre_pooler import SpyreMeanPool

    hidden = torch.arange(10, dtype=torch.float32).view(10, 1).expand(10, 2).contiguous()
    lens = torch.tensor([8, 1, 1], dtype=torch.int64)

    class _Cursor:
        prompt_lens_cpu = lens

        def is_partial_prefill(self) -> bool:
            return False

    class _Meta:
        def get_pooling_cursor(self):
            return _Cursor()

    out = SpyreMeanPool().forward(hidden, _Meta())
    expected = torch.stack(
        [
            hidden[:8].mean(0),
            hidden[8:9].mean(0),
            hidden[9:].mean(0),
        ]
    )
    torch.testing.assert_close(out, expected)


def test_spyre_mean_pool_ignores_trailing_pad_rows():
    """Encoder pad past ``sum(lens)`` must not enter the mean."""
    from spyre_inference.v1.pool.spyre_pooler import SpyreMeanPool

    hidden = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [99.0, 99.0],
            [99.0, 99.0],
        ],
        dtype=torch.float32,
    )
    lens = torch.tensor([2], dtype=torch.int64)

    class _Cursor:
        prompt_lens_cpu = lens

        def is_partial_prefill(self) -> bool:
            return False

    class _Meta:
        def get_pooling_cursor(self):
            return _Cursor()

    out = SpyreMeanPool().forward(hidden, _Meta())
    torch.testing.assert_close(out, hidden[:2].mean(0, keepdim=True))
