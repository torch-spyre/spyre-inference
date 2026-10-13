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

"""SpyreSequencePooler runs the head on the padded row count and trims after it."""

import pytest
import torch
from vllm.model_executor.layers.pooler.seqwise.heads import (
    ClassifierPoolerHead,
    EmbeddingPoolerHead,
)
from vllm.model_executor.layers.pooler.seqwise.methods import CLSPool, LastPool
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingCursor, PoolingMetadata, PoolingStates

from spyre_inference.v1.pool.spyre_pooler import SpyreSequencePooler, patch_pooler_for_spyre

HIDDEN = 8


def _metadata(lens: list[int], task: str) -> PoolingMetadata:
    lens_t = torch.tensor(lens, dtype=torch.int64)
    ends = torch.cumsum(lens_t, dim=0)
    cursor = PoolingCursor(
        first_token_indices_gpu=ends - lens_t,
        last_token_indices_gpu=ends - 1,
        prompt_lens_cpu=lens_t,
        seq_lens_cpu=lens_t,
        num_scheduled_tokens_cpu=lens_t,
    )
    return PoolingMetadata(
        prompt_lens=lens_t,
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[PoolingParams(task=task) for _ in lens],
        pooling_states=[PoolingStates() for _ in lens],
        pooling_cursor=cursor,
    )


@pytest.mark.parametrize("num_reqs", [1, 3, 5, 7])
@pytest.mark.parametrize(
    ("pooling", "head", "task", "first"),
    [
        pytest.param(CLSPool(), EmbeddingPoolerHead(), "embed", True, id="cls-embed"),
        pytest.param(LastPool(), EmbeddingPoolerHead(), "embed", False, id="last-embed"),
        pytest.param(
            CLSPool(),
            ClassifierPoolerHead(classifier=torch.nn.Identity()),
            "classify",
            True,
            id="cls-classify",
        ),
    ],
)
def test_head_sees_padded_rows_and_output_is_trimmed(num_reqs, pooling, head, task, first):
    """A request count that is not a power of two pads the rows the head sees.

    Upstream heads reject a row count that differs from the pooling-param count, so
    a 3- or 5-request batch must still pool -- one output row per real request.
    """
    pooler = SequencePooler(pooling=pooling, head=head)
    patch_pooler_for_spyre(pooler)
    assert type(pooler) is SpyreSequencePooler

    lens = [2 + i for i in range(num_reqs)]
    hidden = torch.randn(sum(lens), HIDDEN)
    out = pooler(hidden, _metadata(lens, task))

    ends = torch.cumsum(torch.tensor(lens), dim=0)
    rows = (ends - torch.tensor(lens)) if first else (ends - 1)
    assert len(out) == num_reqs
    assert torch.equal(torch.stack(list(out)), hidden[rows])
