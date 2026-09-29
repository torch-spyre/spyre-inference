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

"""The rectangular path's grid layout is a contract between two places in the runner:
``_preprocess`` scatters the packed tokens into ``[B*L]``, and ``_unpad_encoder_hidden``
gathers them back for the pooler. They share one row-index table, so a round trip is
the thing worth testing.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

from spyre_inference.models.bert import SpyreBertEmbedding
from spyre_inference.models.roberta import SpyreRobertaEmbedding
from spyre_inference.v1.pool.spyre_pooler import (
    SpyreCLSPool,
    SpyreLastPool,
    set_cls_grid_rows,
)
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    encoder_cls_rows,
    encoder_dense_row_indices,
    expand_packed_to_encoder_grid,
    expand_packed_token_types,
)


@pytest.mark.parametrize(
    "query_lens",
    [
        [64, 64],
        [1, 64, 32],
        [10],
        [64] * 4,
        [7, 1, 63, 64],
    ],
)
def test_expand_then_gather_round_trips(query_lens):
    extent, width = 64, len(query_lens) + 2  # spare batch-pad lanes
    total = sum(query_lens)
    ids = torch.arange(1, total + 1, dtype=torch.int64)
    positions = torch.cat([torch.arange(n, dtype=torch.int64) for n in query_lens])

    grid_ids, grid_pos = expand_packed_to_encoder_grid(
        ids, positions, query_lens, width, extent, pad_token_id=0
    )
    assert grid_ids.shape[0] == width * extent

    rows = encoder_dense_row_indices(query_lens, extent)
    assert torch.equal(grid_ids[rows], ids)
    assert torch.equal(grid_pos[rows], positions)


def test_real_pad_continues_positions_and_batch_pad_restarts():
    """Real pad rows continue the sequence's own positions so a position embedding
    stays in range; batch-pad lanes are filler whose output is never read."""
    extent, width = 64, 3
    query_lens = [4, 2]
    ids = torch.arange(1, 7, dtype=torch.int64)
    positions = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.int64)

    grid_ids, grid_pos = expand_packed_to_encoder_grid(
        ids, positions, query_lens, width, extent, pad_token_id=9
    )

    # Sequence 0: 4 real tokens, then its pad rows carry positions 4..63.
    assert grid_ids[:4].tolist() == [1, 2, 3, 4]
    assert grid_pos[:8].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert grid_ids[4:extent].unique().tolist() == [9]
    # The third lane is pure batch pad: filler ids, positions 0..L-1.
    base = 2 * extent
    assert grid_ids[base : base + extent].unique().tolist() == [9]
    assert grid_pos[base : base + 5].tolist() == [0, 1, 2, 3, 4]
    # Every position stays inside the declared length, which is what keeps an
    # offset position table in bounds.
    assert int(grid_pos.max()) == extent - 1


def test_roberta_offset_is_added_on_the_host_grid():
    """``padding_idx + 1`` lands on real rows, real pad, and batch pad together."""
    extent, width = 64, 2
    query_lens = [3]
    ids = torch.tensor([10, 11, 12], dtype=torch.int64)
    positions = torch.tensor([0, 1, 2], dtype=torch.int64)
    _, grid_pos = expand_packed_to_encoder_grid(
        ids, positions, query_lens, width, extent, position_offset=2
    )
    assert grid_pos[:5].tolist() == [2, 3, 4, 5, 6]
    assert grid_pos[extent : extent + 3].tolist() == [2, 3, 4]
    assert int(grid_pos.max()) == extent - 1 + 2


def test_cls_row_is_the_start_of_each_rectangle_lane():
    assert encoder_cls_rows(3, 64) == [0, 64, 128]
    assert encoder_cls_rows(0, 64) == []


def test_positions_never_exceed_the_declared_length():
    extent, width = 128, 4
    query_lens = [100, 5, 128]
    ids = torch.ones(sum(query_lens), dtype=torch.int64)
    positions = torch.cat([torch.arange(n, dtype=torch.int64) for n in query_lens])
    _, grid_pos = expand_packed_to_encoder_grid(ids, positions, query_lens, width, extent)
    assert int(grid_pos.max()) == extent - 1


def test_more_sequences_than_the_rectangle_is_rejected():
    with pytest.raises(ValueError, match="exceeds batch_bucket"):
        expand_packed_to_encoder_grid(
            torch.zeros(3, dtype=torch.int64),
            torch.zeros(3, dtype=torch.int64),
            [1, 1, 1],
            2,
            64,
        )


def test_a_length_past_the_extent_is_rejected():
    with pytest.raises(ValueError, match="exceeds len_bucket"):
        encoder_dense_row_indices([65], 64)


def test_empty_batch_yields_no_rows():
    assert encoder_dense_row_indices([], 64).numel() == 0


@pytest.mark.parametrize("query_lens", [[4, 2], [64, 64], [1, 64, 32], [7, 1, 63, 64]])
def test_token_types_follow_the_same_rows_as_the_ids(query_lens):
    """Segment ids must land on the tokens they describe. Left packed, every sequence
    past the first pairs with another's tokens -- and silently, since the buffer still
    matches ``input_ids`` in shape."""
    extent, width = 64, len(query_lens) + 2
    total = sum(query_lens)
    ids = torch.arange(1, total + 1, dtype=torch.int64)
    positions = torch.cat([torch.arange(n, dtype=torch.int64) for n in query_lens])
    # Distinct per sequence, so a misplaced run is visible rather than coincidentally equal.
    token_types = torch.cat(
        [torch.full((n,), seq_idx % 2, dtype=torch.int32) for seq_idx, n in enumerate(query_lens)]
    )

    grid_ids, _ = expand_packed_to_encoder_grid(ids, positions, query_lens, width, extent)
    grid_types = expand_packed_token_types(token_types, query_lens, width, extent)

    assert grid_types.shape == grid_ids.shape
    rows = encoder_dense_row_indices(query_lens, extent)
    assert torch.equal(grid_types[rows], token_types)
    # Every pad slot -- interior and batch-pad lane -- is segment 0.
    pad = torch.ones(width * extent, dtype=torch.bool)
    pad[rows] = False
    assert not grid_types[pad].any()


def test_token_types_are_not_left_packed():
    """The bug this guards: a contiguous copy pairs sequence 1's segment ids with
    sequence 0's pad rows."""
    extent, width = 64, 2
    query_lens = [4, 3]
    token_types = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.int32)

    grid = expand_packed_token_types(token_types, query_lens, width, extent)

    assert grid[:4].tolist() == [0, 0, 0, 0]
    assert grid[4:extent].sum() == 0, "sequence 0's pad rows must not carry segment 1"
    assert grid[extent : extent + 3].tolist() == [1, 1, 1]


def _bare_embedding(cls, *, vocab: int, positions: int, types: int, hidden: int):
    """The embedding arithmetic, without VocabParallelEmbedding or a compile context.

    ``_compiled_forward`` only needs the three tables, the layer norm, and the
    compile flag. Leaving compile off runs that body eagerly on CPU.
    """
    emb = cls.__new__(cls)
    nn.Module.__init__(emb)
    emb.word_embeddings = nn.Embedding(vocab, hidden)
    emb.position_embeddings = nn.Embedding(positions, hidden)
    emb.token_type_embeddings = nn.Embedding(types, hidden)
    emb.LayerNorm = nn.LayerNorm(hidden, eps=1e-5)
    emb.spyre_token_type_ids = None
    emb.spyre_compile_enabled = False
    emb.spyre_compiled_kernel = None
    return emb


def _unfused_embedding(emb, input_ids, position_ids, token_type_ids, inputs_embeds=None):
    """The embedding body from before the single compiled forward."""
    if inputs_embeds is None:
        inputs_embeds = emb.word_embeddings(input_ids)
    return emb.LayerNorm(
        inputs_embeds
        + emb.token_type_embeddings(token_type_ids)
        + emb.position_embeddings(position_ids)
    )


def test_fused_bert_embedding_matches_the_unfused_body():
    torch.manual_seed(0)
    emb = _bare_embedding(SpyreBertEmbedding, vocab=8, positions=6, types=2, hidden=4)
    input_ids = torch.tensor([1, 3, 0, 2])
    position_ids = torch.arange(4)
    token_types = torch.tensor([0, 1, 0, 1])
    emb.spyre_token_type_ids = token_types
    embeds = torch.randn(4, 4)

    fused = emb.forward(input_ids, position_ids)
    unfused = _unfused_embedding(emb, input_ids, position_ids, token_types)
    torch.testing.assert_close(fused, unfused)

    fused_embeds = emb.forward(input_ids, position_ids, embeds)
    unfused_embeds = _unfused_embedding(
        emb, input_ids, position_ids, token_types, inputs_embeds=embeds
    )
    torch.testing.assert_close(fused_embeds, unfused_embeds)


def test_fused_roberta_embedding_matches_the_offset_then_gather():
    """The offset stays outside the compiled body. Passing already-offset ids
    matches the old forward, which added ``padding_idx + 1`` itself."""
    torch.manual_seed(1)
    padding_idx = 1
    delta = padding_idx + 1
    emb = _bare_embedding(SpyreRobertaEmbedding, vocab=8, positions=8, types=2, hidden=4)
    emb.padding_idx = padding_idx
    input_ids = torch.tensor([2, 4, 1, 0])
    raw_positions = torch.arange(4)
    token_types = torch.zeros(4, dtype=torch.int64)
    emb.spyre_token_type_ids = token_types

    fused = emb.forward(input_ids, raw_positions + delta)
    unfused = _unfused_embedding(emb, input_ids, raw_positions + delta, token_types)
    torch.testing.assert_close(fused, unfused)
    # The compiled body does not add the offset a second time.
    doubled = _unfused_embedding(emb, input_ids, raw_positions + 2 * delta, token_types)
    assert not torch.equal(fused, doubled)


def _pooling_metadata(lengths: list[int]) -> PoolingMetadata:
    prompt = torch.tensor(lengths, dtype=torch.int64)
    metadata = PoolingMetadata(
        prompt_lens=prompt,
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[PoolingParams(task="embed") for _ in lengths],
        pooling_states=[PoolingStates() for _ in lengths],
    )
    metadata.build_pooling_cursor(
        np.array(lengths, dtype=np.int32),
        seq_lens_cpu=prompt.clone(),
        device=torch.device("cpu"),
    )
    return metadata


def _runner(pooler, grid):
    runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
    runner._encoder_grid = grid
    runner.model = type("_Model", (), {})()
    runner.model.pooler = pooler
    return runner


def test_rectangular_cls_matches_unpad_then_gather():
    """Grid CLS rows name the same vectors the packed cursor names after unpad."""
    extent, width = 8, 3
    query_lens = [3, 1, 2]
    hidden = (
        torch.arange(width * extent, dtype=torch.float32).unsqueeze(1).expand(-1, 4).contiguous()
    )
    metadata = _pooling_metadata(query_lens)
    pooler = SequencePooler(pooling=SpyreCLSPool(), head=nn.Identity())
    runner = _runner(pooler, (extent, width, query_lens))

    assert runner._rectangular_cls(metadata)
    assert not _runner(
        SequencePooler(pooling=SpyreLastPool(), head=nn.Identity()),
        (extent, width, query_lens),
    )._rectangular_cls(metadata)

    packed = runner._unpad_encoder_hidden(hidden, sum(query_lens))
    cls = SpyreCLSPool()
    try:
        set_cls_grid_rows(None)
        from_unpad = cls(packed, metadata)
        set_cls_grid_rows(
            torch.tensor(encoder_cls_rows(len(query_lens), extent), dtype=torch.int64)
        )
        from_grid = cls(hidden, metadata)
    finally:
        set_cls_grid_rows(None)

    torch.testing.assert_close(from_grid, from_unpad)
    # Sequence starts: rows 0, 8 and 16 of the grid.
    assert from_grid[:, 0].tolist() == [0, 8, 16]
