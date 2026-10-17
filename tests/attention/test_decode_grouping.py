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

"""Card-free tests for splitting a batched decode into length groups.

Not named "batched_decode": tests/conftest.py skips those under the tiled walk.
"""

import pytest
import torch
from spyre_testing_plugin.attn_helpers import _build_metadata

BLOCK = 128
KV_HEADS, HEAD = 2, 64


@pytest.fixture()
def decode_config(default_vllm_config, monkeypatch):
    from vllm.config import get_current_vllm_config

    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
    cfg = get_current_vllm_config()
    cfg.scheduler_config.max_num_seqs = 8
    cfg.model_config.max_model_len = 32768
    return cfg


def _build(kv_lens: list[int], query_lens: list[int] | None = None, block_table=None):
    query_lens = query_lens or [1] * len(kv_lens)
    width = 256
    if block_table is None:
        block_table = torch.arange(len(kv_lens) * width, dtype=torch.int32).reshape(-1, width)
    query_start_loc = torch.tensor([0, *query_lens], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return _build_metadata(
        num_query_heads=KV_HEADS,
        num_kv_heads=KV_HEADS,
        head_size=HEAD,
        block_size=BLOCK,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32),
        query_start_loc=query_start_loc,
        block_table=block_table,
        slot_mapping=torch.arange(int(query_start_loc[-1]), dtype=torch.int64),
    )


def _members(md) -> list[list[int]]:
    return [g.rows_cpu.tolist() for g in md.decode_groups]


def test_grouping_is_off_by_default(decode_config):
    md = _build([32000, 2000, 2000, 2000])
    assert md.padded_num_seqs is not None
    assert md.decode_groups is None


def test_same_bucket_batch_stays_one_batch(decode_config, monkeypatch):
    monkeypatch.setenv("SPYRE_BATCHED_DECODE_GROUPING", "1")
    assert _build([30000, 30500, 31000, 30000]).decode_groups is None


def test_uniform_batch_skips_grouping(decode_config, monkeypatch):
    from spyre_inference.v1.attention.backends import spyre_attn

    monkeypatch.setenv("SPYRE_BATCHED_DECODE_GROUPING", "1")

    def fail(*args, **kwargs):
        raise AssertionError("grouping evaluated for a uniform batch")

    monkeypatch.setattr(spyre_attn.SpyreAttentionMetadataBuilder, "_batched_decode_groups", fail)
    md = _build([30000, 29990, 30001, 30000])
    assert md.decode_uniformity == 1.0
    assert md.decode_groups is None


def test_multi_bucket_batch_splits_by_block_bucket(decode_config, monkeypatch):
    monkeypatch.setenv("SPYRE_BATCHED_DECODE_GROUPING", "1")
    md = _build([2000, 32000, 2000, 2000])
    assert _members(md) == [[1], [0, 2, 3]]
    # The single batch is still built, as the fallback.
    assert md.padded_num_seqs == 4


@pytest.mark.parametrize(
    "kv_lens", [[2000, 32000, 2000, 2000], [32000, 16000, 8000, 4000], [9000, 300, 9000, 300, 60]]
)
def test_each_group_matches_its_own_batch(decode_config, monkeypatch, kv_lens):
    """A group's precomputes are those of a batch holding only its members, with its
    query rows mapped back to their place in the full batch."""
    monkeypatch.setenv("SPYRE_BATCHED_DECODE_GROUPING", "1")
    width = 256
    block_table = torch.randperm(len(kv_lens) * width, dtype=torch.int32).reshape(-1, width)
    md = _build(kv_lens, block_table=block_table)
    assert md.decode_groups is not None
    seen = []
    for g in md.decode_groups:
        rows = g.rows_cpu.tolist()
        seen += rows
        alone = _build([kv_lens[r] for r in rows], block_table=block_table[rows])
        assert g.padded_num_seqs == alone.padded_num_seqs
        assert g.blocks_per_chunk == alone.blocks_per_chunk
        assert g.padded_batch_blocks == alone.padded_batch_blocks
        assert torch.equal(g.chunk_page_ids_cpu, alone.chunk_page_ids_cpu)
        assert torch.equal(g.mask_by_chunk_cpu, alone.mask_by_chunk_cpu)
        # Entry j*B+s holds slot s: real slots carry the member's batch row, padding 0.
        slot = torch.arange(g.rep_row_ids_cpu.numel()) % g.padded_num_seqs
        member_rows = torch.tensor(rows + [0] * (g.padded_num_seqs - len(rows)))
        expected = torch.where(slot < len(rows), member_rows[alone.rep_row_ids_cpu.long()], 0)
        assert torch.equal(g.rep_row_ids_cpu, expected.to(torch.int32))
    assert sorted(seen) == list(range(len(kv_lens)))


@pytest.mark.parametrize(("group_cost", "split"), [(2, True), (8, False)])
def test_groups_merge_within_the_dispatch_cost(decode_config, monkeypatch, group_cost, split):
    """256 + 2 blocks: 8 + 1 chunks apart, 16 as one batch, so a dispatch worth 8
    chunks merges them and one worth 2 does not."""
    monkeypatch.setenv("SPYRE_BATCHED_DECODE_GROUPING", "1")
    monkeypatch.setenv("SPYRE_BATCHED_DECODE_GROUP_COST", str(group_cost))
    md = _build([32000, 200])
    assert (md.decode_groups is not None) == split


def test_neighbouring_buckets_merge(decode_config, monkeypatch):
    """3 and 5 blocks cost one chunk each, apart or together."""
    monkeypatch.setenv("SPYRE_BATCHED_DECODE_GROUPING", "1")
    assert _build([300, 600]).decode_groups is None


def test_groups_cover_only_the_decode_prefix(decode_config, monkeypatch):
    monkeypatch.setenv("SPYRE_BATCHED_DECODE_GROUPING", "1")
    md = _build([32000, 2000, 300], query_lens=[1, 1, 100])
    assert md.num_decode_seqs == 2
    assert _members(md) == [[0], [1]]
