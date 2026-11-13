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

"""Card-free tests for packing a batched decode onto lanes."""

from collections import Counter

import pytest
import torch
from spyre_testing_plugin.attn_helpers import _build_metadata

from spyre_inference import envs
from spyre_inference.v1.attention.ops import batched_decode, tile_loop
from spyre_inference.v1.attention.spyre_attn_bucketer import BATCHED_DECODE_ENTRIES

BLOCK = 64
KV_HEADS, Q_PER_KV, HEAD = 2, 2, 64
MAX_NUM_SEQS = 8

RAGGED = [
    [30000, 900, 17000, 64, 5000],
    [100, 200],
    [64] * MAX_NUM_SEQS,
    [32768, 32768, 32768],
    [1, 70, 129],
]


@pytest.fixture()
def decode_config(default_vllm_config, monkeypatch):
    """The Python walk, so the kernel runs on CPU; its mask then has a broadcast KV axis."""
    from vllm.config import get_current_vllm_config

    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
    envs.clear_env_cache()
    monkeypatch.setattr(tile_loop, "USE_FOR_EACH_TILE", False)
    monkeypatch.setattr(batched_decode, "USE_FOR_EACH_TILE", False)
    torch.set_default_device("cpu")
    cfg = get_current_vllm_config()
    cfg.scheduler_config.max_num_seqs = MAX_NUM_SEQS
    cfg.model_config.max_model_len = 32768
    yield
    envs.clear_env_cache()


def _pages(kv_len: int) -> int:
    return -(-kv_len // BLOCK)


def _build(kv_lens: list[int]):
    """A decode batch whose sequences own disjoint, shuffled pages, starting at page 1."""
    # As wide as max_model_len: build() reads each row up to its padded block bucket.
    width = 32768 // BLOCK
    owned = 1 + torch.randperm(sum(_pages(k) for k in kv_lens), dtype=torch.int32)
    block_table = torch.zeros(len(kv_lens), width, dtype=torch.int32)
    start = 0
    for s, k in enumerate(kv_lens):
        block_table[s, : _pages(k)] = owned[start : start + _pages(k)]
        start += _pages(k)
    md = _build_metadata(
        num_query_heads=KV_HEADS * Q_PER_KV,
        num_kv_heads=KV_HEADS,
        head_size=HEAD,
        block_size=BLOCK,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32),
        query_start_loc=torch.arange(len(kv_lens) + 1, dtype=torch.int32),
        block_table=block_table,
        slot_mapping=torch.zeros(len(kv_lens), dtype=torch.int64),
    )
    return md, block_table


def test_a_decode_batch_packs_onto_one_trip_of_lanes(decode_config):
    md, _ = _build([30000, 900, 17000])
    assert md.packed_out_seqs == MAX_NUM_SEQS
    assert (md.padded_num_seqs, md.blocks_per_chunk) == (BATCHED_DECODE_ENTRIES, 1)
    assert md.decode_groups is None, "a packed batch needs no length groups"
    assert md.lane_onehot_cpu is not None
    assert md.lane_onehot_cpu.shape == (KV_HEADS, Q_PER_KV, MAX_NUM_SEQS, BATCHED_DECODE_ENTRIES)


@pytest.mark.parametrize("kv_lens", RAGGED)
def test_every_page_is_gathered_once_by_its_own_lanes(decode_config, kv_lens):
    md, block_table = _build(kv_lens)
    assert md.chunk_page_ids_cpu is not None and md.mask_by_chunk_cpu is not None
    assert md.lane_seq_ids_cpu is not None
    # A gathered page is one whose mask tile lets at least one token through.
    live = (md.mask_by_chunk_cpu[:, :, 0, 0, :] == 0).any(-1)
    for s, k in enumerate(kv_lens):
        lanes = md.lane_seq_ids_cpu == s
        got = Counter(md.chunk_page_ids_cpu[:, lanes][live[:, lanes]].tolist())
        assert got == Counter(block_table[s, : _pages(k)].tolist()), f"sequence {s}"


@pytest.mark.parametrize("kv_lens", RAGGED)
def test_kernel_matches_a_softmax_over_each_sequence(decode_config, kv_lens):
    md, block_table = _build(kv_lens)
    num_pages = 1 + sum(_pages(k) for k in kv_lens)
    k_pages = torch.randn(num_pages, KV_HEADS, BLOCK, HEAD, dtype=torch.float16)
    v_pages = torch.randn(num_pages, KV_HEADS, BLOCK, HEAD, dtype=torch.float16)
    query = torch.randn(len(kv_lens), KV_HEADS * Q_PER_KV, HEAD, dtype=torch.float16)
    scale = HEAD**-0.5

    out = batched_decode.batched_decode_kernel(
        query=query,
        rep_row_ids=md.rep_row_ids_cpu,
        k_pages=k_pages,
        v_pages=v_pages,
        chunk_page_ids=md.chunk_page_ids_cpu,
        mask_by_chunk=md.mask_by_chunk_cpu,
        scale=scale,
        num_seqs=md.padded_num_seqs,
        blocks_per_chunk=md.blocks_per_chunk,
        num_kv_heads=KV_HEADS,
        num_queries_per_kv=Q_PER_KV,
        block_size=BLOCK,
        head_size=HEAD,
        lane_seq_ids=md.lane_seq_ids_cpu,
        lane_onehot=md.lane_onehot_cpu,
    )

    assert out.shape == (MAX_NUM_SEQS, KV_HEADS * Q_PER_KV, HEAD)
    assert torch.isfinite(out).all(), "rows past the batch must stay finite"
    for s, k in enumerate(kv_lens):
        pages = block_table[s, : _pages(k)].long()
        keys = k_pages[pages].float().permute(1, 0, 2, 3).reshape(KV_HEADS, -1, HEAD)[:, :k]
        values = v_pages[pages].float().permute(1, 0, 2, 3).reshape(KV_HEADS, -1, HEAD)[:, :k]
        q = query[s].float().reshape(KV_HEADS, Q_PER_KV, HEAD)
        expected = ((q @ keys.transpose(-1, -2)) * scale).softmax(-1) @ values
        torch.testing.assert_close(out[s].float(), expected.reshape(-1, HEAD), atol=2e-2, rtol=2e-2)
