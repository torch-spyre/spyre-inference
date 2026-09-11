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

"""Batched multi-sequence decode, behind ``SPYRE_BATCHED_DECODE``.

Reads the slot-major page shape only; ``SPYRE_ATTN_HEAD_MAJOR_KV`` disables this path.
"""

import torch


def batched_decode_kernel(
    query,
    query_row_ids,
    k_pages,
    v_pages,
    block_ids,
    mask_by_block,
    scale,
    num_seqs,
    num_blocks,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
    logits_soft_cap=0.0,
    out=None,
):
    """Batched decode kernel; gathers K/V and the query in-graph.

    Gathers one block at a time; block_ids rows must stay stick-aligned, since a
    flat per-block slice does not compile.

    k/v_pages: [num_pages_total, block_size, KV, D] (the raw page cache).
    block_ids: [num_blocks, stick-padded num_seqs] int32, row i holding the
    i-th block's page index per sequence. mask_by_block:
    [num_blocks, num_seqs * KV, 1, block_size], pre-broadcast across KV heads
    by the builder. ``query_row_ids`` None takes the query buffer's first
    num_seqs rows instead of gathering; ``out`` None returns the result instead
    of storing it.
    """
    num_heads = num_kv_heads * num_queries_per_kv
    # Q=1 puts the sequences in rows 0..num_seqs-1; lanes past the batch are
    # -inf-masked and dropped by the caller, so any b_seqs-row prefix serves.
    q_rows = query[:num_seqs] if query_row_ids is None else query.index_select(0, query_row_ids)
    # lower_bmm's 4-D form takes two batch axes, so num_seqs and KV stay separate.
    q = q_rows.reshape(num_seqs, num_kv_heads, num_queries_per_kv, head_size)

    tile_max = None
    tile_sum = None
    tile_output = None

    for i in range(num_blocks):
        # index_select, not `k_pages[page_idx]`: subscripting lowers to
        # aten.index, which upcasts the int32 index to int64 and fails eager.
        page_idx = block_ids[i, 0:num_seqs]
        # Token-major cache page to head-major; a view, so do not add
        # .contiguous() -- merging these axes is what materializes the page.
        k_page = k_pages.index_select(0, page_idx).permute(0, 2, 1, 3)
        v_page = v_pages.index_select(0, page_idx).permute(0, 2, 1, 3)
        # Builder already broadcast across KV heads; split them back out.
        mask_tile = mask_by_block[i].reshape(num_seqs, num_kv_heads, 1, block_size)

        scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Before the mask add: tanh(-inf/cap)*cap is -cap, not -inf, so
            # capping after it would un-mask the padded lanes.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = torch.matmul(tile_probs, v_page)
            tile_sum = tile_probs.sum(dim=-1, keepdim=True)
        else:
            assert tile_max is not None
            assert tile_sum is not None
            assert tile_output is not None
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_output = tile_output * rescale
            tile_sum = tile_sum * rescale
            tile_probs = torch.exp(scores - new_max)
            tile_output += torch.matmul(tile_probs, v_page)
            tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
    attn = (tile_output / tile_sum).reshape(num_seqs, num_heads, head_size)
    if out is not None:
        # The destination prefix starts at offset 0, so torch-spyre#3770 does not
        # apply; rows past the batch are don't-care and kept finite by the builder.
        out[:num_seqs].copy_(attn)
        return out
    return attn
