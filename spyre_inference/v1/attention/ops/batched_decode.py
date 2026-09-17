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

"""Batched multi-sequence decode, behind ``SPYRE_BATCHED_DECODE``."""

import torch


def batched_decode_kernel(
    query,
    rep_row_ids,
    k_pages,
    v_pages,
    chunk_page_ids,
    mask_by_chunk,
    scale,
    num_seqs,
    blocks_per_chunk,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
    logits_soft_cap=0.0,
    out=None,
):
    """Batched decode kernel; gathers K/V and the query in-graph.

    Gathers blocks_per_chunk blocks per sequence per step, so the gather's entry
    axis is entries = num_seqs * blocks_per_chunk. A gather is core-split only on
    that axis, and behind a 1-D index it is counted in whole 32-entry sticks, so
    a narrow 1-D gather has no splittable unit and runs on one core.

    k/v_pages: [num_pages_total, block_size, KV, D] (the raw page cache).
    chunk_page_ids: one [entries, 1] int32 tensor per chunk, entry (s, j) holding
    sequence s's (c * blocks_per_chunk + j)-th page. mask_by_chunk:
    [num_chunks, entries * KV, 1, block_size], pre-broadcast across KV heads by
    the builder. rep_row_ids: [entries] int32, each query row repeated
    blocks_per_chunk times. ``out`` None returns the result instead of storing it.
    """
    num_heads = num_kv_heads * num_queries_per_kv
    entries = num_seqs * blocks_per_chunk
    q = query.index_select(0, rep_row_ids).reshape(
        entries, num_kv_heads, num_queries_per_kv, head_size
    )

    tile_max = None
    tile_sum = None
    tile_output = None

    for c, page_idx in enumerate(chunk_page_ids):
        # Advanced indexing on a [entries, 1] index, not index_select on a 1-D
        # one: behind a 1-D index the entry axis splits in whole 32-entry sticks,
        # so a narrow gather gets one core. It costs the eager path, which
        # _batched_decode_preconditions_met gives up.
        # Token-major cache page to head-major; a view, so do not add
        # .contiguous() -- merging these axes is what materializes the page.
        k_page = k_pages[page_idx].squeeze(1).permute(0, 2, 1, 3)
        v_page = v_pages[page_idx].squeeze(1).permute(0, 2, 1, 3)
        # Builder already broadcast across KV heads; split them back out.
        mask_tile = mask_by_chunk[c].reshape(entries, num_kv_heads, 1, block_size)

        scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Before the mask add: tanh(-inf/cap)*cap is -cap, not -inf, so
            # capping after it would un-mask the padded lanes.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        scores = scores + mask_tile
        # Leading-axis split only: merging a permuted axis pair is what
        # torch-spyre rejects.
        sc = scores.reshape(
            num_seqs, blocks_per_chunk, num_kv_heads, num_queries_per_kv, block_size
        )
        chunk_max = torch.amax(torch.amax(sc, dim=-1, keepdim=True), dim=1, keepdim=True)

        # The running max drives exp(), not the chunk's own: a chunk wholly past
        # a sequence's length is -inf throughout and exp(-inf - -inf) is NaN.
        # Every row has a valid block 0, so the chunk-0 max is finite.
        if c == 0:
            new_max = chunk_max
        else:
            assert tile_max is not None
            new_max = torch.maximum(tile_max, chunk_max)
        probs = torch.exp(sc - new_max)
        # The chunk's slots share one max, so summing them needs no rescale.
        chunk_sum = torch.sum(torch.sum(probs, dim=-1, keepdim=True), dim=1, keepdim=True)
        chunk_out = torch.sum(
            torch.matmul(
                probs.reshape(entries, num_kv_heads, num_queries_per_kv, block_size),
                v_page,
            ).reshape(num_seqs, blocks_per_chunk, num_kv_heads, num_queries_per_kv, head_size),
            dim=1,
            keepdim=True,
        )

        if c == 0:
            tile_max = new_max
            tile_sum = chunk_sum
            tile_output = chunk_out
        else:
            assert tile_max is not None
            assert tile_sum is not None
            assert tile_output is not None
            rescale = torch.exp(tile_max - new_max)
            tile_output = tile_output * rescale + chunk_out
            tile_sum = tile_sum * rescale + chunk_sum
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
    attn = (tile_output / tile_sum).reshape(num_seqs, num_heads, head_size)
    if out is not None:
        # The destination prefix starts at offset 0, so torch-spyre#3770 does not
        # apply; rows past the batch are don't-care and kept finite by the builder.
        out[:num_seqs].copy_(attn)
        return out
    return attn
