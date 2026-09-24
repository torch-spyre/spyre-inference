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

from spyre_inference.v1.attention.ops.tile_loop import USE_FOR_EACH_TILE, walk_tiles


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
    chunk_page_ids: [padded_blocks, num_seqs] int32, row b holding each sequence's
    b-th active page. mask_by_chunk: [padded_blocks, num_seqs, KV or 1, 1,
    block_size], broadcast across the query group in the kernel, and across KV heads
    too unless the tiled walk materialized that axis. rep_row_ids: [entries] int32,
    every query row repeated once per block slot. ``out`` None returns the result
    instead of storing it.
    """
    num_heads = num_kv_heads * num_queries_per_kv
    entries = num_seqs * blocks_per_chunk
    split_index = USE_FOR_EACH_TILE
    q = query.index_select(0, rep_row_ids).reshape(
        entries, num_kv_heads, num_queries_per_kv, head_size
    )

    def reduce_chunk(probs, v_page):
        """Sum the chunk's blocks; its slots share one max, so this needs no rescale."""
        chunk_sum = torch.sum(torch.sum(probs, dim=-1, keepdim=True), dim=0, keepdim=True)
        chunk_out = torch.sum(
            torch.matmul(
                probs.reshape(entries, num_kv_heads, num_queries_per_kv, block_size),
                v_page,
            ).reshape(blocks_per_chunk, num_seqs, num_kv_heads, num_queries_per_kv, head_size),
            dim=0,
            keepdim=True,
        )
        return chunk_sum, chunk_out

    def chunk_body(carry, tiles):
        page_ids, mask_rows, k_pages, v_pages, q = tiles
        if split_index:
            # The tile is chunk-major, so shape[3:] still carries the KV axis.
            page_ids = page_ids[0]
            mask_rows = mask_rows.reshape(blocks_per_chunk, num_seqs, *mask_rows.shape[3:])
        # Advanced indexing on the [block-slot, sequence] tile, not index_select on a
        # 1-D one: behind a 1-D index the entry axis splits in whole 32-entry sticks,
        # so a narrow gather gets one core, and flattening the tile first needs an
        # unsupported int32 staging layout. It costs the eager path, which
        # _batched_decode_preconditions_met gives up.
        # Token-major cache page to head-major; a view, so do not add
        # .contiguous() -- merging these axes is what materializes the page.
        k_page = (
            k_pages[page_ids]
            .reshape(entries, block_size, num_kv_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        v_page = (
            v_pages[page_ids]
            .reshape(entries, block_size, num_kv_heads, head_size)
            .permute(0, 2, 1, 3)
        )
        scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Before the mask add: tanh(-inf/cap)*cap is -cap, not -inf, so
            # capping after it would un-mask the padded lanes.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        # Leading-axis split only: merging a permuted axis pair is what torch-spyre
        # rejects, and the mask's advancing read window must stay unflattened.
        sc = scores.reshape(
            blocks_per_chunk, num_seqs, num_kv_heads, num_queries_per_kv, block_size
        )
        sc = sc + mask_rows
        chunk_max = torch.amax(torch.amax(sc, dim=-1, keepdim=True), dim=0, keepdim=True)

        # The running max drives exp(), not the chunk's own: a chunk wholly past
        # a sequence's length is -inf throughout and exp(-inf - -inf) is NaN.
        # Every row has a valid block 0, so the chunk-0 max is finite.
        # `carry is None` is required for SPYRE_ATTN_FOR_EACH_TILE=0
        if carry is None:
            chunk_sum, chunk_out = reduce_chunk(torch.exp(sc - chunk_max), v_page)
            return (chunk_max, chunk_sum, chunk_out), None

        tile_max, tile_sum, tile_output = carry
        # Read tile_max before the maximum that supersedes it, or the tiled lowering
        # copies the whole carry every trip. Identical to exp(tile_max - new_max).
        rescale = torch.exp(-torch.relu(chunk_max - tile_max))
        new_max = torch.maximum(tile_max, chunk_max)
        chunk_sum, chunk_out = reduce_chunk(torch.exp(sc - new_max), v_page)
        return (
            new_max,
            tile_sum * rescale + chunk_sum,
            tile_output * rescale + chunk_out,
        ), None

    state_shape = (1, num_seqs, num_kv_heads, num_queries_per_kv, 1)
    state_kwargs = {"dtype": q.dtype, "device": q.device}
    (_, tile_sum, tile_output), _ = walk_tiles(
        chunk_body,
        (chunk_page_ids, mask_by_chunk, k_pages, v_pages, q),
        dims=(0, 0, None, None, None),
        tile_size=1 if split_index else blocks_per_chunk,
        init=(
            torch.full(state_shape, float("-inf"), **state_kwargs),
            torch.zeros(state_shape, **state_kwargs),
            torch.zeros(
                (1, num_seqs, num_kv_heads, num_queries_per_kv, head_size),
                **state_kwargs,
            ),
        ),
    )
    attn = (tile_output / tile_sum).reshape(num_seqs, num_heads, head_size)
    if out is not None:
        # The destination prefix starts at offset 0, so torch-spyre#3770 does not
        # apply; rows past the batch are don't-care and kept finite by the builder.
        out[:num_seqs].copy_(attn)
        return out
    return attn
