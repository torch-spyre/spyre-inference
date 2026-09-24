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

"""Per-sequence decode attention over a head-major KV cache, keeping the page LX-resident.

Two shape choices keep a gathered page in LX rather than round-tripping it through HBM,
over the cache folded to ``[pages * kv, block_size, D]``: the page is gathered on
(page, kv_head) so the gather's split lands per kv head, an output axis of ``probs @ V``
the consumer can mirror; and the query groups fold into the row axis, since the batched GQA
form leaves the page with two batch dims and Inductor clones it out to a query-group axis it
does not have (torch-spyre#4123).
"""

import torch

from spyre_inference.v1.attention.ops.tile_loop import walk_tiles


def page_attn_head_major_decode_kernel(
    query,
    query_row_index,
    k_pages,
    v_pages,
    page_index_table,
    kv_row_pool,
    mask_stack,
    scale,
    num_blocks,
    padded_query_len,
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    logits_soft_cap=0.0,
    out=None,
):
    """Decode (Q=1) attention with the query groups folded into the row axis.

    Heads are kv-major, so the fold is a reshape, and it needs no head gather. The page
    walk goes through `walk_tiles`, which holds one block body rather than an unrolled
    copy per page when SPYRE_ATTN_FOR_EACH_TILE is set.

    Expected shapes:
        query: [num_tokens, num_heads, head_size], the whole batch's query
        query_row_index: [padded_query_len] int32 device tensor of this sequence's
            absolute query rows.
        k_pages / v_pages: [num_pages_total * num_kv_heads, block_size, head_size]
        page_index_table: [num_blocks, INT32_ELEMS_PER_STICK] int32 device tensor whose
            row i holds the i-th active block's page id in column 0, the base's table
            unchanged.
        kv_row_pool: [num_pages_total, num_kv_heads, 1] int32 device tensor of every
            page's folded-cache rows, ``page * num_kv_heads + kv``. Gathered rather than
            computed (int32 arithmetic has no device op mapping) or sliced out of a
            per-block table (an int32 argument's nonzero storage offset still reads as 0,
            torch-spyre#3770). Pure cache geometry, so it is built once.
        mask_stack: [num_blocks, padded_query_len, block_size], tiled on dim 0.
        out: buffer to store into, or None to return the result instead.

    Returns [padded_query_len, num_heads, head_size], or ``out``.
    """
    assert padded_query_len == 1, "decode kernel is specialized for a single query row"
    num_queries_per_kv = num_heads // num_kv_heads

    q = query.index_select(0, query_row_index).reshape(num_kv_heads, num_queries_per_kv, head_size)

    def block_body(carry, tiles):
        page_index, k_pages, v_pages, mask_tile, q, kv_row_pool = tiles
        # A tile is readable whole or as the single element `page_index[0, 0:1]` is; a
        # [num_kv_heads]-wide read of a wider row is neither, so the rows come from the
        # pool. See the `kv_row_pool` docstring.
        kv_rows = kv_row_pool.index_select(0, page_index[0, 0:1])
        # Subscripting, not index_select, which takes only a 1-D index: that puts the entry
        # axis on the index's own stick axis, splittable only in whole 32-entry sticks.
        # [num_kv_heads, 1] lets the split land per kv head.
        k_page = k_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)
        v_page = v_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)

        scores = torch.matmul(q, k_page.permute(0, 2, 1)) * scale
        if logits_soft_cap > 0.0:
            # Before the mask add: tanh(-inf/cap)*cap is -cap, not -inf, so capping after
            # it would un-mask the padded lanes.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        # At one query row the mask is head-independent, so its [1, block_size] tile
        # broadcasts across the folded group axis.
        scores = scores + mask_tile[0]
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        # `carry is None` is required for SPYRE_ATTN_FOR_EACH_TILE=0
        if carry is None:
            probs = torch.exp(scores - scores_max)
            return (
                scores_max,
                probs.sum(dim=-1, keepdim=True),
                torch.matmul(probs, v_page),
            ), None

        tile_max, tile_sum, tile_output = carry
        # Read tile_max before the maximum that supersedes it, or the tiled lowering
        # copies the whole carry every trip. Identical to exp(tile_max - new_max).
        rescale = torch.exp(-torch.relu(scores_max - tile_max))
        new_max = torch.maximum(tile_max, scores_max)
        probs = torch.exp(scores - new_max)
        return (
            new_max,
            tile_sum * rescale + probs.sum(dim=-1, keepdim=True),
            tile_output * rescale + torch.matmul(probs, v_page),
        ), None

    state_shape = (num_kv_heads, num_queries_per_kv, 1)
    state_kwargs = {"dtype": q.dtype, "device": q.device}
    (_, tile_sum, tile_output), _ = walk_tiles(
        block_body,
        (page_index_table[:num_blocks], k_pages, v_pages, mask_stack[:num_blocks], q, kv_row_pool),
        dims=(0, None, None, 0, None, None),
        tile_size=1,
        init=(
            torch.full(state_shape, float("-inf"), **state_kwargs),
            torch.zeros(state_shape, **state_kwargs),
            torch.zeros((num_kv_heads, num_queries_per_kv, head_size), **state_kwargs),
        ),
    )
    attn = (tile_output / tile_sum).reshape(1, num_heads, head_size)
    if out is not None:
        out.index_copy_(0, query_row_index, attn)
        return out
    return attn
