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


def page_attn_head_major_decode_kernel(
    query,
    query_row_index,
    k_pages,
    v_pages,
    kv_index_tables,
    mask_tiles,
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

    Heads are kv-major, so the fold is a reshape, and it needs no head gather. Under
    `dynamic=False` Dynamo specializes on every non-tensor argument, so the page loop is
    unrolled per variant.

    Expected shapes:
        query: [num_tokens, num_heads, head_size], the whole batch's query
        query_row_index: [padded_query_len] int32 device tensor of this sequence's
            absolute query rows.
        k_pages / v_pages: [num_pages_total * num_kv_heads, block_size, head_size]
        kv_index_tables: per active block, a [num_kv_heads, 1] int32 device tensor of that
            block's ``page * num_kv_heads + kv`` rows. One real tensor per block, not a
            slice of a table: an index tensor reaches the hardware as a tensor argument,
            so a slice's nonzero storage offset is dropped (torch-spyre#3770).
        mask_tiles: [num_blocks], each [padded_query_len, block_size]
        out: buffer to store into, or None to return the result instead.

    Returns [padded_query_len, num_heads, head_size], or ``out``.
    """
    assert padded_query_len == 1, "decode kernel is specialized for a single query row"
    num_queries_per_kv = num_heads // num_kv_heads

    q = query.index_select(0, query_row_index).reshape(num_kv_heads, num_queries_per_kv, head_size)

    tile_max = None
    tile_sum = None
    tile_out = None

    for i in range(num_blocks):
        # Subscripting, not index_select, which takes only a 1-D index: that puts the
        # entry axis on the index's own stick axis, splittable only in whole 32-entry
        # sticks. [num_kv_heads, 1] lets the split land per kv head.
        kv_rows = kv_index_tables[i]
        k_page = k_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)
        v_page = v_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)

        scores = torch.matmul(q, k_page.permute(0, 2, 1)) * scale
        if logits_soft_cap > 0.0:
            # Before the mask add: tanh(-inf/cap)*cap is -cap, not -inf, so capping after
            # it would un-mask the padded lanes.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        # At one query row the mask is head-independent, so its [1, block_size] tile
        # broadcasts across the folded group axis.
        scores = scores + mask_tiles[i]
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            probs = torch.exp(scores - scores_max)
            tile_max = scores_max
            tile_out = torch.matmul(probs, v_page)
            tile_sum = probs.sum(dim=-1, keepdim=True)
        else:
            assert tile_max is not None
            assert tile_sum is not None
            assert tile_out is not None
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_out = tile_out * rescale
            tile_sum = tile_sum * rescale
            probs = torch.exp(scores - new_max)
            tile_out = tile_out + torch.matmul(probs, v_page)
            tile_sum = tile_sum + probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    assert tile_out is not None and tile_sum is not None
    attn = (tile_out / tile_sum).reshape(1, num_heads, head_size)
    if out is not None:
        out.index_copy_(0, query_row_index, attn)
        return out
    return attn
