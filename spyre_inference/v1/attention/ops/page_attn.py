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

"""Per-sequence paged attention over the KV cache."""

import torch


def page_attn_kernel(
    query,
    query_row_index,
    k_pages,
    v_pages,
    page_index_table,
    mask_tiles,
    scale,
    num_blocks,
    padded_query_len,
    num_heads,
    num_kv_heads,
    head_size,
    logits_soft_cap=0.0,
    alibi_bias_tiles=None,
    out=None,
):
    """Online softmax attention over ``num_blocks`` KV pages.

    Under `dynamic=False` Dynamo specializes on every non-tensor argument. The
    page walk uses `for_each_tile`, so its graph has one block body rather than an
    unrolled copy per page.

    Expected shapes:
        query: [num_tokens, num_heads, head_size], the whole batch's query
        query_row_index: int32 device tensor whose first padded_query_len
            entries are this sequence's absolute query rows.
        k_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
        v_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
        page_index_table: [num_blocks, INT32_ELEMS_PER_STICK] int32 device
            tensor, row i holding the i-th active block's page index at
            column 0.
        mask_tiles: [num_blocks, padded_query_len, block_size] additive masks,
            stacked block-major for `for_each_tile`.
        alibi_bias_tiles: [num_blocks, num_kv_heads, num_queries_per_kv, 1,
            block_size],
            or None for no ALiBi. The query-axis dim is 1 because softmax absorbs
            per-query-row constants; see the derivation at the bias-tile
            construction site in _online_softmax_attention.
        out: buffer to store into, or None to return the result instead.

    Returns [padded_query_len, num_heads, head_size], or ``out`` when this
    kernel stored the result itself.
    """
    num_queries_per_kv = num_heads // num_kv_heads
    # A compiled region reads a view from offset 0, ignoring storage_offset
    # (torch-spyre#3770), so the rows are gathered here rather than sliced outside.
    q_rows = query.index_select(0, query_row_index[:padded_query_len])
    q = (
        q_rows.unsqueeze(0)
        .transpose(1, 2)
        .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
    )

    from torch_spyre._inductor.wsr import for_each_tile

    # `for_each_tile` consumes tensor axes, so the baseline's per-block lists
    # arrive stacked on dim 0. The table, page gather, score calculation, and
    # online-softmax update otherwise follow the baseline loop body directly.
    operands = [
        page_index_table[:num_blocks],
        k_pages,
        v_pages,
        mask_tiles[:num_blocks],
        q,
    ]
    dims: list[int | None] = [0, None, None, 0, None]
    if alibi_bias_tiles is not None:
        operands.append(alibi_bias_tiles[:num_blocks])
        dims.append(0)

    def block_body(carry, tiles):
        tile_max, tile_sum, tile_output = carry
        if alibi_bias_tiles is not None:
            page_index, k_pages, v_pages, mask_tile, q, alibi_bias_tile = tiles
        else:
            page_index, k_pages, v_pages, mask_tile, q = tiles

        page_idx = page_index[0, 0:1]
        k_page = k_pages.index_select(0, page_idx)
        v_page = v_pages.index_select(0, page_idx)
        k_page_4d = k_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)
        v_page_4d = v_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)

        scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Pull logits into (-cap, +cap) before the mask add so masked
            # positions still map cleanly to -inf. Applied before the ALiBi
            # bias so the positional term is not squashed by the tanh.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        if alibi_bias_tiles is not None:
            # ALiBi bias slope[h] * (kv_pos - context_len). The additive
            # mask_tile below uses finfo.min for masked positions, so this
            # bias cannot un-mask them.
            scores = scores + alibi_bias_tile[0]
        scores = scores + mask_tile[0]

        scores_max = torch.amax(scores, dim=-1, keepdim=True)
        new_max = torch.maximum(tile_max, scores_max)
        rescale = torch.exp(tile_max - new_max)
        tile_probs = torch.exp(scores - new_max)
        new_sum = tile_sum * rescale + tile_probs.sum(dim=-1, keepdim=True)
        new_output = tile_output * rescale + torch.matmul(tile_probs, v_page_4d)
        return (new_max, new_sum, new_output), None

    state_shape = (num_kv_heads, num_queries_per_kv, padded_query_len, 1)
    state_kwargs = {"dtype": q.dtype, "device": q.device}
    (_, tile_sum, tile_output), _ = for_each_tile(
        block_body,
        tuple(operands),
        dims=tuple(dims),
        tile_size=1,
        init=(
            torch.full(state_shape, float("-inf"), **state_kwargs),
            torch.zeros(state_shape, **state_kwargs),
            torch.zeros(
                (num_kv_heads, num_queries_per_kv, padded_query_len, head_size),
                **state_kwargs,
            ),
        ),
    )
    attn = tile_output / tile_sum
    attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
    attn = attn.reshape(padded_query_len, num_heads, head_size)
    if out is not None:
        # `out` and `query` are both indexed by absolute token row. Storing the
        # full padded extent keeps the sequence's real query_len out of the
        # arguments, so it is not specialized on; rows past it duplicate the
        # sequence's last row, so index_copy_'s undefined write order for
        # duplicate indices is harmless.
        out.index_copy_(0, query_row_index[:padded_query_len], attn[:padded_query_len])
        return out
    return attn
