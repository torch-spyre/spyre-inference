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
    page_group=1,
    alibi_bias_tiles=None,
    out=None,
):
    """Online softmax attention over ``num_blocks`` KV pages.

    Under `dynamic=False` Dynamo specializes on every non-tensor argument, so the
    page loop is unrolled per variant.

    Expected shapes:
        query: [num_tokens, num_heads, head_size], the whole batch's query
        query_row_index: int32 device tensor whose first padded_query_len
            entries are this sequence's absolute query rows.
        k_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
        v_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
        page_index_table: [num_blocks, INT32_ELEMS_PER_STICK] int32 device
            tensor, row i holding the i-th active block's page index at
            column 0.
        mask_tiles: [num_blocks]
        page_group: adjacent pages folded into one online-softmax update, widening
            its score matmul to [padded_query_len, page_group * block_size]. 1
            keeps one page per update; the final group may be shorter.
        alibi_bias_tiles: list of [num_kv_heads, num_queries_per_kv, 1, block_size],
            or None for no ALiBi. The query-axis dim is 1 because softmax absorbs
            per-query-row constants — see the derivation at the bias-tile
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

    tile_max = None
    tile_sum = None
    tile_output = None

    for group_start in range(0, num_blocks, page_group):
        group_end = min(group_start + page_group, num_blocks)
        width = group_end - group_start

        if width == 1:
            # index_select, not `k_pages[page_idx]`: subscripting lowers to
            # aten.index, which upcasts the int32 index to int64 and fails eager.
            page_idx = page_index_table[group_start, 0:1]
            k_page = k_pages.index_select(0, page_idx)
            v_page = v_pages.index_select(0, page_idx)
            # Token-major page to head-major for the matmuls; permutes on device.
            k_page_4d = k_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)
            v_page_4d = v_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)

            mask_tile = mask_tiles[group_start]
            bias_tile = None if alibi_bias_tiles is None else alibi_bias_tiles[group_start]
        else:
            # The whole group in one gather, its pages joined along the token axis.
            # The column slice stays inside the traced region, so it is a real op
            # rather than a view crossing the argument boundary (torch-spyre#3770).
            page_idx = page_index_table[group_start:group_end, 0]
            k_group = k_pages.index_select(0, page_idx)
            v_group = v_pages.index_select(0, page_idx)
            group_tokens = width * k_group.shape[1]
            k_page_4d = (
                k_group.permute(2, 0, 1, 3)
                .reshape(num_kv_heads, group_tokens, head_size)
                .unsqueeze(1)
            )
            v_page_4d = (
                v_group.permute(2, 0, 1, 3)
                .reshape(num_kv_heads, group_tokens, head_size)
                .unsqueeze(1)
            )

            mask_tile = torch.cat(mask_tiles[group_start:group_end], dim=-1)
            bias_tile = (
                None
                if alibi_bias_tiles is None
                else torch.cat(alibi_bias_tiles[group_start:group_end], dim=-1)
            )

        scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Pull logits into (-cap, +cap) before the mask add so masked
            # positions still map cleanly to -inf. Applied before the ALiBi
            # bias so the positional term is not squashed by the tanh.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        if bias_tile is not None:
            # ALiBi bias slope[h] * (kv_pos - context_len). The additive
            # mask_tile below uses finfo.min for masked positions, so this
            # bias cannot un-mask them.
            scores = scores + bias_tile
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if group_start == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = torch.matmul(tile_probs, v_page_4d)
            tile_sum = tile_probs.sum(dim=-1, keepdim=True)
        else:
            # Later groups only reachable after the first initialized these.
            assert tile_max is not None
            assert tile_sum is not None
            assert tile_output is not None
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_output = tile_output * rescale
            tile_sum = tile_sum * rescale
            tile_probs = torch.exp(scores - new_max)
            tile_output += torch.matmul(tile_probs, v_page_4d)
            tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
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
