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

"""Paged attention over a head-major KV cache for a query wider than one token.

``page_attn_head_major_decode`` buys LX page residency by shaping the gather and the query
around the folded cache. Past one query token the page transfer that buys is amortised over
every query row, so this kernel spends it instead: batched GQA over
``[kv_head, group, query, D]``, one accumulator, store fuses.
"""

import torch

from spyre_inference.v1.attention.ops.sinks import normalize
from spyre_inference.v1.attention.ops.tile_loop import walk_tiles


def page_attn_head_major_prefill_kernel(
    *,
    query,
    query_row_index,
    k_pages,
    v_pages,
    page_index_table,
    mask_stack,
    scale,
    num_blocks,
    padded_query_len,
    num_heads,
    num_kv_heads,
    head_size,
    logits_soft_cap=0.0,
    out=None,
    v_scale=None,
    sinks=None,
    page_group=1,
):
    """Online softmax attention over ``num_blocks`` pages of the unfolded cache.

    The page walk goes through `walk_tiles`. Shapes are
    ``page_attn_head_major_decode``'s, except:
        page_index_table: [num_blocks, INT32_ELEMS_PER_STICK] int32 device tensor, row i
            holding the i-th active block's page index at column 0, indexing
            ``[num_blocks_total, num_kv_heads, block_size, head_size]``.
        mask_stack: [num_blocks, padded_query_len, block_size], tiled on dim 0.
        v_scale: float V scale of a float8 cache, None for a model-dtype one; the caller
            folds k_scale into ``scale``.
        sinks: [num_kv_heads, num_queries_per_kv, 1] per-head sink logits, or None.
        page_group: pages per online-softmax update; must divide num_blocks.
    """
    if page_group < 1 or num_blocks % page_group:
        raise ValueError(
            f"page_group={page_group} must be a positive divisor of num_blocks={num_blocks}"
        )
    grouped = page_group > 1
    num_queries_per_kv = num_heads // num_kv_heads

    # Gathered, not sliced outside: since torch-spyre#4449 a view's storage_offset is a
    # Dynamo graph guard, and q_start varies, so a slice would compile one kernel per batch
    # layout -- test_spyre_compile_input_offset_specialises_the_graph. The builder now
    # creates this table at exactly padded_query_len rows.
    q_rows = query.index_select(0, query_row_index)
    q = (
        q_rows.unsqueeze(0)
        .transpose(1, 2)
        .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
    )
    if grouped:
        q = q.unsqueeze(0)

    # Both walks tile tensor axes, so what an unrolled walk read per block arrives
    # stacked on dim 0.
    operands = (page_index_table[:num_blocks], k_pages, v_pages, mask_stack[:num_blocks], q)
    dims: tuple[int | None, ...] = (0, None, None, 0, None)

    def fold_group(tile, reduce):
        return reduce(tile, dim=0, keepdim=True) if grouped else tile

    def block_body(carry, tiles):
        page_index, k_pages, v_pages, mask_tile, q = tiles

        # One row of the unfolded cache: the folded per-kv-head gather exists to split for LX
        # residency. index_select, not subscripting, which lowers to aten.index and fails eager.
        if grouped:
            page_idx = page_index[:, 0]
            # The group stays a batch axis: merging it into the token axis crosses the kv axis,
            # which is not viewable and lowers to a strided copy the backend rejects.
            k_page = k_pages.index_select(0, page_idx).unsqueeze(2)
            v_page = v_pages.index_select(0, page_idx).unsqueeze(2)
            mask = mask_tile.unsqueeze(1).unsqueeze(1)
        else:
            page_idx = page_index[0, 0:1]
            k_page = k_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
            v_page = v_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
            mask = mask_tile[0]
        if k_page.dtype != q.dtype:
            k_page = k_page.to(q.dtype)
            v_page = v_page.to(q.dtype)

        scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Before the mask add: tanh(-inf/cap)*cap is -cap, not -inf, so capping after it
            # would un-mask the padded lanes.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        scores = scores + mask
        scores_max = fold_group(torch.amax(scores, dim=-1, keepdim=True), torch.amax)

        # `carry is None` is required for SPYRE_ATTN_FOR_EACH_TILE=0
        if carry is None:
            tile_probs = torch.exp(scores - scores_max)
            return (
                scores_max,
                fold_group(tile_probs.sum(dim=-1, keepdim=True), torch.sum),
                fold_group(torch.matmul(tile_probs, v_page), torch.sum),
            ), None

        tile_max, tile_sum, tile_output = carry
        # Read tile_max before the maximum that supersedes it, or the tiled lowering
        # copies the whole carry every trip. Identical to exp(tile_max - new_max).
        rescale = torch.exp(-torch.relu(scores_max - tile_max))
        new_max = torch.maximum(tile_max, scores_max)
        tile_probs = torch.exp(scores - new_max)
        new_sum = tile_sum * rescale + fold_group(tile_probs.sum(dim=-1, keepdim=True), torch.sum)
        new_output = tile_output * rescale + fold_group(torch.matmul(tile_probs, v_page), torch.sum)
        return (new_max, new_sum, new_output), None

    group_axis = (1,) if grouped else ()
    state_shape = (*group_axis, num_kv_heads, num_queries_per_kv, padded_query_len, 1)
    state_kwargs = {"dtype": q.dtype, "device": q.device}
    (tile_max, tile_sum, tile_output), _ = walk_tiles(
        block_body,
        operands,
        dims=dims,
        tile_size=page_group,
        init=(
            torch.full(state_shape, float("-inf"), **state_kwargs),
            torch.zeros(state_shape, **state_kwargs),
            torch.zeros(
                (*group_axis, num_kv_heads, num_queries_per_kv, padded_query_len, head_size),
                **state_kwargs,
            ),
        ),
    )
    if grouped:
        tile_max, tile_sum, tile_output = (t.squeeze(0) for t in (tile_max, tile_sum, tile_output))
    # The state has a query-row axis the sinks broadcast over.
    attn = normalize(
        tile_max, tile_sum, tile_output, None if sinks is None else sinks.unsqueeze(-1)
    )
    attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
    attn = attn.reshape(padded_query_len, num_heads, head_size)
    if v_scale is not None and v_scale != 1.0:
        attn = attn * v_scale
    if out is not None:
        # Storing the full padded extent keeps this sequence's real query_len out of the
        # arguments, so it is not specialized on.
        out.index_copy_(0, query_row_index, attn)
        return out
    return attn
