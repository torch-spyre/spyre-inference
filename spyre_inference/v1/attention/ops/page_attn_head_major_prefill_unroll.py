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

Opt-in alternative to ``page_attn_head_major_prefill``, selected only when
``SPYRE_ATTN_FOR_EACH_TILE=0``: walks the block axis with a plain Python ``for``
loop (unrolled at trace time) instead of ``for_each_tile``, and carries the LX-hint
placement from the pre-``for_each_tile`` ``prefill-kv-lx-residency`` branch (tip
``2a24ac9``, "land Q's post-gather materialization in LX residency") verbatim,
adapted onto today's stacked-tensor ``page_index_table``/``mask_stack`` call
signature. ``for_each_tile`` requires ``fullgraph=True``; this plain loop does not,
and its hints only apply to this unrolled shape -- see ``spyre_head_major_attn.py``
for the dispatch.
"""

import torch
from torch_spyre._inductor import config as _spyre_config  # noqa: F401
from torch_spyre._inductor import spyre_hint

from spyre_inference.v1.attention.ops.page_attn_head_major_prefill import prefill_work_div

# Greedy, co-optimizer off: ~5% faster, mostly from splitting a query copy over 32
# cores, not 1. But these writes are process-wide, so they switch the whole model to
# greedy, which seems to avoid the bug test_kernel_cache's fresh-lowering strict xfail
# expects, making it XPASS. This option is disabled, i.e. commented out, for now
# because this kernel is opt-in and an alternative to the default, and that test's
# intent under other solvers is unclear. May re-enable if the test's intent is
# clarified or the targeted bug is resolved; keep it at module scope.
# _spyre_config.co_optimizing_lx_planning = False
# _spyre_config.layout_solver = "greedy"


def page_attn_head_major_prefill_unroll_kernel(
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
):
    """Online softmax attention over ``num_blocks`` pages of the unfolded cache.

    Shapes are ``page_attn_head_major_decode``'s, except:
        page_index_table: [num_blocks, INT32_ELEMS_PER_STICK] int32 device tensor, row i
            holding the i-th active block's page index at column 0, indexing
            ``[num_blocks_total, num_kv_heads, block_size, head_size]``.
        mask_stack: [num_blocks, padded_query_len, block_size].
    """
    num_queries_per_kv = num_heads // num_kv_heads
    hkv_split, t_split = prefill_work_div(num_kv_heads, padded_query_len)
    matmul_split = {"T": t_split, "Hkv": hkv_split}

    # Gathered, not sliced outside: since torch-spyre#4449 a view's storage_offset is a
    # Dynamo graph guard, and q_start varies, so a slice would compile one kernel per batch
    # layout -- test_spyre_compile_input_offset_specialises_the_graph. The builder now
    # creates this table at exactly padded_query_len rows.
    q_rows = query.index_select(0, query_row_index)
    q_view = (
        q_rows.unsqueeze(0)
        .transpose(1, 2)
        .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
    )
    with spyre_hint(named_dims=["Hkv", "Hq_kv", "T", "D"], work_div=matmul_split):
        q = q_view * 1.0

    def _hinted_matmul(a, b):
        with spyre_hint(work_div=matmul_split):
            return torch.matmul(a, b)

    tile_max = None
    tile_sum = None
    tile_output = None

    for i in range(num_blocks):
        # One row of the unfolded cache: the folded per-kv-head gather exists to split for LX
        # residency. index_select, not subscripting, which lowers to aten.index and fails eager.
        page_idx = page_index_table[i, 0:1]
        with spyre_hint(named_dims=["Hkv", "Hq_kv", "St", "D"], work_div=matmul_split):
            k_page = k_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
            v_page = v_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
        mask_tile = mask_stack[i]

        scores = _hinted_matmul(q, k_page.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Before the mask add: tanh(-inf/cap)*cap is -cap, not -inf, so capping after it
            # would un-mask the padded lanes.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = _hinted_matmul(tile_probs, v_page)
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
            tile_output = tile_output + _hinted_matmul(tile_probs, v_page)
            tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
    attn = tile_output / tile_sum
    attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
    attn = attn.reshape(padded_query_len, num_heads, head_size)
    if out is not None:
        # Storing the full padded extent keeps this sequence's real query_len out of the
        # arguments, so it is not specialized on.
        out.index_copy_(0, query_row_index, attn)
        return out
    return attn
