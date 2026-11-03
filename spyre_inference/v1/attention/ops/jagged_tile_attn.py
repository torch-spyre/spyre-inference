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

"""Packed attention with an inner page loop and one store per query tile."""

import torch
from torch_spyre._inductor.wsr import for_each_tile

from spyre_inference.v1.attention.ops.jagged_page_attn import _compensated_add


def jagged_tile_attn_kernel(
    query,
    k_pages,
    v_pages,
    query_indices,
    output_indices,
    page_indices,
    query_bounds,
    key_offsets,
    scale: float,
    *,
    head_major: bool = False,
    logits_soft_cap: float = 0.0,
    out=None,
):
    query_capacity, num_heads, head_size = query.shape
    num_kv_heads = k_pages.shape[1] if head_major else k_pages.shape[2]
    group_size = num_heads // num_kv_heads
    query_tile_size = query_indices.shape[1]
    block_size = key_offsets.shape[2]
    # Keep the reduced axis: squeezing it forces repeated restickification
    # between query-oriented reductions and feature-oriented matmul outputs.
    state_shape = (num_kv_heads, group_size, query_tile_size, 1)
    output_shape = (num_kv_heads, group_size, query_tile_size, head_size)
    state_kwargs = {"dtype": query.dtype, "device": query.device}
    sum_scale = 1.0 / block_size
    floor = torch.finfo(query.dtype).min

    def query_body(destination, tiles):
        q_ids, out_ids, page_ids, q_bounds, k_pos, query, k_pages, v_pages = tiles
        q = query.index_select(0, q_ids[0]).transpose(0, 1)
        q = q.reshape(num_kv_heads, group_size, query_tile_size, head_size)

        def page_body(carry, page_tiles):
            tile_max, tile_sum, tile_sum_error, tile_output, tile_output_error = carry
            page_id, bounds, offsets, q, k_pages, v_pages = page_tiles
            page = page_id[0, 0:1]
            k = k_pages.index_select(0, page).squeeze(0)
            v = v_pages.index_select(0, page).squeeze(0)
            if not head_major:
                k = k.permute(1, 0, 2)
                v = v.permute(1, 0, 2)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            upper = bounds[0, 0].unsqueeze(1).expand(query_tile_size, block_size) * 1.0
            lower = bounds[0, 1].unsqueeze(1).expand(query_tile_size, block_size) * 1.0
            offsets = offsets[0].unsqueeze(0) * 1.0
            valid_keys = (offsets >= 0).reshape(1, 1, -1, 1)
            k = torch.where(valid_keys, k, 0.0)
            v = torch.where(valid_keys, v, 0.0)
            visible = (offsets <= upper) & (offsets > lower)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if logits_soft_cap > 0.0:
                scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
            scores = torch.where(visible, scores, floor)
            scores_max = scores.amax(dim=-1, keepdim=True)
            rescale = torch.exp(-torch.relu(scores_max - tile_max))
            new_max = torch.maximum(tile_max, scores_max)
            probs = torch.where(visible, torch.exp(scores - new_max), 0.0)
            new_sum, new_sum_error = _compensated_add(
                tile_sum, tile_sum_error, probs.sum(dim=-1, keepdim=True) * sum_scale, rescale
            )
            new_output, new_output_error = _compensated_add(
                tile_output,
                tile_output_error,
                torch.matmul(probs, v) * sum_scale,
                rescale,
            )
            return (new_max, new_sum, new_sum_error, new_output, new_output_error), None

        (_, tile_sum, _, tile_output, _), _ = for_each_tile(
            page_body,
            (page_ids[0], q_bounds[0], k_pos[0], q, k_pages, v_pages),
            dims=(0, 0, 0, None, None, None),
            tile_size=1,
            init=(
                torch.full(state_shape, floor, **state_kwargs),
                torch.zeros(state_shape, **state_kwargs),
                torch.zeros(state_shape, **state_kwargs),
                torch.zeros(output_shape, **state_kwargs),
                torch.zeros(output_shape, **state_kwargs),
            ),
        )
        result = tile_output / tile_sum.clamp_min(sum_scale)
        result = result.reshape(num_heads, query_tile_size, head_size).transpose(0, 1)
        if out is not None:
            return None, result
        return destination.index_copy(0, out_ids[0], result), None

    output, tiles = for_each_tile(
        query_body,
        (
            query_indices,
            output_indices,
            page_indices,
            query_bounds,
            key_offsets,
            query,
            k_pages,
            v_pages,
        ),
        dims=(0, 0, 0, 0, 0, None, None, None),
        tile_size=1,
        init=(
            torch.zeros((query_capacity + query_tile_size, num_heads, head_size), **state_kwargs)
            if out is None
            else None
        ),
        out_dim=None if out is None else 0,
    )
    if out is not None:
        out.index_copy_(0, output_indices.reshape(-1), tiles)
        return out
    return output
