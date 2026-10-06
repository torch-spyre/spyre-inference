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

"""Grouped packed queries with parallel page slots and a final softmax merge."""

import torch
from torch_spyre._inductor.wsr import for_each_tile

from spyre_inference.v1.attention.ops.jagged_page_attn import _compensated_add


def jagged_decode_attn_kernel(
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
    entries = page_indices.shape[2]
    queries_per_group = output_indices.shape[1]
    page_slots = entries // queries_per_group
    block_size = key_offsets.shape[-1]
    state_shape = (entries, num_kv_heads, group_size, 1)
    output_shape = (entries, num_kv_heads, group_size, head_size)
    state_kwargs = {"dtype": query.dtype, "device": query.device}
    sum_scale = 1.0 / block_size
    floor = torch.finfo(query.dtype).min

    def query_body(destination, tiles):
        q_ids, out_ids, page_ids, q_bounds, k_pos, query, k_pages, v_pages = tiles
        q = query.index_select(0, q_ids[0, :entries])
        q = q.reshape(entries, num_kv_heads, group_size, head_size)

        def page_body(carry, page_tiles):
            tile_max, tile_sum, tile_sum_error, tile_output, tile_output_error = carry
            page_ids, bounds, offsets, q, k_pages, v_pages = page_tiles
            page_ids, bounds, offsets = page_ids[0, 0], bounds[0, 0], offsets[0, 0]
            # Keep each page index in its own stick so page slots can split
            # across cores, as in the existing batched head-major decode.
            indices = page_ids[:, 0:1].clone(memory_format=torch.contiguous_format)
            k = k_pages[indices].squeeze(1)
            if not head_major:
                k = k.permute(0, 2, 1, 3)
            upper = bounds[:, 0].reshape(entries, 1, 1, 1)
            lower = bounds[:, 1].reshape(entries, 1, 1, 1)
            offsets = offsets.reshape(entries, 1, 1, block_size)
            valid_keys = (offsets >= 0).reshape(entries, 1, block_size, 1)
            zero = torch.zeros((), **state_kwargs)
            masked_score = torch.full((), floor, **state_kwargs)
            k = torch.where(valid_keys, k, zero)
            visible = (offsets <= upper) & (offsets > lower)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if logits_soft_cap > 0.0:
                scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
            scores = torch.where(visible, scores, masked_score)
            scores_max = scores.amax(dim=-1, keepdim=True)
            rescale = torch.exp(-torch.relu(scores_max - tile_max))
            new_max = torch.maximum(tile_max, scores_max)
            probs = torch.where(visible, torch.exp(scores - new_max), zero)
            new_sum, new_sum_error = _compensated_add(
                tile_sum, tile_sum_error, probs.sum(dim=-1, keepdim=True) * sum_scale, rescale
            )
            v = v_pages[indices].squeeze(1)
            if not head_major:
                v = v.permute(0, 2, 1, 3)
            v = torch.where(valid_keys, v, zero)
            new_output, new_output_error = _compensated_add(
                tile_output,
                tile_output_error,
                torch.matmul(probs, v) * sum_scale,
                rescale,
            )
            return (new_max, new_sum, new_sum_error, new_output, new_output_error), None

        (tile_max, tile_sum, _, tile_output, _), _ = for_each_tile(
            page_body,
            (page_ids, q_bounds, k_pos, q, k_pages, v_pages),
            dims=(1, 1, 1, None, None, None),
            tile_size=1,
            init=(
                torch.full(state_shape, floor, **state_kwargs),
                torch.zeros(state_shape, **state_kwargs),
                torch.zeros(state_shape, **state_kwargs),
                torch.zeros(output_shape, **state_kwargs),
                torch.zeros(output_shape, **state_kwargs),
            ),
        )
        if page_slots == 1:
            denominator = tile_sum
            numerator = tile_output
        else:
            tile_max = tile_max.reshape(page_slots, queries_per_group, num_kv_heads, group_size, 1)
            tile_sum = tile_sum.reshape(page_slots, queries_per_group, num_kv_heads, group_size, 1)
            tile_output = tile_output.reshape(
                page_slots, queries_per_group, num_kv_heads, group_size, head_size
            )
            merged_max = tile_max.amax(dim=0, keepdim=True)
            weights = torch.exp(tile_max - merged_max)
            denominator = (weights * tile_sum).sum(dim=0)
            numerator = (weights * tile_output).sum(dim=0)
        result = (numerator / denominator.clamp_min(sum_scale)).reshape(
            queries_per_group, num_heads, head_size
        )
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
            torch.zeros((query_capacity + 1, num_heads, head_size), **state_kwargs)
            if out is None
            else None
        ),
        out_dim=None if out is None else 0,
    )
    if out is not None:
        # Short rows occupy separate int32 sticks; flattening their index
        # layout creates unsupported sub-stick modulo expressions.
        for group in range(output_indices.shape[0]):
            start = group * queries_per_group
            out.index_copy_(0, output_indices[group], tiles[start : start + queries_per_group])
        return out
    return output
