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

"""Independent page partitions followed by a separate decode reduction."""

import torch
from torch_spyre._inductor.wsr import for_each_tile

from spyre_inference.v1.attention.ops.jagged_page_attn import _compensated_add


def jagged_decode_partials_kernel(
    query,
    k_pages,
    v_pages,
    query_indices,
    page_indices,
    query_bounds,
    key_offsets,
    scale: float,
    *,
    head_major: bool = False,
    logits_soft_cap: float = 0.0,
    reduce_pages: bool = False,
):
    num_heads, head_size = query.shape[1:]
    num_kv_heads = k_pages.shape[1] if head_major else k_pages.shape[2]
    group_size = num_heads // num_kv_heads
    entries = page_indices.shape[2]
    block_size = key_offsets.shape[-1]
    state_kwargs = {"dtype": query.dtype, "device": query.device}
    sum_scale = 1.0 / block_size
    floor = torch.finfo(query.dtype).min
    reduce_pages = reduce_pages and page_indices.shape[1] > 1
    state_shape = (entries, num_kv_heads, group_size, 1)
    output_shape = (entries, num_kv_heads, group_size, head_size)

    def query_body(_, tiles):
        q_ids, page_ids, q_bounds, k_pos, query, k_pages, v_pages = tiles
        q = query.index_select(0, q_ids[0, :entries])
        q = q.reshape(entries, num_kv_heads, group_size, head_size)

        def page_body(carry, page_tiles):
            page_ids, bounds, offsets, q, k_pages, v_pages = page_tiles
            page_ids, bounds, offsets = page_ids[0, 0], bounds[0, 0], offsets[0, 0]
            indices = page_ids[:, 0:1].clone(memory_format=torch.contiguous_format)
            k = k_pages[indices].squeeze(1)
            v = v_pages[indices].squeeze(1)
            if not head_major:
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
            upper = bounds[:, 0].reshape(entries, 1, 1, 1)
            lower = bounds[:, 1].reshape(entries, 1, 1, 1)
            offsets = offsets.reshape(entries, 1, 1, block_size)
            valid_keys = (offsets >= 0).reshape(entries, 1, block_size, 1)
            zero = torch.zeros((), **state_kwargs)
            masked_score = torch.full((), floor, **state_kwargs)
            k = torch.where(valid_keys, k, zero)
            v = torch.where(valid_keys, v, zero)
            visible = (offsets <= upper) & (offsets > lower)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if logits_soft_cap > 0.0:
                scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
            scores = torch.where(visible, scores, masked_score)
            page_max = scores.amax(dim=-1, keepdim=True)
            if reduce_pages:
                tile_max, tile_sum, tile_sum_error, tile_output, tile_output_error = carry
                rescale = torch.exp(-torch.relu(page_max - tile_max))
                new_max = torch.maximum(tile_max, page_max)
                probs = torch.where(visible, torch.exp(scores - new_max), zero)
                new_sum, new_sum_error = _compensated_add(
                    tile_sum,
                    tile_sum_error,
                    probs.sum(dim=-1, keepdim=True) * sum_scale,
                    rescale,
                )
                new_output, new_output_error = _compensated_add(
                    tile_output,
                    tile_output_error,
                    torch.matmul(probs, v) * sum_scale,
                    rescale,
                )
                return (new_max, new_sum, new_sum_error, new_output, new_output_error), None
            probs = torch.where(visible, torch.exp(scores - page_max), zero)
            page_sum = probs.sum(dim=-1, keepdim=True) * sum_scale
            page_output = torch.matmul(probs, v) * sum_scale
            return None, (
                page_max.unsqueeze(0),
                page_sum.unsqueeze(0),
                page_output.unsqueeze(0),
            )

        state, partials = for_each_tile(
            page_body,
            (page_ids, q_bounds, k_pos, q, k_pages, v_pages),
            dims=(1, 1, 1, None, None, None),
            tile_size=1,
            out_dim=None if reduce_pages else 0,
            init=(
                torch.full(state_shape, floor, **state_kwargs),
                torch.zeros(state_shape, **state_kwargs),
                torch.zeros(state_shape, **state_kwargs),
                torch.zeros(output_shape, **state_kwargs),
                torch.zeros(output_shape, **state_kwargs),
            )
            if reduce_pages
            else None,
        )
        if reduce_pages:
            tile_max, tile_sum, _, tile_output, _ = state
            partials = tuple(t.unsqueeze(0) for t in (tile_max, tile_sum, tile_output))
        return None, tuple(partial.unsqueeze(0) for partial in partials)

    _, partials = for_each_tile(
        query_body,
        (query_indices, page_indices, query_bounds, key_offsets, query, k_pages, v_pages),
        dims=(0, 0, 0, 0, None, None, None),
        tile_size=1,
        out_dim=0,
    )
    return partials


def jagged_decode_merge_kernel(
    partial_max,
    partial_sum,
    partial_output,
    output_indices,
    query_capacity: int,
    block_size: int,
):
    tasks, chunks, entries, num_kv_heads, group_size, head_size = partial_output.shape
    groups = output_indices.shape[0]
    queries_per_group = output_indices.shape[1]
    pages = tasks * chunks * entries // (groups * queries_per_group)
    sum_scale = 1.0 / block_size

    state_shape = (groups, pages, queries_per_group, num_kv_heads, group_size, 1)
    maxima = partial_max.reshape(state_shape)
    sums = partial_sum.reshape(state_shape)
    numerators = partial_output.reshape(*state_shape[:-1], head_size)
    merged_max = maxima.amax(dim=1, keepdim=True)
    weights = torch.exp(maxima - merged_max)
    denominator = (weights * sums).sum(dim=1)
    numerator = (weights * numerators).sum(dim=1)
    result = (numerator / denominator.clamp_min(sum_scale)).reshape(
        groups, queries_per_group, num_kv_heads * group_size, head_size
    )
    output = torch.zeros(
        (query_capacity + 1, num_kv_heads * group_size, head_size),
        dtype=partial_output.dtype,
        device=partial_output.device,
    )
    if groups == 1:
        return output.index_copy(0, output_indices[0], result[0])

    # Flattening short index rows would merge separately padded int32 sticks.
    def scatter_group(destination, tiles):
        indices, rows = tiles
        return destination.index_copy(0, indices[0], rows[0]), None

    output, _ = for_each_tile(
        scatter_group,
        (output_indices, result),
        dims=(0, 0),
        tile_size=1,
        init=output,
    )
    return output
