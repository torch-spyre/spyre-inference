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

"""Packed paged attention with one counted loop over runtime page visits."""

import torch
from torch_spyre._inductor.wsr import for_each_tile


def _compensated_add(total, error, increment, rescale):
    scaled = total * rescale
    delta = increment - error * rescale
    updated = scaled + delta
    return updated, (updated - scaled) - delta


def jagged_page_attn_kernel(
    query,
    k_pages,
    v_pages,
    query_indices,
    output_indices,
    page_indices,
    query_bounds,
    key_offsets,
    first_page,
    scale: float,
    *,
    head_major: bool = False,
    logits_soft_cap: float = 0.0,
):
    """Return packed attention output followed by reserved sink rows."""
    query_capacity, num_heads, head_size = query.shape
    num_kv_heads = k_pages.shape[1] if head_major else k_pages.shape[2]
    group_size = num_heads // num_kv_heads
    query_tile_size = query_indices.shape[1]
    if query_tile_size % 32:
        raise ValueError(
            "flat attention requires whole index sticks; use nested attention for one-row tiles"
        )
    block_size = key_offsets.shape[1]
    state_shape = (num_kv_heads, group_size, query_tile_size)
    state_kwargs = {"dtype": query.dtype, "device": query.device}
    sum_scale = 1.0 / block_size
    floor = torch.finfo(query.dtype).min

    def body(carry, tiles):
        old_max, old_sum, old_sum_error, old_output, old_output_error, destination = carry
        q_ids, out_ids, page_ids, q_bounds, k_pos, first, query, k_pages, v_pages = tiles
        q = query.index_select(0, q_ids[0]).transpose(0, 1)
        q = q.reshape(num_kv_heads, group_size, query_tile_size, head_size)
        page = page_ids[0, 0:1]
        k = k_pages.index_select(0, page).squeeze(0)
        v = v_pages.index_select(0, page).squeeze(0)
        if not head_major:
            k = k.permute(1, 0, 2)
            v = v.permute(1, 0, 2)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        # Two vectors on different stick axes cannot broadcast directly. The
        # full current tile can be restickified; the work tables stay compact.
        upper = q_bounds[0, 0].unsqueeze(1).expand(query_tile_size, block_size) * 1.0
        lower = q_bounds[0, 1].unsqueeze(1).expand(query_tile_size, block_size) * 1.0
        k_pos = k_pos[0].unsqueeze(0) * 1.0
        valid_keys = (k_pos >= 0).reshape(1, 1, -1, 1)
        # Zero weights alone cannot suppress NaNs in unwritten cache slots.
        k = torch.where(valid_keys, k, 0.0)
        v = torch.where(valid_keys, v, 0.0)
        visible = (k_pos <= upper) & (k_pos > lower)

        first = first[0, 0:1].reshape(1, 1, 1)
        tile_max = torch.where(first != 0, floor, old_max)
        tile_sum = torch.where(first != 0, 0.0, old_sum)
        tile_sum_error = torch.where(first != 0, 0.0, old_sum_error)
        tile_output = torch.where(first.unsqueeze(-1) != 0, 0.0, old_output)
        tile_output_error = torch.where(first.unsqueeze(-1) != 0, 0.0, old_output_error)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        scores = torch.where(visible, scores, floor)
        scores_max = scores.amax(dim=-1)
        rescale = torch.exp(-torch.relu(scores_max - tile_max))
        new_max = torch.maximum(tile_max, scores_max)
        probs = torch.where(visible, torch.exp(scores - new_max.unsqueeze(-1)), 0.0)
        # Correct rounding lost by each page addition without widening to fp32.
        # The common scale also keeps a 128K denominator in IEEE fp16's range.
        new_sum, new_sum_error = _compensated_add(
            tile_sum, tile_sum_error, probs.sum(dim=-1) * sum_scale, rescale
        )
        new_output, new_output_error = _compensated_add(
            tile_output,
            tile_output_error,
            torch.matmul(probs, v) * sum_scale,
            rescale.unsqueeze(-1),
        )
        # An all-masked tile has zero weight, including its padded Q lanes.
        result = new_output / new_sum.clamp_min(sum_scale).unsqueeze(-1)
        result = result.reshape(num_heads, query_tile_size, head_size).transpose(0, 1)
        destination = destination.index_copy(0, out_ids[0], result)
        return (new_max, new_sum, new_sum_error, new_output, new_output_error, destination), None

    (_, _, _, _, _, output), _ = for_each_tile(
        body,
        (
            query_indices,
            output_indices,
            page_indices,
            query_bounds,
            key_offsets,
            first_page,
            query,
            k_pages,
            v_pages,
        ),
        dims=(0, 0, 0, 0, 0, 0, None, None, None),
        tile_size=1,
        init=(
            torch.full(state_shape, floor, **state_kwargs),
            torch.zeros(state_shape, **state_kwargs),
            torch.zeros(state_shape, **state_kwargs),
            torch.zeros((*state_shape, head_size), **state_kwargs),
            torch.zeros((*state_shape, head_size), **state_kwargs),
            torch.zeros((query_capacity + query_tile_size, num_heads, head_size), **state_kwargs),
        ),
    )
    return output
