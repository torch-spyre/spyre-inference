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

import gc

import pytest
import torch

from spyre_inference.v1.attention.jagged_plan import (
    JaggedPlanWorkspace,
    build_jagged_decode_plan,
    build_jagged_mixed_plan,
    build_jagged_plan,
    build_jagged_tile_plan,
    partition_jagged_decode_plan,
)


@pytest.mark.parametrize("width", [1, 64, 128])
@pytest.mark.parametrize("window", [None, 129])
@pytest.mark.parametrize("causal", [False, True])
def test_direct_plan_matches_flat_page_visits(width, window, causal):
    starts = torch.tensor([0, 1, 4, 4, 69], dtype=torch.int32)
    lengths = torch.tensor([1025, 777, 0, 413], dtype=torch.int32)
    pages = torch.randperm(64, generator=torch.Generator().manual_seed(9)).reshape(4, 16)
    kwargs = dict(query_capacity=193, query_tile_size=width, causal=causal, sliding_window=window)
    flat = build_jagged_plan(starts, lengths, pages, 128, **kwargs)
    builder = build_jagged_decode_plan if width == 1 else build_jagged_tile_plan
    plan = builder(starts, lengths, pages, 128, **kwargs)
    q_ids, out_ids, page_ids, bounds, offsets = plan.tensors
    if width == 1:
        groups, chunks, entries, stick = page_ids.shape
        queries = out_ids.shape[1]
        slots = entries // queries
        q_ids = q_ids[:, :queries].reshape(-1, 1)
        out_ids = out_ids.reshape(-1, 1)
        page_ids = (
            page_ids.reshape(groups, chunks, slots, queries, stick)
            .permute(0, 3, 1, 2, 4)
            .reshape(-1, chunks * slots, stick)
        )
        bounds = (
            bounds.reshape(groups, chunks, slots, queries, 2, 1)
            .permute(0, 3, 1, 2, 4, 5)
            .reshape(-1, chunks * slots, 2, 1)
        )
        offsets = (
            offsets.reshape(groups, chunks, slots, queries, 128)
            .permute(0, 3, 1, 2, 4)
            .reshape(-1, chunks * slots, 128)
        )

    first = flat.first_page[: flat.num_work_items, 0].nonzero().flatten().tolist()
    last = [*first[1:], flat.num_work_items]
    assert plan.num_query_tiles == len(first)
    assert plan.num_work_items == flat.num_work_items
    for tile, (start, end) in enumerate(zip(first, last, strict=True)):
        torch.testing.assert_close(q_ids[tile], flat.query_indices[start], rtol=0, atol=0)
        torch.testing.assert_close(out_ids[tile], flat.output_indices[end - 1], rtol=0, atol=0)
        for actual, expected in (
            (page_ids, flat.page_indices),
            (bounds, flat.query_bounds),
            (offsets, flat.key_offsets),
        ):
            torch.testing.assert_close(
                actual[tile, : end - start], expected[start:end], rtol=0, atol=0
            )
        assert torch.all(page_ids[tile, end - start :] == 0)
        assert torch.all(bounds[tile, end - start :] == -1)
        assert torch.all(offsets[tile, end - start :] == -1)
    assert torch.all(out_ids[len(first) :] >= 193)
    assert torch.all(bounds[len(first) :] == -1)
    assert torch.all(offsets[len(first) :] == -1)


def test_workspace_reuses_released_plans_and_preserves_live_plans():
    workspace = JaggedPlanWorkspace()
    starts = torch.tensor([0, 1, 2], dtype=torch.int32)
    lengths = torch.tensor([1025, 701], dtype=torch.int32)
    pages = torch.arange(32, dtype=torch.int32).reshape(2, 16)
    kwargs = dict(query_capacity=65, workspace=workspace, max_parallel_entries=64)
    first = build_jagged_decode_plan(starts, lengths, pages, 128, **kwargs)
    first_ids = tuple(t.data_ptr() for t in first.tensors)
    first_values = tuple(t.clone() for t in first.tensors)
    second = build_jagged_decode_plan(starts, lengths + 1, pages + 1, 128, **kwargs)
    assert all(
        a.data_ptr() != b.data_ptr() for a, b in zip(first.tensors, second.tensors, strict=True)
    )
    for actual, expected in zip(first.tensors, first_values, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    del first
    gc.collect()
    recycled = build_jagged_decode_plan(starts, lengths, pages + 2, 128, **kwargs)
    assert tuple(t.data_ptr() for t in recycled.tensors) == first_ids
    independent = build_jagged_decode_plan(
        starts, lengths, pages + 2, 128, query_capacity=65, max_parallel_entries=64
    )
    for actual, expected in zip(recycled.tensors, independent.tensors, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_partition_keeps_workspace_tables_alive():
    workspace = JaggedPlanWorkspace()
    starts = torch.tensor([0, 1, 2], dtype=torch.int32)
    lengths = torch.tensor([1025, 701], dtype=torch.int32)
    pages = torch.arange(32, dtype=torch.int32).reshape(2, 16)
    kwargs = dict(query_capacity=65, workspace=workspace, max_parallel_entries=4)
    source = build_jagged_decode_plan(starts, lengths, pages, 128, **kwargs)
    partitioned = partition_jagged_decode_plan(source, pages_per_partition=2)
    expected = tuple(t.clone() for t in partitioned.tensors)
    del source
    gc.collect()
    replacement = build_jagged_decode_plan(starts, lengths + 1, pages + 1, 128, **kwargs)
    assert replacement.page_indices.data_ptr() != partitioned.page_indices.data_ptr()
    for actual, value in zip(partitioned.tensors, expected, strict=True):
        torch.testing.assert_close(actual, value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "builder", [build_jagged_tile_plan, build_jagged_decode_plan, build_jagged_mixed_plan]
)
def test_direct_plan_rejects_invalid_active_page_but_allows_unused_page(builder):
    starts = torch.tensor([0, 1], dtype=torch.int32)
    lengths = torch.tensor([128], dtype=torch.int32)
    pages = torch.tensor([[2, -1]], dtype=torch.int32)
    builder(starts, lengths, pages, 128, query_capacity=65)
    with pytest.raises(ValueError, match="nonnegative"):
        builder(starts, lengths + 1, pages, 128, query_capacity=65)


@pytest.mark.parametrize("builder", [build_jagged_tile_plan, build_jagged_decode_plan])
def test_direct_empty_plan_has_no_live_output(builder):
    plan = builder(
        torch.tensor([0]),
        torch.empty(0, dtype=torch.int32),
        torch.empty((0, 0), dtype=torch.int32),
        128,
        query_capacity=65,
    )
    assert plan.num_query_tiles == plan.num_work_items == 0
    assert torch.all(plan.output_indices >= 65)
    assert torch.all(plan.query_bounds == -1)
    assert torch.all(plan.key_offsets == -1)
