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

"""Static-capacity page visits for packed, variable-length query sequences."""

import weakref
from collections import OrderedDict
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from spyre_inference.v1.attention.ops.layout import INT32_ELEMS_PER_STICK

QUERY_TILE_SIZE = 64
FP16_ELEMS_PER_STICK = 64
MAX_BLOCK_SIZE = 1024


@dataclass(frozen=True)
class JaggedAttentionPlan:
    query_indices: torch.Tensor
    output_indices: torch.Tensor
    page_indices: torch.Tensor
    query_bounds: torch.Tensor
    key_offsets: torch.Tensor
    first_page: torch.Tensor
    num_work_items: int
    query_capacity: int
    causal: bool
    sliding_window: int | None

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.query_indices,
            self.output_indices,
            self.page_indices,
            self.query_bounds,
            self.key_offsets,
            self.first_page,
        )


@dataclass(frozen=True)
class JaggedTilePlan:
    query_indices: torch.Tensor
    output_indices: torch.Tensor
    page_indices: torch.Tensor
    query_bounds: torch.Tensor
    key_offsets: torch.Tensor
    num_query_tiles: int
    num_work_items: int
    query_capacity: int
    _source_plan: "JaggedTilePlan | None" = field(default=None, repr=False, compare=False)

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.query_indices,
            self.output_indices,
            self.page_indices,
            self.query_bounds,
            self.key_offsets,
        )


@dataclass(frozen=True)
class JaggedMixedPlan:
    groups: tuple[JaggedTilePlan, ...]
    query_capacity: int

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(tensor for group in self.groups for tensor in group.tensors)


@dataclass(frozen=True)
class JaggedPlanVariant:
    query_tile_size: int
    tile_capacity: int
    page_capacity: int


def jagged_plan_variants(
    max_tokens, max_seqs, max_seq_len, block_size, query_capacity, sliding_window=None
):
    """Enumerate the independent group shapes needed by automatic mixed dispatch."""
    max_width = min(512, 1 << ((query_capacity - 1).bit_length() - 1))
    costs: dict[int, dict[int, int]] = {}
    for count in range(1, min(max_tokens, max_seq_len) + 1):
        width = 1 if count <= 8 else min(max_width, max(64, 1 << (count - 1).bit_length()))
        tiles = (count + width - 1) // width
        costs.setdefault(width, {}).setdefault(tiles, count)
    max_pages = (max_seq_len + block_size - 1) // block_size
    if sliding_window is not None:
        max_pages = min(
            max_pages,
            (sliding_window + min(max_tokens, max_seq_len) + block_size - 1) // block_size + 1,
        )
    page_capacities = [1 << power for power in range((max_pages - 1).bit_length() + 1)]
    variants = []
    for width, request_costs in sorted(costs.items()):
        states = {0: 0}
        max_tiles = 0
        for _ in range(max_seqs):
            next_states = {}
            for tiles, cost in states.items():
                for added_tiles, added_cost in request_costs.items():
                    total = cost + added_cost
                    if total <= max_tokens:
                        n = tiles + added_tiles
                        next_states[n] = min(next_states.get(n, max_tokens + 1), total)
            if not next_states:
                break
            max_tiles = max(max_tiles, max(next_states))
            states = next_states
        variants.extend(
            JaggedPlanVariant(width, 1 << power, pages)
            for power in range((max_tiles - 1).bit_length() + 1)
            for pages in page_capacities
        )
    return variants


@dataclass
class _PlanStorage:
    tensors: tuple[torch.Tensor, ...]
    owner: weakref.ReferenceType[JaggedTilePlan] | None = None


class JaggedPlanWorkspace:
    """Lease reusable table storage until its owning plan is released."""

    def __init__(self):
        self._storage: OrderedDict[tuple, list[_PlanStorage]] = OrderedDict()

    def acquire(self, shapes: tuple[tuple[int, ...], ...]) -> _PlanStorage:
        # Retaining a table tensor requires retaining its owning plan.
        entries = self._storage.setdefault(shapes, [])
        self._storage.move_to_end(shapes)
        if len(self._storage) > 8:
            self._storage.popitem(last=False)
        for entry in entries:
            if entry.owner is None or entry.owner() is None:
                return entry
        dtypes = (torch.int32, torch.int32, torch.int32, torch.float16, torch.float16)
        entry = _PlanStorage(
            tuple(
                torch.empty(shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes, strict=True)
            )
        )
        entries.append(entry)
        return entry


def _packed_inputs(
    query_start_loc, seq_lens, block_table, block_size, query_capacity, sliding_window
):
    if block_size <= 0 or block_size % FP16_ELEMS_PER_STICK or block_size > MAX_BLOCK_SIZE:
        raise ValueError("block_size must be a multiple of 64 and at most 1024")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    if query_start_loc.ndim != 1 or seq_lens.ndim != 1:
        raise ValueError("query_start_loc and seq_lens must be one-dimensional")
    starts = query_start_loc.detach().cpu().numpy().astype(np.int64, copy=False)
    lengths = seq_lens.detach().cpu().numpy().astype(np.int64, copy=False)
    pages = block_table.detach().cpu().numpy().astype(np.int64, copy=False)
    if len(starts) != len(lengths) + 1 or not len(starts) or starts[0] != 0:
        raise ValueError(
            "query_start_loc must contain one cumulative length per sequence plus zero"
        )
    query_lens = np.diff(starts)
    if np.any((query_lens < 0) | (query_lens > lengths)):
        raise ValueError("each query length must be between zero and its KV sequence length")
    if pages.ndim != 2 or pages.shape[0] < len(lengths):
        raise ValueError("block_table must have one row per sequence")
    if np.any((lengths + block_size - 1) // block_size > pages.shape[1]):
        raise ValueError("block_table is too short for seq_lens")
    if starts[-1] > query_capacity:
        raise ValueError("query_capacity must cover the packed input")
    return starts, lengths, pages, query_lens


def _build_group_plan(
    inputs,
    sequences,
    block_size,
    query_capacity,
    width,
    tile_capacity,
    page_capacity,
    causal,
    sliding_window,
    max_parallel_entries,
    workspace,
):
    if width <= 0 or (width != 1 and width % INT32_ELEMS_PER_STICK):
        raise ValueError("query tiles must be one row or contain whole int32 index sticks")
    if query_capacity <= width:
        raise ValueError("query_capacity must cover the packed input and exceed one query tile")
    starts, lengths, pages, query_lens = inputs
    counts = (query_lens[sequences] + width - 1) // width
    seqs = np.repeat(sequences, counts)
    tile_starts = np.repeat(np.cumsum(counts) - counts, counts)
    local = (np.arange(len(seqs)) - tile_starts) * width
    q_start = starts[seqs] + local
    q_count = np.minimum(width, query_lens[seqs] - local)
    q_pos = lengths[seqs] - query_lens[seqs] + local
    first_key = (
        np.maximum(0, q_pos - sliding_window + 1)
        if sliding_window is not None
        else np.zeros_like(q_pos)
    )
    first_block = first_key // block_size
    end_key = q_pos + q_count if causal else lengths[seqs]
    page_counts = (end_key + block_size - 1) // block_size - first_block
    real_tiles = len(seqs)
    required_tiles = max(1, real_tiles)
    required_pages = max(1, int(page_counts.max(initial=0)))
    if tile_capacity is None:
        tile_capacity = 1 << (required_tiles - 1).bit_length()
    if page_capacity is None:
        page_capacity = 1 << (required_pages - 1).bit_length()
    if tile_capacity < required_tiles or page_capacity < required_pages:
        raise ValueError("tile and page capacities must cover all query tiles and page visits")

    if max_parallel_entries is None:
        table_shape = (tile_capacity, page_capacity)
        query_shape = output_shape = (tile_capacity, width)
        tile_ids = np.arange(tile_capacity)[:, None]
        page_slots = np.arange(page_capacity)[None, :]
    else:
        entries = min(max_parallel_entries, page_capacity & -page_capacity)
        queries_per_group = min(entries, tile_capacity & -tile_capacity)
        slots = entries // queries_per_group
        groups = tile_capacity // queries_per_group
        chunks = page_capacity // slots
        table_shape = (groups, chunks, entries)
        query_shape = (groups, max(INT32_ELEMS_PER_STICK, entries))
        output_shape = (groups, queries_per_group)
        lane = np.arange(entries)
        tile_ids = (
            np.arange(groups)[:, None, None] * queries_per_group
            + lane[None, None, :] % queries_per_group
        )
        page_slots = (
            np.arange(chunks)[None, :, None] * slots + lane[None, None, :] // queries_per_group
        )

    shapes = (
        query_shape,
        output_shape,
        (*table_shape, INT32_ELEMS_PER_STICK),
        (*table_shape, 2, width),
        (*table_shape, block_size),
    )
    storage = (workspace or JaggedPlanWorkspace()).acquire(shapes)
    q_ids, out_ids, page_ids, q_bounds, k_offsets = (t.numpy() for t in storage.tensors)
    fallback_row = int(starts[sequences[0]]) if len(sequences) else 0
    q_ids.fill(fallback_row)
    page_ids.fill(0)
    q_bounds.fill(-1)
    k_offsets.fill(-1)
    q_lanes = np.arange(width)
    output_rows = np.broadcast_to(query_capacity + q_lanes, (tile_capacity, width)).copy()
    query_rows = np.full((tile_capacity, width), fallback_row, dtype=np.int32)
    valid_q = q_lanes[None, :] < q_count[:, None]
    query_rows[:real_tiles] = np.where(valid_q, q_start[:, None] + q_lanes, fallback_row)
    output_rows[:real_tiles] = np.where(valid_q, query_rows[:real_tiles], query_capacity + q_lanes)
    out_ids[:] = output_rows.reshape(output_shape)
    if max_parallel_entries is None:
        q_ids[:] = query_rows
    else:
        q_ids[:, :entries] = query_rows.reshape(groups, queries_per_group)[
            :, lane % queries_per_group
        ]

    if real_tiles:
        # Positions stay int64 until page-relative clipping, preserving 128K
        # contexts without asking DL16 to represent absolute positions.
        pad = tile_capacity - real_tiles
        seqs = np.pad(seqs, (0, pad))
        q_count = np.pad(q_count, (0, pad))
        q_pos = np.pad(q_pos, (0, pad))
        first_block = np.pad(first_block, (0, pad))
        page_counts = np.pad(page_counts, (0, pad))
        active = (tile_ids < real_tiles) & (page_slots < page_counts[tile_ids])
        logical = first_block[tile_ids] + page_slots
        safe_logical = np.where(active, logical, 0)
        physical = pages[seqs[tile_ids], safe_logical]
        if np.any(active & (physical < 0)):
            raise ValueError("active block_table entries must be nonnegative")
        page_ids[:] = np.where(active, physical, 0)[..., None]
        live_rows = active[..., None] & (q_lanes < q_count[tile_ids][..., None])
        relative = q_pos[tile_ids][..., None] + q_lanes - logical[..., None] * block_size
        upper = (
            np.clip(relative, -1, block_size - 1).astype(np.float32) if causal else block_size - 1
        )
        q_bounds[..., 0, :] = np.where(live_rows, upper, -1)
        if sliding_window is not None:
            lower = np.clip(relative - sliding_window, -1, block_size - 1).astype(np.float32)
            q_bounds[..., 1, :] = np.where(live_rows, lower, -1)
        k_lanes = np.arange(block_size)
        valid_keys = lengths[seqs[tile_ids]] - logical * block_size
        k_offsets[:] = np.where(
            active[..., None] & (k_lanes < valid_keys[..., None]),
            k_lanes.astype(np.float16),
            np.float16(-1),
        )

    plan = JaggedTilePlan(
        query_indices=storage.tensors[0],
        output_indices=storage.tensors[1],
        page_indices=storage.tensors[2],
        query_bounds=storage.tensors[3],
        key_offsets=storage.tensors[4],
        num_query_tiles=real_tiles,
        num_work_items=int(page_counts.sum()),
        query_capacity=query_capacity,
    )
    storage.owner = weakref.ref(plan)
    return plan


def build_jagged_mixed_plan(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    *,
    query_capacity: int,
    query_tile_size: int = 0,
    causal: bool = True,
    sliding_window: int | None = None,
    decode_threshold: int = 8,
    max_parallel_entries: int = INT32_ELEMS_PER_STICK,
    workspace: JaggedPlanWorkspace | None = None,
) -> JaggedMixedPlan:
    """Build final query-width groups directly in the original packed row space."""
    if query_tile_size not in (0, 64, 128, 256, 512):
        raise ValueError("mixed query tiles must be automatic or 64/128/256/512 rows")
    if decode_threshold < 1:
        raise ValueError("decode_threshold must be positive")
    if max_parallel_entries < 1 or max_parallel_entries & (max_parallel_entries - 1):
        raise ValueError("max_parallel_entries must be a positive power of two")
    inputs = _packed_inputs(
        query_start_loc, seq_lens, block_table, block_size, query_capacity, sliding_window
    )
    query_lens = inputs[3]
    max_width = max(64, min(512, 1 << (max(query_capacity - 1, 1).bit_length() - 1)))
    groups: dict[int, list[int]] = {}
    for seq, count in enumerate(query_lens.tolist()):
        if not count:
            continue
        width = (
            1
            if count <= decode_threshold
            else query_tile_size or min(max_width, max(64, 1 << (count - 1).bit_length()))
        )
        groups.setdefault(width, []).append(seq)
    plans = [
        _build_group_plan(
            inputs,
            np.array(seqs),
            block_size,
            query_capacity,
            width,
            None,
            None,
            causal,
            sliding_window,
            max_parallel_entries if width == 1 else None,
            workspace,
        )
        for width, seqs in sorted(groups.items())
    ]
    return JaggedMixedPlan(tuple(plans), query_capacity)


def build_jagged_tile_plan(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    *,
    query_capacity: int,
    query_tile_size: int = QUERY_TILE_SIZE,
    tile_capacity: int | None = None,
    page_capacity: int | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
    workspace: JaggedPlanWorkspace | None = None,
) -> JaggedTilePlan:
    """Build nested query/page tables without materializing a flat visit plan."""
    inputs = _packed_inputs(
        query_start_loc, seq_lens, block_table, block_size, query_capacity, sliding_window
    )
    return _build_group_plan(
        inputs,
        np.arange(len(inputs[1])),
        block_size,
        query_capacity,
        query_tile_size,
        tile_capacity,
        page_capacity,
        causal,
        sliding_window,
        None,
        workspace,
    )


def build_jagged_decode_plan(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    *,
    query_capacity: int,
    query_tile_size: int = 1,
    tile_capacity: int | None = None,
    page_capacity: int | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
    max_parallel_entries: int = INT32_ELEMS_PER_STICK,
    workspace: JaggedPlanWorkspace | None = None,
) -> JaggedTilePlan:
    """Write grouped query/chunk/entry tables directly into their final layout."""
    if query_tile_size != 1:
        raise ValueError("page-parallel attention requires one-row query tiles")
    if max_parallel_entries < 1 or max_parallel_entries & (max_parallel_entries - 1):
        raise ValueError("max_parallel_entries must be a positive power of two")
    inputs = _packed_inputs(
        query_start_loc, seq_lens, block_table, block_size, query_capacity, sliding_window
    )
    return _build_group_plan(
        inputs,
        np.arange(len(inputs[1])),
        block_size,
        query_capacity,
        1,
        tile_capacity,
        page_capacity,
        causal,
        sliding_window,
        max_parallel_entries,
        workspace,
    )


def partition_jagged_decode_plan(plan: JaggedTilePlan, pages_per_partition: int) -> JaggedTilePlan:
    """Group independent page partitions as query tasks for the partials kernel."""
    if pages_per_partition < 1 or pages_per_partition & (pages_per_partition - 1):
        raise ValueError("pages_per_partition must be a positive power of two")
    if plan.page_indices.ndim != 4:
        raise ValueError("split decode requires a grouped decode plan")
    if pages_per_partition == 1:
        return plan
    groups, chunks, entries, width = plan.page_indices.shape
    steps = min(pages_per_partition, chunks & -chunks)
    partitions = chunks // steps
    tasks = groups * partitions
    return replace(
        plan,
        query_indices=plan.query_indices.repeat_interleave(partitions, dim=0),
        page_indices=plan.page_indices.reshape(tasks, steps, entries, width),
        query_bounds=plan.query_bounds.reshape(tasks, steps, entries, 2, 1),
        key_offsets=plan.key_offsets.reshape(tasks, steps, entries, -1),
        _source_plan=plan,
    )


def build_jagged_plan(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    *,
    query_capacity: int,
    query_tile_size: int = QUERY_TILE_SIZE,
    work_capacity: int | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
) -> JaggedAttentionPlan:
    """Build flat page visits over packed token rows."""
    if block_size <= 0 or query_tile_size <= 0:
        raise ValueError("block_size and query_tile_size must be positive")
    if query_tile_size != 1 and query_tile_size % INT32_ELEMS_PER_STICK:
        raise ValueError("query tiles must be one row or contain whole int32 index sticks")
    if block_size % INT32_ELEMS_PER_STICK:
        raise ValueError("KV tiles must contain whole int32 index sticks")
    if block_size % FP16_ELEMS_PER_STICK or block_size > MAX_BLOCK_SIZE:
        raise ValueError("block_size must be a multiple of 64 and at most 1024")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    if query_start_loc.ndim != 1 or seq_lens.ndim != 1:
        raise ValueError("query_start_loc and seq_lens must be one-dimensional")
    starts = query_start_loc.to(device="cpu", dtype=torch.int64).tolist()
    lengths = seq_lens.to(device="cpu", dtype=torch.int64).tolist()
    pages = block_table.to(device="cpu", dtype=torch.int64)
    if len(starts) != len(lengths) + 1 or not starts or starts[0] != 0:
        raise ValueError(
            "query_start_loc must contain one cumulative length per sequence plus zero"
        )
    if pages.ndim != 2 or pages.shape[0] < len(lengths):
        raise ValueError("block_table must have one row per sequence")
    if query_capacity <= query_tile_size or starts[-1] > query_capacity:
        raise ValueError("query_capacity must cover the packed input and exceed one query tile")

    visits: list[tuple[int, int, int, int, int, int, bool, bool]] = []
    for seq, kv_len in enumerate(lengths):
        q_len = starts[seq + 1] - starts[seq]
        if not 0 <= q_len <= kv_len:
            raise ValueError("each query length must be between zero and its KV sequence length")
        if (kv_len + block_size - 1) // block_size > pages.shape[1]:
            raise ValueError("block_table is too short for seq_lens")
        context = kv_len - q_len
        for local_start in range(0, q_len, query_tile_size):
            count = min(query_tile_size, q_len - local_start)
            q_pos = context + local_start
            first_key = max(0, q_pos - sliding_window + 1) if sliding_window is not None else 0
            end_key = q_pos + count if causal else kv_len
            first_block = first_key // block_size
            end_block = (end_key + block_size - 1) // block_size
            for logical_page in range(first_block, end_block):
                physical_page = int(pages[seq, logical_page])
                if physical_page < 0:
                    raise ValueError("active block_table entries must be nonnegative")
                visits.append(
                    (
                        starts[seq] + local_start,
                        count,
                        q_pos,
                        kv_len,
                        physical_page,
                        logical_page * block_size,
                        logical_page == first_block,
                        logical_page == end_block - 1,
                    )
                )

    required = max(1, len(visits))
    if work_capacity is None:
        work_capacity = 1 << (required - 1).bit_length()
    if work_capacity < required:
        raise ValueError(f"work_capacity={work_capacity} cannot hold {len(visits)} page visits")

    q_indices = torch.zeros((work_capacity, query_tile_size), dtype=torch.int32)
    sink_rows = torch.arange(query_capacity, query_capacity + query_tile_size, dtype=torch.int32)
    out_indices = sink_rows.expand(work_capacity, -1).clone()
    page_indices = torch.zeros((work_capacity, INT32_ELEMS_PER_STICK), dtype=torch.int32)
    # SEN169 fp16 exactly represents these page-relative bounds. Absolute
    # positions are subtracted on the host, before clipping or conversion.
    q_bounds = torch.full((work_capacity, 2, query_tile_size), -1, dtype=torch.float16)
    k_offsets = torch.full((work_capacity, block_size), -1, dtype=torch.float16)
    first_page = torch.ones((work_capacity, FP16_ELEMS_PER_STICK), dtype=torch.float16)
    q_lanes = torch.arange(query_tile_size, dtype=torch.int32)
    k_lanes = torch.arange(block_size, dtype=torch.int32)
    for step, (q_start, count, q_pos, kv_len, page, k_start, first, last) in enumerate(visits):
        q_indices[step, :count] = q_start + q_lanes[:count]
        relative = q_lanes[:count].to(torch.int64) + (q_pos - k_start)
        upper = relative.clamp(-1, block_size - 1) if causal else block_size - 1
        q_bounds[step, 0, :count] = upper
        if sliding_window is not None:
            q_bounds[step, 1, :count] = (relative - sliding_window).clamp(-1, block_size - 1)
        if last:
            out_indices[step, :count] = q_indices[step, :count]
        page_indices[step].fill_(page)
        valid_keys = min(block_size, kv_len - k_start)
        k_offsets[step, :valid_keys] = k_lanes[:valid_keys]
        first_page[step].fill_(int(first))
    return JaggedAttentionPlan(
        q_indices,
        out_indices,
        page_indices,
        q_bounds,
        k_offsets,
        first_page,
        len(visits),
        query_capacity,
        causal,
        sliding_window,
    )
