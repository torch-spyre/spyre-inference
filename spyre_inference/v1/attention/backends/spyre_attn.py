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

"""Paged KV-cache attention backend for Spyre using a dense page tensor and online softmax."""

import bisect
import contextlib
import functools
import time
from dataclasses import dataclass, field
from typing import ClassVar, NamedTuple

import torch
from torch._dynamo.utils import counters
from vllm.config import CompilationMode, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention import attn_layer
from spyre_inference.v1.attention.spyre_attn_bucketer import (
    SpyreAttnBucket,
    SpyreAttnBucketer,
)

logger = init_logger(__name__)

# When set, wraps forward(), _online_softmax_attention() and the batched
# decode K/V/mask gather blocks in torch.profiler.record_function spans for
# kineto trace capture. Off by default: the spans are not free, so a profiled
# run is not wall-clock comparable to a default one.
_ATTN_PROFILING = envs.SPYRE_ATTN_PROFILING


def _record_function(name: str):
    def decorator(fn):
        if not _ATTN_PROFILING:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with torch.profiler.record_function(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


@contextlib.contextmanager
def _record_block(name: str):
    """Gated counterpart to _record_function for inline blocks.

    Same SPYRE_ATTN_PROFILING gate; a no-op when profiling is off so the
    span carries no cost on the default path.
    """
    if not _ATTN_PROFILING:
        yield
        return
    with torch.profiler.record_function(name):
        yield


# Elements per stick for int32 (128-byte stick / 4 bytes). Page-index rows are
# padded to this width so each row starts on a stick boundary; see
# SpyreAttentionMetadata.page_index_tables.
INT32_ELEMS_PER_STICK = 32


# Batches below this fall back to the per-seq loop: the batched matmul's
# padded-row overhead exceeds the per-seq cost at small N.
_MIN_BATCHED_SEQS = 4


def _powers_of_two_up_to(n: int, start: int = 1) -> tuple[int, ...]:
    """Powers of 2 in [start, n], plus n itself if it is not already a power of 2.

    ``start`` is rounded up to a power of 2 first, keeping a pure doubling
    sequence. A ``start`` above ``n`` yields just ``(n,)``.
    """
    if n < 1:
        return ()
    v = 1
    while v < start:
        v *= 2
    result = []
    while v < n:
        result.append(v)
        v *= 2
    result.append(n)
    return tuple(result)


def _find_bucket(n: int, buckets: tuple[int, ...]) -> int | None:
    """Smallest bucket >= n, or None when n exceeds the top bucket."""
    idx = bisect.bisect_left(buckets, n)
    if idx < len(buckets):
        return buckets[idx]
    return None


class SpyrePagedKVCache(NamedTuple):
    """Per-layer paged KV cache for the Spyre backend.

    Each field is one dense tensor of shape
    [num_blocks, block_size, num_kv_heads, head_size] on the Spyre device,
    matching `SpyreAttentionBackend.get_kv_cache_shape`.

    NamedTuple (not dataclass) because it is a tuple at runtime, so unpacking
    (`k_pages, v_pages = cache`) traces cleanly under Dynamo without relying on
    attribute access on a custom object.

    Allocated by `TorchSpyreModelRunner.initialize_kv_cache_tensors` and
    consumed by `SpyreAttentionImpl.forward`. vLLM's `bind_kv_cache` types
    the relay path as `dict[str, torch.Tensor]`; see the suppression at the
    `bind_kv_cache(...)` call site for why that type-hole is benign.
    """

    k_pages: torch.Tensor
    v_pages: torch.Tensor


def slot_major_kv_layout(num_slots: int, num_kv_heads: int, head_size: int, dtype: torch.dtype):
    """Slot-axis-outermost layout. The default tiled layout spreads the slot index
    across two device dims, making the indirect store write to the wrong rows
    (torch-spyre#3705)."""
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick

    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_slots, num_kv_heads, sticks, eps],
        stride_map=[num_kv_heads * sticks * eps, sticks * eps, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


def _reshape_and_cache_kernel(key, value, k_slots, v_slots, slot_mapping):
    k_slots.index_copy_(0, slot_mapping, key)
    v_slots.index_copy_(0, slot_mapping, value)


def _alibi_tile_shape(
    num_kv_heads: int, num_queries_per_kv: int, block_size: int
) -> tuple[int, int, int, int]:
    """Shape of one per-block ALiBi bias tile; see _online_softmax_attention's derivation."""
    return (num_kv_heads, num_queries_per_kv, 1, block_size)


def _mirror_mask_tiles(
    tiles_cpu: list[list[torch.Tensor]], device: torch.device
) -> list[list[torch.Tensor]]:
    """Mirror per-block mask tiles to `device`, one transfer per distinct tile.

    `_get_zero_tile` hands the same CPU tensor to every interior block, so
    keying on `id()` collapses those to a single H2D transfer instead of one
    per block. `tiles_cpu` keeps strong references for the whole call, so no
    id can be recycled mid-flight, and sharing one device buffer across blocks
    is safe because mask tiles are read-only by contract (see
    `_get_zero_tile`).
    """
    mirrored: dict[int, torch.Tensor] = {}
    tiles_device: list[list[torch.Tensor]] = []
    for seq_tiles in tiles_cpu:
        row: list[torch.Tensor] = []
        for tile in seq_tiles:
            dev_tile = mirrored.get(id(tile))
            if dev_tile is None:
                dev_tile = convert(tile, device=device)
                mirrored[id(tile)] = dev_tile
            row.append(dev_tile)
        tiles_device.append(row)
    return tiles_device


# ---------------------------------------------------------------------------
# Compilable kernels
# ---------------------------------------------------------------------------


def _stick_aligned_len(n: int) -> int:
    """Round n up to a whole number of int32 sticks (see INT32_ELEMS_PER_STICK)."""
    return (n + INT32_ELEMS_PER_STICK - 1) // INT32_ELEMS_PER_STICK * INT32_ELEMS_PER_STICK


def _build_query_row_tables(
    attn_metadata: "SpyreAttentionMetadata", device: torch.device
) -> list[torch.Tensor]:
    """Build query gather/dest row tables for the whole batch.

    Builds one contiguous CPU tensor for the whole batch and mirrors it to the
    Spyre device with a single convert() call, then clones per-sequence rows so
    each compiled kernel sees an offset-0 buffer (torch-spyre#3770).
    """
    num_seqs = attn_metadata.num_seqs
    starts = attn_metadata.query_start_loc[:num_seqs].cpu()
    lens = attn_metadata.query_start_loc[1 : num_seqs + 1].cpu() - starts
    aligned_query_lens = attn_metadata.aligned_query_lens
    max_index_len = max((_stick_aligned_len(al) for al in aligned_query_lens), default=0)
    rows = torch.zeros(num_seqs, max_index_len, dtype=torch.int32)
    for s, aligned in enumerate(aligned_query_lens):
        last_real = max(int(lens[s]) - 1, 0)
        rows[s, :aligned] = (starts[s] + torch.arange(aligned).clamp(max=last_real)).to(torch.int32)
    rows_dev = convert(rows, device=device)
    # Per-seq clones keep offset 0 for the compiled kernel (torch-spyre#3770), each
    # narrowed to the width the recorder traced for that query length, not the batch max.
    return [
        rows_dev[s, : _stick_aligned_len(aligned_query_lens[s])].clone() for s in range(num_seqs)
    ]


def _page_attn_kernel(
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

    for i in range(num_blocks):
        # index_select, not `k_pages[page_idx]`: subscripting lowers to
        # aten.index, which upcasts the int32 index to int64 and fails eager.
        page_idx = page_index_table[i, 0:1]
        k_page = k_pages.index_select(0, page_idx)
        v_page = v_pages.index_select(0, page_idx)
        # Token-major page to head-major for the matmuls; permutes on device.
        k_page_4d = k_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)
        v_page_4d = v_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)

        mask_tile = mask_tiles[i]

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
            scores = scores + alibi_bias_tiles[i]
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = torch.matmul(tile_probs, v_page_4d)
            tile_sum = tile_probs.sum(dim=-1, keepdim=True)
        else:
            # i > 0 only reachable after the i == 0 branch initialized these.
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


def _batched_decode_kernel(
    query,
    query_row_ids,
    k_pages,
    v_pages,
    block_ids,
    mask_by_block,
    scale,
    num_seqs,
    num_blocks,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
    logits_soft_cap=0.0,
    out=None,
):
    """Batched decode kernel; gathers K/V and the query in-graph.

    Gathers one block at a time; block_ids rows must stay stick-aligned, since a
    flat per-block slice does not compile.

    k/v_pages: [num_pages_total, block_size, KV, D] (the raw page cache).
    block_ids: [num_blocks, stick-padded num_seqs] int32, row i holding the
    i-th block's page index per sequence. mask_by_block:
    [num_blocks, num_seqs * KV, 1, block_size], pre-broadcast across KV heads
    by the builder. ``query_row_ids`` None takes the query buffer's first
    num_seqs rows instead of gathering; ``out`` None returns the result instead
    of storing it.
    """
    num_heads = num_kv_heads * num_queries_per_kv
    # Q=1 puts the sequences in rows 0..num_seqs-1; lanes past the batch are
    # -inf-masked and dropped by the caller, so any b_seqs-row prefix serves.
    q_rows = query[:num_seqs] if query_row_ids is None else query.index_select(0, query_row_ids)
    # lower_bmm's 4-D form takes two batch axes, so num_seqs and KV stay separate.
    q = q_rows.reshape(num_seqs, num_kv_heads, num_queries_per_kv, head_size)

    tile_max = None
    tile_sum = None
    tile_output = None

    for i in range(num_blocks):
        # index_select, not `k_pages[page_idx]`: subscripting lowers to
        # aten.index, which upcasts the int32 index to int64 and fails eager.
        page_idx = block_ids[i, 0:num_seqs]
        # Token-major cache page to head-major; a view, so do not add
        # .contiguous() -- merging these axes is what materializes the page.
        k_page = k_pages.index_select(0, page_idx).permute(0, 2, 1, 3)
        v_page = v_pages.index_select(0, page_idx).permute(0, 2, 1, 3)
        # Builder already broadcast across KV heads; split them back out.
        mask_tile = mask_by_block[i].reshape(num_seqs, num_kv_heads, 1, block_size)

        scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Before the mask add: tanh(-inf/cap)*cap is -cap, not -inf, so
            # capping after it would un-mask the padded lanes.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = torch.matmul(tile_probs, v_page)
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
            tile_output += torch.matmul(tile_probs, v_page)
            tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
    attn = (tile_output / tile_sum).reshape(num_seqs, num_heads, head_size)
    if out is not None:
        # The destination prefix starts at offset 0, so torch-spyre#3770 does not
        # apply; rows past the batch are don't-care and kept finite by the builder.
        out[:num_seqs].copy_(attn)
        return out
    return attn


# Attention compiles separately from the model's fullgraph capture, which can't
# hold the per-sequence Python loop around these.
_page_attn_compiled = torch.compile(_page_attn_kernel, dynamic=False)
_batched_decode_compiled = torch.compile(_batched_decode_kernel, dynamic=False)

_warmup_complete = False


def mark_warmup_complete() -> None:
    """Arm the late-compile warning, once warmup has claimed full variant coverage."""
    global _warmup_complete
    _warmup_complete = True


def _call_kernel(label: str, fn, *args):
    """Dispatch a kernel, warning if it compiles once warmup has claimed coverage.

    Dynamo's counter is process-wide but attributable across just this call: a
    compiled region runs no eager ops, and torch-spyre compiles every eager aten op.
    That assumes nothing else compiles concurrently on another thread, which holds for
    a single-tenant serving process; if it ever stops holding, the cost is a spurious
    warning, not a wrong result.
    """
    if not _warmup_complete:
        return fn(*args)
    before = counters["stats"]["unique_graphs"]
    result = fn(*args)
    if counters["stats"]["unique_graphs"] != before:
        logger.warning_once(
            "%s compiled outside warmup, which costs a full Inductor compile mid-request. "
            "Re-run with TORCH_LOGS=recompiles to see which guard failed.",
            label,
        )
    return result


@dataclass
class SpyreAttentionMetadata(AttentionMetadata):
    """Metadata for paged online-softmax attention on Spyre."""

    # Total real (non-padding) tokens across all sequences. Used to slice
    # q/k/v to actual tokens before processing (input may have padding).
    num_actual_tokens: int

    # Number of sequences in this batch.
    num_seqs: int

    # Maximum query length among all sequences (raw, unaligned).
    max_query_len: int

    # Maximum KV sequence length among all sequences (raw, unaligned).
    max_seq_len: int

    # Per-sequence KV lengths. [num_seqs]
    seq_lens: torch.Tensor

    # Cumulative query lengths for varlen layout. query_start_loc[i]
    # is the start offset of sequence i in the flat q/k/v buffer.
    # [num_seqs + 1], last entry = total tokens.
    query_start_loc: torch.Tensor

    # Block table mapping logical blocks to physical pages.
    # [num_seqs, max_num_blocks_per_seq]
    block_table: torch.Tensor

    # Number of KV tokens per physical page.
    block_size: int

    # Flat mapping from token index to its position in the KV cache
    # (physical_block_index * block_size + block_offset). [num_actual_tokens]
    slot_mapping: torch.Tensor

    # True when causal masking is needed (prefill/mixed, i.e. max_query_len > 1).
    # Decode steps (max_query_len=1) don't need explicit causal masking because
    # the online softmax over KV pages naturally only attends to past tokens.
    apply_causal_mask: bool = False

    # Number of KV heads (for GQA).
    num_kv_heads: int = 0

    # Number of query heads.
    num_heads: int = 0

    # Pre-tiled additive attention mask. attention_mask_tiles[seq_idx][i]
    # gives the mask tile for the i-th ACTIVE block of one sequence (indexed
    # by position within active_block_indices[seq_idx], not by absolute block
    # index). Each tile: [aligned_query_lens[seq_idx], block_size] on CPU. When
    # sliding_window is None, active == all blocks and the layout is
    # equivalent to indexing by absolute block index.
    attention_mask_tiles: list[list[torch.Tensor]] | None = None

    # For each sequence: absolute block indices whose mask is not fully
    # `-inf` (blocks that contribute to at least one query's attention).
    # None means all blocks are active (sliding_window is None, or the
    # window covers the whole sequence). When set, len(active_block_indices[s])
    # matches len(attention_mask_tiles[s]).
    active_block_indices: list[list[int]] | None = None

    # Per-sequence query_len rounded up onto the bucketer's query buckets
    # (1 for a decoding sequence), for stable kernel compilation.
    aligned_query_lens: list[int] = field(default_factory=list)

    # Per-sequence padded active-block count, rounded up onto the recorder's
    # buckets; equals len(attention_mask_tiles[s]). None on the sliding-window
    # path, which is left unpadded (see build()).
    padded_num_blocks: list[int] | None = None

    # Gather indices for the paged attention loop, one row per active block:
    # [num_seqs, max_active_blocks, INT32_ELEMS_PER_STICK] int32 with the page
    # index at [s, b, 0]. Each index needs its own stick-wide row to compile,
    # which is why block_table cannot serve as the index. The device mirror is
    # filled by the first forward(), since the builder's device is CPU.
    # One table per sequence at its own active-block count, materialized once per
    # step: a batch-max width would put max(num_active) into the kernel's guards.
    page_index_tables_cpu: list[torch.Tensor] | None = None
    page_index_tables: list[torch.Tensor] | None = None

    # Absolute query rows per sequence: gather sources in `query`, and store
    # destinations in `output`. One offset-0 tensor each, as above. Rows past
    # query_len repeat the sequence's last real row; the mask discards them.
    query_row_tables: list[torch.Tensor] | None = None

    # Device mirror of attention_mask_tiles, filled once per step by forward().
    attention_mask_tiles_device: list[list[torch.Tensor]] | None = None

    # Batched-decode precomputes. None-valued when the batch is ineligible
    # (callers fall back to the per-seq loop). query_row_ids is int64 because
    # Spyre's index_copy_ requires int64. mask_by_block is pre-permuted for cheap
    # axis-0 slicing in the dispatch.
    padded_num_seqs: int | None = None
    padded_batch_blocks: int | None = None
    query_row_ids_cpu: torch.Tensor | None = None  # [B_seqs] int64
    query_row_ids_dev: torch.Tensor | None = None
    block_ids_padded_cpu: torch.Tensor | None = None  # [B_blocks, padded B_seqs] int32
    block_ids_padded_dev: torch.Tensor | None = None
    mask_by_block_cpu: torch.Tensor | None = None  # [B_blocks, B_seqs * KV, 1, block_size] fp16
    mask_by_block_dev: torch.Tensor | None = None

    # Encoder scatter dest ``[T]`` (int32 on Spyre) and gather unpack.
    # Filled on the first layer of a step (page_index_tables pattern).
    encoder_q_pack_idx: torch.Tensor | None = None
    encoder_kv_pack_idx: torch.Tensor | None = None
    encoder_unpack_idx: torch.Tensor | None = None
    # Packed SDPA grid. ``None`` until layer 0; fused B=1 still sets these so
    # later layers skip rebuild. Do not H2D a ``[B, 1, L, L]`` mask — forward
    # only needs this pair plus ``encoder_key_pad_mask``.
    encoder_pack_batch: int | None = None
    encoder_pack_len: int | None = None
    encoder_fused_sdpa: bool = False
    # Host-built dense key-pad ``[B * KV, 1, L, L]`` on the target device.
    # ``None`` on the fused path. ``[BH, 1, 1, L]`` does not broadcast onto
    # encoder scores ``[BH, G, L, L]`` (eager add; query axis ``1 → L``).
    encoder_key_pad_mask: torch.Tensor | None = None
    # Slot-major scatter scratch ``[B*L+1, H, D]``. Alloc once per step; ``zero_``
    # before each pack so pad slots stay empty. K and V must not share a buffer:
    # at ``Hkv == 1`` ``permute.contiguous`` is a no-op view, so packing V into
    # K's workspace would silently overwrite ``k_batched``. Q still differs
    # under GQA.
    encoder_q_workspace: torch.Tensor | None = None
    encoder_kv_workspace: torch.Tensor | None = None
    encoder_v_workspace: torch.Tensor | None = None

    @property
    def query_lens(self) -> torch.Tensor:
        """Per-sequence query lengths, derived from query_start_loc. [num_seqs]"""
        return self.query_start_loc[1:] - self.query_start_loc[:-1]


class SpyreAttentionMetadataBuilder(AttentionMetadataBuilder[SpyreAttentionMetadata]):
    """Builds attention metadata — only the attention mask is precomputed."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # The attn_bucketer below derives its buckets from cache_config's block
        # size, so a disagreement here would miss the recorded set.
        assert kv_cache_spec.block_size == vllm_config.cache_config.block_size, (
            f"kv cache spec block_size={kv_cache_spec.block_size} disagrees with "
            f"cache_config.block_size={vllm_config.cache_config.block_size}; the "
            "attention padding buckets are derived from the latter."
        )
        self.block_size = kv_cache_spec.block_size
        self.head_size = kv_cache_spec.head_size
        self.sliding_window = getattr(kv_cache_spec, "sliding_window", None)
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {self.sliding_window}")

        # Validate block_size alignment: Spyre stick size is 128 bytes (64 fp16 elements).
        # block_size must be a multiple of 64 to avoid restickification errors during
        # torch.compile.
        if self.block_size % 64 != 0:
            raise ValueError(
                f"block_size must be a multiple of 64 for the Spyre paged attention "
                f"backend. Got block_size={self.block_size}, head_size={self.head_size}. "
            )

        model_config = vllm_config.model_config
        self.num_heads = model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        # `model_config.dtype` is typed `ModelDType | torch.dtype`, but
        # `TorchSpyrePlatform.check_and_update_config` rejects anything but
        # `torch.float16`/`torch.bfloat16` upstream, so it's always a real
        # torch.dtype here.
        assert isinstance(model_config.dtype, torch.dtype)
        self.model_dtype: torch.dtype = model_config.dtype

        # Shared zero tiles for interior active blocks, whose mask is all-zeros.
        # Keyed by query width: a mixed batch pads its sequences to several.
        self._zero_tiles: dict[int, torch.Tensor] = {}

        static_ctx = vllm_config.compilation_config.static_forward_context
        self._slot_mapping = attn_layer.install(
            static_ctx[name] for name in layer_names if name in static_ctx
        )

        # Buckets for the batched decode fast path. One compiled kernel
        # per bucket. TODO: expose as engine args if configurability is needed.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_num_blocks_per_seq = (
            model_config.max_model_len + self.block_size - 1
        ) // self.block_size
        self._num_seqs_buckets: tuple[int, ...] = _powers_of_two_up_to(max_num_seqs)
        self._num_blocks_buckets: tuple[int, ...] = _powers_of_two_up_to(max_num_blocks_per_seq)

        # Owned here, not by the recorder, so a bucket build() can emit is
        # always a bucket that was compiled: the warmup recorder reads this
        # same instance back (spyre_model_runner._record_attention_graphs)
        # rather than constructing a second one that could drift.
        self._attn_bucketer = SpyreAttnBucketer(vllm_config)

    def _get_zero_tile(self, aligned_query_len: int) -> torch.Tensor:
        """Return (or create) the shared all-zero mask tile for interior blocks.

        The returned tensor is reused by reference across all interior blocks
        of every sequence padded to this width. Callers must treat it as
        read-only: any in-place mutation would corrupt every interior tile
        simultaneously. This is safe today because attention kernels only read
        mask tiles.
        """
        tile = self._zero_tiles.get(aligned_query_len)
        if tile is None:
            tile = torch.zeros((aligned_query_len, self.block_size), dtype=self.model_dtype)
            self._zero_tiles[aligned_query_len] = tile
        return tile

    def _pad_num_blocks(self, num_blocks: int) -> int:
        """Round an active-block count up onto the recorder's num_blocks buckets.

        block_table is allocated at width ceil(max_model_len / block_size) for
        the engine's lifetime, so the padded bucket always fits.
        """
        if num_blocks == 0:
            # A fully-masked padded tile would divide by a zero softmax
            # denominator, so zero real blocks must stay zero.
            return 0
        padded = SpyreAttnBucketer._round_up(num_blocks, self._attn_bucketer.num_blocks_buckets)
        # Unreachable: the top bucket covers ceil(max_model_len / block_size),
        # and num_blocks here is bounded by the same max_model_len.
        assert padded is not None, (
            f"num_blocks={num_blocks} exceeds the largest recorded bucket "
            f"{self._attn_bucketer.num_blocks_buckets[-1]}, which should cover "
            f"ceil(max_model_len / block_size)."
        )
        assert padded >= num_blocks
        return padded

    def _build_attention_mask(
        self,
        seq_lens: torch.Tensor,
        query_lens: torch.Tensor,
        apply_causal_mask: bool,
        aligned_query_len: int,
        aligned_max_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build additive attention mask on Spyre for the non-sliding-window path.

        Vectorized over the sequences it is given, so a caller with several
        query widths in one batch calls it once per width rather than padding
        every sequence to the widest.

        Sliding-window sequences take a different path: see
        _build_active_tiles_with_skip.

        Returns:
            - mask: [len(seq_lens), aligned_query_len, aligned_max_seq_len] additive mask
        """
        assert self.sliding_window is None

        q_pos = torch.arange(aligned_query_len, device=device)
        kv_pos = torch.arange(aligned_max_seq_len, device=device)

        # Padded query rows are clamped to query_len - 1 rather than masked out, so
        # they reproduce the last real row. _build_query_row_tables clamps the gather
        # identically, so they also receive the same query vector.
        q_pos = torch.minimum(q_pos.unsqueeze(0), (query_lens - 1).clamp(min=0).unsqueeze(1))
        kv_valid = kv_pos.unsqueeze(0) < seq_lens.unsqueeze(1)
        attend = kv_valid.unsqueeze(1).expand(-1, aligned_query_len, -1)

        # Causal mask: prevent attending to future tokens during generation
        if apply_causal_mask:
            context_lens = seq_lens - query_lens
            causal_limit = (context_lens.unsqueeze(1) + q_pos).unsqueeze(2)
            kv_pos_exp = kv_pos.unsqueeze(0).unsqueeze(0)
            causal_ok = kv_pos_exp <= causal_limit
            attend = attend & causal_ok

        # Convert to additive mask: finfo.min for masked positions, 0 for valid
        mask_bool = ~attend

        mask_additive = torch.where(
            mask_bool,
            torch.tensor(torch.finfo(self.model_dtype).min, dtype=self.model_dtype, device=device),
            torch.tensor(0.0, dtype=self.model_dtype, device=device),
        )

        return mask_additive

    def _build_single_tile(
        self,
        block_idx: int,
        kv_len: int,
        query_len: int,
        context_len: int,
        aligned_query_len: int,
        apply_causal_mask: bool,
    ) -> torch.Tensor:
        """Build the additive mask tile for one (sequence, block) pair.

        Returns a [aligned_query_len, block_size] CPU tensor.

        Only called for boundary blocks that require real mask content:
          - lower-boundary blocks (window-start cutoff falls inside them for
            at least one query), and
          - the upper-boundary block (last block: KV padding, plus causal
            during prefill).
        Interior blocks reuse the shared zero tile instead.
        """
        block_size = self.block_size
        mask_min = torch.finfo(self.model_dtype).min

        # KV positions covered by this block. May extend past kv_len (handled
        # by the kv_valid mask below).
        kv_start = block_idx * block_size
        kv_end = kv_start + block_size

        q_pos = torch.arange(aligned_query_len)  # [aligned_query_len]
        kv_pos = torch.arange(kv_start, kv_end)  # [block_size]

        # Padded query rows are clamped to query_len - 1, matching the gather in
        # _build_query_row_tables; see _build_attention_mask.
        q_pos = q_pos.clamp(max=max(query_len - 1, 0))
        kv_valid = kv_pos < kv_len  # [block_size]
        attend = kv_valid.unsqueeze(0).expand(aligned_query_len, -1)  # [Q, B]

        # Causal mask (prefill only): query at absolute position
        # context_len + q_pos can only attend to KV positions <= that value.
        if apply_causal_mask:
            causal_limit = context_len + q_pos  # [aligned_query_len]
            attend = attend & (kv_pos.unsqueeze(0) <= causal_limit.unsqueeze(1))

        # Sliding window: per-query window_start.
        assert self.sliding_window is not None
        abs_q_pos = context_len + q_pos  # [aligned_query_len]
        window_start = (abs_q_pos - self.sliding_window + 1).clamp(min=0)
        attend = attend & (kv_pos.unsqueeze(0) >= window_start.unsqueeze(1))

        mask_bool = ~attend
        return torch.where(
            mask_bool,
            torch.tensor(mask_min, dtype=self.model_dtype),
            torch.tensor(0.0, dtype=self.model_dtype),
        )

    def _build_active_tiles_with_skip(
        self,
        kv_len: int,
        query_len: int,
        context_len: int,
        aligned_query_len: int,
        apply_causal_mask: bool,
    ) -> tuple[list[int], list[torch.Tensor]]:
        """Return (active_block_indices, mask_tiles) using arithmetic block-skip.

        active_block_indices: absolute block indices whose mask contributes
        to at least one query's attention (i.e. inside the window of the
        earliest query).
        mask_tiles: one tile per active block, in the same order.

        Block classification:
          - [0, first_active):
                entirely outside every query's window; skipped.
          - [first_active, last_lower_boundary]:
                lower-boundary blocks — the window cutoff falls inside them
                for at least one query. Real tile with per-query-row cutoffs.
                In decode (query_len == 1) this collapses to a single block.
          - (last_lower_boundary, last_causal_interior]:
                interior blocks — fully inside every query's window AND fully
                below the earliest query's causal limit. Mask is all-zero.
          - (last_causal_interior, last_block):
                causal-boundary blocks — inside every window, but early
                queries have causal cutoffs falling inside them (prefill
                only). Real tile.
          - last_block:
                upper-boundary block — always has KV padding (and causal
                cutoffs during prefill). Real tile.

        When any of the boundary ranges overlap (short kv_len, single-block
        sequence, etc.) real tiles are built for the union — never zero tiles.
        """
        assert self.sliding_window is not None
        block_size = self.block_size
        num_blocks = (kv_len + block_size - 1) // block_size

        # Earliest query (q_pos=0) has window
        # [max(0, context_len - W + 1), context_len].
        # Latest query (q_pos=query_len-1) has window
        # [max(0, kv_len - W), kv_len - 1].
        # A block is fully outside every query's window when its highest KV
        # position is below the earliest query's window start.
        # NOTE: using the EARLIEST query's window (not the latest, kv_len - W)
        # is required for prefill correctness. In a prefill batch with
        # query_len > 1, early queries have earlier windows and their
        # in-window blocks would otherwise be incorrectly dropped. For decode
        # (query_len == 1) both formulas coincide.
        earliest_window_start = max(0, context_len - self.sliding_window + 1)
        latest_window_start = max(0, kv_len - self.sliding_window)

        first_active = earliest_window_start // block_size
        # Every block from first_active up to the block containing the
        # latest window start can have a per-query cutoff falling inside it.
        last_lower_boundary = latest_window_start // block_size
        # A block is fully below the earliest query's causal limit
        # (abs_pos = context_len) iff (b + 1) * block_size - 1 <= context_len.
        # For decode (no causal mask) all blocks satisfy this trivially.
        if apply_causal_mask:
            last_causal_interior = (context_len + 1) // block_size - 1
        else:
            last_causal_interior = num_blocks - 1
        last_block = num_blocks - 1

        active_bs = list(range(first_active, num_blocks))
        if not active_bs:
            return [], []

        zero_tile = self._get_zero_tile(aligned_query_len)
        tiles: list[torch.Tensor] = []

        for b in active_bs:
            is_lower_boundary = b <= last_lower_boundary
            is_upper_boundary = (b == last_block) and not is_lower_boundary
            is_causal_boundary = apply_causal_mask and b > last_causal_interior and b != last_block
            if is_lower_boundary or is_upper_boundary or is_causal_boundary:
                tiles.append(
                    self._build_single_tile(
                        b,
                        kv_len,
                        query_len,
                        context_len,
                        aligned_query_len,
                        apply_causal_mask,
                    )
                )
            else:
                # Interior block: entirely within every query's window,
                # entirely filled with valid KV tokens, and (for prefill)
                # entirely below the earliest query's causal limit.
                # Mask is all-zero.
                tiles.append(zero_tile)

        return active_bs, tiles

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SpyreAttentionMetadata:
        """Build attention metadata from common metadata."""

        seq_lens = common_attn_metadata.seq_lens
        query_start_loc = common_attn_metadata.query_start_loc
        max_seq_len = common_attn_metadata.max_seq_len
        max_query_len = common_attn_metadata.max_query_len
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        causal = common_attn_metadata.causal
        if isinstance(causal, torch.Tensor):
            causal = bool(causal.item())
        # Batch-level flag: True iff the batch contains at least one prefill
        # sequence (max_query_len > 1). For decode sequences (query_len == 1)
        # in a mixed batch, the causal constraint is subsumed by the KV
        # validity mask (the single query at position context_len can only
        # attend to KV positions [0, kv_len) = [0, context_len]), so applying
        # the causal mask to them is a correct no-op.
        apply_causal_mask = causal and max_query_len > 1

        num_seqs = common_attn_metadata.num_reqs
        query_lens = query_start_loc[1 : num_seqs + 1] - query_start_loc[:num_seqs]

        aligned_query_lens: list[int] = []
        for query_len in query_lens.tolist():
            if query_len <= 1:
                aligned_query_lens.append(1)
                continue
            # Round to a recorded bucket so the kernel this sequence needs was
            # already compiled during warmup. The top query bucket is at least
            # max_num_batched_tokens, so a miss here means a batch outside the
            # scheduler's own contract; asserted rather than left unpadded,
            # since this sizes the mask tiles and query-row gather.
            aligned = self._attn_bucketer.find_query_bucket(query_len)
            assert aligned is not None, (
                f"no query bucket for query_len={query_len}; top bucket is "
                f"{self._attn_bucketer.query_buckets[-1]}, which should be at least "
                f"max_num_batched_tokens."
            )
            aligned_query_lens.append(aligned)

        block_size = self.block_size
        attention_mask_tiles: list[list[torch.Tensor]] = []
        active_block_indices: list[list[int]] | None = None

        padded_num_blocks: list[int] | None = None
        real_num_blocks: list[int] = []

        if self.sliding_window is None:
            # seq_lens itself is NOT padded: it feeds the mask's kv_valid cutoff,
            # causal context_len, and ALiBi offset, all of which need the true
            # length. The padded block count is carried separately.
            for s in range(num_seqs):
                n = (int(seq_lens[s].item()) + block_size - 1) // block_size
                real_num_blocks.append(n)
            padded_num_blocks = [self._pad_num_blocks(n) for n in real_num_blocks]

            # Padded tiles need no special construction — kv_valid = kv_pos <
            # seq_lens already emits finfo.min past the true length.
            # Each tile is [aligned_query_lens[s], block_size] -- _record_one
            # builds same-shaped zero tiles by hand; keep them in sync.
            attention_mask_tiles = [[] for _ in range(num_seqs)]
            for aligned_query_len in sorted(set(aligned_query_lens)):
                group = [s for s in range(num_seqs) if aligned_query_lens[s] == aligned_query_len]
                mask_cpu = self._build_attention_mask(
                    seq_lens[group],
                    query_lens[group],
                    # Width 1 is exactly the query_len == 1 sequences, whose causal
                    # constraint is subsumed by the kv_valid cutoff.
                    apply_causal_mask and aligned_query_len > 1,
                    aligned_query_len,
                    # Covers every block forward() iterates for these sequences.
                    max(padded_num_blocks[s] for s in group) * block_size,
                    torch.device("cpu"),
                )
                for row, s in enumerate(group):
                    # `.contiguous()` is a no-op on a [1, N] slice, leaving
                    # stride(0) == the mask width and a nonzero storage offset
                    # reaching a compiled kernel (torch-spyre#3770), so clone.
                    attention_mask_tiles[s] = [
                        mask_cpu[row, :, b * block_size : (b + 1) * block_size].clone(
                            memory_format=torch.contiguous_format
                        )
                        for b in range(padded_num_blocks[s])
                    ]
            # active_block_indices stays None, so forward iterates all blocks.
        else:
            # Sliding window: arithmetic block-skip. Blocks entirely outside
            # every query's window are dropped; interior blocks share a
            # zero mask tile; only boundary blocks get real per-query cutoffs.
            # Left unpadded (padded_num_blocks stays None): len(active_bs) is a
            # window-width quantity, already near-constant across decode steps.
            # TODO: give this its own window-width buckets if it ever needs recording.
            active_block_indices = []
            query_lens_list = query_lens.tolist()
            seq_lens_list = seq_lens.tolist()

            for s in range(num_seqs):
                kv_len_s = int(seq_lens_list[s])
                query_len_s = int(query_lens_list[s])
                context_len_s = kv_len_s - query_len_s

                active_bs, tiles = self._build_active_tiles_with_skip(
                    kv_len_s,
                    query_len_s,
                    context_len_s,
                    aligned_query_lens[s],
                    apply_causal_mask and aligned_query_lens[s] > 1,
                )
                active_block_indices.append(active_bs)
                attention_mask_tiles.append(tiles)

        # Sized per sequence, each its own allocation: a batch-max width becomes a
        # Dynamo guard the key cannot carry, and a fresh buffer matches _record_one's.
        num_active = [len(tiles) for tiles in attention_mask_tiles]
        page_index_tables_cpu = []
        for s, n in enumerate(num_active):
            blocks_s = slice(n) if active_block_indices is None else active_block_indices[s]
            table = torch.zeros(n, INT32_ELEMS_PER_STICK, dtype=torch.int32)
            table[:, 0] = block_table[s, blocks_s]
            page_index_tables_cpu.append(table)

        # Padded to match key/value by upstream once forward_includes_kv_cache_update is
        # False, so the traced write keeps one shape per bucket, not one per token count.
        self._slot_mapping.publish(slot_mapping)

        # Batched-decode precomputes: only when Q=1 and num_seqs is within the
        # buckets. None-valued fields signal fallback. Sliding-window batches are
        # eligible: the kernel reads a precomputed mask, so a window only shrinks
        # the active block set.
        padded_num_seqs = None
        padded_batch_blocks = None
        query_row_ids_cpu = None
        block_ids_padded_cpu = None
        mask_by_block_cpu = None
        if max_query_len == 1 and num_seqs >= _MIN_BATCHED_SEQS:
            # Real counts, not padded: this path has its own buckets, so an
            # inflated count would only push it onto a larger bucket for no
            # reason. Safe because padding only appends blocks. Under a window
            # real_num_blocks is empty and the tiles are the unpadded active
            # blocks, so num_active is already the real count.
            blocks_per_seq = real_num_blocks if active_block_indices is None else num_active
            b_seqs = _find_bucket(num_seqs, self._num_seqs_buckets)
            b_blocks = _find_bucket(max(blocks_per_seq), self._num_blocks_buckets)
            if b_seqs is not None and b_blocks is not None:
                padded_num_seqs = b_seqs
                padded_batch_blocks = b_blocks

                # int64 (not int32): Spyre's index_copy_ requires int64 indices.
                query_row_ids_cpu = torch.zeros(b_seqs, dtype=torch.int64)
                query_row_ids_cpu[:num_seqs] = query_start_loc[:num_seqs].to(torch.int64)
                # Guards the identity scatter used by _run_batched_decode_dispatch.
                assert query_row_ids_cpu[:num_seqs].tolist() == list(range(num_seqs))

                # Rows padded to the stick width: a narrower inner dim emits a
                # Mod(d0, ...) stick expression the inductor rejects.
                block_ids_padded_cpu = torch.zeros(
                    b_blocks, _stick_aligned_len(b_seqs), dtype=torch.int32
                )
                for s, n in enumerate(blocks_per_seq):
                    n_use = min(n, b_blocks)
                    # Position i is the i-th ACTIVE block, matching the mask tiles.
                    blocks_s = (
                        range(n_use)
                        if active_block_indices is None
                        else active_block_indices[s][:n_use]
                    )
                    for b, abs_b in enumerate(blocks_s):
                        block_ids_padded_cpu[b, s] = block_table[s, abs_b]

                # -inf on padded rows/blocks and past-kv-len positions; 0 on
                # valid positions. Broadcast to KV heads and reshape to the
                # kernel input shape [B_blocks, B_seqs * KV, 1, block_size].
                mask_bs_bb = torch.full(
                    (b_seqs, b_blocks, block_size),
                    float("-inf"),
                    dtype=self.model_dtype,
                )
                for s in range(num_seqs):
                    n_use = min(blocks_per_seq[s], b_blocks)
                    for b in range(n_use):
                        mask_bs_bb[s, b] = attention_mask_tiles[s][b][0]
                # A row past the batch is -inf in every block, so its softmax is NaN and
                # the in-graph store would publish it. A real row always has a valid
                # block 0, so its padded blocks can stay -inf and contribute zero.
                # Holds under a window too: first_active <= num_blocks - 1.
                mask_bs_bb[num_seqs:, 0] = torch.finfo(self.model_dtype).min
                # 4-D, not 5-D: the kernel slices dim 0 per block, and a dim-0 slice
                # of a 5-D base fails torch-spyre layout propagation.
                mask_by_block_cpu = (
                    mask_bs_bb.permute(1, 0, 2)
                    .unsqueeze(2)
                    .expand(b_blocks, b_seqs, self.num_kv_heads, block_size)
                    .reshape(b_blocks, b_seqs * self.num_kv_heads, 1, block_size)
                    .contiguous()
                )

        return SpyreAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_seqs=common_attn_metadata.num_reqs,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_table=block_table,
            block_size=self.block_size,
            slot_mapping=slot_mapping,
            apply_causal_mask=apply_causal_mask,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            attention_mask_tiles=attention_mask_tiles,
            active_block_indices=active_block_indices,
            page_index_tables_cpu=page_index_tables_cpu,
            aligned_query_lens=aligned_query_lens,
            padded_num_blocks=padded_num_blocks,
            padded_num_seqs=padded_num_seqs,
            padded_batch_blocks=padded_batch_blocks,
            query_row_ids_cpu=query_row_ids_cpu,
            block_ids_padded_cpu=block_ids_padded_cpu,
            mask_by_block_cpu=mask_by_block_cpu,
        )


class SpyreAttentionBackend(AttentionBackend):
    """Paged KV-cache attention backend for Spyre."""

    accept_output_buffer: bool = True
    # False tells upstream the attention op does not write KV; attn_layer.py does, and
    # upstream inserts its own unified_kv_cache_update for layers attn_layer declines.
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        # For checkpoints that overflow fp16 (see TorchSpyrePlatform._default_dtype).
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Spyre stick size is 128 bytes; tensors are transferred as float16 (2 bytes),
        # so block_size must be a multiple of 64 (= 128 / 2) to satisfy stick alignment.
        # This matches the constraint on head_size in supports_head_size().
        return [MultipleOf(64)]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["SpyreAttentionImpl"]:
        return SpyreAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["SpyreAttentionMetadataBuilder"]:
        return SpyreAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # K and V are separate tensors in SpyrePagedKVCache, each with the same
        # shape. The base vLLM API expects a single tuple here; callers like
        # get_kv_cache_block_dim and KV-transfer code index into it directly.
        return (num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Spyre stick size is 128 bytes; tensors are transferred as float16 (2 bytes),
        # so head_size must be a multiple of 64 (= 128 / 2) to satisfy stick alignment.
        return head_size % 64 == 0

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes


class SpyreAttentionImpl(AttentionImpl[SpyreAttentionMetadata]):
    """Online-softmax paged attention iterating over KV pages.

    KV cache is a tuple (k_pages, v_pages) where each is one dense tensor of
    shape [num_blocks, block_size, num_kv_heads, head_size] on Spyre. Pages are
    read by indirect access, indexing the dense tensor with a device-resident
    page index. No gather masks.

    On Spyre, the per-page attention loop and reshape_and_cache are compiled
    via torch.compile, with their loop counts passed as arguments.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        # `== STOCK`, not `!= NONE`: a bare CompilationConfig (e.g. the unit-test
        # fixture) leaves mode unset (Python None), which `!= NONE` would wrongly
        # treat as compiled. The platform resolves compiled runs to STOCK.
        _mode = get_current_vllm_config().compilation_config.mode
        self._compile_attn = _mode == CompilationMode.STOCK_TORCH_COMPILE

        # Resolved before the ALiBi slopes below, which are built at this dtype.
        # TorchSpyrePlatform.check_and_update_config enforces float16 or bfloat16.
        _dtype = get_current_vllm_config().model_config.dtype
        self.model_dtype: torch.dtype = _dtype if isinstance(_dtype, torch.dtype) else torch.float16

        # ALiBi slopes: per-head linear-bias coefficients (BLOOM/MPT style).
        # Reshape once to [num_kv_heads, num_queries_per_kv, 1, 1] so the
        # per-block bias construction in _online_softmax_attention broadcasts
        # cleanly against the score-tile shape.
        if alibi_slopes is not None:
            slopes_t = torch.tensor(alibi_slopes, dtype=self.model_dtype)
            if slopes_t.numel() != num_heads:
                raise ValueError(
                    f"alibi_slopes must have length num_heads={num_heads}, got {slopes_t.numel()}"
                )
            self.alibi_slopes: torch.Tensor | None = slopes_t.view(
                num_kv_heads, self.num_queries_per_kv, 1, 1
            )
        else:
            self.alibi_slopes = None

        # Normalise the API's Optional[float] into a plain float so the kernel
        # can bake it as a closure constant. logits_soft_cap == 0.0 disables
        # soft-capping (kernel takes the same path as upstream).
        self.logits_soft_cap: float = 0.0 if logits_soft_cap is None else float(logits_soft_cap)

        # Always compiled: eager index_copy_ rejects an int32 index and falls
        # back to CPU with an int64 one.
        self._reshape_fn = torch.compile(_reshape_and_cache_kernel, dynamic=False)

        self._attn_fn = _page_attn_compiled if self._compile_attn else _page_attn_kernel
        self._decode_fn = _batched_decode_compiled if self._compile_attn else _batched_decode_kernel

        self._kv_slots: SpyrePagedKVCache | None = None

        # Constant for the run, so the kernel's arguments never carry the model
        # graph's token count. The +1 keeps every gather a strict subset: selecting
        # a whole source faults the device (torch-spyre#4033).
        self.staging_rows: int = (
            get_current_vllm_config().scheduler_config.max_num_batched_tokens + 1
        )
        self._staging: tuple[torch.Tensor, torch.Tensor] | None = None

        logger.debug_once(
            "Using SpyreAttentionBackend with a dense paged KV cache and indirect page gather"
        )

    def _staging_buffers(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Constant-shaped query and output buffers the kernel is called on.

        The kernel cannot take the caller's buffers: their row count is the model
        graph's token bucket, which Dynamo then guards on, so no recorded variant
        ever matches. Nor can it take a bucket-sized view of them -- a compiled
        kernel reads its arguments from storage offset 0 (torch-spyre#3770), so a
        slice past row 0 reads the wrong storage. Allocated whole (hence at offset
        0) and reused, at one size for the whole run.
        """
        if self._staging is None:
            shape = (self.staging_rows, self.num_heads, self.head_size)
            self._staging = (
                convert(torch.zeros(shape, dtype=self.model_dtype), device=device),
                convert(torch.zeros(shape, dtype=self.model_dtype), device=device),
            )
        return self._staging

    def staging_buffers(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Public accessor so ``attn_layer`` can stage inside the traced graph."""
        return self._staging_buffers(device)

    def _assert_query_fits_staging(self, padded_query_len: int) -> None:
        assert padded_query_len < self.staging_rows, (
            f"padded_query_len={padded_query_len} needs a query buffer wider than "
            "itself; a gather selecting its whole source faults the device"
        )

    def _batched_decode_preconditions_met(self, attn_metadata: "SpyreAttentionMetadata") -> bool:
        # Off by default: the batched matmul pads every sequence row up to the
        # bucket width, and that overhead is uncharacterised at the smallest
        # bucket (num_seqs == _MIN_BATCHED_SEQS), where there is no headroom.
        # Set SPYRE_BATCHED_DECODE=1 to restore the path.
        if not envs.SPYRE_BATCHED_DECODE:
            return False
        # Layer 0's builder gates on max_query_len and the bucket lattice;
        # we add ALiBi, which the batched kernel doesn't implement.
        if attn_metadata.padded_num_seqs is None:
            return False
        return self.alibi_slopes is None

    # `kv_cache` widens the base's `torch.Tensor` to `SpyrePagedKVCache`,
    # which `TorchSpyreModelRunner.initialize_kv_cache_tensors` allocates
    # and `bind_kv_cache` smuggles through a dict typed `dict[str, Tensor]`.
    # The matching pair of overrides preserves the runtime contract; ty
    # cannot see the co-evolution.
    @_record_function("spyre_attn::forward")
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: SpyrePagedKVCache,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output

        k_pages, v_pages = kv_cache
        _target_device = k_pages.device

        # Only the first layer of a step pays for the device mirror.
        if attn_metadata.page_index_tables is None:
            tables_cpu = attn_metadata.page_index_tables_cpu
            assert tables_cpu is not None
            # Fresh offset-0 allocations, matching what _record_one traces.
            attn_metadata.page_index_tables = [
                convert(table, device=_target_device) for table in tables_cpu
            ]
        if attn_metadata.attention_mask_tiles_device is None:
            tiles_cpu = attn_metadata.attention_mask_tiles
            assert tiles_cpu is not None, (
                "attention_mask_tiles must be precomputed by the metadata builder"
            )
            attn_metadata.attention_mask_tiles_device = _mirror_mask_tiles(
                tiles_cpu, _target_device
            )

        # The KV write is not here: attn_layer.py traces it for the layers it splits,
        # and upstream's own unified_kv_cache_update op covers the rest.

        # Mirror batched-decode precomputes to device once per step, only for
        # layers whose impl can actually use the batched kernel (skips ALiBi
        # and soft-cap layers).
        if (
            self._batched_decode_preconditions_met(attn_metadata)
            and attn_metadata.query_row_ids_dev is None
        ):
            assert attn_metadata.query_row_ids_cpu is not None
            assert attn_metadata.block_ids_padded_cpu is not None
            assert attn_metadata.mask_by_block_cpu is not None
            attn_metadata.query_row_ids_dev = convert(
                attn_metadata.query_row_ids_cpu, device=_target_device
            )
            attn_metadata.block_ids_padded_dev = convert(
                attn_metadata.block_ids_padded_cpu, device=_target_device
            )
            attn_metadata.mask_by_block_dev = convert(
                attn_metadata.mask_by_block_cpu, device=_target_device
            )

        output = self._online_softmax_attention(
            query,
            k_pages,
            v_pages,
            attn_metadata,
            output,
            _target_device,
        )

        return output

    def record_graphs(
        self,
        device: torch.device,
        bucketer: "SpyreAttnBucketer",
        kv_cache: SpyrePagedKVCache,
    ) -> int:
        """Compile every attention variant the bucketer enumerates.

        Called from warmup, after the KV cache exists, since the kernel
        ``index_select``s real pages. Dynamo traces on the first *call*, so each
        variant is invoked once here.

        Returns the number of variants invoked. A failing variant is logged and
        skipped, not raised, so it can't take down engine startup; dispatch
        falls back to compiling it on first use.
        """
        if not self._compile_attn:
            return 0

        k_pages, v_pages = kv_cache
        num_pages, block_size = k_pages.shape[0], k_pages.shape[1]
        variants = bucketer.variants()
        t_start = time.time()

        # Belt-and-suspenders: platform._raise_dynamo_recompile_limits already
        # raises this globally, but bump it here too in case that hasn't run.
        prev_limit = torch._dynamo.config.accumulated_recompile_limit
        torch._dynamo.config.accumulated_recompile_limit = max(  # ty: ignore[invalid-assignment]
            prev_limit, 4 * len(variants) + 64
        )

        logger.info("Recording %d attention variants for layer...", len(variants))
        try:
            recorded = self._record_all(variants, k_pages, v_pages, num_pages, block_size, device)
        finally:
            torch._dynamo.config.accumulated_recompile_limit = prev_limit  # ty: ignore[invalid-assignment]

        logger.info(
            "Recorded %d/%d attention variants in %.2fs.",
            recorded,
            len(variants),
            time.time() - t_start,
        )
        return recorded

    def _record_all(
        self,
        variants: "list[SpyreAttnBucket]",
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        num_pages: int,
        block_size: int,
        device: torch.device,
    ) -> int:
        recorded = 0
        for i, bucket in enumerate(variants, start=1):
            if bucket.num_blocks > num_pages:
                # The buckets are sized from max_model_len; a small KV allocation
                # cannot host that many distinct pages to gather.
                continue
            t0 = time.time()
            try:
                self._record_one(bucket, k_pages, v_pages, block_size, device)
            except Exception:
                logger.warning(
                    "Attention variant %s failed to record; it will compile on first use instead.",
                    bucket,
                    exc_info=True,
                )
                continue
            recorded += 1
            logger.debug(
                "  [%d/%d] recorded %s in %.2fs",
                i,
                len(variants),
                bucket,
                time.time() - t0,
            )
        return recorded

    def _record_one(
        self,
        bucket: "SpyreAttnBucket",
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        block_size: int,
        device: torch.device,
    ) -> None:
        """Trace one variant on dummy args matching the kernel's shape contract."""
        q_len = bucket.padded_query_len
        self._assert_query_fits_staging(q_len)

        # Trace on the buffers dispatch passes: Dynamo also guards device layout and
        # dispatch key set, which a same-shaped fresh allocation would not match.
        q_staging, out_staging = self._staging_buffers(device)
        query = q_staging

        # Row table width is stick-aligned exactly as _build_query_row_tables does.
        index_len = _stick_aligned_len(q_len)
        rows = torch.zeros(index_len, dtype=torch.int32)
        rows[:q_len] = torch.arange(q_len, dtype=torch.int32)
        row_table = convert(rows, device=device)

        page_index_table = convert(
            torch.zeros(bucket.num_blocks, INT32_ELEMS_PER_STICK, dtype=torch.int32),
            device=device,
        )

        # All-zero additive tiles: values are irrelevant to tracing, and zero is
        # the one choice that cannot leave a row fully masked (which would make
        # tile_sum 0 and attn NaN).
        mask_tiles = [
            convert(torch.zeros(q_len, block_size, dtype=self.model_dtype), device=device)
            for _ in range(bucket.num_blocks)
        ]

        alibi_bias_tiles = None
        if self.alibi_slopes is not None:
            tile_shape = _alibi_tile_shape(self.num_kv_heads, self.num_queries_per_kv, block_size)
            alibi_bias_tiles = [
                convert(
                    torch.zeros(tile_shape, dtype=self.model_dtype),
                    device=device,
                )
                for _ in range(bucket.num_blocks)
            ]

        self._attn_fn(
            query,
            row_table,
            k_pages,
            v_pages,
            page_index_table,
            mask_tiles,
            self.scale,
            bucket.num_blocks,
            q_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.logits_soft_cap,
            alibi_bias_tiles,
            out_staging,
        )

    def kv_slot_views(self, kv_cache: SpyrePagedKVCache) -> SpyrePagedKVCache:
        """Slot-major views of the pages, built once outside any graph.

        Inductor cannot lower a store through a view of a Spyre-layout tensor created
        inside a graph.
        """
        if self._kv_slots is None:
            k_pages, v_pages = kv_cache
            shape = (-1, k_pages.shape[2], k_pages.shape[3])
            self._kv_slots = SpyrePagedKVCache(k_pages.view(shape), v_pages.view(shape))
        return self._kv_slots

    def do_kv_cache_update(
        self,
        layer: AttentionLayer | None,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: SpyrePagedKVCache,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter new K/V tokens into their cache slots.

        Returns the mutated slot-major K view, which the caller hands to the attention
        op to order the scatter before the read.
        """
        # A source on the wrong device falls back to CPU silently, without raising.
        assert key.device.type == kv_cache[0].device.type, (
            f"kv cache update source is on {key.device.type}, pages on {kv_cache[0].device.type}"
        )

        k_slots, v_slots = self.kv_slot_views(kv_cache)
        # Eager index_copy_ rejects an int32 index and silently falls back to CPU with an
        # int64 one, so this always goes through the compiled artifact.
        self._reshape_fn(key, value, k_slots, v_slots, slot_mapping)
        # Only k_slots is returned, but Inductor fuses both index_copy_ calls into one
        # kernel, so ordering the read after it covers the V write too.
        return k_slots

    def _run_batched_decode_dispatch(
        self,
        query_dev: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        # Spyre-lowering shapes drive several structural choices here:
        # (1) block_ids_padded_cpu rows are stick-padded, so the kernel's row
        # slice emits no Mod(d0, ...) stick expression; (2) K/V/q keep (B_seqs, KV)
        # as the two batch axes lower_bmm allows; (3) the kernel's per-block
        # index stays at Dynamo-trace time (torch-spyre would emit
        # Mod(d0, num_blocks) for a runtime .select); (4) result scatter is a
        # single contiguous copy_ at offset 0, valid because Q=1 forces
        # query_row_ids_cpu[:num_seqs] == range(num_seqs) (asserted in builder).
        b_seqs = attn_metadata.padded_num_seqs
        b_blocks = attn_metadata.padded_batch_blocks
        num_seqs = attn_metadata.num_seqs
        num_heads = self.num_heads
        head_size = self.head_size
        block_size = attn_metadata.block_size

        assert b_seqs is not None and b_blocks is not None
        assert attn_metadata.query_row_ids_dev is not None
        assert attn_metadata.block_ids_padded_dev is not None
        assert attn_metadata.mask_by_block_dev is not None

        # K/V are gathered in-graph by the kernel: an out-of-graph per-block slice
        # has storage_offset > 0, which a compiled kernel reads as 0
        # (torch-spyre#3770), so each block needed a .clone().
        block_ids = attn_metadata.block_ids_padded_dev

        # Short of b_seqs rows only when the runner's compile bucket is tighter than
        # the power-of-two seq bucket; the kernel slices a prefix otherwise.
        needs_gather = query_dev.shape[0] < b_seqs
        # The kernel writes b_seqs rows, so output must have them. Re-checked per call:
        # vLLM hands out a fresh buffer per layer.
        store_out = (
            self._compile_attn
            and not needs_gather
            and output.shape[0] >= b_seqs
            and output.dtype == query_dev.dtype
            and output.storage_offset() == 0
            and output.is_contiguous()
        )
        result = _call_kernel(
            "batched decode attention",
            self._decode_fn,
            query_dev,
            attn_metadata.query_row_ids_dev if needs_gather else None,
            k_pages,
            v_pages,
            block_ids,
            attn_metadata.mask_by_block_dev,
            self.scale,
            b_seqs,
            b_blocks,
            self.num_kv_heads,
            self.num_queries_per_kv,
            block_size,
            self.head_size,
            self.logits_soft_cap,
            output if store_out else None,
        )
        if store_out:
            return

        # Q=1 makes query_row_ids_cpu[:num_seqs] == range(num_seqs), so the
        # scatter is a contiguous prefix write at (0, 0). Neither per-row
        # slice-assign (spyre::copy_from_d2d specialises on (src_off, dst_off)
        # via @compile_once and can return a stale binary) nor index_copy_
        # (CPU-fallback segfaults on vLLM output buffers) is safe here.
        result_flat = result.reshape(b_seqs, num_heads, head_size)
        src_block = result_flat[:num_seqs].clone()
        output[:num_seqs].copy_(src_block)

    @_record_function("spyre_attn::online_softmax")
    def _online_softmax_attention(
        self,
        query_dev: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,
        _target_device: torch.device,
    ) -> torch.Tensor:
        """FlashAttention-style online softmax iterating over KV pages (varlen).

        Handles multiple sequences using query_start_loc for the varlen layout.
        k_pages/v_pages are dense [num_blocks, block_size, num_kv_heads,
        head_size] tensors on Spyre; each iteration gathers one page with a
        one-element int32 device index, then feeds it to bmm without slicing.

        Writes results directly into the caller's output buffer in-place.

        The whole query buffer is passed through; each kernel gathers its own
        sequence's rows and reshapes them to the 4D form it expects.

        Args:
            query_dev: Query on the target device, [num_tokens, num_heads, D].
        """
        block_size = attn_metadata.block_size

        num_seqs = attn_metadata.num_seqs
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        mask_tiles_all = attn_metadata.attention_mask_tiles_device
        active_block_indices_all = attn_metadata.active_block_indices
        padded_num_blocks = attn_metadata.padded_num_blocks
        aligned_query_lens = attn_metadata.aligned_query_lens
        page_index_tables = attn_metadata.page_index_tables
        # Let the kernel write its output buffer directly, saving a copy per layer.
        store_out = self._compile_attn
        assert mask_tiles_all is not None, (
            "attention_mask_tiles_device must be mirrored by forward()"
        )
        assert page_index_tables is not None, "page_index_tables must be mirrored by forward()"

        if self._batched_decode_preconditions_met(attn_metadata):
            self._run_batched_decode_dispatch(query_dev, k_pages, v_pages, attn_metadata, output)
            return output

        # Mirrors the batch layout row for row, so the absolute query_start_loc
        # offsets in the row tables still apply.
        q_staging, out_staging = self._staging_buffers(_target_device)
        # attn_layer stages inside the traced graph where it can be; then the
        # buffers arrive here already staged and both copies are already done.
        pre_staged = query_dev is q_staging
        batch_rows = self.staging_rows
        if not pre_staged:
            batch_rows = query_dev.shape[0]
            assert batch_rows <= self.staging_rows, (
                f"batch has {batch_rows} rows, above max_num_batched_tokens="
                f"{self.staging_rows}; the staging buffers cannot hold it"
            )
            q_staging[:batch_rows] = query_dev
        # Where this loop writes. With a fused store the kernel writes the staging
        # buffer and it is copied back once, after the loop.
        dest = out_staging if store_out else output

        self._assert_query_fits_staging(max(aligned_query_lens, default=1))

        for seq_idx in range(num_seqs):
            # Most-naive implementation: no parallelization
            # over sequences or GQA optimization
            q_start = int(query_start_loc[seq_idx].item())
            q_end = int(query_start_loc[seq_idx + 1].item())
            query_len = q_end - q_start
            kv_len = int(seq_lens[seq_idx].item())

            # Restrict to active (non-fully-masked) blocks when sliding window
            # is set. Otherwise all blocks are active, padded up onto the
            # recorder's buckets by build() (trailing padded blocks are fully
            # masked, hence inert), so the num_blocks key below hits a variant
            # warmup already traced.
            if active_block_indices_all is not None:
                active_bs = active_block_indices_all[seq_idx]
            elif padded_num_blocks is not None:
                active_bs = list(range(padded_num_blocks[seq_idx]))
            else:
                active_bs = list(range((kv_len + block_size - 1) // block_size))

            if len(active_bs) == 0:
                # Every KV position is outside every query's window. Attention
                # over the empty set is undefined; write zeros.
                dest[q_start:q_end] = 0.0
                continue

            # Wider than the kernel's num_blocks, so its shape is a Dynamo guard the
            # cache key misses. Narrowing belongs before the convert, not here.
            page_index_table = page_index_tables[seq_idx]
            # mask_tiles_all[seq_idx] is indexed by position within active_bs.
            mask_tiles = mask_tiles_all[seq_idx][: len(active_bs)]
            # A short slice here would silently hand the kernel a wrong shape.
            assert len(mask_tiles) == len(active_bs)

            # ALiBi bias tiles: slope[h] * (kv_pos - context_len), one per block.
            #
            # The full ALiBi form is slope[h] * (kv_pos - (context_len + q_rel)),
            # which varies over both query and KV positions. The (context_len + q_rel)
            # term is a per-query-row constant, and softmax is invariant under adding
            # any per-row constant to its input (numerator and denominator both pick
            # up the same exp() factor). We therefore drop it and keep only the
            # kv-dependent term — the softmax output is bit-identical to the full
            # form, and each tile stays 1D over KV (block_size floats per head)
            # instead of 2D (aligned_query_len * block_size).
            #
            # Padded blocks get a tile too (the loop iterates active_bs); their
            # values stay finite (slopes are small negative powers of two) and
            # saturate under the mask's finfo.min, so they stay inert.
            #
            # Matches vllm/v1/attention/ops/triton_attention_helpers.py::apply_alibi_to_score
            # (alibi_offset = seq_offset - context_len) — the production Triton path.
            #
            # Shape enforced via _alibi_tile_shape, matching _record_one's dummy tiles.
            alibi_bias_tiles: list[torch.Tensor] | None = None
            if self.alibi_slopes is not None:
                context_len = kv_len - query_len
                alibi_bias_tiles = []
                for b in active_bs:
                    kv_pos = torch.arange(
                        b * block_size,
                        (b + 1) * block_size,
                        dtype=self.model_dtype,
                    )
                    rel = (kv_pos - context_len).view(1, 1, 1, block_size)
                    bias = self.alibi_slopes * rel
                    alibi_bias_tiles.append(convert(bias, device=_target_device))

            if attn_metadata.query_row_tables is None:
                attn_metadata.query_row_tables = _build_query_row_tables(
                    attn_metadata, _target_device
                )
            row_table = attn_metadata.query_row_tables[seq_idx]

            # Run attention on target device
            result = _call_kernel(
                "page attention",
                self._attn_fn,
                q_staging,
                row_table,
                k_pages,
                v_pages,
                page_index_table,
                mask_tiles,
                self.scale,
                len(active_bs),
                aligned_query_lens[seq_idx],
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                self.logits_soft_cap,
                alibi_bias_tiles,
                out_staging if store_out else None,
            )

            assert result.dtype == output.dtype
            if store_out:
                # The kernel wrote `out_staging` itself; copied back below.
                continue
            dest[q_start:q_end] = result[:query_len]

        if store_out and not pre_staged:
            output.copy_(out_staging[:batch_rows])

        return output


def allocate_staging_buffers(
    static_forward_context: dict[str, object], device: torch.device
) -> None:
    """Allocate every attention layer's staging buffers before anything is traced.

    ``attn_layer`` reaches them from inside the block graph, so a lazy allocation
    check would become one of that graph's guards: True while warmup compiles and
    False once serving starts, which recompiles every block.
    """
    for layer in static_forward_context.values():
        impl = getattr(layer, "impl", None)
        if isinstance(impl, SpyreAttentionImpl):
            impl.staging_buffers(device)
