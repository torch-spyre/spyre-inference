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

import functools
import os
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import torch
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

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention import attn_layer

logger = init_logger(__name__)

# When set, wraps forward() and _online_softmax_attention()
# in torch.profiler.record_function spans for kineto trace capture.
_ATTN_PROFILING = os.environ.get("SPYRE_ATTN_PROFILING", "0") == "1"


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


# TODO: Make these hyperparameters configurable
# KV length alignment: KV tensors are padded to the next multiple of this value.
# Because torch.compile treats shapes as static constants, every distinct kv_len
# triggers a full recompile. Aligning to 256 buckets sequence lengths into tiers
# (256, 512, 768, ...) so only the first request at each tier pays compilation cost,
# rather than recompiling on every decode step.
KV_LENGTH_ALIGNMENT = 256

# Query chunk size for padding - ensures consistent tensor sizes for Spyre compilation.
# TODO: decode sequences in a mixed batch still pad to this; only decode-only
# batches skip it.
QUERY_CHUNK_SIZE = 32

# Elements per stick for int32 (128-byte stick / 4 bytes). Page-index rows are
# padded to this width so each row starts on a stick boundary; see
# SpyreAttentionMetadata.page_index_tables.
INT32_ELEMS_PER_STICK = 32


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
    across two device dims, so a 1-KV-head cache cannot be work-divided under the
    256 MB per-core span limit; it also makes the indirect store write to the wrong
    rows (torch-spyre#3705)."""
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick

    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_slots, num_kv_heads, sticks, eps],
        stride_map=[num_kv_heads * sticks * eps, sticks * eps, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


def _maybe_compile(fn, compile_enabled: bool):
    """Compile `fn` when enabled. Attention compiles separately from the model's
    fullgraph capture, which can't hold its per-sequence Python loop.
    """
    if compile_enabled:
        return torch.compile(fn, dynamic=False)
    return fn


def _reshape_and_cache_kernel(key, value, k_slots, v_slots, slot_mapping):
    k_slots.index_copy_(0, slot_mapping, key)
    v_slots.index_copy_(0, slot_mapping, value)


# ---------------------------------------------------------------------------
# Compilable factory functions
# ---------------------------------------------------------------------------


def _build_query_row_tables(
    attn_metadata: "SpyreAttentionMetadata", device: torch.device
) -> list[torch.Tensor]:
    num_seqs = attn_metadata.num_seqs
    aligned = attn_metadata.aligned_max_query_len
    index_len = (
        (aligned + INT32_ELEMS_PER_STICK - 1) // INT32_ELEMS_PER_STICK * INT32_ELEMS_PER_STICK
    )
    starts = attn_metadata.query_start_loc[:num_seqs].cpu()
    lens = attn_metadata.query_start_loc[1 : num_seqs + 1].cpu() - starts
    q_pos = torch.arange(aligned)
    rows = torch.zeros(num_seqs, index_len, dtype=torch.int32)
    rows[:, :aligned] = (
        starts.unsqueeze(1) + torch.minimum(q_pos.unsqueeze(0), (lens - 1).unsqueeze(1))
    ).to(torch.int32)
    return [convert(rows[s].contiguous(), device=device) for s in range(num_seqs)]


def _create_compilable_page_attn(
    num_blocks: int,
    padded_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    has_alibi: bool = False,
    logits_soft_cap: float = 0.0,
    store_mode: str = "none",
    store_len: int = 0,
    needs_gather: bool = True,
):
    """Create online softmax attention over a fixed number of pages for torch.compile.

    Dynamo unrolls the loop because num_blocks, padded_query_len, has_alibi,
    logits_soft_cap, store_mode, store_len, and needs_gather are closure constants.
    """

    num_queries_per_kv = num_heads // num_kv_heads

    def specialized_paged_attn_kernel(
        query,
        query_row_index,
        k_pages,
        v_pages,
        page_index_table,
        mask_tiles,
        scale,
        alibi_bias_tiles=None,
        out=None,
    ):
        """
        This kernels specializes for num_blocks and padded_query_len.

        Expected shapes:
            query: [num_tokens, num_heads, head_size], the whole batch's query
            query_row_index: int32 device tensor whose first padded_query_len
                entries are this sequence's absolute query rows. None only when
                neither the gather nor an indexed store needs it.
            k_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
            v_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
            page_index_table: [num_blocks, INT32_ELEMS_PER_STICK] int32 device
                tensor, row i holding the i-th active block's page index at
                column 0.
            mask_tiles: [num_blocks]
            alibi_bias_tiles: list of [num_kv_heads, num_queries_per_kv, 1, block_size]
                (only when has_alibi=True; None otherwise). The query-axis dim
                is 1 because softmax absorbs per-query-row constants — see
                the derivation at the bias-tile construction site in
                _online_softmax_attention.

            out: with store_mode != "none", the caller's buffer to write into.

        Returns [padded_query_len, num_heads, head_size], or ``out`` when this
        kernel stored the result itself.
        """
        # A compiled region reads a view from offset 0, ignoring storage_offset
        # (torch-spyre#3770), so the rows are gathered here rather than sliced outside.
        # A gather selecting its whole source instead faults the device
        # (RAS ComputeHardwareError 0x7b1b, torch-spyre#4033), hence needs_gather.
        q_rows = (
            query.index_select(0, query_row_index[:padded_query_len]) if needs_gather else query
        )
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
            if has_alibi:
                # ALiBi bias slope[h] * (kv_pos - context_len). The additive
                # mask_tile below uses finfo.min for masked positions, so this
                # bias cannot un-mask them.
                assert alibi_bias_tiles is not None
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
        if store_mode == "copy":
            assert out is not None
            out.copy_(attn[:store_len])
            return out
        if store_mode == "index":
            # `out` and `query` are both indexed by absolute token row.
            assert out is not None and query_row_index is not None
            out.index_copy_(0, query_row_index[:store_len], attn[:store_len])
            return out
        return attn

    return specialized_paged_attn_kernel


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
    # index). Each tile: [aligned_max_query_len, block_size] on CPU. When
    # sliding_window is None, active == all blocks and the layout is
    # equivalent to indexing by absolute block index.
    attention_mask_tiles: list[list[torch.Tensor]] | None = None

    # For each sequence: absolute block indices whose mask is not fully
    # `-inf` (blocks that contribute to at least one query's attention).
    # None means all blocks are active (sliding_window is None, or the
    # window covers the whole sequence). When set, len(active_block_indices[s])
    # matches len(attention_mask_tiles[s]).
    active_block_indices: list[list[int]] | None = None

    # Global aligned query length for stable kernel compilation.
    # max_query_len rounded up to QUERY_CHUNK_SIZE (32). All queries are
    # padded to this length so the compiled attention kernel receives
    # consistent tensor shapes across steps and sequences.
    aligned_max_query_len: int = 0

    # Global aligned KV sequence length for stable kernel compilation.
    # max_seq_len rounded up to KV_LENGTH_ALIGNMENT (256). The KV mask
    # dimension is padded to this length so recompilation only happens
    # per 256-token tier, not per distinct sequence length.
    aligned_max_seq_len: int = 0

    # Gather indices for the paged attention loop, one row per active block:
    # [num_seqs, max_active_blocks, INT32_ELEMS_PER_STICK] int32 with the page
    # index at [s, b, 0]. Each index needs its own stick-wide row to compile,
    # which is why block_table cannot serve as the index. The device mirror is
    # filled by the first forward(), since the builder's device is CPU.
    # One tensor per sequence, materialized once per step: a compiled kernel reads
    # its inputs from offset 0, ignoring storage_offset (torch-spyre#3770).
    page_index_table_cpu: torch.Tensor | None = None
    page_index_tables: list[torch.Tensor] | None = None

    # Absolute query rows per sequence: gather sources in `query`, and store
    # destinations in `output`. One offset-0 tensor each, as above. Rows past
    # query_len repeat the sequence's last real row; the mask discards them.
    query_row_tables: list[torch.Tensor] | None = None

    # Device mirror of attention_mask_tiles, filled once per step by forward().
    attention_mask_tiles_device: list[list[torch.Tensor]] | None = None

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
        # `torch.float16` upstream so it's always a real torch.dtype here.
        assert isinstance(model_config.dtype, torch.dtype)
        self.model_dtype: torch.dtype = model_config.dtype

        # Shared zero tile reused for interior active blocks (fully inside the
        # window, so their mask is all-zeros). Allocated lazily on first use
        # and resized if aligned_max_query_len or block_size changes across
        # calls.
        self._zero_tile: torch.Tensor | None = None
        self._zero_tile_shape: tuple[int, int] = (0, 0)

        static_ctx = vllm_config.compilation_config.static_forward_context
        self._slot_mapping = attn_layer.install(
            static_ctx[name] for name in layer_names if name in static_ctx
        )

    def _get_zero_tile(self, aligned_max_query_len: int) -> torch.Tensor:
        """Return (or create) the shared all-zero mask tile for interior blocks.

        The returned tensor is reused by reference across all interior blocks
        and sequences in a batch. Callers must treat it as read-only: any
        in-place mutation would corrupt every interior tile simultaneously.
        This is safe today because attention kernels only read mask tiles.
        """
        shape = (aligned_max_query_len, self.block_size)
        if self._zero_tile is None or self._zero_tile_shape != shape:
            self._zero_tile = torch.zeros(shape, dtype=self.model_dtype)
            self._zero_tile_shape = shape
        return self._zero_tile

    def _build_attention_mask(
        self,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        apply_causal_mask: bool,
        max_query_len: int,
        aligned_max_query_len: int,
        aligned_max_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build additive attention mask on Spyre for the non-sliding-window path.

        All sequences share the same aligned_max_query_len so every mask tile
        has a uniform query dimension — this avoids per-sequence kernel
        specializations.

        Sliding-window sequences take a different path: see
        _build_active_tiles_with_skip.

        Returns:
            - mask: [num_seqs, aligned_max_query_len, aligned_max_seq_len] additive mask
        """
        assert self.sliding_window is None
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        num_seqs = len(seq_lens)

        q_pos = torch.arange(max_query_len, device=device)
        kv_pos = torch.arange(aligned_max_seq_len, device=device)

        # Padding mask: valid positions are within actual sequence/query lengths
        q_valid = q_pos.unsqueeze(0) < query_lens.unsqueeze(1)
        kv_valid = kv_pos.unsqueeze(0) < seq_lens.unsqueeze(1)
        attend = q_valid.unsqueeze(2) & kv_valid.unsqueeze(1)

        # Causal mask: prevent attending to future tokens during generation
        if apply_causal_mask:
            context_lens = seq_lens - query_lens
            causal_limit = (context_lens.unsqueeze(1) + q_pos.unsqueeze(0)).unsqueeze(2)
            kv_pos_exp = kv_pos.unsqueeze(0).unsqueeze(0)
            causal_ok = kv_pos_exp <= causal_limit
            attend = attend & causal_ok

        # Convert to additive mask: finfo.min for masked positions, 0 for valid
        mask_bool = ~attend  # [num_seqs, max_query_len, aligned_max_seq_len]

        if aligned_max_query_len > max_query_len:
            padding = torch.ones(
                num_seqs,
                aligned_max_query_len - max_query_len,
                aligned_max_seq_len,
                dtype=torch.bool,
                device=device,
            )
            mask_bool = torch.cat([mask_bool, padding], dim=1)

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
        aligned_max_query_len: int,
        apply_causal_mask: bool,
    ) -> torch.Tensor:
        """Build the additive mask tile for one (sequence, block) pair.

        Returns a [aligned_max_query_len, block_size] CPU tensor.

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

        q_pos = torch.arange(aligned_max_query_len)  # [aligned_max_query_len]
        kv_pos = torch.arange(kv_start, kv_end)  # [block_size]

        # Padding mask: query rows beyond query_len are fully masked;
        # KV columns beyond kv_len are fully masked.
        q_valid = q_pos < query_len  # [aligned_max_query_len]
        kv_valid = kv_pos < kv_len  # [block_size]
        attend = q_valid.unsqueeze(1) & kv_valid.unsqueeze(0)  # [Q, B]

        # Causal mask (prefill only): query at absolute position
        # context_len + q_pos can only attend to KV positions <= that value.
        if apply_causal_mask:
            causal_limit = context_len + q_pos  # [aligned_max_query_len]
            attend = attend & (kv_pos.unsqueeze(0) <= causal_limit.unsqueeze(1))

        # Sliding window: per-query window_start.
        assert self.sliding_window is not None
        abs_q_pos = context_len + q_pos  # [aligned_max_query_len]
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
        aligned_max_query_len: int,
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

        zero_tile = self._get_zero_tile(aligned_max_query_len)
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
                        aligned_max_query_len,
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

        # A decode-only batch needs no padding at all: every query_len is 1.
        if max_query_len == 1:
            aligned_max_query_len = 1
        else:
            aligned_max_query_len = (
                (max_query_len + QUERY_CHUNK_SIZE - 1) // QUERY_CHUNK_SIZE * QUERY_CHUNK_SIZE
            )
        aligned_max_seq_len = (
            (max_seq_len + KV_LENGTH_ALIGNMENT - 1) // KV_LENGTH_ALIGNMENT * KV_LENGTH_ALIGNMENT
        )

        num_seqs = common_attn_metadata.num_reqs
        block_size = self.block_size
        attention_mask_tiles: list[list[torch.Tensor]] = []
        active_block_indices: list[list[int]] | None = None

        if self.sliding_window is None:
            # No sliding window: build the full additive mask and split it into
            # per-block tiles (one tile per absolute block index).
            mask_cpu = self._build_attention_mask(
                seq_lens,
                query_start_loc,
                apply_causal_mask,
                max_query_len,
                aligned_max_query_len,
                aligned_max_seq_len,
                torch.device("cpu"),
            )
            # Pre-tile the mask: split into per-block tiles.
            # Query dimension is uniform (aligned_max_query_len) for all sequences,
            # so tiling only follows the KV dimension.
            for s in range(num_seqs):
                seq_tiles: list[torch.Tensor] = []
                kv_len_s = int(seq_lens[s].item())
                num_blocks_s = (kv_len_s + block_size - 1) // block_size
                for b in range(num_blocks_s):
                    col_start = b * block_size
                    col_end = col_start + block_size
                    tile = mask_cpu[s, :aligned_max_query_len, col_start:col_end]
                    seq_tiles.append(tile.contiguous())
                attention_mask_tiles.append(seq_tiles)
            # active_block_indices stays None, so forward iterates all blocks.
        else:
            # Sliding window: arithmetic block-skip. Blocks entirely outside
            # every query's window are dropped; interior blocks share a
            # zero mask tile; only boundary blocks get real per-query cutoffs.
            active_block_indices = []
            query_lens_list = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
            seq_lens_list = seq_lens.tolist()

            for s in range(num_seqs):
                kv_len_s = int(seq_lens_list[s])
                query_len_s = int(query_lens_list[s])
                context_len_s = kv_len_s - query_len_s

                active_bs, tiles = self._build_active_tiles_with_skip(
                    kv_len_s,
                    query_len_s,
                    context_len_s,
                    aligned_max_query_len,
                    apply_causal_mask,
                )
                active_block_indices.append(active_bs)
                attention_mask_tiles.append(tiles)

        # Gather indices for the attention loop, one row per active block.
        num_active = [len(tiles) for tiles in attention_mask_tiles]
        page_index_table_cpu = torch.zeros(
            num_seqs, max(num_active), INT32_ELEMS_PER_STICK, dtype=torch.int32
        )
        for s, n in enumerate(num_active):
            blocks_s = slice(n) if active_block_indices is None else active_block_indices[s]
            page_index_table_cpu[s, :n, 0] = block_table[s, blocks_s]

        # Padded to match key/value by upstream once forward_includes_kv_cache_update is
        # False, so the traced write keeps one shape per bucket, not one per token count.
        self._slot_mapping.publish(slot_mapping)

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
            page_index_table_cpu=page_index_table_cpu,
            aligned_max_query_len=aligned_max_query_len,
            aligned_max_seq_len=aligned_max_seq_len,
        )


class SpyreAttentionBackend(AttentionBackend):
    """Paged KV-cache attention backend for Spyre."""

    accept_output_buffer: bool = True
    # False tells upstream the attention op does not write KV; attn_layer.py does, and
    # upstream inserts its own unified_kv_cache_update for layers attn_layer declines.
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
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
    via torch.compile with fixed iteration counts. A dict
    caches compiled variants per unique loop length.
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

        # ALiBi slopes: per-head linear-bias coefficients (BLOOM/MPT style).
        # Reshape once to [num_kv_heads, num_queries_per_kv, 1, 1] so the
        # per-block bias construction in _online_softmax_attention broadcasts
        # cleanly against the score-tile shape.
        if alibi_slopes is not None:
            slopes_t = torch.tensor(alibi_slopes, dtype=torch.float16)
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

        # Compiled attention loops, keyed by
        # (num_blocks, padded_query_len, store_mode, store_len, needs_gather)
        self._attn_fns: dict[tuple[int, int, str, int, bool], object] = {}

        self._kv_slots: SpyrePagedKVCache | None = None

        logger.debug_once(
            "Using SpyreAttentionBackend with a dense paged KV cache and indirect page gather"
        )

    def _get_attn_fn(
        self,
        num_blocks: int,
        padded_query_len: int,
        store_mode: str = "none",
        store_len: int = 0,
        needs_gather: bool = True,
    ):
        # self.alibi_slopes and self.logits_soft_cap are fixed per instance, so
        # has_alibi and logits_soft_cap don't need to be part of the cache key.
        key = (num_blocks, padded_query_len, store_mode, store_len, needs_gather)
        if key not in self._attn_fns:
            self._attn_fns[key] = _maybe_compile(
                _create_compilable_page_attn(
                    num_blocks,
                    padded_query_len,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_size,
                    has_alibi=self.alibi_slopes is not None,
                    logits_soft_cap=self.logits_soft_cap,
                    store_mode=store_mode,
                    store_len=store_len,
                    needs_gather=needs_gather,
                ),
                self._compile_attn,
            )
        return self._attn_fns[key]

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
        num_actual_tokens = attn_metadata.num_actual_tokens

        # Only the first layer of a step pays for the device mirror.
        if attn_metadata.page_index_tables is None:
            table_cpu = attn_metadata.page_index_table_cpu
            assert table_cpu is not None
            attn_metadata.page_index_tables = [
                convert(table_cpu[s].contiguous(), device=_target_device)
                for s in range(table_cpu.shape[0])
            ]
        if attn_metadata.attention_mask_tiles_device is None:
            tiles_cpu = attn_metadata.attention_mask_tiles
            assert tiles_cpu is not None, (
                "attention_mask_tiles must be precomputed by the metadata builder"
            )
            attn_metadata.attention_mask_tiles_device = [
                [convert(t, device=_target_device) for t in seq_tiles] for seq_tiles in tiles_cpu
            ]

        # The KV write is not here: attn_layer.py traces it for the layers it splits,
        # and upstream's own unified_kv_cache_update op covers the rest.
        output = self._online_softmax_attention(
            query[:num_actual_tokens],
            k_pages,
            v_pages,
            attn_metadata,
            output,
            _target_device,
        )

        return output

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
        aligned_max_query_len = attn_metadata.aligned_max_query_len
        page_index_tables = attn_metadata.page_index_tables
        # Folds the per-layer eager slice-assign into the attention jobplan.
        # Re-checked per call: vLLM hands out a fresh buffer per layer.
        fused_store_ok = (
            self._compile_attn
            and output.dtype == query_dev.dtype
            # A compiled kernel reads its arguments from offset 0: torch-spyre#3770.
            and output.storage_offset() == 0
            and output.is_contiguous()
        )
        assert mask_tiles_all is not None, (
            "attention_mask_tiles_device must be mirrored by forward()"
        )
        assert page_index_tables is not None, "page_index_tables must be mirrored by forward()"

        for seq_idx in range(num_seqs):
            # Most-naive implementation: no parallelization
            # over sequences or GQA optimization
            q_start = int(query_start_loc[seq_idx].item())
            q_end = int(query_start_loc[seq_idx + 1].item())
            query_len = q_end - q_start
            kv_len = int(seq_lens[seq_idx].item())

            num_blocks_needed = (kv_len + block_size - 1) // block_size

            # Restrict to active (non-fully-masked) blocks when sliding window
            # is set. When active_block_indices_all is None (no sliding), all
            # blocks are active in their natural order.
            if active_block_indices_all is not None:
                active_bs = active_block_indices_all[seq_idx]
            else:
                active_bs = list(range(num_blocks_needed))

            if len(active_bs) == 0:
                # Every KV position is outside every query's window. Attention
                # over the empty set is undefined; write zeros.
                output[q_start:q_end] = 0.0
                continue

            page_index_table = page_index_tables[seq_idx]
            # mask_tiles_all[seq_idx] is indexed by position within active_bs.
            mask_tiles = mask_tiles_all[seq_idx][: len(active_bs)]

            # ALiBi bias tiles: slope[h] * (kv_pos - context_len), one per block.
            #
            # The full ALiBi form is slope[h] * (kv_pos - (context_len + q_rel)),
            # which varies over both query and KV positions. The (context_len + q_rel)
            # term is a per-query-row constant, and softmax is invariant under adding
            # any per-row constant to its input (numerator and denominator both pick
            # up the same exp() factor). We therefore drop it and keep only the
            # kv-dependent term — the softmax output is bit-identical to the full
            # form, and each tile stays 1D over KV (block_size floats per head)
            # instead of 2D (aligned_max_query_len * block_size).
            #
            # Matches vllm/v1/attention/ops/triton_attention_helpers.py::apply_alibi_to_score
            # (alibi_offset = seq_offset - context_len) — the production Triton path.
            #
            # Per-tile shape: [num_kv_heads, num_queries_per_kv, 1, block_size].
            alibi_bias_tiles: list[torch.Tensor] | None = None
            if self.alibi_slopes is not None:
                context_len = kv_len - query_len
                alibi_bias_tiles = []
                for b in active_bs:
                    kv_pos = torch.arange(
                        b * block_size,
                        (b + 1) * block_size,
                        dtype=torch.float16,
                    )
                    rel = (kv_pos - context_len).view(1, 1, 1, block_size)
                    bias = self.alibi_slopes * rel
                    alibi_bias_tiles.append(convert(bias, device=_target_device))

            # Selecting the whole buffer faults the device; see the kernel's gather.
            needs_gather = not (
                q_start == 0
                and query_len == aligned_max_query_len
                and query_dev.shape[0] == aligned_max_query_len
            )

            store_mode = "none"
            if fused_store_ok:
                if output.shape[0] == query_len:
                    # Owns every row, so the store is a plain copy_: no index, and
                    # no relayout of the destination.
                    store_mode = "copy"
                elif output.shape[0] > 1:
                    # Owns only a slice, so the store must be indirect. A single-row
                    # destination is excluded: index_copy_ writes nothing there
                    # (torch-spyre#4007).
                    store_mode = "index"

            row_table = None
            if needs_gather or store_mode == "index":
                if attn_metadata.query_row_tables is None:
                    attn_metadata.query_row_tables = _build_query_row_tables(
                        attn_metadata, _target_device
                    )
                row_table = attn_metadata.query_row_tables[seq_idx]

            # Run attention on target device
            attn_fn = self._get_attn_fn(
                len(active_bs),
                aligned_max_query_len,
                store_mode=store_mode,
                store_len=query_len,
                needs_gather=needs_gather,
            )
            result = attn_fn(
                query_dev,
                row_table,
                k_pages,
                v_pages,
                page_index_table,
                mask_tiles,
                self.scale,
                alibi_bias_tiles=alibi_bias_tiles,
                out=output if store_mode != "none" else None,
            )

            assert result.dtype == output.dtype
            if store_mode != "none":
                # The kernel wrote `output` itself; `result` is that same buffer.
                continue
            output[q_start:q_end] = result[:query_len]

        return output
