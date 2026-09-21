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
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec, EncoderOnlyAttentionSpec

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention import attn_layer
from spyre_inference.v1.attention.ops.batched_decode import batched_decode_kernel
from spyre_inference.v1.attention.ops.layout import (
    INT32_ELEMS_PER_STICK,
    slot_major_kv_layout,
    stick_aligned_len,
)
from spyre_inference.v1.attention.ops.page_attn import page_attn_kernel
from spyre_inference.v1.attention.ops.reshape_and_cache import reshape_and_cache_kernel
from spyre_inference.v1.attention.spyre_attn_bucketer import (
    _MIN_BATCHED_SEQS,
    SpyreAttnBatchedDecodeBucket,
    SpyreAttnBucket,
    SpyreAttnBucketer,
    batched_decode_chunking,
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


# mean/max decode block count: low means the batch pads short sequences up to a
# much longer one. Calibrated from the crossover sweep; 0.0 disables.
_BATCHED_DECODE_MIN_UNIFORMITY: float = 0.0


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


def _build_query_row_tables(
    attn_metadata: "SpyreAttentionMetadata", device: torch.device
) -> list[torch.Tensor]:
    """Build query gather/dest row tables for the whole batch.

    Each row is converted on its own so it lands at storage offset 0
    (torch-spyre#3770); slicing a batched device tensor instead hits
    spyre::copy_from_d2d, which recompiles per (shape, src_off, dst_off).
    """
    num_seqs = attn_metadata.num_seqs
    starts = attn_metadata.query_start_loc[:num_seqs].cpu()
    lens = attn_metadata.query_start_loc[1 : num_seqs + 1].cpu() - starts
    aligned_query_lens = attn_metadata.aligned_query_lens
    tables = []
    for s, aligned in enumerate(aligned_query_lens):
        # Width the recorder traced for this query length, not the batch max.
        index_len = stick_aligned_len(aligned)
        row = torch.zeros(index_len, dtype=torch.int32)
        last_real = max(int(lens[s]) - 1, 0)
        row[:aligned] = (starts[s] + torch.arange(aligned).clamp(max=last_real)).to(torch.int32)
        tables.append(convert(row, device=device))
    return tables


# Attention compiles separately from the model's fullgraph capture, which can't
# hold the per-sequence Python loop around these.
_page_attn_compiled = torch.compile(page_attn_kernel, dynamic=False)
_batched_decode_compiled = torch.compile(batched_decode_kernel, dynamic=False)

_warmup_complete = False


def mark_warmup_complete() -> None:
    """Arm the late-compile warning, once warmup has claimed full variant coverage."""
    global _warmup_complete
    _warmup_complete = True


def is_warmup_complete() -> bool:
    """Whether ``mark_warmup_complete`` has run. For diagnostics, not control flow."""
    return _warmup_complete


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
    # which is why block_table cannot serve as the index.
    # One table per sequence at its own active-block count, materialized once per
    # step: a batch-max width would put max(num_active) into the kernel's guards.
    page_index_tables_cpu: list[torch.Tensor] | None = None

    # Per sequence, the device index table each kernel gathers pages with, built from
    # page_index_tables_cpu on the step's first layer since the builder's device is CPU.
    # Its shape is the kernel's business, so a caller driving two impls over one step's
    # metadata has to clear this between them.
    kernel_index_tables: list | None = None

    # Absolute query rows per sequence: gather sources in `query`, and store
    # destinations in `output`. One offset-0 tensor each, as above. Rows past
    # query_len repeat the sequence's last real row; the mask discards them.
    query_row_tables: list[torch.Tensor] | None = None

    # Device mirror of attention_mask_tiles, filled once per step by forward().
    attention_mask_tiles_device: list[list[torch.Tensor]] | None = None

    # Batched-decode precomputes. None-valued when the batch is ineligible
    # (callers fall back to the per-seq loop). entries = B_seqs * blocks_per_chunk.
    num_decode_seqs: int = 0  # leading decode-only seqs; == num_seqs for pure-decode batches
    num_decode_tokens: int = 0  # == num_decode_seqs since each decode contributes one token
    decode_uniformity: float = 0.0  # mean/max decode block count; 0.0 when batched path ineligible
    padded_num_seqs: int | None = None
    padded_batch_blocks: int | None = None
    blocks_per_chunk: int | None = None
    rep_row_ids_cpu: torch.Tensor | None = None  # [entries] int32
    rep_row_ids_dev: torch.Tensor | None = None
    chunk_page_ids_cpu: list[torch.Tensor] | None = None  # num_chunks x [entries, 1] int32
    # Built by build_chunk_index_tables, so the shape is the kernel's (as above).
    chunk_page_ids_dev: list[torch.Tensor] | None = None
    mask_by_chunk_cpu: torch.Tensor | None = None  # [num_chunks, entries * KV, 1, block] fp16
    mask_by_chunk_dev: torch.Tensor | None = None

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
    # Host-built key-pad ``[B * KV, 1, 1, L]`` on the target device. ``None`` on
    # the fused path. Broadcast onto encoder scores ``[BH, G, L, L]`` by the
    # compiled add in ``_packed_pv``; a dense ``[BH, 1, L, L]`` was needed only
    # while that add was eager.
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
        # No KV cache, so build() skips every block-, page- and tile-shaped field.
        self._encoder_only = isinstance(kv_cache_spec, EncoderOnlyAttentionSpec)
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
        # From the spec, like block_size and head_size above: gemma-4 resolves KV heads
        # per layer, so this group's count need not be the model-level one.
        self.num_kv_heads = kv_cache_spec.num_kv_heads
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

        # record_graphs() enumerates this same instance, so the buckets warmup
        # compiles are exactly the ones build() can round onto.
        self._attn_bucketer = SpyreAttnBucketer(vllm_config)

        self._init_reorder_batch_threshold(
            reorder_batch_threshold=1 if envs.SPYRE_BATCHED_DECODE else None
        )

    @property
    def attn_bucketer(self) -> SpyreAttnBucketer:
        return self._attn_bucketer

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
        padded = self._attn_bucketer.find_blocks_bucket(num_blocks)
        # The buckets cover ceil(max_model_len / block_size), and the scheduler
        # bounds seq_lens by max_model_len. Encoder-only builds, whose profiling
        # seq_lens come from the token budget instead, never reach here.
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

        if self._encoder_only:
            # Everything below is KV-cache state -- block counts, per-block mask
            # tiles, page-index tables, batched-decode precomputes -- and
            # SpyreEncoderAttentionImpl reads none of it, deriving its own packed
            # grid from seq_lens and query_start_loc. Besides the per-step waste,
            # _pad_num_blocks would reject the batch: its ladder comes from
            # max_model_len while a pooling model's body buckets come from
            # max_num_batched_tokens, and warmup profiles every sequence at the full
            # body size (a 256-token sentence-transformer asks for 512).
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
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
            )

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
                if aligned_query_len == 1:
                    # One unbind beats a __getitem__ per tile at decode tile counts; the
                    # clones stay, for storage offset 0 (torch-spyre#3770).
                    # Width 1 only: wider needs a permute that copies the whole mask.
                    tiles_by_block = mask_cpu.reshape(
                        len(group), mask_cpu.shape[-1] // block_size, 1, block_size
                    )
                    for row, s in enumerate(group):
                        attention_mask_tiles[s] = [
                            tile.clone(memory_format=torch.contiguous_format)
                            for tile in tiles_by_block[row].unbind(0)[: padded_num_blocks[s]]
                        ]
                    continue
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
        # Dynamo guard the key cannot carry.
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

        num_decode_seqs, _, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold or 1,
            treat_short_extends_as_decodes=common_attn_metadata.is_prefilling is None,
        )
        padded_num_seqs = None
        padded_batch_blocks = None
        blocks_per_chunk = None
        rep_row_ids_cpu = None
        chunk_page_ids_cpu = None
        mask_by_chunk_cpu = None
        decode_uniformity = 0.0
        if num_decode_seqs >= _MIN_BATCHED_SEQS:
            # Real counts for the decode prefix only — same reasoning as before.
            blocks_per_seq = real_num_blocks if active_block_indices is None else num_active

            decode_blocks = blocks_per_seq[:num_decode_seqs]
            b_seqs = self._attn_bucketer.find_sequence_bucket(num_decode_seqs)
            b_blocks = self._attn_bucketer.find_blocks_bucket(max(decode_blocks))

            if b_seqs is not None and b_blocks is not None:
                # Mean/max block count: how uniform the contexts are, independent of
                # bucket round-up (which padding the denser ladder addresses instead).
                decode_uniformity = (sum(decode_blocks) / num_decode_seqs) / max(decode_blocks)
                padded_num_seqs = b_seqs
                # Padding columns gather page 0 under an all--inf mask and
                # contribute zero; chunk 0 still holds every real row's block 0,
                # so the running max stays finite.
                blocks_per_chunk, num_chunks = batched_decode_chunking(b_seqs, b_blocks)
                padded_batch_blocks = num_chunks * blocks_per_chunk
                assert padded_batch_blocks >= b_blocks
                entries = b_seqs * blocks_per_chunk

                query_row_ids = torch.zeros(b_seqs, dtype=torch.int32)
                query_row_ids[:num_decode_seqs] = query_start_loc[:num_decode_seqs].to(torch.int32)
                # Guards the identity scatter used by _run_batched_decode_dispatch.
                assert query_row_ids[:num_decode_seqs].tolist() == list(range(num_decode_seqs))
                rep_row_ids_cpu = query_row_ids.repeat_interleave(blocks_per_chunk)

                block_ids_padded = torch.zeros(padded_batch_blocks, b_seqs, dtype=torch.int32)
                bt = block_table[:num_decode_seqs].to(torch.int32)
                if active_block_indices is None:
                    n_use_list = [min(n, b_blocks) for n in decode_blocks]
                    w = min(b_blocks, bt.shape[1])
                    cols = torch.arange(w)
                    in_range = cols.unsqueeze(0) < torch.tensor(
                        n_use_list, dtype=torch.int64
                    ).unsqueeze(1)
                    block_ids_padded[:w, :num_decode_seqs] = (bt[:, :w] * in_range).t()
                else:
                    # Position i is the i-th ACTIVE block, matching the mask tiles.
                    for s, abs_blocks in enumerate(active_block_indices[:num_decode_seqs]):
                        n_use = min(len(abs_blocks), b_blocks)
                        for b, abs_b in enumerate(abs_blocks[:n_use]):
                            block_ids_padded[b, s] = bt[s, abs_b]
                # Entry order (s, j), s major, matching rep_row_ids and the mask.
                # One tensor per chunk, not slices of a stack: an index tensor
                # reaches the device as a real argument and a view's storage
                # offset is dropped (torch-spyre#3770), so a sliced chunk c > 0
                # would silently gather chunk 0's pages. Probed by
                # test_spyre_compile_input_honors_storage_offset; when that
                # strict xfail flips, one stacked tensor also collapses the
                # per-chunk H2D transfers into one.
                chunk_page_ids_cpu = [
                    block_ids_padded[c * blocks_per_chunk : (c + 1) * blocks_per_chunk]
                    .t()
                    .reshape(entries, 1)
                    .contiguous()
                    for c in range(num_chunks)
                ]

                # -inf on padded rows/blocks and past-kv-len positions; 0 on
                # valid positions. Broadcast to KV heads and reshape to the
                # kernel input shape [num_chunks, entries * KV, 1, block_size].
                mask_bs_bb = torch.full(
                    (b_seqs, padded_batch_blocks, block_size),
                    float("-inf"),
                    dtype=self.model_dtype,
                )
                for s in range(num_decode_seqs):
                    n_use = min(blocks_per_seq[s], b_blocks)
                    if n_use:
                        mask_bs_bb[s, :n_use] = torch.stack(
                            [attention_mask_tiles[s][b][0] for b in range(n_use)]
                        )
                # A row past the batch is -inf in every block, so its softmax is NaN and
                # the in-graph store would publish it. A real row always has a valid
                # block 0, so its padded blocks can stay -inf and contribute zero.
                # Holds under a window too: first_active <= num_blocks - 1.
                mask_bs_bb[num_decode_seqs:, 0] = torch.finfo(self.model_dtype).min
                # 4-D, not 5-D: the kernel slices dim 0 per chunk, and a dim-0
                # slice of a 5-D base fails torch-spyre layout propagation.
                mask_by_chunk_cpu = (
                    mask_bs_bb.reshape(b_seqs, num_chunks, blocks_per_chunk, block_size)
                    .permute(1, 0, 2, 3)
                    .unsqueeze(3)
                    .expand(num_chunks, b_seqs, blocks_per_chunk, self.num_kv_heads, block_size)
                    .reshape(num_chunks, entries * self.num_kv_heads, 1, block_size)
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
            num_decode_seqs=num_decode_seqs,
            num_decode_tokens=num_decode_tokens,
            decode_uniformity=decode_uniformity,
            padded_num_seqs=padded_num_seqs,
            padded_batch_blocks=padded_batch_blocks,
            blocks_per_chunk=blocks_per_chunk,
            rep_row_ids_cpu=rep_row_ids_cpu,
            chunk_page_ids_cpu=chunk_page_ids_cpu,
            mask_by_chunk_cpu=mask_by_chunk_cpu,
        )

    def build_for_variant(self, bucket: SpyreAttnBucket) -> SpyreAttentionMetadata:
        """Metadata for the one-sequence batch that dispatches to ``bucket``."""
        query_len = self._attn_bucketer.min_real_query_len(bucket.padded_query_len)
        kv_len = bucket.num_blocks * self.block_size
        assert query_len <= kv_len, f"{bucket} pairs a query length no sequence can reach"
        query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
        # Every block points at page 0, vLLM's null block: nothing real is read.
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc,
                seq_lens=torch.tensor([kv_len], dtype=torch.int32),
                num_reqs=1,
                num_actual_tokens=query_len,
                max_query_len=query_len,
                max_seq_len=kv_len,
                block_table_tensor=torch.zeros(1, bucket.num_blocks, dtype=torch.int32),
                slot_mapping=torch.zeros(query_len, dtype=torch.int64),
                causal=True,
                is_prefilling=torch.tensor([query_len > 1]),
            ),
        )

    def build_for_batched_decode_variant(
        self, bucket: SpyreAttnBatchedDecodeBucket
    ) -> SpyreAttentionMetadata:
        """Metadata for the all-decode batch that dispatches to ``bucket``.

        ``num_seqs`` sequences of one query token each, all at the same length, so
        ``find_sequence_bucket`` and ``find_blocks_bucket`` return the bucket's own
        values and ``decode_uniformity`` is 1.
        """
        num_seqs = bucket.num_seqs
        kv_len = bucket.num_blocks * self.block_size
        query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32)
        metadata = self.build(
            common_prefix_len=0,
            common_attn_metadata=CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc,
                seq_lens=torch.full((num_seqs,), kv_len, dtype=torch.int32),
                num_reqs=num_seqs,
                num_actual_tokens=num_seqs,
                max_query_len=1,
                max_seq_len=kv_len,
                # Every block points at page 0, vLLM's null block: nothing real is read.
                block_table_tensor=torch.zeros(num_seqs, bucket.num_blocks, dtype=torch.int32),
                slot_mapping=torch.zeros(num_seqs, dtype=torch.int64),
                causal=True,
                is_prefilling=torch.zeros(num_seqs, dtype=torch.bool),
            ),
        )
        assert metadata.padded_num_seqs is not None, (
            f"build() declined the batched path for {bucket}; the recorded variant would "
            "not be the one dispatch reaches"
        )
        return metadata


class SpyreAttentionBackend(AttentionBackend):
    """Paged KV-cache attention backend for Spyre."""

    accept_output_buffer: bool = True
    # False tells upstream the attention op does not write KV; attn_layer.py does, and
    # upstream inserts its own unified_kv_cache_update for layers attn_layer declines.
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        # Only reachable through an explicit `--dtype bfloat16`; the platform's own
        # default is float16 for every model.
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
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

        # The kernels read a page at model dtype and Spyre has no cast on the way in, so
        # the two 2-byte dtypes are not interchangeable per-cache.
        if kv_cache_dtype not in ("auto", str(self.model_dtype).removeprefix("torch.")):
            raise ValueError(
                f"kv_cache_dtype={kv_cache_dtype} does not match the model dtype "
                f"{self.model_dtype} on Spyre; use 'auto'."
            )

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
        self._reshape_fn = torch.compile(reshape_and_cache_kernel, dynamic=False)

        self._attn_fn = _page_attn_compiled if self._compile_attn else page_attn_kernel
        # Always the compiled variant: the 2-D page index lowers to aten.index,
        # which fails eager, so _batched_decode_preconditions_met declines the
        # whole path when self._compile_attn is False.
        self._decode_fn = _batched_decode_compiled

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

        Row-outermost: the kernels gather from and scatter into the row axis, which the
        default tiled layout would relayout whole per call.
        """
        if self._staging is None:
            shape = (self.staging_rows, self.num_heads, self.head_size)
            self._staging = (
                convert(torch.zeros(shape, dtype=self.model_dtype), device, row_major=True),
                convert(torch.zeros(shape, dtype=self.model_dtype), device, row_major=True),
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

    def _batched_decode_supported(self) -> bool:
        """The batch-independent preconditions, so the warmup recorder can share them."""
        # Batches below _MIN_BATCHED_SEQS take the per-seq loop regardless: the
        # num_seqs ladder starts there, so they have no batched variant to
        # dispatch to. Set SPYRE_BATCHED_DECODE=0 to force the loop for all sizes.
        if not envs.SPYRE_BATCHED_DECODE:
            return False
        # The 2-D page index lowers to aten.index, which upcasts the int32 index
        # to int64 and fails eager; eager takes the per-seq loop instead.
        if not self._compile_attn:
            return False
        # The batched kernel doesn't implement ALiBi.
        return self.alibi_slopes is None

    def _batched_decode_preconditions_met(self, attn_metadata: "SpyreAttentionMetadata") -> bool:
        if not self._batched_decode_supported():
            return False
        # Layer 0's builder gates on the decode count and the bucket lattice.
        if attn_metadata.padded_num_seqs is None:
            return False
        return attn_metadata.decode_uniformity >= _BATCHED_DECODE_MIN_UNIFORMITY

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
            and attn_metadata.rep_row_ids_dev is None
        ):
            assert attn_metadata.rep_row_ids_cpu is not None
            assert attn_metadata.chunk_page_ids_cpu is not None
            assert attn_metadata.mask_by_chunk_cpu is not None
            attn_metadata.rep_row_ids_dev = convert(
                attn_metadata.rep_row_ids_cpu, device=_target_device
            )
            attn_metadata.chunk_page_ids_dev = self.build_chunk_index_tables(
                attn_metadata, _target_device
            )
            attn_metadata.mask_by_chunk_dev = convert(
                attn_metadata.mask_by_chunk_cpu, device=_target_device
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
        layer: AttentionLayer,
        kv_cache: SpyrePagedKVCache,
        builder: "SpyreAttentionMetadataBuilder",
    ) -> int:
        """Compile every attention variant the builder's bucketer enumerates.

        Called from warmup, after the KV cache exists, since the kernel
        ``index_select``s real pages. A failing variant is logged and skipped, not
        raised, so it cannot take down engine startup; dispatch then compiles it on
        first use.
        """
        if not self._compile_attn:
            return 0

        num_pages = kv_cache[0].shape[0]
        variants = builder.attn_bucketer.variants()
        decode_variants = (
            builder.attn_bucketer.batched_decode_variants()
            if self._batched_decode_supported()
            else []
        )
        t_start = time.time()

        # Belt-and-suspenders: platform._raise_dynamo_recompile_limits already
        # raises this globally, but bump it here too in case that hasn't run.
        prev_limit = torch._dynamo.config.accumulated_recompile_limit
        torch._dynamo.config.accumulated_recompile_limit = max(  # ty: ignore[invalid-assignment]
            prev_limit, 4 * (len(variants) + len(decode_variants)) + 64
        )

        logger.info(
            "Recording %d per-seq + %d batched-decode attention variants for layer...",
            len(variants),
            len(decode_variants),
        )
        try:
            recorded = self._record_all(variants, layer, kv_cache, builder, num_pages)
            recorded_decode = self._record_batched_all(
                decode_variants, layer, kv_cache, builder, num_pages
            )
        finally:
            torch._dynamo.config.accumulated_recompile_limit = prev_limit

        if recorded == 0 and variants:
            # Recording nothing is a broken pass, not a degenerate bucket set: the
            # fallback is a full Inductor compile on every shape mid-serving.
            logger.warning_once(
                "Recorded none of the %d attention variants; every shape will compile "
                "on first use. The per-variant warnings above carry the reason.",
                len(variants),
            )
        logger.info(
            "Recorded %d/%d per-seq and %d/%d batched-decode attention variants in %.2fs.",
            recorded,
            len(variants),
            recorded_decode,
            len(decode_variants),
            time.time() - t_start,
        )
        return recorded + recorded_decode

    def _record_all(
        self,
        variants: "list[SpyreAttnBucket]",
        layer: AttentionLayer,
        kv_cache: SpyrePagedKVCache,
        builder: "SpyreAttentionMetadataBuilder",
        num_pages: int,
    ) -> int:
        recorded: set[SpyreAttnBucket] = set()
        for i, bucket in enumerate(variants, start=1):
            if bucket.num_blocks > num_pages:
                # The buckets are sized from max_model_len; a memory-constrained cache
                # allocates fewer pages than that many distinct blocks to gather.
                continue
            t0 = time.time()
            try:
                realized = self._record_one(bucket, layer, kv_cache, builder, recorded)
            except Exception:
                logger.warning(
                    "Attention variant %s failed to record; it will compile on first use instead.",
                    bucket,
                    exc_info=True,
                )
                continue
            if realized is None:
                continue
            recorded.add(realized)
            logger.debug(
                "  [%d/%d] recorded %s in %.2fs",
                i,
                len(variants),
                realized,
                time.time() - t0,
            )
        return len(recorded)

    def _record_one(
        self,
        bucket: "SpyreAttnBucket",
        layer: AttentionLayer,
        kv_cache: SpyrePagedKVCache,
        builder: "SpyreAttentionMetadataBuilder",
        recorded: "set[SpyreAttnBucket]",
    ) -> "SpyreAttnBucket | None":
        """Trace the kernel ``bucket`` needs; None if ``build()`` realized one already traced."""
        attn_metadata = builder.build_for_variant(bucket)
        assert attn_metadata.attention_mask_tiles is not None
        realized = SpyreAttnBucket(
            num_blocks=len(attn_metadata.attention_mask_tiles[0]),
            padded_query_len=attn_metadata.aligned_query_lens[0],
        )
        # Several requested buckets realize onto one kernel: a sliding window leaves
        # the block count unpadded. Without a window build() rounds onto the bucketer's
        # own buckets, so a mismatch means the two have drifted and dispatch can ask
        # for a kernel warmup never recorded.
        if realized != bucket and builder.sliding_window is None:
            logger.warning(
                "Attention variant %s realized as %s without a sliding window; the "
                "bucketer and build() have diverged and some shapes will compile on "
                "first use.",
                bucket,
                realized,
            )
        if realized in recorded:
            return None

        # The staging buffers take the same pre-staged path attn_layer takes, so no
        # extra copies get traced. key/value are unused: attn_layer does the KV write.
        q_staging, out_staging = self._staging_buffers(kv_cache[0].device)
        self.forward(layer, q_staging, q_staging, q_staging, kv_cache, attn_metadata, out_staging)
        return realized

    def _record_batched_all(
        self,
        variants: "list[SpyreAttnBatchedDecodeBucket]",
        layer: AttentionLayer,
        kv_cache: SpyrePagedKVCache,
        builder: "SpyreAttentionMetadataBuilder",
        num_pages: int,
    ) -> int:
        # Checked once, outside the try below: the store these recordings trace needs
        # staging at least as wide as the widest bucket, and swallowing that would
        # leave dispatch tracing the fallback scatter instead.
        widest = max((v.num_seqs for v in variants), default=0)
        assert self.staging_rows >= widest, (
            f"staging buffers hold {self.staging_rows} rows, below the widest num_seqs "
            f"bucket {widest}; the recorded variants would not be the ones dispatch reaches"
        )

        recorded: set[tuple[int, int, int]] = set()
        for i, bucket in enumerate(variants, start=1):
            t0 = time.time()
            try:
                realized = self._record_batched_one(
                    bucket, layer, kv_cache, builder, recorded, num_pages
                )
            except Exception:
                logger.warning(
                    "Batched decode variant %s failed to record; it will compile on first "
                    "use instead.",
                    bucket,
                    exc_info=True,
                )
                continue
            if realized is None:
                continue
            recorded.add(realized)
            logger.debug(
                "  [%d/%d] recorded %s in %.2fs",
                i,
                len(variants),
                realized,
                time.time() - t0,
            )
        return len(recorded)

    def _record_batched_one(
        self,
        bucket: "SpyreAttnBatchedDecodeBucket",
        layer: AttentionLayer,
        kv_cache: SpyrePagedKVCache,
        builder: "SpyreAttentionMetadataBuilder",
        recorded: "set[tuple[int, int, int]]",
        num_pages: int,
    ) -> "tuple[int, int, int] | None":
        """Trace the batched kernel ``bucket`` needs; None if already traced or unreachable.

        Returns the ``(num_seqs, blocks_per_chunk, num_chunks)`` key the metadata
        ``build()`` produced actually dispatches on, not ``bucket``'s own.
        """
        attn_metadata = builder.build_for_batched_decode_variant(bucket)
        assert attn_metadata.padded_num_seqs is not None
        assert attn_metadata.blocks_per_chunk is not None
        assert attn_metadata.chunk_page_ids_cpu is not None
        realized = (
            attn_metadata.padded_num_seqs,
            attn_metadata.blocks_per_chunk,
            len(attn_metadata.chunk_page_ids_cpu),
        )
        # Several requested buckets realize onto one kernel: a sliding window leaves
        # the block count unpadded. Without a window build() rounds onto the bucketer's
        # own buckets, so a mismatch means the two have drifted and dispatch can ask
        # for a kernel warmup never recorded.
        requested = (bucket.num_seqs, bucket.blocks_per_chunk, bucket.num_chunks)
        if realized != requested and builder.sliding_window is None:
            logger.warning(
                "Batched decode variant %s realized as %s without a sliding window; the "
                "bucketer and build() have diverged and some shapes will compile on "
                "first use.",
                requested,
                realized,
            )
        # The kernel gathers `entries` pages per chunk, not num_blocks of them, and
        # selecting a whole source faults the device (torch-spyre#4033). A sliding
        # window leaves the block count unpadded, so this keys on what build()
        # realized, not the bucket's window-agnostic count, to skip only the
        # variants dispatch cannot reach either.
        if attn_metadata.padded_num_seqs * attn_metadata.blocks_per_chunk >= num_pages:
            return None
        if realized in recorded:
            return None

        q_staging, out_staging = self._staging_buffers(kv_cache[0].device)
        self.forward(layer, q_staging, q_staging, q_staging, kv_cache, attn_metadata, out_staging)
        return realized

    @classmethod
    def allocate_pages(
        cls, num_blocks: int, spec: AttentionSpec, device: torch.device
    ) -> SpyrePagedKVCache:
        """Allocate the paged K/V tensors in the layout this impl's kernels read."""
        # Host-allocated then transferred: only .to() takes a device_layout.
        dtype = spec.dtype
        layout = slot_major_kv_layout(
            num_blocks * spec.block_size, spec.num_kv_heads, spec.head_size, dtype
        )
        shape = (num_blocks, spec.block_size, spec.num_kv_heads, spec.head_size)
        return SpyrePagedKVCache(
            k_pages=torch.zeros(shape, dtype=dtype).to(device, device_layout=layout),  # ty: ignore[no-matching-overload]
            v_pages=torch.zeros(shape, dtype=dtype).to(device, device_layout=layout),  # ty: ignore[no-matching-overload]
        )

    def kv_write_index(self, slot_mapping: torch.Tensor, device: torch.device):
        """Mirror a host slot mapping to the index ``do_kv_cache_update`` takes.

        Token-major puts a token's heads in one contiguous run, so the slot is the index.
        """
        return convert(slot_mapping, device=device)

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
        # Spyre-lowering shapes drive several structural choices here: K/V/q keep
        # (entries, KV) as the two batch axes lower_bmm allows; the kernel's
        # per-chunk index stays at Dynamo-trace time (torch-spyre would emit
        # Mod(d0, num_chunks) for a runtime .select); and the result scatter is a
        # single contiguous copy_ at offset 0, valid because the decode prefix's
        # query rows are range(num_decode_seqs) (asserted in the builder).
        b_seqs = attn_metadata.padded_num_seqs
        blocks_per_chunk = attn_metadata.blocks_per_chunk
        num_decode_seqs = attn_metadata.num_decode_seqs
        num_heads = self.num_heads
        head_size = self.head_size
        block_size = attn_metadata.block_size

        assert b_seqs is not None and blocks_per_chunk is not None
        assert attn_metadata.rep_row_ids_dev is not None
        assert attn_metadata.chunk_page_ids_dev is not None
        assert attn_metadata.mask_by_chunk_dev is not None

        # The kernel's store writes out[:b_seqs] -- its num_seqs parameter receives
        # b_seqs, not the real count -- so the destination needs that many rows and a
        # plain offset-0 layout; the scatter below covers every other case, including
        # a query buffer narrower than the seq bucket. Re-checked per call: vLLM hands
        # out a fresh buffer per layer.
        store_out = (
            query_dev.shape[0] >= b_seqs
            and output.shape[0] >= b_seqs
            and output.dtype == query_dev.dtype
            and output.storage_offset() == 0
            and output.is_contiguous()
        )
        result = self._run_batched_decode(
            query_dev,
            attn_metadata.rep_row_ids_dev,
            k_pages,
            v_pages,
            attn_metadata.chunk_page_ids_dev,
            attn_metadata.mask_by_chunk_dev,
            b_seqs,
            blocks_per_chunk,
            block_size,
            output if store_out else None,
        )
        if store_out:
            return

        # The decode prefix's query rows are range(num_decode_seqs), so the scatter
        # is a contiguous prefix write at (0, 0). Neither per-row
        # slice-assign (spyre::copy_from_d2d specialises on (src_off, dst_off)
        # via @compile_once and can return a stale binary) nor index_copy_
        # (CPU-fallback segfaults on vLLM output buffers) is safe here.
        result_flat = result.reshape(b_seqs, num_heads, head_size)
        src_block = result_flat[:num_decode_seqs].clone()
        output[:num_decode_seqs].copy_(src_block)

    def _run_batched_decode(
        self,
        query_dev: torch.Tensor,
        rep_row_ids: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        chunk_index_tables: list[torch.Tensor],
        mask_by_chunk: torch.Tensor,
        b_seqs: int,
        blocks_per_chunk: int,
        block_size: int,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the batch's decode attention. The point where a subclass swaps kernels."""
        return _call_kernel(
            "batched decode attention",
            self._decode_fn,
            query_dev,
            rep_row_ids,
            k_pages,
            v_pages,
            chunk_index_tables,
            mask_by_chunk,
            self.scale,
            b_seqs,
            blocks_per_chunk,
            self.num_kv_heads,
            self.num_queries_per_kv,
            block_size,
            self.head_size,
            self.logits_soft_cap,
            out,
        )

    def build_chunk_index_tables(
        self, attn_metadata: "SpyreAttentionMetadata", device: torch.device
    ) -> list[torch.Tensor]:
        """Per chunk, the device index table the batched kernel gathers pages with."""
        tables_cpu = attn_metadata.chunk_page_ids_cpu
        assert tables_cpu is not None, "chunk_page_ids_cpu must come from the builder"
        # Fresh offset-0 allocations (torch-spyre#3770).
        return [convert(table, device=device) for table in tables_cpu]

    def build_index_tables(
        self, attn_metadata: "SpyreAttentionMetadata", device: torch.device
    ) -> list:
        """Per sequence, the device index table this impl's kernel gathers pages with."""
        tables_cpu = attn_metadata.page_index_tables_cpu
        assert tables_cpu is not None, "page_index_tables_cpu must come from the builder"
        # Fresh offset-0 allocations (torch-spyre#3770).
        return [convert(table, device=device) for table in tables_cpu]

    def index_tables(self, attn_metadata: "SpyreAttentionMetadata", device: torch.device) -> list:
        """`build_index_tables`, memoized so only the step's first layer pays for it."""
        if attn_metadata.kernel_index_tables is None:
            attn_metadata.kernel_index_tables = self.build_index_tables(attn_metadata, device)
        return attn_metadata.kernel_index_tables

    def _run_page_attn(
        self,
        query: torch.Tensor,
        row_table: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        index_table,
        mask_tiles: list[torch.Tensor],
        num_blocks: int,
        padded_query_len: int,
        alibi_bias_tiles: list[torch.Tensor] | None,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one sequence's page attention. The point where a subclass swaps kernels."""
        return _call_kernel(
            "page attention",
            self._attn_fn,
            query,
            row_table,
            k_pages,
            v_pages,
            index_table,
            mask_tiles,
            self.scale,
            num_blocks,
            padded_query_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.logits_soft_cap,
            alibi_bias_tiles,
            out,
        )

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
        index_tables = self.index_tables(attn_metadata, _target_device)
        # Let the kernel write its output buffer directly, saving a copy per layer.
        store_out = self._compile_attn
        assert mask_tiles_all is not None, (
            "attention_mask_tiles_device must be mirrored by forward()"
        )

        num_decode_seqs = attn_metadata.num_decode_seqs
        batched_done = False
        if self._batched_decode_preconditions_met(attn_metadata):
            self._run_batched_decode_dispatch(query_dev, k_pages, v_pages, attn_metadata, output)
            if num_decode_seqs == num_seqs:
                return output
            batched_done = True

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

        # In a mixed batch, skip the decode rows already written by the batched kernel.
        seq_start = num_decode_seqs if batched_done else 0

        for seq_idx in range(seq_start, num_seqs):
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
            index_table = index_tables[seq_idx]
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
            result = self._run_page_attn(
                q_staging,
                row_table,
                k_pages,
                v_pages,
                index_table,
                mask_tiles,
                len(active_bs),
                aligned_query_lens[seq_idx],
                alibi_bias_tiles,
                out_staging if store_out else None,
            )

            assert result.dtype == output.dtype
            if store_out:
                # The kernel wrote `out_staging` itself; copied back below.
                continue
            dest[q_start:q_end] = result[:query_len]

        if store_out and not pre_staged:
            if batched_done:
                # Decode rows are already in output; copy only the prefill suffix.
                prefill_token_start = attn_metadata.num_decode_tokens
                output[prefill_token_start:batch_rows].copy_(
                    out_staging[prefill_token_start:batch_rows]
                )
            else:
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
