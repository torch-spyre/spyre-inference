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

"""Paged KV-cache attention backend for Spyre using list-of-pages and online softmax."""

import functools
from dataclasses import dataclass
from typing import Callable, ClassVar, NamedTuple

import os

import torch

from spyre_inference.custom_ops.utils import convert

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.config.cache import CacheDType
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

logger = init_logger(__name__)

# When set, wraps forward(), _reshape_and_cache(), and _online_softmax_attention()
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


# Force torch.compile(dynamic=False) on the Spyre attention/reshape kernels
# regardless of the vLLM compilation config. Used to evaluate the compiled path
# on Spyre, where CompilationMode.NONE otherwise makes _maybe_compile a no-op.
# Default: off (unset or "0").
_FORCE_COMPILE_ATTN = os.environ.get("SPYRE_FORCE_COMPILE_ATTN", "0") == "1"

# TODO: Make these hyperparameters configurable
# KV length alignment: KV tensors are padded to the next multiple of this value.
# Because torch.compile treats shapes as static constants, every distinct kv_len
# triggers a full recompile. Aligning to 256 buckets sequence lengths into tiers
# (256, 512, 768, ...) so only the first request at each tier pays compilation cost,
# rather than recompiling on every decode step.
KV_LENGTH_ALIGNMENT = 256

# Query chunk size for padding - ensures consistent tensor sizes for Spyre compilation
# TODO: decode tokens (max_query_len=1) are always padded to 32, which is wasteful.
# Explore a separate decode kernel path that doesn't need query padding, or use
# a smaller alignment (e.g. QUERY_CHUNK_SIZE=1) for single-token decode steps.
QUERY_CHUNK_SIZE = 32


class SpyrePagedKVCache(NamedTuple):
    """Per-layer paged KV cache for the Spyre backend.

    NamedTuple (not dataclass) because it is a tuple at runtime — Dynamo
    specializes tuple subscripts at trace time, which is what makes the
    compile-unrolled per-page loop in `_create_compilable_page_attn` work.
    A regular dataclass would cross an unverified line with Dynamo's tracing
    of attribute access on custom objects. Index-by-int and unpacking
    (`k_pages, v_pages = cache`) keep working unchanged.

    Allocated by `TorchSpyreModelRunner.initialize_kv_cache_tensors` and
    consumed by `SpyreAttentionImpl.forward`. vLLM's `bind_kv_cache` types
    the relay path as `dict[str, torch.Tensor]`; see the suppression at the
    `bind_kv_cache(...)` call site for why that type-hole is benign.
    """

    k_pages: list[torch.Tensor]
    v_pages: list[torch.Tensor]


def _overwrite(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: list[int],
    offsets: list[int],
) -> None:
    """Write input into output at the specified position (in-place)."""
    if output.device.type == "spyre":
        # `torch.ops.spyre.overwrite` is dynamically registered, so its
        # signature is opaque to the type checker (ParamSpec resolves to `...`).
        torch.ops.spyre.overwrite(
            input,  # ty: ignore[invalid-argument-type]
            output,  # ty: ignore[invalid-argument-type]
            dims,  # ty: ignore[invalid-argument-type]
            offsets,  # ty: ignore[invalid-argument-type]
        )
    else:
        # intended behaviour on cpu
        sliced_t = output
        for i, dim in enumerate(dims):
            sliced_t = torch.narrow(sliced_t, dim, offsets[i], 1)
        sliced_t.copy_(input)


def _indirect_matmul_mock(
    a: torch.Tensor | list[torch.Tensor],
    address_or_index_of_a: int | torch.Tensor | None,
    b: torch.Tensor | list[torch.Tensor],
    address_or_index_of_b: int | torch.Tensor | None,
    # we need the option to transform a and/or b, after the indirect access
    transform_a: Callable | None = None,
    transform_b: Callable | None = None,
) -> torch.Tensor:
    """mock implementation for custom indirect matmul

    address_or_index_of_ : this can be both: index if running on the CPU or if
                           the outer-dimension of the tensors are lists. Or then
                           absolute addresses if it is supported on Spyre.
                           The semantic is always to access ONE slice of the outer-most
                           dimension: I.e. if we have a 2D tensor like [8, 32, 32], then
                           the address_or_index_of would access the first dimension here
                           (which has the size of 8) and return one element of shape
                           [1, 32, 32]. Same example for a 3D tensor: if tensor a
                           would be e.g. [8, 32, 8, 128], and the
                           address_or_index_of_a is 5, then this matmul would
                           access the 5th slice of Tensor a and return a slice of
                           shape [1, 32, 8, 128].

    transform_ : This is an optional torch-compilable function to transform (e.g.
                 transpose/rotate) the tensor-slice after it was loaded via
                 the indirect access before the matmul happens.

    """
    current_device = a[0].device.type  # works with tensors and lists (?)
    if current_device == "spyre":
        # constraints for now -> this should change with true indirect access
        # on the cpu, it also works with "true" indirect access, meaning a/b being tensors
        assert isinstance(a, list) or address_or_index_of_a is None, (
            "here needs to be true indirect access"
        )
        assert isinstance(b, list) or address_or_index_of_b is None, (
            "here needs to be true indirect access"
        )

    # resolving indirect access
    # it is important here that this DOES NOT RESULT in new tensors being realized in DRAM
    # hence, it has to be views like here
    if isinstance(a, list) or (isinstance(a, torch.Tensor) and address_or_index_of_a is not None):
        if isinstance(address_or_index_of_a, torch.Tensor):
            assert len(address_or_index_of_a) == 1, "for now, we support only one page at a time"
            idx_a = int(address_or_index_of_a.item())
        else:
            assert address_or_index_of_a is not None
            idx_a = address_or_index_of_a
        # pytorch syntax is the same like for python lists here
        a = a[idx_a]
        if transform_a:
            a = transform_a(a)
    if isinstance(b, list) or (isinstance(b, torch.Tensor) and address_or_index_of_b is not None):
        if isinstance(address_or_index_of_b, torch.Tensor):
            assert len(address_or_index_of_b) == 1, "for now, we support only one page at a time"
            idx_b = int(address_or_index_of_b.item())
        else:
            assert address_or_index_of_b is not None
            idx_b = address_or_index_of_b
        b = b[idx_b]
        if transform_b:
            b = transform_b(b)

    # do the actual matmul
    output = torch.matmul(a, b)
    return output


def _maybe_compile(fn):
    """Compile fn unless vLLM's compilation config disables it.

    Mirrors the gating in CustomOp.maybe_compile.
    """
    from vllm.config import get_current_vllm_config_or_none
    from vllm.config.compilation import CompilationMode

    config = get_current_vllm_config_or_none()
    if config is None:
        return fn
    cfg = config.compilation_config
    if cfg.mode == CompilationMode.NONE or cfg.backend == "eager":
        return fn
    return torch.compile(fn, dynamic=False)


def _maybe_compile_attn(fn):
    """Like _maybe_compile, but also honors the SPYRE_FORCE_COMPILE_ATTN
    escape hatch.

    Used only for the online-softmax attention kernel — the reshape/cache
    kernel is *not* covered, because forcing compile on it currently hits an
    unsupported torch-spyre Inductor path (missing device_tensor_layout on
    graph input). Flip _get_reshape_fn to call this helper too once that gap
    is resolved.
    """
    if _FORCE_COMPILE_ATTN:
        return torch.compile(fn, dynamic=False)
    return _maybe_compile(fn)


# ---------------------------------------------------------------------------
# Compilable factory functions
# ---------------------------------------------------------------------------


def _create_compilable_reshape_and_cache(num_tokens: int):
    """Create a reshape_and_cache with fixed token count for torch.compile.

    Dynamo unrolls the loop because num_tokens is a closure constant.
    """

    def specialized_reshape_and_cache_kernel(
        key,
        value,
        k_pages,
        v_pages,
        block_indices,
        block_offsets,
        target_device,
    ):
        for t in range(num_tokens):
            k_tok = convert(key[t].unsqueeze(1).contiguous(), target_device)
            v_tok = convert(value[t].unsqueeze(1).contiguous(), target_device)
            _overwrite(k_tok, k_pages[block_indices[t]], [1], [block_offsets[t]])
            _overwrite(v_tok, v_pages[block_indices[t]], [1], [block_offsets[t]])

    return specialized_reshape_and_cache_kernel


def _create_compilable_page_attn(
    num_blocks: int,
    padded_query_len: int,
    has_alibi: bool = False,
    logits_soft_cap: float = 0.0,
):
    """Create online softmax attention over a fixed number of pages for torch.compile.

    Dynamo unrolls the loop because num_blocks, padded_query_len, has_alibi, and
    logits_soft_cap are closure constants.
    """

    def specialized_paged_attn_kernel(
        q,
        k_pages,
        v_pages,
        page_indices,
        mask_tiles,
        scale,
        alibi_bias_tiles=None,
    ):
        """
        This kernels specializes for num_blocks and padded_query_len.

        Expected shapes:
            k_pages: list of [num_kv_heads, block_size, head_size]
            v_pages: list of [num_kv_heads, block_size, head_size]
            page_indices: [num_blocks]
            mask_tiles: [num_blocks]
            alibi_bias_tiles: list of [num_kv_heads, num_queries_per_kv, 1, block_size]
                (only when has_alibi=True; None otherwise). The query-axis dim
                is 1 because softmax absorbs per-query-row constants — see
                the derivation at the bias-tile construction site in
                _online_softmax_attention.
        """

        tile_max = None
        tile_sum = None
        tile_output = None

        for i in range(num_blocks):
            page_idx = page_indices[i]
            # Syntax with views and indirect access
            # (i.e. instead of _indirect_matmul_mock)
            # k_page = k_pages[page_idx]
            # v_page = v_pages[page_idx]
            # k_page_4d = k_page.unsqueeze(1)
            # v_page_4d = v_page.unsqueeze(1)

            mask_tile = mask_tiles[i]

            # scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
            # NOTE: for true "varlen" layout, q would be
            # an indirect access too (avoided here for simplicity...)
            scores = _indirect_matmul_mock(
                q, None, k_pages, page_idx, transform_b=lambda t: t.unsqueeze(1).transpose(-2, -1)
            )
            scores *= scale
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
                # tile_output = torch.matmul(tile_probs, v_page_4d)
                tile_output = _indirect_matmul_mock(
                    tile_probs, None, v_pages, page_idx, transform_b=lambda t: t.unsqueeze(1)
                )
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
                # tile_output = tile_output + torch.matmul(tile_probs, v_page_4d)
                tile_output += _indirect_matmul_mock(
                    tile_probs, None, v_pages, page_idx, transform_b=lambda t: t.unsqueeze(1)
                )
                tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
                tile_max = new_max

        assert tile_output is not None and tile_sum is not None
        return tile_output / tile_sum

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

    # Precomputed from slot_mapping to avoid CPU round-trips during forward:
    # each entry is the physical page index for one token.
    slot_block_indices: list[int]

    # Precomputed from slot_mapping: offset within the page for each token.
    slot_block_offsets: list[int]

    # True when causal masking is needed (prefill/mixed, i.e. max_query_len > 1).
    # Decode steps (max_query_len=1) don't need explicit causal masking because
    # the online softmax over KV pages naturally only attends to past tokens.
    apply_causal_mask: bool = False

    # Number of KV heads (for GQA).
    num_kv_heads: int = 0

    # Number of query heads.
    num_heads: int = 0

    # Pre-tiled additive attention mask. attention_mask_tiles[seq_idx][block_idx]
    # gives the mask tile for one KV page of one sequence.
    # Each tile: [aligned_max_query_len, block_size] on CPU.
    attention_mask_tiles: list[list[torch.Tensor]] | None = None

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
                f"block_size must be a multiple of 64 for the list-based attention "
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
        """Build additive attention mask on Spyre.

        All sequences share the same aligned_max_query_len so every mask tile
        has a uniform query dimension — this avoids per-sequence kernel
        specializations.

        Returns:
            - mask: [num_seqs, aligned_max_query_len, aligned_max_seq_len] additive mask
        """
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

        # Sliding window mask: limit attention to recent tokens
        # For query at absolute position p, attend to [max(0, p - sliding_window + 1), p]
        # Example: p=5, window=4 → attend to [2,3,4,5] (4 tokens)
        # Note: computed on max_query_len; padding applied after (lines 454-462)
        if self.sliding_window is not None:
            context_lens = seq_lens - query_lens  # [num_seqs]
            # Absolute query position: context_len + q_pos
            # Window start: max(0, absolute_pos - sliding_window + 1)
            window_start = (
                context_lens.unsqueeze(1) + q_pos.unsqueeze(0) - self.sliding_window + 1
            ).clamp(min=0)
            kv_pos_exp = kv_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, aligned_max_seq_len]
            window_ok = kv_pos_exp >= window_start.unsqueeze(
                2
            )  # [num_seqs, max_query_len, aligned_max_seq_len]
            attend = attend & window_ok

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
        apply_causal_mask = causal and max_query_len > 1

        aligned_max_query_len = (
            (max_query_len + QUERY_CHUNK_SIZE - 1) // QUERY_CHUNK_SIZE * QUERY_CHUNK_SIZE
        )
        aligned_max_seq_len = (
            (max_seq_len + KV_LENGTH_ALIGNMENT - 1) // KV_LENGTH_ALIGNMENT * KV_LENGTH_ALIGNMENT
        )

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
        num_seqs = common_attn_metadata.num_reqs
        block_size = self.block_size
        attention_mask_tiles: list[list[torch.Tensor]] = []
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

        # Precompute slot indices on CPU to avoid CPU round-trip during forward
        sm_cpu = slot_mapping.detach().cpu()
        slot_block_indices = (sm_cpu // self.block_size).tolist()
        slot_block_offsets = (sm_cpu % self.block_size).tolist()

        # NOTE: since the outer loop of the paged attention implementation
        #  runs on the CPU (list-based), most meta-data also remains on CPU
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
            slot_block_indices=slot_block_indices,
            slot_block_offsets=slot_block_offsets,
            apply_causal_mask=apply_causal_mask,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            attention_mask_tiles=attention_mask_tiles,
            aligned_max_query_len=aligned_max_query_len,
            aligned_max_seq_len=aligned_max_seq_len,
        )


class SpyreAttentionBackend(AttentionBackend):
    """Paged KV-cache attention backend for Spyre."""

    accept_output_buffer: bool = True
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
        return [  # ty: ignore[invalid-return-type]
            (num_blocks, block_size, num_kv_heads, head_size),
            (num_blocks, block_size, num_kv_heads, head_size),
        ]

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

    KV cache is a tuple (k_pages, v_pages) where each is a list of tensors
    of shape [num_kv_heads, block_size, head_size] on Spyre. No monolithic
    cache tensor, no gather masks.

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

        # Compiled function caches (keyed by iteration count for reshape, and
        # by (num_blocks, padded_query_len) for the per-page attention loop)
        self._reshape_fns: dict[int, object] = {}
        self._attn_fns: dict[tuple[int, int], object] = {}

        logger.debug_once("Using SpyreAttentionBackend with LIST-BASED online softmax")

    def _get_reshape_fn(self, num_tokens: int):
        if num_tokens not in self._reshape_fns:
            # Deliberately uses _maybe_compile (not _maybe_compile_attn):
            # SPYRE_FORCE_COMPILE_ATTN must not force-compile this kernel.
            # torch.compile of _create_compilable_reshape_and_cache currently
            # fails inside torch-spyre's Inductor pass with
            # `Unsupported: missing device_tensor_layout on graph input arg0_1`.
            # Move to _maybe_compile_attn once that gap is resolved.
            self._reshape_fns[num_tokens] = _maybe_compile(
                _create_compilable_reshape_and_cache(num_tokens)
            )
        return self._reshape_fns[num_tokens]

    def _get_attn_fn(self, num_blocks: int, padded_query_len: int):
        # self.alibi_slopes and self.logits_soft_cap are fixed per instance, so
        # has_alibi and logits_soft_cap don't need to be part of the cache key.
        key = (num_blocks, padded_query_len)
        if key not in self._attn_fns:
            self._attn_fns[key] = _maybe_compile_attn(
                _create_compilable_page_attn(
                    num_blocks,
                    padded_query_len,
                    has_alibi=self.alibi_slopes is not None,
                    logits_soft_cap=self.logits_soft_cap,
                )
            )
        return self._attn_fns[key]

    # `kv_cache` widens the base's `torch.Tensor` to `SpyrePagedKVCache`,
    # which `TorchSpyreModelRunner.initialize_kv_cache_tensors` allocates
    # and `bind_kv_cache` smuggles through a dict typed `dict[str, Tensor]`.
    # The matching pair of overrides preserves the runtime contract; ty
    # cannot see the co-evolution.
    @_record_function("spyre_attn::forward")
    def forward(  # ty: ignore[invalid-method-override]
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
        # Derive target device from the KV pages — query may arrive on CPU
        # (e.g. in unit tests) while pages live on the real Spyre device.
        _target_device = k_pages[0].device
        num_actual_tokens = attn_metadata.num_actual_tokens

        # Spyre slicing corrupts memory, so
        # bring q/k/v to CPU once for all slicing below; per-token slices get
        # transferred to Spyre individually inside the scatter and attention paths.
        key_cpu = convert(key, "cpu")
        value_cpu = convert(value, "cpu")
        query_cpu = convert(query, "cpu")

        # Step 1: Reshape and cache — write new tokens into pages
        self._reshape_and_cache(
            key_cpu[:num_actual_tokens],
            value_cpu[:num_actual_tokens],
            k_pages,
            v_pages,
            attn_metadata.slot_block_indices[:num_actual_tokens],
            attn_metadata.slot_block_offsets[:num_actual_tokens],
            _target_device,
        )

        # Step 2: Online softmax attention over pages (varlen)
        output = self._online_softmax_attention(
            query_cpu[:num_actual_tokens],
            k_pages,
            v_pages,
            attn_metadata,
            output,
            _target_device,
        )

        return output

    @_record_function("spyre_attn::reshape_and_cache")
    def _reshape_and_cache(
        self,
        key_cpu: torch.Tensor,
        value_cpu: torch.Tensor,
        k_pages: list[torch.Tensor],
        v_pages: list[torch.Tensor],
        block_indices: list[int],
        block_offsets: list[int],
        _target_device: torch.device,
    ) -> None:
        """Write new K/V tokens into their respective pages.

        key, value: [num_tokens, num_kv_heads, head_size]
        k_pages, v_pages: list[Tensor], each [num_kv_heads, block_size, head_size]
        block_indices, block_offsets: precomputed from slot_mapping in metadata builder
        """
        num_tokens = key_cpu.shape[0]

        # Force CPU contiguous: value from QKV split-along-last-dim is
        # non-contiguous; transferring a non-contiguous CPU tensor to Spyre
        # silently corrupts data (see custom_ops/silu_and_mul.py).
        key_cpu = key_cpu.contiguous()
        value_cpu = value_cpu.contiguous()

        fn = self._get_reshape_fn(num_tokens)
        fn(key_cpu, value_cpu, k_pages, v_pages, block_indices, block_offsets, _target_device)

    @_record_function("spyre_attn::online_softmax")
    def _online_softmax_attention(
        self,
        query_cpu: torch.Tensor,
        k_pages: list[torch.Tensor],
        v_pages: list[torch.Tensor],
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,
        _target_device: torch.device,
    ) -> torch.Tensor:
        """FlashAttention-style online softmax iterating over KV pages (varlen).

        Handles multiple sequences using query_start_loc for the varlen layout.
        Each k_page/v_page is [num_kv_heads, block_size, head_size] — a complete
        tensor on Spyre, passed to bmm directly without slicing.

        Writes results directly into the caller's output buffer in-place.
        """
        num_heads = self.num_heads
        head_size = self.head_size
        num_kv_heads = self.num_kv_heads
        num_queries_per_kv = self.num_queries_per_kv
        block_size = attn_metadata.block_size

        num_seqs = attn_metadata.num_seqs
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        mask_tiles_all = attn_metadata.attention_mask_tiles
        aligned_max_query_len = attn_metadata.aligned_max_query_len
        assert mask_tiles_all is not None, (
            "attention_mask_tiles must be precomputed by the metadata builder"
        )

        # Scattering into `output` on Spyre dim=0 has no working primitive:
        # `output[i:j] = ...` and `narrow().copy_(...)` silently write to row 0;
        # `torch.ops.spyre.overwrite` is deprecated and its compile_once wrapper
        # compiles one SDSC binary per unique offset, which recurses past the
        # dynamo cache limit once vLLM's model compile has filled it. Raising
        # the limit unblocks short tests but compiles N binaries for a
        # query_len=N prefill, which doesn't scale to long contexts. Stage the
        # result on CPU and bulk-copy at the end of the per-sequence loop.
        # Revisit when torch-spyre lands symbolic-offset overwrite
        # (torch-spyre#220 / #1371-3).
        output_cpu = torch.zeros_like(output, device="cpu")

        for seq_idx in range(num_seqs):
            # Most-naive implementation: no parallelization
            # over sequences or GQA optimization
            q_start = int(query_start_loc[seq_idx].item())
            q_end = int(query_start_loc[seq_idx + 1].item())
            query_len = q_end - q_start
            kv_len = int(seq_lens[seq_idx].item())

            q_seq = query_cpu[q_start:q_end]

            # Pad query to global aligned_max_query_len (uniform for all seqs)
            if aligned_max_query_len > query_len:
                q_seq = torch.nn.functional.pad(
                    q_seq,
                    (0, 0, 0, 0, 0, aligned_max_query_len - query_len),
                    mode="constant",
                    value=0.0,
                )

            # Reshape: [padded_query_len, num_heads, head_size]
            #   → [num_kv_heads, num_queries_per_kv, padded_query_len, head_size]
            q = q_seq.unsqueeze(0).transpose(1, 2).contiguous()
            q = q.reshape(num_kv_heads, num_queries_per_kv, aligned_max_query_len, head_size)

            # TODO: MHA (num_queries_per_kv=1) currently fails due to a Spyre compiler
            # bug in layout propagation through transpose operations. The compiler's
            # deadcode elimination pass fails with stride/index mismatches when
            # handling the degenerate dimension in MHA. GQA (num_queries_per_kv > 1)
            # works correctly. See error: "cannot restickify any input layout of y
            # to carry y_var=d2" in propagate_layouts.py:341

            num_blocks_needed = (kv_len + block_size - 1) // block_size
            page_indices = [int(block_table[seq_idx, i]) for i in range(num_blocks_needed)]
            # mask_tiles = [m.to(_target_device) for m in mask_tiles_all[seq_idx]]
            mask_tiles = [convert(m, device=_target_device) for m in mask_tiles_all[seq_idx]]

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
                for b in range(num_blocks_needed):
                    kv_pos = torch.arange(
                        b * block_size,
                        (b + 1) * block_size,
                        dtype=torch.float16,
                    )
                    rel = (kv_pos - context_len).view(1, 1, 1, block_size)
                    bias = self.alibi_slopes * rel
                    alibi_bias_tiles.append(convert(bias, device=_target_device))

            # Run attention on target device
            q_dev = convert(q, device=_target_device)
            attn_fn = self._get_attn_fn(num_blocks_needed, aligned_max_query_len)
            result = attn_fn(
                q_dev,
                k_pages,
                v_pages,
                page_indices,
                mask_tiles,
                self.scale,
                alibi_bias_tiles=alibi_bias_tiles,
            )

            # Reshape back: [num_kv_heads, num_queries_per_kv, padded_query_len, head_size]
            #   → [query_len, num_heads, head_size]
            # Pull result to CPU (Spyre transpose+contiguous on the head axes
            # is broken) and write into the CPU staging buffer; one bulk H2D
            # at the end of the loop replaces the per-token writes.
            result_cpu = convert(result, "cpu", output.dtype)
            result_cpu = result_cpu.reshape(1, num_heads, aligned_max_query_len, head_size)
            result_cpu = result_cpu.transpose(1, 2).contiguous()
            output_cpu[q_start:q_end] = result_cpu[0, :query_len, :, :]

        output.copy_(convert(output_cpu, device=_target_device))
        return output
