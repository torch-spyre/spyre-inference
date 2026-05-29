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

"""Paged KV-cache attention backend for Spyre using list-of-pages and online softmax.

This backend implements attention using individual page tensors (one per KV block)
and FlashAttention-style online softmax that iterates over pages. Each page is a
complete tensor [num_kv_heads, block_size, head_size] on the target device, passed
directly to bmm without any slicing.

Architecture:
  - KV cache: two separate lists (k_pages, v_pages), managed by the model runner.
  - reshape_and_cache: writes new K/V tokens into pages via _overwrite (all on device).
  - Attention: online softmax iterating over pages — no gather/materialization step.
"""

from dataclasses import dataclass
from typing import ClassVar, Callable

import torch

from spyre_inference.custom_ops.utils import convert

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

KV_LENGTH_ALIGNMENT = 256
QUERY_CHUNK_SIZE = 32


def _overwrite(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: list[int],
    offsets: list[int],
) -> None:
    """Write input into output at the specified position (in-place)."""
    if output.device.type == "spyre":
        torch.ops.spyre.overwrite(input, output, dims, offsets)
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
            assert len(address_or_index_of_a) == 1, (
                "for now, we support only one page at a time"
            )
            address_or_index_of_a = address_or_index_of_a.item()
        # pytorch syntax is the same like for python lists here
        a = a[address_or_index_of_a]
        if transform_a:
            a = transform_a(a)
    if isinstance(b, list) or (isinstance(b, torch.Tensor) and address_or_index_of_b is not None):
        if isinstance(address_or_index_of_b, torch.Tensor):
            assert len(address_or_index_of_b) == 1, (
                "for now, we support only one page at a time"
            )
            address_or_index_of_b = address_or_index_of_b.item()
        b = b[address_or_index_of_b]
        if transform_b:
            b = transform_b(b)

    # do the actual matmul
    output = torch.matmul(a, b)
    return output


def _maybe_compile(fn):
    """Compile fn unless vLLM's compilation config disables it.

    Mirrors the gating in CustomOp.maybe_compile without requiring CustomOp
    inheritance: returns fn unchanged when compilation mode is NONE or the
    backend is "eager", otherwise wraps it with torch.compile.
    """
    from vllm.config import get_cached_compilation_config
    from vllm.config.compilation import CompilationMode

    cfg = get_cached_compilation_config()
    if cfg.mode == CompilationMode.NONE:
        return fn
    if cfg.backend == "eager":
        return fn
    return torch.compile(fn, dynamic=False)


# ---------------------------------------------------------------------------
# Compilable factory functions (PR #5 pattern)
# ---------------------------------------------------------------------------


def _create_compilable_reshape_and_cache(num_tokens: int):
    """Create a reshape_and_cache with fixed token count for torch.compile.

    Dynamo unrolls the loop because num_tokens is a closure constant.
    """

    def specialized_reshape_and_cache_kernel(
        key, value, k_pages, v_pages, block_indices, block_offsets
    ):
        # this kernels specializes (i.e. compiles for specific constants) for num_tokens
        for t in range(num_tokens):
            block_idx = block_indices[t]
            block_offset = block_offsets[t]
            k_tok = key[t].unsqueeze(1)
            v_tok = value[t].unsqueeze(1)
            _overwrite(k_tok, k_pages[block_idx], [1], [block_offset])
            _overwrite(v_tok, v_pages[block_idx], [1], [block_offset])

    return specialized_reshape_and_cache_kernel


def _create_compilable_page_attn(num_blocks: int, padded_query_len: int):
    """Create online softmax attention over a fixed number of pages for torch.compile.

    Dynamo unrolls the loop because num_blocks and padded_query_len are closure constants.
    """

    def specialized_paged_attn_kernel(q, k_pages, v_pages, page_indices, mask_tiles, scale):
        """
        This kernels specializes for num_blocks and padded_query_len.

        Expected shapes:
            k_pages: list of [num_kv_heads, block_size, head_size]
            v_pages: list of [num_kv_heads, block_size, head_size]
            page_indices: [num_blocks]
            mask_tiles: [num_blocks]
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
            scores = scores + mask_tile
            scores_max = scores.max(dim=-1, keepdim=True)[0]

            if i == 0:
                tile_max = scores_max
                tile_probs = torch.exp(scores - tile_max)
                # tile_output = torch.matmul(tile_probs, v_page_4d)
                tile_output = _indirect_matmul_mock(
                    tile_probs, None, v_pages, page_idx, transform_b=lambda t: t.unsqueeze(1)
                )
                tile_sum = tile_probs.sum(dim=-1, keepdim=True)
            else:
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

        return tile_output / tile_sum

    return specialized_paged_attn_kernel


@dataclass
class SpyreAttentionMetadata(AttentionMetadata):
    """Metadata for paged online-softmax attention on Spyre."""

    num_actual_tokens: int
    num_seqs: int
    max_query_len: int
    max_seq_len: int

    seq_lens: torch.Tensor  # [num_seqs]
    query_start_loc: torch.Tensor  # [num_seqs + 1]

    block_table: torch.Tensor  # [num_seqs, max_num_blocks_per_seq]
    block_size: int

    slot_mapping: torch.Tensor  # [num_actual_tokens]

    # Precomputed from slot_mapping (avoids CPU round-trip during forward)
    slot_block_indices: list[int] | None = None
    slot_block_offsets: list[int] | None = None

    apply_causal_mask: bool = False

    num_kv_heads: int = 0
    num_heads: int = 0

    # Pre-tiled attention mask: attention_mask_tiles[seq_idx][block_idx]
    # Each tile: [num_kv_heads, 1, padded_query_len, block_size] on target device
    attention_mask_tiles: list[list[torch.Tensor]] | None = None

    # Per-sequence padded query length for stable kernel compilation
    padded_query_lens: torch.Tensor | None = None  # [num_seqs] on CPU

    aligned_max_seq_len: int = 0

    @property
    def query_lens(self) -> torch.Tensor:
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

        model_config = vllm_config.model_config
        self.num_heads = model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)

        # Device is passed from model runner (spyre or cpu)
        self._target_device = device
        self._target_dtype = torch.float16

    def _build_attention_mask(
        self,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        apply_causal_mask: bool,
        max_query_len: int,
        aligned_max_seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build additive attention mask on Spyre.

        Returns:
            - mask_4d: [num_seqs * num_kv_heads, 1, padded_query_len, aligned_max_seq_len]
            - padded_query_lens: [num_seqs] per-sequence padded query lengths
        """
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        num_seqs = len(seq_lens)
        num_kv_heads = self.num_kv_heads

        # Compute per-sequence padded query length
        padded_query_lens = (
            (query_lens + QUERY_CHUNK_SIZE - 1) // QUERY_CHUNK_SIZE * QUERY_CHUNK_SIZE
        )  # [num_seqs]

        # Use global max for mask tensor allocation
        padded_query_len = int(padded_query_lens.max().item())

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

        # Convert to additive mask: -65504 for masked, 0 for valid (float16 min)
        mask_bool = ~attend  # [num_seqs, max_query_len, aligned_max_seq_len]

        if padded_query_len > max_query_len:
            padding = torch.ones(
                num_seqs,
                padded_query_len - max_query_len,
                aligned_max_seq_len,
                dtype=torch.bool,
                device=device,
            )
            mask_bool = torch.cat([mask_bool, padding], dim=1)

        mask_additive = torch.where(
            mask_bool,
            torch.tensor(-65504.0, dtype=self._target_dtype, device=device),
            torch.tensor(0.0, dtype=self._target_dtype, device=device),
        )

        mask_4d = (
            mask_additive.unsqueeze(1)
            .expand(-1, num_kv_heads, -1, -1)
            .reshape(num_seqs * num_kv_heads, 1, padded_query_len, aligned_max_seq_len)
            .contiguous()
        )

        return mask_4d, padded_query_lens

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SpyreAttentionMetadata:
        seq_lens = common_attn_metadata.seq_lens
        query_start_loc = common_attn_metadata.query_start_loc
        max_seq_len = common_attn_metadata.max_seq_len
        max_query_len = common_attn_metadata.max_query_len
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        apply_causal_mask = common_attn_metadata.causal and max_query_len > 1

        aligned_max_seq_len = (
            (max_seq_len + KV_LENGTH_ALIGNMENT - 1) // KV_LENGTH_ALIGNMENT * KV_LENGTH_ALIGNMENT
        )

        mask_4d_cpu, padded_query_lens = self._build_attention_mask(
            seq_lens,
            query_start_loc,
            apply_causal_mask,
            max_query_len,
            aligned_max_seq_len,
            torch.device("cpu"),
        )

        # Pre-tile the mask: split into per-sequence, per-block tiles.
        # This handles partial final pages (padding mask) and per-page attention.
        num_seqs = common_attn_metadata.num_reqs
        num_kv_heads = self.num_kv_heads
        block_size = self.block_size
        attention_mask_tiles: list[list[torch.Tensor]] = []
        for s in range(num_seqs):
            seq_tiles: list[torch.Tensor] = []
            row_start = s * num_kv_heads
            row_end = row_start + num_kv_heads
            kv_len_s = int(seq_lens[s].item())
            num_blocks_s = (kv_len_s + block_size - 1) // block_size
            # Use per-sequence padded query length for mask tile slicing
            padded_query_len_s = int(padded_query_lens[s].item())
            for b in range(num_blocks_s):
                col_start = b * block_size
                col_end = col_start + block_size
                tile = mask_4d_cpu[row_start:row_end, :, :padded_query_len_s, col_start:col_end]
                # Sanity check: ensure tile has correct shape
                assert tile.shape[2] == padded_query_len_s, (
                    f"Tile shape mismatch: expected {padded_query_len_s}, "
                    f"got {tile.shape[2]}"
                )
                seq_tiles.append(tile.contiguous().to(self.device))
            attention_mask_tiles.append(seq_tiles)

        # Precompute slot indices on CPU to avoid CPU round-trip during forward
        sm_cpu = slot_mapping.detach().cpu()
        slot_block_indices = (sm_cpu // self.block_size).tolist()
        slot_block_offsets = (sm_cpu % self.block_size).tolist()

        # NOTE: since the outer loop of the paged attention implementaiton
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
            padded_query_lens=padded_query_lens,
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
        return [MultipleOf(1)]

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
        # TODO this should be:
        #     return (num_blocks, 2, block_size, num_kv_heads, head_size)
        # but for the lists it is actually
        return [
            (num_blocks, block_size, num_kv_heads, head_size),
            (num_blocks, block_size, num_kv_heads, head_size),
        ]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
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
    via torch.compile with fixed iteration counts (PR #5 pattern). A dict
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

        self._target_device: torch.device | None = None
        self._target_dtype = torch.float16

        # Compiled function caches (keyed by iteration count)
        self._reshape_fns: dict[int, object] = {}
        self._attn_fns: dict[int, object] = {}

        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi slopes not supported yet")
        if sliding_window is not None:
            raise NotImplementedError("Sliding window not supported yet")
        if logits_soft_cap is not None:
            raise NotImplementedError("Logits soft cap not supported yet")

    def _get_reshape_fn(self, num_tokens: int):
        if num_tokens not in self._reshape_fns:
            self._reshape_fns[num_tokens] = _maybe_compile(
                _create_compilable_reshape_and_cache(num_tokens)
            )
        return self._reshape_fns[num_tokens]

    def _get_attn_fn(self, num_blocks: int, padded_query_len: int):
        key = (num_blocks, padded_query_len)
        if key not in self._attn_fns:
            self._attn_fns[key] = _maybe_compile(
                _create_compilable_page_attn(num_blocks, padded_query_len)
            )
        return self._attn_fns[key]

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[list[torch.Tensor], list[torch.Tensor]],
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided"

        if attn_metadata is None:
            return output

        k_pages, v_pages = kv_cache
        num_actual_tokens = attn_metadata.num_actual_tokens

        # Derive target device from the pages themselves
        if self._target_device is None:
            self._target_device = k_pages[0].device

        # Step 1: Reshape and cache — write new tokens into pages
        self._reshape_and_cache(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            k_pages,
            v_pages,
            attn_metadata.slot_block_indices[:num_actual_tokens],
            attn_metadata.slot_block_offsets[:num_actual_tokens],
        )

        # Step 2: Online softmax attention over pages (varlen)
        output = self._online_softmax_attention(
            query[:num_actual_tokens],
            k_pages,
            v_pages,
            attn_metadata,
            output,
        )

        return output

    def _reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_pages: list[torch.Tensor],
        v_pages: list[torch.Tensor],
        block_indices: list[int],
        block_offsets: list[int],
    ) -> None:
        """Write new K/V tokens into their respective pages.

        key, value: [num_tokens, num_kv_heads, head_size]
        k_pages, v_pages: list[Tensor], each [num_kv_heads, block_size, head_size]
        block_indices, block_offsets: precomputed from slot_mapping in metadata builder
        """
        num_tokens = key.shape[0]

        # Ensure tensors are on the target device and in the correct dtype
        k_dev = convert(key, self._target_device, self._target_dtype)
        v_dev = convert(value, self._target_device, self._target_dtype)

        fn = self._get_reshape_fn(num_tokens)
        fn(k_dev, v_dev, k_pages, v_pages, block_indices, block_offsets)

    def _online_softmax_attention(
        self,
        query: torch.Tensor,
        k_pages: list[torch.Tensor],
        v_pages: list[torch.Tensor],
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,
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
        padded_query_lens = attn_metadata.padded_query_lens  # [num_seqs]

        for seq_idx in range(num_seqs):
            # Most-naive implementation: no parallelization
            # over sequences or GQA optimization
            q_start = int(query_start_loc[seq_idx].item())
            q_end = int(query_start_loc[seq_idx + 1].item())
            query_len = q_end - q_start
            kv_len = int(seq_lens[seq_idx].item())

            # Use per-sequence padded query length
            padded_query_len = int(padded_query_lens[seq_idx].item())

            q_seq = query[q_start:q_end]

            # Pad query to per-sequence padded_query_len
            if padded_query_len > query_len:
                q_seq = torch.nn.functional.pad(
                    q_seq,
                    (0, 0, 0, 0, 0, padded_query_len - query_len),
                    mode="constant",
                    value=0.0,
                )

            # Reshape: [padded_query_len, num_heads, head_size]
            #   → [num_kv_heads, num_queries_per_kv, padded_query_len, head_size]
            q = q_seq.unsqueeze(0).transpose(1, 2).contiguous()
            q = q.reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)

            num_blocks_needed = (kv_len + block_size - 1) // block_size
            page_indices = [int(block_table[seq_idx, i]) for i in range(num_blocks_needed)]
            mask_tiles = [m.to(self._target_device) for m in mask_tiles_all[seq_idx]]

            # Run attention on target device
            q_dev = convert(q, self._target_device, self._target_dtype)
            attn_fn = self._get_attn_fn(num_blocks_needed, padded_query_len)
            result = attn_fn(q_dev, k_pages, v_pages, page_indices, mask_tiles, self.scale)

            # Reshape back: [num_kv_heads, num_queries_per_kv, padded_query_len, head_size]
            #   → [query_len, num_heads, head_size]
            result = result.reshape(1, num_heads, padded_query_len, head_size)
            result = result.transpose(1, 2).contiguous()
            seq_result = result[0, :query_len, :, :]

            # Convert to output dtype and write into output buffer
            seq_result = convert(seq_result, self._target_device, output.dtype)
            for i in range(seq_result.shape[0]):
                _overwrite(seq_result[i : i + 1], output, [0], [q_start + i])

        return output
