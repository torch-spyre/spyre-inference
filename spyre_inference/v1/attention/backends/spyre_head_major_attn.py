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

"""Paged attention over a head-major KV cache: ``SPYRE_ATTN_KV_LAYOUT=head_major``.

Storing a page as ``[num_kv_heads, block_size, head_size]`` drops the permute the
token-major kernels do before the matmuls, and pays for it in the KV write, whose
per-token destinations are one head apart rather than contiguous. The cache is decomposed
so a gathered page stays LX-resident; see ``page_attn_head_major_decode``. Past one query
token that residency stops paying, and ``page_attn_head_major_prefill`` runs instead; across
sequences at decode, ``batched_decode_head_major`` gathers whole pages, which are LX-resident
too, so it needs none of the per-sequence kernel's ``(page, kv_head)`` fold.

Everything above the cache's memory is shared with ``spyre_attn``; the places that touch
it — advertised shape, allocation, kernels, index tables — are duplicated rather than
parameterised. This layout does not carry ALiBi.
"""

from collections.abc import Callable
from contextvars import copy_context
from typing import ClassVar, TypeVar

import torch
from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.kv_cache_interface import AttentionSpec

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    FP8_KV_CACHE_DTYPES,
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadata,
    SpyrePagedKVCache,
)
from spyre_inference.v1.attention.ops.batched_decode_head_major import (
    batched_decode_head_major_kernel,
)
from spyre_inference.v1.attention.ops.layout import (
    head_major_kv_layout,
    temporary_chunk_major_page_index_layout,
)
from spyre_inference.v1.attention.ops.page_attn_head_major_decode import (
    page_attn_head_major_decode_kernel,
)
from spyre_inference.v1.attention.ops.page_attn_head_major_prefill import (
    page_attn_head_major_prefill_kernel,
)
from spyre_inference.v1.attention.ops.reshape_and_cache_head_major import (
    reshape_and_cache_head_major_kernel,
)
from spyre_inference.v1.attention.ops.tile_loop import USE_FOR_EACH_TILE
from spyre_inference.v1.worker import compile_guard

logger = init_logger(__name__)

# Compiled apart from the token-major kernels: same reason those are compiled at module
# scope, and a shared artifact would guard on the page shape either way.
# Kernels already specialise per padded_query_len, so dispatching per regime adds no compiles.
#
# The kernels that walk their pages tiled need fullgraph because of ``for_each_tile``.
_page_attn_prefill_compiled = torch.compile(
    page_attn_head_major_prefill_kernel, dynamic=False, fullgraph=USE_FOR_EACH_TILE
)
_page_attn_decode_compiled = torch.compile(
    page_attn_head_major_decode_kernel, dynamic=False, fullgraph=USE_FOR_EACH_TILE
)
_batched_decode_compiled = torch.compile(
    batched_decode_head_major_kernel, dynamic=False, fullgraph=USE_FOR_EACH_TILE
)

# Warmup's recorder covers these, so a compile afterwards is a coverage gap.
compile_guard.watch(
    page_attn_head_major_prefill_kernel, "page attention prefill kernel (head-major)"
)
compile_guard.watch(page_attn_head_major_decode_kernel, "page attention decode kernel (head-major)")
compile_guard.watch(batched_decode_head_major_kernel, "batched decode kernel (head-major)")
compile_guard.watch(reshape_and_cache_head_major_kernel, "reshape_and_cache kernel (head-major)")

_SPYRE_CORES = 32
_LX_ATTN_CORES = 8
_T = TypeVar("_T")


def _lx_max_cores(output_units: int) -> int:
    """Core cap for one LX attention compile, 0 for uncapped.

    Capping is needed only when the bmm's output axes (num_kv_heads * padded_query_len)
    cannot fill the cores alone, since filling them then means K-splitting the reduction
    and a gather cannot mirror a split on a value table's data dim.
    """
    override = envs.SPYRE_ATTN_MAX_CORES
    if override:
        return override
    return 0 if output_units >= _SPYRE_CORES else _LX_ATTN_CORES


def _run_with_capped_cores(output_units: int, fn: Callable[..., _T], *args) -> _T:
    # work_division reads config.sencores per compile, which is what keeps the rest of
    # the model uncapped.
    max_cores = _lx_max_cores(output_units)
    if not max_cores:
        return fn(*args)
    from torch_spyre._inductor import config as ts_config

    def run():
        ts_config.sencores = max_cores
        return fn(*args)

    # Config overrides are context-local; leave the caller's unset state intact.
    return copy_context().run(run)


class SpyreHeadMajorAttentionBackend(SpyreAttentionBackend):
    """Head-major variant of the paged KV-cache backend."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        *SpyreAttentionBackend.supported_kv_cache_dtypes,
        *FP8_KV_CACHE_DTYPES,
    ]

    @staticmethod
    def get_impl_cls() -> type["SpyreHeadMajorAttentionImpl"]:
        return SpyreHeadMajorAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, num_kv_heads, block_size, head_size)


class SpyreHeadMajorAttentionImpl(SpyreAttentionImpl):
    """Online-softmax paged attention over a ``[num_blocks, KV, block_size, D]`` cache.

    A float8 cache stores ``K / k_scale`` and ``V / v_scale``, dequantized on read.
    """

    supports_fp8_kv_cache: ClassVar[bool] = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # At construction: forward() runs past a custom-op boundary that loses the config.
        self.block_size: int = get_current_vllm_config().cache_config.block_size

        self._reshape_fn = torch.compile(reshape_and_cache_head_major_kernel, dynamic=False)
        # Always the compiled kernel, even under --enforce-eager: the gather that keeps a
        # page LX-resident is a 2-D subscript, which lowers to aten.index and fails eager
        # by upcasting the int32 index to int64. Attention compiles in its own domain, so
        # this leaves the rest of the model eager.
        self._decode_attn_fn = _page_attn_decode_compiled
        self._decode_fn = _batched_decode_compiled
        if self.alibi_slopes is not None:
            raise NotImplementedError(
                "ALiBi is not supported on the head-major KV layout; use the "
                "token-major layout (SPYRE_ATTN_KV_LAYOUT=token_major)."
            )
        self._folded: SpyrePagedKVCache | None = None
        self._kv_row_pool_device: torch.Tensor | None = None

        logger.info_once(
            "Using SpyreHeadMajorAttentionBackend with a head-major paged KV cache, "
            "LX-resident pages"
        )

    @classmethod
    def allocate_pages(
        cls, num_blocks: int, spec: AttentionSpec, device: torch.device
    ) -> SpyrePagedKVCache:
        # Upstream sizes an fp8 cache as uint8 bytes; the pages hold e4m3 values.
        dtype = torch.float8_e4m3fn if spec.dtype == torch.uint8 else spec.dtype
        layout = head_major_kv_layout(
            num_blocks * spec.num_kv_heads, spec.block_size, spec.head_size, dtype
        )
        shape = (num_blocks, spec.num_kv_heads, spec.block_size, spec.head_size)
        pages = [
            torch.empty(shape, dtype=dtype, device=device, device_layout=layout)  # ty: ignore[no-matching-overload]
            for _ in range(2)
        ]
        # Through a byte view: the device fill has no fp8 pattern.
        for p in pages:
            p.view(torch.uint8).zero_()
        return SpyrePagedKVCache(*pages)

    def _kernel_kv_scales(self) -> tuple[float, ...]:
        """Trailing (k_scale, v_scale) store arguments; none for a model-dtype cache."""
        if not self.kv_cache_fp8:
            return ()
        assert self._kv_scales is not None, (
            "KV scales unresolved: the runner reads them at load (_resolve_kv_scales)"
        )
        return self._kv_scales

    def _read_scales(self) -> tuple[float, tuple[float, ...]]:
        """The softmax scale with k_scale folded in, and the trailing (v_scale,) argument."""
        if not self.kv_cache_fp8:
            return self.scale, ()
        k_scale, v_scale = self._kernel_kv_scales()
        return self.scale * k_scale, (v_scale,)

    def kv_write_index(
        self, slot_mapping: torch.Tensor, device: torch.device
    ) -> list[torch.Tensor]:
        """Head h of the token at ``block * block_size + offset`` lives at row
        ``(block * num_kv_heads + h) * block_size + offset``.

        One tensor per head, not rows of one ``[KV, T]`` tensor: a row of a 2-D index is
        followed for its first int32 stick only (torch-spyre#3851), so every head drops the
        tokens past it -- a short-token test passes while long prefill corrupts.
        """
        block = torch.div(slot_mapping, self.block_size, rounding_mode="floor")
        base = block * self.num_kv_heads * self.block_size + slot_mapping % self.block_size
        return [
            convert(base + h * self.block_size, device=device) for h in range(self.num_kv_heads)
        ]

    def kv_slot_views(self, kv_cache: SpyrePagedKVCache) -> SpyrePagedKVCache:
        """One row per (block, kv_head, token), which is what ``kv_write_index`` indexes."""
        if self._kv_slots is None:
            k_pages, v_pages = kv_cache
            shape = (-1, k_pages.shape[3])
            self._kv_slots = SpyrePagedKVCache(k_pages.view(shape), v_pages.view(shape))
        return self._kv_slots

    def _folded_pages(self, k_pages: torch.Tensor, v_pages: torch.Tensor) -> SpyrePagedKVCache:
        """The cache as [pages * kv_head, block_size, head_size]; free under this layout."""
        if self._folded is None:
            shape = (k_pages.shape[0] * k_pages.shape[1], k_pages.shape[2], k_pages.shape[3])
            self._folded = SpyrePagedKVCache(k_pages.view(shape), v_pages.view(shape))
        return self._folded

    def _kv_row_pool(self, num_pages: int, device: torch.device) -> torch.Tensor:
        """Every page's rows in the folded cache, for the decode kernel to gather from.

        A page's rows are ``page * num_kv_heads + kv``. The kernel gathers
        them out of this pool rather than reading a per-block table, which also settles the
        question the unrolled walk had to work around: an index argument that is a row of
        a stacked table loses its layout (torch-spyre#3851,
        test_spyre_scatter_index_row_of_2d_table), and an in-graph slice of a stacked table
        gathers the wrong rows at that shape (test_spyre_in_graph_slice_of_stacked_kv_row_index).
        One pool serves every block.

        Pure cache geometry, so it is built once rather than per step, unlike the index
        tables -- which retires one H2D transfer per active block per sequence per step.
        """
        if self._kv_row_pool_device is None:
            rows = torch.arange(num_pages * self.num_kv_heads, dtype=torch.int32)
            self._kv_row_pool_device = convert(
                rows.reshape(num_pages, self.num_kv_heads, 1), device=device
            )
        return self._kv_row_pool_device

    def _tiled_batched_decode_supported(self) -> bool:
        # The split index below uploads one page ID per stick.
        return True

    def _mirror_batched_decode_indices(
        self, attn_metadata: "SpyreAttentionMetadata", device: torch.device
    ) -> None:
        # TEMPORARY until torch-spyre#4603 is validated: each page ID gets its own
        # stick. Preserve the CPU metadata shapes used by the recorder and dispatch.
        if not USE_FOR_EACH_TILE:
            return super()._mirror_batched_decode_indices(attn_metadata, device)
        assert attn_metadata.rep_row_ids_cpu is not None
        assert attn_metadata.chunk_page_ids_cpu is not None
        assert attn_metadata.mask_by_chunk_cpu is not None
        b = attn_metadata.padded_num_seqs
        bpc = attn_metadata.blocks_per_chunk
        assert b is not None and bpc is not None
        e = bpc * b
        c = attn_metadata.chunk_page_ids_cpu.shape[0] // bpc
        idx = (
            attn_metadata.chunk_page_ids_cpu.reshape(c, bpc, b)
            .reshape(c, e, 1)
            .to(torch.int32)
            .contiguous()
        )
        attn_metadata.chunk_page_ids_dev = idx.to(  # ty: ignore[no-matching-overload]
            device, device_layout=temporary_chunk_major_page_index_layout(c, e)
        )
        attn_metadata.rep_row_ids_dev = convert(attn_metadata.rep_row_ids_cpu, device=device)
        attn_metadata.mask_by_chunk_dev = convert(
            attn_metadata.mask_by_chunk_cpu.reshape(
                c, bpc, b, *attn_metadata.mask_by_chunk_cpu.shape[2:]
            ),
            device=device,
        )

    def _run_batched_decode(
        self,
        query_dev: torch.Tensor,
        rep_row_ids: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        chunk_index_tables: torch.Tensor,
        mask_by_chunk: torch.Tensor,
        b_seqs: int,
        blocks_per_chunk: int,
        block_size: int,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        scale, v_args = self._read_scales()
        return _run_with_capped_cores(
            b_seqs * blocks_per_chunk * self.num_kv_heads,
            self._decode_fn,
            query_dev,
            rep_row_ids,
            k_pages,
            v_pages,
            chunk_index_tables,
            mask_by_chunk,
            scale,
            b_seqs,
            blocks_per_chunk,
            self.num_kv_heads,
            self.num_queries_per_kv,
            block_size,
            self.head_size,
            self.logits_soft_cap,
            out,
            *v_args,
        )

    def _run_page_attn(
        self,
        query: torch.Tensor,
        row_table: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        index_table: torch.Tensor,
        mask_stack: torch.Tensor,
        num_blocks: int,
        padded_query_len: int,
        alibi_stack: torch.Tensor | None,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        # Both kernels below index `row_table` whole, so a wrong width is a shape mismatch
        # at trace time — see the base's `_run_page_attn`, which this replaces rather than
        # extends.
        assert row_table.shape == (padded_query_len,), (
            f"row table {tuple(row_table.shape)} must be 1D of padded_query_len {padded_query_len}"
        )
        # Both kernels tile `index_table` on dim 0, so a short table is a trip-count
        # mismatch rather than a shape error.
        assert index_table.shape[0] == num_blocks, (
            f"index table has {index_table.shape[0]} rows for {num_blocks} blocks"
        )
        scale, v_args = self._read_scales()
        # Beyond one query token the page transfer LX residency saves is amortised over every
        # query row, and the unrolling it costs is not.
        if padded_query_len > 1:
            return _run_with_capped_cores(
                self.num_kv_heads * padded_query_len,
                _page_attn_prefill_compiled,
                query,
                row_table,
                k_pages,
                v_pages,
                index_table,
                mask_stack,
                scale,
                num_blocks,
                padded_query_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                self.logits_soft_cap,
                out,
                *v_args,
            )

        k_folded, v_folded = self._folded_pages(k_pages, v_pages)
        kv_row_pool = self._kv_row_pool(k_pages.shape[0], query.device)
        # The folded kernel carries num_heads output units; lifting the cap for it
        # measured no difference, so it is left as is.
        return _run_with_capped_cores(
            self.num_kv_heads * padded_query_len,
            self._decode_attn_fn,
            query,
            row_table,
            k_folded,
            v_folded,
            index_table,
            kv_row_pool,
            mask_stack,
            scale,
            num_blocks,
            padded_query_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.block_size,
            self.logits_soft_cap,
            out,
            *v_args,
        )

    # `slot_mapping` narrows the base's single index tensor to the per-head list
    # `kv_write_index` publishes; ty cannot see that the pair co-evolves.
    def do_kv_cache_update(  # ty: ignore[invalid-method-override]
        self,
        layer: AttentionLayer | None,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: SpyrePagedKVCache,
        slot_mapping: list[torch.Tensor],
    ) -> torch.Tensor:
        # A source on the wrong device falls back to CPU silently, without raising.
        assert key.device.type == kv_cache[0].device.type, (
            f"kv cache update source is on {key.device.type}, pages on {kv_cache[0].device.type}"
        )
        k_rows, v_rows = self.kv_slot_views(kv_cache)
        self._reshape_fn(key, value, k_rows, v_rows, slot_mapping, *self._kernel_kv_scales())
        # Only k_rows is returned; Inductor fuses the stores into one kernel, so
        # ordering the read after it covers the V write too.
        return k_rows
