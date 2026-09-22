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
sequences at decode, ``batched_decode_head_major`` gathers whole pages for the same reason.

Everything above the cache's memory is shared with ``spyre_attn``; the places that touch
it — advertised shape, allocation, kernels, index tables — are duplicated rather than
parameterised. This layout does not carry ALiBi.
"""

import contextlib

import torch
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.kv_cache_interface import AttentionSpec

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadata,
    SpyrePagedKVCache,
    _call_kernel,
)
from spyre_inference.v1.attention.ops.batched_decode_head_major import (
    batched_decode_head_major_kernel,
)
from spyre_inference.v1.attention.ops.layout import head_major_kv_layout
from spyre_inference.v1.attention.ops.page_attn_head_major_decode import (
    page_attn_head_major_decode_kernel,
)
from spyre_inference.v1.attention.ops.page_attn_head_major_prefill import (
    page_attn_head_major_prefill_kernel,
)
from spyre_inference.v1.attention.ops.reshape_and_cache_head_major import (
    reshape_and_cache_head_major_kernel,
)
from spyre_inference.v1.worker import compile_guard

logger = init_logger(__name__)

# Compiled apart from the token-major kernels: same reason those are compiled at module
# scope, and a shared artifact would guard on the page shape either way.
# Kernels already specialise per padded_query_len, so dispatching per regime adds no compiles.
_page_attn_prefill_compiled = torch.compile(page_attn_head_major_prefill_kernel, dynamic=False)
_page_attn_decode_compiled = torch.compile(page_attn_head_major_decode_kernel, dynamic=False)
_batched_decode_compiled = torch.compile(batched_decode_head_major_kernel, dynamic=False)

# Warmup's recorder covers these, so a compile afterwards is a coverage gap.
compile_guard.watch(
    page_attn_head_major_prefill_kernel, "page attention prefill kernel (head-major)"
)
compile_guard.watch(page_attn_head_major_decode_kernel, "page attention decode kernel (head-major)")
compile_guard.watch(batched_decode_head_major_kernel, "batched decode kernel (head-major)")
compile_guard.watch(reshape_and_cache_head_major_kernel, "reshape_and_cache kernel (head-major)")

_SPYRE_CORES = 32
_LX_ATTN_CORES = 8


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


@contextlib.contextmanager
def _capped_cores(output_units: int):
    # work_division reads config.sencores per compile, which is what keeps the rest of
    # the model uncapped.
    max_cores = _lx_max_cores(output_units)
    if not max_cores:
        yield
        return
    from torch_spyre._inductor import config as ts_config

    prev = ts_config.sencores
    ts_config.sencores = max_cores
    try:
        yield
    finally:
        ts_config.sencores = prev


class SpyreHeadMajorAttentionBackend(SpyreAttentionBackend):
    """Head-major variant of the paged KV-cache backend."""

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
    """Online-softmax paged attention over a ``[num_blocks, KV, block_size, D]`` cache."""

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
                "ALiBi is not supported on the head-major KV layout; use the default "
                "token-major layout (SPYRE_ATTN_KV_LAYOUT=token_major)."
            )
        self._folded: SpyrePagedKVCache | None = None

        logger.info_once(
            "Using SpyreHeadMajorAttentionBackend with a head-major paged KV cache, "
            "LX-resident pages"
        )

    @classmethod
    def allocate_pages(
        cls, num_blocks: int, spec: AttentionSpec, device: torch.device
    ) -> SpyrePagedKVCache:
        dtype = spec.dtype
        layout = head_major_kv_layout(
            num_blocks * spec.num_kv_heads, spec.block_size, spec.head_size, dtype
        )
        shape = (num_blocks, spec.num_kv_heads, spec.block_size, spec.head_size)
        return SpyrePagedKVCache(
            k_pages=torch.zeros(shape, dtype=dtype).to(device, device_layout=layout),  # ty: ignore[no-matching-overload]
            v_pages=torch.zeros(shape, dtype=dtype).to(device, device_layout=layout),  # ty: ignore[no-matching-overload]
        )

    def kv_write_index(
        self, slot_mapping: torch.Tensor, device: torch.device
    ) -> list[torch.Tensor]:
        """Head h of the token at ``block * block_size + offset`` lives at row
        ``(block * num_kv_heads + h) * block_size + offset``.

        One offset-0 tensor per head, not rows of one ``[KV, T]`` tensor: a view's storage
        offset is dropped on the way to the device (torch-spyre#3770). That corruption is
        shape-dependent — correct while a row fits one int32 stick, every head past it
        silently wrong — so a short-token test passes while long prefill corrupts.
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

    # The base publishes one table per sequence; this layout needs two, so the pair travels
    # together and `_run_page_attn` picks the one its kernel reads.
    def build_index_tables(  # ty: ignore[invalid-method-override]
        self, attn_metadata: SpyreAttentionMetadata, device: torch.device
    ) -> list[tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Per sequence, per active block, that block's ``page * num_kv_heads + kv`` rows,
        paired with the page id alone for the wide-query kernel.

        One [KV, 1] tensor per block, not rows of one table: an index tensor reaches the
        hardware as a tensor argument, so a slice's nonzero storage offset is dropped and
        every block would gather block 0 (torch-spyre#3770).
        """
        tables_cpu = attn_metadata.page_index_tables_cpu
        assert tables_cpu is not None, "page_index_tables_cpu must come from the builder"
        heads = torch.arange(self.num_kv_heads, dtype=torch.int32).reshape(self.num_kv_heads, 1)
        query_lens = attn_metadata.aligned_query_lens
        return [
            (
                [
                    convert(int(pages[b, 0]) * self.num_kv_heads + heads, device=device)
                    for b in range(pages.shape[0])
                ],
                # Only a wide query reads these, and building them for a decode step would
                # add an H2D transfer per page to the path this layout exists to speed up.
                [
                    convert(torch.tensor([int(pages[b, 0])], dtype=torch.int32), device=device)
                    for b in range(pages.shape[0])
                ]
                if query_lens[s] > 1
                else [],
            )
            for s, pages in enumerate(tables_cpu)
        ]

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
        with _capped_cores(b_seqs * blocks_per_chunk * self.num_kv_heads):
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
        # Both kernels below index `row_table` whole, so a wrong width is a shape mismatch
        # at trace time — see the base's `_run_page_attn`, which this replaces rather than
        # extends.
        assert row_table.shape == (padded_query_len,), (
            f"row table {tuple(row_table.shape)} must be 1D of padded_query_len {padded_query_len}"
        )
        k_folded, v_folded = self._folded_pages(k_pages, v_pages)
        kv_row_table, page_table = index_table
        # Beyond one query token the page transfer LX residency saves is amortised over every
        # query row, and the unrolling it costs is not.
        if padded_query_len > 1:
            with _capped_cores(self.num_kv_heads * padded_query_len):
                return _call_kernel(
                    "page attention (prefill)",
                    _page_attn_prefill_compiled,
                    query,
                    row_table,
                    k_pages,
                    v_pages,
                    page_table,
                    mask_tiles,
                    self.scale,
                    num_blocks,
                    padded_query_len,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_size,
                    self.block_size,
                    self.logits_soft_cap,
                    out,
                )

        # The folded kernel carries num_heads output units; lifting the cap for it
        # measured no difference, so it is left as is.
        with _capped_cores(self.num_kv_heads * padded_query_len):
            return _call_kernel(
                "page attention",
                self._decode_attn_fn,
                query,
                row_table,
                k_folded,
                v_folded,
                kv_row_table,
                mask_tiles,
                self.scale,
                num_blocks,
                padded_query_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                self.block_size,
                self.logits_soft_cap,
                out,
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
        self._reshape_fn(key, value, k_rows, v_rows, slot_mapping)
        # Only k_rows is returned; Inductor fuses the stores into one kernel, so
        # ordering the read after it covers the V write too.
        return k_rows
