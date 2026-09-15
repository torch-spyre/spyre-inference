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
per-token destinations are one head apart rather than contiguous.

Everything above the cache's memory is shared with ``spyre_attn``; the four places that
touch it — advertised shape, allocation, kernels, store index — are duplicated rather
than parameterised.
"""

import torch
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.kv_cache_interface import AttentionSpec

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyrePagedKVCache,
)
from spyre_inference.v1.attention.ops.batched_decode_head_major import (
    batched_decode_head_major_kernel,
)
from spyre_inference.v1.attention.ops.layout import head_major_kv_layout
from spyre_inference.v1.attention.ops.page_attn_head_major import page_attn_head_major_kernel
from spyre_inference.v1.attention.ops.reshape_and_cache_head_major import (
    reshape_and_cache_head_major_kernel,
)

logger = init_logger(__name__)

# Compiled apart from the token-major kernels: same reason those are compiled at module
# scope, and a shared artifact would guard on the page shape either way.
_page_attn_compiled = torch.compile(page_attn_head_major_kernel, dynamic=False)
_batched_decode_compiled = torch.compile(batched_decode_head_major_kernel, dynamic=False)


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
        self._attn_fn = _page_attn_compiled if self._compile_attn else page_attn_head_major_kernel
        self._decode_fn = _batched_decode_compiled

        logger.info_once("Using SpyreHeadMajorAttentionBackend with a head-major paged KV cache")

    @classmethod
    def allocate_pages(
        cls, num_blocks: int, spec: AttentionSpec, device: torch.device
    ) -> SpyrePagedKVCache:
        layout = head_major_kv_layout(
            num_blocks * spec.num_kv_heads * spec.block_size, spec.head_size, torch.float16
        )
        shape = (num_blocks, spec.num_kv_heads, spec.block_size, spec.head_size)
        return SpyrePagedKVCache(
            k_pages=torch.zeros(shape, dtype=torch.float16).to(device, device_layout=layout),  # ty: ignore[no-matching-overload]
            v_pages=torch.zeros(shape, dtype=torch.float16).to(device, device_layout=layout),  # ty: ignore[no-matching-overload]
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
