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

"""Accessor for the Spyre list-of-pages KV cache used by the KV connector.

The Spyre attention backend allocates each layer's KV cache as
``SpyrePagedKVCache(k_pages, v_pages)`` — a NamedTuple of two equal-length
lists of per-page tensors of shape ``[num_kv_heads, block_size, head_size]``
(see ``spyre_inference/v1/attention/backends/spyre_attn.py``). vLLM's
``bind_kv_cache`` relays those objects through a dict typed
``dict[str, torch.Tensor]``, so the connector receives page lists where it
nominally expects monolithic tensors.

This module bridges that gap without importing vLLM, NIXL, or torch_spyre:
it detects the page-list layout, exposes geometry, and reads/writes whole
pages in the same block convention as the heap accessor
(``[block_size, num_kv_heads, head_size]`` on CPU), so the connector's
staging store interoperates with both paths.
"""

from __future__ import annotations

from itertools import chain
from typing import Any

import torch

# Per-page tensor rank in the Spyre paged layout
# [num_kv_heads, block_size, head_size].
_PAGE_RANK = 3


def is_paged_layer_cache(layer_cache: Any) -> bool:
    """Return True iff ``layer_cache`` looks like ``(k_pages, v_pages)``.

    Accepts plain tuples/lists of two non-empty equal-length page lists of
    3-D tensors, as well as the typed ``SpyrePagedKVCache`` NamedTuple
    (which is a tuple at runtime). Plain staging tensors return False.
    """
    if isinstance(layer_cache, torch.Tensor):
        return False
    if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) != 2:
        return False
    k_pages, v_pages = layer_cache
    if not isinstance(k_pages, list) or not isinstance(v_pages, list):
        return False
    if not k_pages or len(k_pages) != len(v_pages):
        return False
    return all(
        isinstance(page, torch.Tensor) and page.dim() == _PAGE_RANK
        for page in chain(k_pages, v_pages)
    )


class SpyrePagedKVCacheAccessor:
    """Uniform page read/write access over per-layer Spyre page lists.

    Geometry (layers, pages, heads, block size, head dim, dtype, device) is
    inferred from the registered cache objects — nothing is hard-coded.
    Block tensors cross this boundary in the heap-accessor convention
    ``[block_size, num_kv_heads, head_size]`` on CPU; pages stay in their
    native ``[num_kv_heads, block_size, head_size]`` layout on their device.
    """

    def __init__(self, layer_caches: dict[str, Any]) -> None:
        if not layer_caches:
            raise ValueError("No layer caches provided")

        self._layer_names = sorted(layer_caches)
        self._caches: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}

        ref_shape: tuple[int, ...] | None = None
        ref_dtype: torch.dtype | None = None
        ref_device: torch.device | None = None
        ref_pages: int | None = None
        for name in self._layer_names:
            cache = layer_caches[name]
            if not is_paged_layer_cache(cache):
                raise ValueError(f"Layer {name!r} is not a (k_pages, v_pages) page list")
            k_pages, v_pages = cache
            for page in chain(k_pages, v_pages):
                shape = tuple(page.shape)
                if ref_shape is None:
                    ref_shape, ref_dtype, ref_device = shape, page.dtype, page.device
                elif shape != ref_shape or page.dtype != ref_dtype or page.device != ref_device:
                    raise ValueError(
                        f"Inconsistent page geometry: layer {name!r} has "
                        f"{shape}/{page.dtype}/{page.device}, "
                        f"expected {ref_shape}/{ref_dtype}/{ref_device}"
                    )
            if ref_pages is None:
                ref_pages = len(k_pages)
            elif len(k_pages) != ref_pages:
                raise ValueError(f"Layer {name!r} has {len(k_pages)} pages, expected {ref_pages}")
            self._caches[name] = (k_pages, v_pages)

        assert (
            ref_shape is not None
            and ref_dtype is not None
            and ref_device is not None
            and ref_pages is not None
        )
        self._num_kv_heads, self._block_size, self._head_dim = ref_shape
        self._dtype = ref_dtype
        self._num_pages = ref_pages
        self._device = ref_device

    @classmethod
    def try_from_kv_caches(cls, kv_caches: dict[str, Any]) -> SpyrePagedKVCacheAccessor | None:
        """Build an accessor if every registered layer is a page list."""
        if not kv_caches:
            return None
        if not all(is_paged_layer_cache(c) for c in kv_caches.values()):
            return None
        return cls(kv_caches)

    @property
    def layer_names(self) -> list[str]:
        return list(self._layer_names)

    @property
    def num_layers(self) -> int:
        return len(self._layer_names)

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @property
    def num_kv_heads(self) -> int:
        return self._num_kv_heads

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def page_shape(self) -> tuple[int, int, int]:
        return (self._num_kv_heads, self._block_size, self._head_dim)

    @property
    def block_shape(self) -> tuple[int, int, int]:
        return (self._block_size, self._num_kv_heads, self._head_dim)

    @property
    def page_nbytes(self) -> int:
        return (
            self._num_kv_heads
            * self._block_size
            * self._head_dim
            * torch.tensor([], dtype=self._dtype).element_size()
        )

    def _pages(self, layer_name: str, kv_kind: str) -> list[torch.Tensor]:
        kind = kv_kind.lower()
        if kind not in ("k", "v"):
            raise ValueError(f"kv_kind must be 'k' or 'v', got {kv_kind!r}")
        try:
            cache = self._caches[layer_name]
        except KeyError:
            raise KeyError(f"Unknown layer {layer_name!r}") from None
        return cache[0] if kind == "k" else cache[1]

    def page(self, layer_name: str, kv_kind: str, page_id: int) -> torch.Tensor:
        """Native page tensor [num_kv_heads, block_size, head_size], in place."""
        pages = self._pages(layer_name, kv_kind)
        if not 0 <= page_id < self._num_pages:
            raise IndexError(f"page_id {page_id} out of range [0, {self._num_pages})")
        return pages[page_id]

    def validate_block(self, values: torch.Tensor) -> None:
        if tuple(values.shape) != self.block_shape:
            raise ValueError(
                f"Expected block tensor shape {self.block_shape}, got {tuple(values.shape)}"
            )
        if values.dtype != self._dtype:
            raise ValueError(f"Expected block dtype {self._dtype}, got {values.dtype}")

    def read_block(self, *, layer_name: str, kv_kind: str, page_id: int) -> torch.Tensor:
        """Copy one page to CPU as [block_size, num_kv_heads, head_size]."""
        page = self.page(layer_name, kv_kind, page_id)
        return page.to("cpu").permute(1, 0, 2).contiguous()

    def write_block(
        self,
        *,
        layer_name: str,
        kv_kind: str,
        page_id: int,
        values: torch.Tensor,
    ) -> None:
        """Overwrite one page from a CPU block [block_size, num_kv_heads, head_size].

        The full page is replaced in one device copy; pages are never sliced
        (Spyre slicing of device tensors is unreliable — see spyre_attn.py).
        """
        self.validate_block(values)
        page = self.page(layer_name, kv_kind, page_id)
        page.copy_(values.permute(1, 0, 2).contiguous().to(page.device))

    def transfer_units(self, block_ids: list[int]) -> dict[str, Any]:
        """Describe the per-page transfer units for a set of block ids.

        One unit per (layer, kind, page): the page granularity NIXL or any
        future transport must use, since pages are independent allocations
        with no contiguous [num_pages, ...] buffer to address.
        """
        valid = [b for b in block_ids if 0 <= b < self._num_pages]
        units = self.num_layers * 2 * len(valid)
        return {
            "unit": "page",
            "unit_shape": self.page_shape,
            "unit_nbytes": self.page_nbytes,
            "units_total": units,
            "total_nbytes": units * self.page_nbytes,
            "invalid_block_ids": [b for b in block_ids if b not in valid],
        }

    def describe(self) -> dict[str, Any]:
        return {
            "layout": "list_of_pages",
            "num_layers": self.num_layers,
            "num_pages": self._num_pages,
            "page_shape": self.page_shape,
            "dtype": str(self._dtype),
            "device": str(self._device),
            "page_nbytes": self.page_nbytes,
        }
