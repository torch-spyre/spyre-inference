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

"""Spyre variant of vLLM's OffloadingConnector.

Upstream `OffloadingConnectorWorker.register_kv_caches` canonicalizes each
layer's cache by asserting it is one `torch.Tensor` and reinterpreting its
storage (`untyped_storage()` + `.set_()` + `as_strided`). Spyre fails both: a
layer is bound to a `SpyrePagedKVCache(k_pages, v_pages)` 2-tuple, and storage
reinterpretation is unsupported on Spyre device tensors.

Canonicalization runs before `spec.get_worker()`, so no spec-level hook can fix
it — hence this subclass, which overrides exactly that one method.
"""

from collections.abc import Mapping

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)

from spyre_inference.v1.kv_offload.upstream_compat import OffloadingConnectorWorker

logger = init_logger(__name__)


def spyre_paged_to_canonical(
    kv_caches: Mapping[str, object],
    kv_cache_config: KVCacheConfig,
) -> CanonicalKVCaches:
    """Canonicalize Spyre paged KV caches without touching tensor storage.

    Each layer is bound to a `SpyrePagedKVCache(k_pages, v_pages)`, two dense
    `[num_blocks, block_size, num_kv_heads, head_size]` device tensors. Upstream
    wants `(num_blocks, page_size_bytes)` views of one storage; we instead emit
    one `CanonicalKVCacheTensor` per *pages* tensor, flattened on the trailing
    dims with `.view()`, which is metadata-only and device-safe.

    K and V therefore become two canonical tensors per physical cache, each
    carrying half of `AttentionSpec.page_size_bytes` (upstream's page size spans
    both). Layers sharing one `SpyrePagedKVCache` share both tensor indices.

    Kept as a single function so upstream drift has one blast radius.
    """
    specs = _specs_by_layer(kv_cache_config)

    tensors: list[CanonicalKVCacheTensor] = []
    # id(SpyrePagedKVCache) -> the (k_idx, v_idx) it contributed to `tensors`,
    # so layers sharing a physical cache reuse the same entries.
    indices_by_cache: dict[int, list[int]] = {}
    refs_by_layer: dict[str, list[CanonicalKVCacheRef]] = {}

    for layer_name, spec in specs.items():
        if not isinstance(spec, AttentionSpec):
            raise NotImplementedError(
                f"Spyre KV offloading supports AttentionSpec layers only; "
                f"layer {layer_name!r} has {type(spec).__name__}"
            )

        cache = kv_caches[layer_name]
        k_pages, v_pages = _unpack_paged_cache(layer_name, cache)

        # Upstream's page spans K and V together; ours are separate tensors.
        if spec.page_size_bytes % 2:
            raise ValueError(
                f"layer {layer_name!r}: odd page_size_bytes "
                f"{spec.page_size_bytes} cannot be split across K and V"
            )
        half_page = spec.page_size_bytes // 2
        _check_page_size(layer_name, k_pages, half_page)

        cache_id = id(cache)
        if cache_id not in indices_by_cache:
            indices_by_cache[cache_id] = [
                _append_flat_tensor(tensors, pages, half_page) for pages in (k_pages, v_pages)
            ]

        # mapping=None: the byte layout is device-private, so it is not
        # certified as parallelism-agnostic and stays worker-local.
        refs_by_layer[layer_name] = [
            CanonicalKVCacheRef(tensor_idx=idx, page_size_bytes=half_page, mapping=None)
            for idx in indices_by_cache[cache_id]
        ]

    group_data_refs = [
        [ref for layer_name in group.layer_names for ref in refs_by_layer[layer_name]]
        for group in kv_cache_config.kv_cache_groups
    ]
    return CanonicalKVCaches(tensors=tensors, group_data_refs=group_data_refs)


def _specs_by_layer(kv_cache_config: KVCacheConfig) -> dict[str, object]:
    specs: dict[str, object] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        per_layer = (
            group_spec.kv_cache_specs if isinstance(group_spec, UniformTypeKVCacheSpecs) else {}
        )
        for layer_name in group.layer_names:
            specs[layer_name] = per_layer.get(layer_name, group_spec)
    return specs


def _unpack_paged_cache(layer_name: str, cache: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Duck-type SpyrePagedKVCache without importing the attention stack."""
    if not (isinstance(cache, tuple) and len(cache) == 2):
        raise TypeError(
            f"layer {layer_name!r}: expected a SpyrePagedKVCache 2-tuple, got "
            f"{type(cache).__name__}"
        )
    k_pages, v_pages = cache
    for name, pages in (("k_pages", k_pages), ("v_pages", v_pages)):
        if not isinstance(pages, torch.Tensor):
            raise TypeError(
                f"layer {layer_name!r}: {name} is {type(pages).__name__}, expected a Tensor"
            )
    if k_pages.shape != v_pages.shape or k_pages.dtype != v_pages.dtype:
        raise ValueError(
            f"layer {layer_name!r}: k/v pages disagree — "
            f"{tuple(k_pages.shape)}/{k_pages.dtype} vs "
            f"{tuple(v_pages.shape)}/{v_pages.dtype}"
        )
    return k_pages, v_pages


def _check_page_size(layer_name: str, pages: torch.Tensor, expected_bytes: int) -> None:
    actual = pages[0].numel() * pages.element_size()
    if actual != expected_bytes:
        raise ValueError(
            f"layer {layer_name!r}: device page is {actual} bytes but the spec "
            f"implies {expected_bytes} per K/V half"
        )


def _append_flat_tensor(
    tensors: list[CanonicalKVCacheTensor],
    pages: torch.Tensor,
    page_size_bytes: int,
) -> int:
    """Append `pages` flattened to (num_blocks, -1) and return its index."""
    if not pages.is_contiguous():
        raise ValueError("Spyre KV pages must be contiguous to canonicalize")
    tensors.append(
        CanonicalKVCacheTensor(
            tensor=pages.view(pages.shape[0], -1),
            page_size_bytes=page_size_bytes,
        )
    )
    return len(tensors) - 1


class SpyreOffloadingConnectorWorker(OffloadingConnectorWorker):
    """Worker overriding only `register_kv_caches` for the paged-list layout."""

    def __init__(self, *args, **kwargs):
        # Upstream has already changed this signature once (it gained
        # `vllm_config` as a 2nd positional arg); forward blindly.
        super().__init__(*args, **kwargs)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        canonical = spyre_paged_to_canonical(kv_caches, self.kv_cache_config)
        logger.info(
            "Spyre KV offloading: canonicalized %d layer(s) into %d tensor(s)",
            len(kv_caches),
            len(canonical.tensors),
        )
        self._init_worker(canonical)


class SpyreOffloadingConnector(OffloadingConnector):
    """OffloadingConnector wired to the Spyre worker. Scheduler side untouched.

    Registered via `KVTransferConfig.kv_connector_module_path`; no upstream patch.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        if self.connector_worker is not None:
            self.connector_worker = SpyreOffloadingConnectorWorker(
                self.connector_worker.spec,
                vllm_config,
                kv_cache_config,
            )
