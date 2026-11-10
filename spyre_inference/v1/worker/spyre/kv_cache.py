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

"""MRV2: SpyreModelRunnerV2.initialize_kv_cache body
(vllm/v1/worker/gpu/model_runner.py:GPUModelRunner.initialize_kv_cache).

Upstream's initialize_kv_cache runs as-is, so every attribute it sets stays
upstream's. Its two module-level calls have no override hook, so they're swapped
for the call: init_attn_backend gains the window split (attn_groups.py), also where
the speculator's set_attn imports it for the draft layers, and init_kv_cache
allocates per-layer SpyrePagedKVCache pages on Spyre, draft layers included.

The pages go on runner.spyre_kv_caches; upstream gets an empty dict back, since it
filters by ``cache.device``, which the page tuple lacks. Only the KV connector and
prefix-cache block copies read it, and neither is supported: init() refuses a KV
transfer config and the runner refuses the copies (model_runner.py).
"""

import contextlib
from collections.abc import Iterator
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner as upstream_model_runner
from vllm.v1.worker.gpu.attn_utils import get_shared_kv_cache_layers
from vllm.v1.worker.gpu.spec_decode import speculator as upstream_speculator
from vllm.v1.worker.utils import bind_kv_cache_to_layers

from spyre_inference.v1.worker.kv_cache_setup import allocate_spyre_kv_caches
from spyre_inference.v1.worker.spyre import attn_groups

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2


@contextlib.contextmanager
def _patched(runner: "SpyreModelRunnerV2", called: set[str] | None = None) -> Iterator[None]:
    called = set() if called is None else called
    init_attn_backend = upstream_model_runner.init_attn_backend
    draft_init_attn_backend = upstream_speculator.init_attn_backend
    init_kv_cache = upstream_model_runner.init_kv_cache

    def spyre_init_attn_backend(*args: Any, **kwargs: Any) -> Any:
        return attn_groups.init(init_attn_backend, *args, **kwargs)

    def spyre_draft_init_attn_backend(*args: Any, **kwargs: Any) -> Any:
        called.add("draft_init_attn_backend")
        return attn_groups.init(draft_init_attn_backend, *args, **kwargs)

    def spyre_init_kv_cache(
        forward_context: dict[str, Any],
        kv_cache_config: KVCacheConfig,
        device: Any,
        kernel_block_sizes: list[int],
        vllm_config: Any,
        kv_cache_allocation_context: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        with kv_cache_allocation_context or nullcontext():
            kv_caches = allocate_spyre_kv_caches(
                kv_cache_config,
                forward_context,
                get_shared_kv_cache_layers(vllm_config),
                runner.spyre_device,
            )
        bindable = {name: c for name, c in kv_caches.items() if name in forward_context}
        bind_kv_cache_to_layers(
            bindable,  # ty: ignore[invalid-argument-type]
            forward_context,
            kv_cache_groups=kv_cache_config.kv_cache_groups,
        )
        runner.spyre_kv_caches = dict(kv_caches)
        return {}

    upstream_model_runner.init_attn_backend = spyre_init_attn_backend  # ty: ignore[invalid-assignment]
    upstream_speculator.init_attn_backend = spyre_draft_init_attn_backend  # ty: ignore[invalid-assignment]
    upstream_model_runner.init_kv_cache = spyre_init_kv_cache  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        upstream_model_runner.init_attn_backend = init_attn_backend
        upstream_speculator.init_attn_backend = draft_init_attn_backend
        upstream_model_runner.init_kv_cache = init_kv_cache


def init(runner: "SpyreModelRunnerV2", kv_cache_config: KVCacheConfig, **kwargs: Any) -> None:
    """Upstream's initialize_kv_cache with the Spyre attention groups and pages."""
    if not kwargs.get("is_profiling", False) and has_kv_transfer_group():
        raise NotImplementedError(
            "KV connectors are not supported on the Spyre V2 runner (#1240): the "
            "connector would register no KV caches."
        )
    runner.spyre_kv_caches = {}
    called: set[str] = set()
    with _patched(runner, called):
        upstream_model_runner.GPUModelRunner.initialize_kv_cache(runner, kv_cache_config, **kwargs)
    # The swaps ride on module globals; a vLLM bump that calls one through another
    # module would otherwise bind plain CPU tensors, or skip the draft's window split,
    # without an error.
    if kv_cache_config.kv_cache_tensors and not runner.spyre_kv_caches:
        raise RuntimeError(
            "initialize_kv_cache allocated no Spyre KV pages: upstream no longer calls "
            "init_kv_cache through vllm.v1.worker.gpu.model_runner, so kv_cache.py's swap "
            "did not apply."
        )
    if runner.speculator is not None and "draft_init_attn_backend" not in called:
        raise RuntimeError(
            "The speculator built its attention groups without kv_cache.py's swap: "
            "upstream no longer calls init_attn_backend through "
            "vllm.v1.worker.gpu.spec_decode.speculator."
        )
