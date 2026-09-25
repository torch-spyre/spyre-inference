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

import pytest
import torch
from spyre_inference.v1.kv_offload.connector import spyre_paged_to_canonical
from torch_spyre._C import SharedHostPool  # type: ignore[attr-defined]
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.kv_offload.base import CanonicalKVCaches, GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from spyre_inference.v1.kv_offload.worker import SpyreOffloadingWorker

NUM_BLOCKS = 4
BLOCK_SIZE = 8
NUM_KV_HEADS = 2
HEAD_SIZE = 64
LAYERS = ["layer.0", "layer.1"]

# One canonical tensor per layer per K/V. One page is a block's K (or V).
NUM_TENSORS = 2 * len(LAYERS)
PAGE_BYTES = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * 2

DEV_BLOCKS = [1, 2]
HOST_BLOCKS = [0, 3]
POOL_NAME = "test_worker_dispatch"


def _spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE, dtype=torch.float16
    )


def _paged_cache() -> tuple[torch.Tensor, torch.Tensor]:
    """A stand-in for SpyrePagedKVCache: a 2-tuple of dense page tensors."""
    shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    return (
        torch.zeros(shape, dtype=torch.float16, device="spyre"),
        torch.zeros(shape, dtype=torch.float16, device="spyre"),
    )


def _config(layer_names, spec) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)],
    )


@pytest.fixture
def kv_cache() -> CanonicalKVCaches:
    spec = _spec()
    kv_caches = {name: _paged_cache() for name in LAYERS}
    canonical = spyre_paged_to_canonical(kv_caches, _config(LAYERS, spec))
    # Fill the canonical tensors with known values for testing.
    for i, cache in enumerate(canonical.tensors):
        cache.tensor.fill_(i + 1)
    return canonical


@pytest.fixture
def pool() -> SharedHostPool:
    # NUM_BLOCKS * NUM_TENSORS is the total number of slots in the pool.
    return SharedHostPool.create_or_attach(POOL_NAME, NUM_BLOCKS * NUM_TENSORS, PAGE_BYTES)


@pytest.fixture
def gpu_spec() -> GPULoadStoreSpec:
    return GPULoadStoreSpec(DEV_BLOCKS, group_sizes=[len(DEV_BLOCKS)], block_indices=[0])


@pytest.fixture
def host_spec() -> CPULoadStoreSpec:
    return CPULoadStoreSpec(block_ids=HOST_BLOCKS)


def test_submit_store_and_load(
    kv_cache: CanonicalKVCaches,
    pool: SharedHostPool,
    gpu_spec: GPULoadStoreSpec,
    host_spec: CPULoadStoreSpec,
):
    worker = SpyreOffloadingWorker(kv_cache, pool)

    # Arbitrary job_id for testing.
    job_id = 42

    # Test storing device blocks 1 and 2 to host blocks 0 and 3.
    assert worker.submit_store(job_id, gpu_spec, host_spec) is True
    assert [(job.job_id, job.success) for job in worker.get_finished()] == [(job_id, True)]

    # Clone kv_cache to then compare the host tensors to the original canonical tensors.
    expected = [cache.tensor.clone() for cache in kv_cache.tensors]

    # Now fill the kv_cache tensors at DEV_BLOCKS with 0s so then we can copy back.
    for block_cache in kv_cache.tensors:
        for dev_blk_id in DEV_BLOCKS:
            zeroes = torch.zeros_like(block_cache.tensor[dev_blk_id])
            block_cache.tensor[dev_blk_id].copy_(zeroes)

    # Test loading host blocks 0 and 3 from device blocks 1 and 2.
    assert worker.submit_load(job_id, host_spec, gpu_spec) is True
    assert [(job.job_id, job.success) for job in worker.get_finished()] == [(job_id, True)]

    # Check kv_cache tensors at DEV_BLOCKS should match the original canonical tensors.
    for cache, expected_cache in zip(kv_cache.tensors, expected):
        assert torch.equal(cache.tensor.to("cpu"), expected_cache.to("cpu"))


def test_get_finished_drains(
    kv_cache: CanonicalKVCaches,
    pool: SharedHostPool,
    gpu_spec: GPULoadStoreSpec,
    host_spec: CPULoadStoreSpec,
):
    worker = SpyreOffloadingWorker(kv_cache, pool)

    # Submit a store job.
    assert worker.submit_store(1, host_spec, gpu_spec) is True

    # Check that get_finished() returns the finished job.
    assert [job.job_id for job in worker.get_finished()] == [1]

    # After calling get_finished(), the finished jobs list should be drained.
    assert worker.get_finished() == []
