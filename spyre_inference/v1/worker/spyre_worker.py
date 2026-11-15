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

"""A Torch Spyre worker class."""

import logging
import os
from contextlib import AbstractContextManager, nullcontext

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import MemorySnapshot
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
from vllm.v1.core.kv_cache_planning import (
    _max_memory_usage_bytes_from_groups,
    _pool_bytes_per_block_from_builder,
    get_kv_cache_groups_from_builder,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.utils import request_memory
from vllm.v1.worker.worker_base import CompilationTimes

from spyre_inference import envs, keep_sigterm_handler
from spyre_inference.custom_ops import register_all
from spyre_inference.models import register_models
from spyre_inference.platform import _raise_dynamo_recompile_limits
from spyre_inference.threading_config import apply_in_process_threading
from spyre_inference.v1.worker import compile_guard
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

logger = init_logger(__name__)


# flex's MAX_REGION_SIZE: the allocator splits device memory into regions of this size
# (six on a 96 GiB card), and never splits a tensor across two.
_SPYRE_REGION_BYTES = 16 << 30


def _kv_block_layout(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> tuple[int, list[int], int] | None:
    """``(pool bytes per block, bytes per block of each Spyre KV tensor, blocks per full
    request)``, or None without a KV cache.

    With ``disable_hybrid_kv_cache_manager`` every layer lands in one group and the pool
    bytes match the tensors' sum. They differ for a Mamba hybrid, which keeps the hybrid
    manager: upstream overlays its groups on one buffer and needs ``pool bytes`` per block, while
    Spyre allocates every layer's own page per block, unpadded where upstream pads (a
    Mamba state). The per-request count is upstream's grouping, so a Mamba group holds
    one block and a sliding-window group its window.
    """
    if not kv_cache_spec:
        return None
    builder = get_kv_cache_config_builder(vllm_config)
    # The engine groups the same specs in get_global_kv_cache_groups and logs from there;
    # this replay would repeat its warnings (a hybrid model's disabled hybrid manager) on
    # every worker.
    grouping_logger = logging.getLogger(get_kv_cache_groups_from_builder.__module__)
    was_disabled, grouping_logger.disabled = grouping_logger.disabled, True
    try:
        groups = get_kv_cache_groups_from_builder(builder, vllm_config, dict(kv_cache_spec))
    finally:
        grouping_logger.disabled = was_disabled
    if not groups:
        return None
    pool_bytes = _pool_bytes_per_block_from_builder(builder, vllm_config, groups)
    # Attention and Mamba specs pad their page to a shared size; Spyre's pages do not.
    tensors: list[int] = []
    for group in groups:
        for name in group.layer_names:
            spec = _layer_spec(group, name)
            page = getattr(spec, "real_page_size_bytes", spec.page_size_bytes)
            # An attention page holds K and V, which allocate_pages allocates apart.
            tensors += [page // 2] * 2 if isinstance(spec, AttentionSpec) else [page]
    per_request = _max_memory_usage_bytes_from_groups(builder, vllm_config, groups) // pool_bytes
    return pool_bytes, tensors, per_request


def _layer_spec(group: KVCacheGroupSpec, layer_name: str) -> KVCacheSpec:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.kv_cache_specs[layer_name]
    return spec


def _get_spyre_pcie_address(local_rank: int) -> str:
    requested_devices = envs.SPYRE_DEVICES
    if not requested_devices:
        return "unknown"

    requested_indices = [index.strip() for index in requested_devices.split(",") if index.strip()]
    if local_rank >= len(requested_indices):
        return "unknown"

    return os.environ.get(f"AIU_WORLD_RANK_{requested_indices[local_rank]}", "unknown")


def allow_private_use1_profiler_activity() -> None:
    """Allow PrivateUse1 in vLLM's validate_worker_profiler_config for the "cpu" device type.

    Spyre reports device_type "cpu", for which vLLM only accepts the CPU activity, so
    `torch_profiler_activities` containing PrivateUse1 would fail at worker start.
    Temporary until vLLM lets a platform declare its profiler activities.
    """
    from vllm.profiler import wrapper

    supported = wrapper._SUPPORTED_TORCH_PROFILER_ACTIVITIES
    if "PrivateUse1" not in supported["cpu"]:
        supported["cpu"] = supported["cpu"] | {"PrivateUse1"}
        logger.debug("Allowed PrivateUse1 in vLLM's torch profiler activities for cpu")


allow_private_use1_profiler_activity()


class TorchSpyreWorker(Worker):
    """A worker class that executes the model on IBM's Spyre device.

    Inherits from Worker (gpu_worker) directly — Spyre is not a CPU device
    and does not need any of the CPU-specific init (NUMA binding,
    torch.ops._C.init_cpu_memory_env, host-RAM profiling) that CPUWorker
    provides. The distributed init, random seed, and model runner
    construction are handled here.
    """

    # Resolved in init_device, before the weights load, so an invalid
    # SPYRE_COMPILE_GUARD fails there rather than after warmup.
    _compile_guard_level = compile_guard.CompileGuardLevel.OFF

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        # Worker.load_model wraps weight loading in a memory pool context
        # that calls get_mem_allocator_instance(). That only short-circuits
        # to nullcontext() when current_platform.is_cpu() returns True; our
        # platform reports OOT, so the upstream check falls through and raises.
        # Spyre weights live on-device, not in a host-side cumem allocator.
        return nullcontext()

    def init_device(self) -> None:
        apply_in_process_threading(self.parallel_config)

        # The env vars spyreccl reads. `setdefault` leaves torchrun-supplied
        # values intact.
        # DP>1 is rejected in TorchSpyrePlatform.check_and_update_config,
        # so parallel_config.world_size is the global rank count and
        # LOCAL_WORLD_SIZE == WORLD_SIZE on a single node. Revisit once
        # multi-node TP is supported.
        world_size = self.vllm_config.parallel_config.world_size
        os.environ.setdefault("RANK", str(self.rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(self.local_rank))
        os.environ.setdefault("LOCAL_WORLD_SIZE", str(world_size))

        # A no-op when `import torch` already autoloaded the backend; needed when
        # TORCH_DEVICE_BACKEND_AUTOLOAD=0 disabled that.
        import torch_spyre

        with keep_sigterm_handler():
            torch_spyre._autoload()

        # torch_spyre's autoload sets cache_size_limit=1024, undoing the limits
        # platform.py raised at import.
        _raise_dynamo_recompile_limits()

        # Pin this worker to its assigned card before the spyreccl
        # backend is constructed in `init_process_group`.
        torch.spyre.set_device(self.local_rank)
        logger.info(
            "Spyre worker device selection: "
            "local_rank=%d requested SPYRE_DEVICES=%r pcie_address=%s",
            self.local_rank,
            os.environ.get("SPYRE_DEVICES"),
            _get_spyre_pcie_address(self.local_rank),
        )

        # Parsed here rather than at the arm() call below: a typo raises, and after
        # warmup that costs the minutes the model just spent compiling. init_device
        # runs before the weights load, so an invalid value fails in seconds.
        self._compile_guard_level = compile_guard.parse_level(envs.SPYRE_COMPILE_GUARD)

        # Register all the custom ops and Spyre model architectures here when a
        # worker is created. This has to happen before the model is loaded, so
        # that all the layers will be swapped out with the custom
        # implementations for spyre. Both also run from the `vllm.general_plugins`
        # entry point, which `VLLM_PLUGINS` allowlists commonly leave out.
        register_all()
        register_models()

        # Initialize the distributed environment.
        from vllm.platforms import current_platform

        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # As Worker.init_device does, after the distributed init. The platform reports
        # device_type "cpu", so the snapshot is given the Spyre device explicitly.
        self.init_snapshot = MemorySnapshot(device=self._spyre_memory_device())
        self.requested_memory = request_memory(self.init_snapshot, self.cache_config)
        logger.debug("worker init memory snapshot: %r", self.init_snapshot)

        # Construct the model runner directly — no monkey-patching needed.
        self.model_runner = TorchSpyreModelRunner(
            self.vllm_config,
            torch.device("spyre"),
        )

    def _spyre_memory_device(self) -> torch.device:
        # getMemoryInfo checks the index against the card the runtime opened, which is
        # LOCAL_RANK under TP, not spyre:0.
        return torch.device("spyre", torch.accelerator.current_device_index())

    def determine_available_memory(self) -> int:
        """Upstream's Worker sizing from free device memory, in blocks Spyre can allocate.

        The runner's profile_run is a no-op, so what upstream measures is what loading
        consumed; a forward's activations come out of the 1 - gpu_memory_utilization
        headroom.

        Upstream turns the returned bytes into blocks by its pool's bytes per block,
        while Spyre gives every layer its own page per block (``allocate_spyre_kv_caches``),
        which costs more wherever groups would share a page. So the measured device bytes
        are converted into blocks at Spyre's cost per block and handed back in upstream's
        units.
        """
        available = super().determine_available_memory()
        layout = _kv_block_layout(self.vllm_config, self.model_runner.get_kv_cache_spec())
        if layout is None or available <= 0:
            return available
        pool_bytes, tensor_bytes, per_request = layout
        blocks = available // sum(tensor_bytes)
        if not self.cache_config.kv_cache_memory_bytes:
            blocks = min(blocks, self._region_safe_blocks(tensor_bytes))
        batch_blocks = self.scheduler_config.max_num_seqs * per_request + 1
        # Upstream replaces the measured count with the override in get_kv_cache_configs.
        override = self.cache_config.num_gpu_blocks_override
        num_blocks = blocks if override is None else override
        if 0 < num_blocks < batch_blocks:
            logger.warning(
                "The KV cache holds %d of the %d blocks that %d sequences at "
                "max_model_len=%d need; requests queue for pages instead. Raise "
                "%s or lower --max-num-seqs / --max-model-len to fill the batch.",
                num_blocks,
                batch_blocks,
                self.scheduler_config.max_num_seqs,
                self.model_config.max_model_len,
                "--gpu-memory-utilization" if override is None else "--num-gpu-blocks-override",
            )
        return blocks * pool_bytes

    def _region_safe_blocks(self, tensor_bytes: list[int]) -> int:
        """The most blocks whose KV tensors fit the allocator's regions, wherever the
        weights sit in them.

        A tensor fails only once every region has less room than it, so it is placed
        while the free bytes left reach num_regions times its size; one largest tensor
        spared per other region guarantees that for the last. A deep model's many small
        tensors stay within the budget anyway; a shallow one's few large ones do not:
        four layers at an 86 GiB budget make eight 10.75 GiB tensors, one per region, of
        which six fit.
        """
        free, total = torch.accelerator.get_memory_info(self.init_snapshot.device_)
        num_regions = cdiv(total, _SPYRE_REGION_BYTES)
        return free // (sum(tensor_bytes) + (num_regions - 1) * max(tensor_bytes))

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # FIXME: Work around for https://github.com/torch-spyre/torch-spyre/issues/1420
        # Ensure registration of Spyre decompositions before FX Graph tracing
        import time

        import torch._inductor.decomposition
        from torch_spyre._inductor.decompositions import spyre_decompositions

        for op, impl in spyre_decompositions.items():
            if "addm" in op.name():
                logger.warning(
                    "FIXME: Adding %s decomposition to work-around torch-spyre crash", op.name()
                )
                torch._inductor.decomposition.decompositions[op] = impl

        warmup_start_time = time.perf_counter()
        self.model_runner.warming_up_model()
        self.compilation_config.compilation_time = time.perf_counter() - warmup_start_time
        # Arm right after warm-up, at the level init_device already validated.
        compile_guard.arm(self._compile_guard_level)
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def sleep(self, level: int = 1) -> None:
        # No sleep mode on Spyre: the GPU worker's CuMemAllocator offload does not apply.
        pass

    def wake_up(self, tags: list[str] | None = None) -> None:
        # Nothing to restore, since sleep() keeps everything resident.
        pass
