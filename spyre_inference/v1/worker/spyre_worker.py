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

import os
from contextlib import AbstractContextManager, nullcontext

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.kv_cache_utils import (
    _max_memory_usage_bytes_from_groups,
    _pool_bytes_per_block,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec, UniformTypeKVCacheSpecs
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


def _kv_block_layout(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> tuple[int, int, int] | None:
    """``(pool bytes per block, Spyre bytes per block, batch-area blocks)``, or None
    without a KV cache.

    Grouping is upstream's (``get_kv_cache_groups``), so sliding-window and Mamba
    groups count what they really hold per request. Upstream overlays its groups on one
    buffer and needs ``pool bytes`` per block; Spyre allocates every layer's own page per
    block, the unpadded one where upstream pads (a Mamba state). The batch area is
    max_num_seqs requests at max_model_len plus the null block.
    """
    if not kv_cache_spec:
        return None
    groups = get_kv_cache_groups(vllm_config, dict(kv_cache_spec))
    if not groups:
        return None
    pool_bytes = _pool_bytes_per_block(vllm_config, groups)
    spyre_bytes = sum(
        _unpadded_page_bytes(_layer_spec(group, name))
        for group in groups
        for name in group.layer_names
    )
    per_request = _max_memory_usage_bytes_from_groups(vllm_config, groups) // pool_bytes
    batch_blocks = vllm_config.scheduler_config.max_num_seqs * per_request + 1
    return pool_bytes, spyre_bytes, batch_blocks


def _layer_spec(group: KVCacheGroupSpec, layer_name: str) -> KVCacheSpec:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.kv_cache_specs[layer_name]
    return spec


def _unpadded_page_bytes(spec: KVCacheSpec) -> int:
    # Attention and Mamba specs pad their page to a shared size; the Spyre allocation
    # does not.
    return getattr(spec, "real_page_size_bytes", spec.page_size_bytes)


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
        """Upstream's Worker sizing from free device memory, capped at the pages a full
        batch at max_model_len can use.

        The runner's profile_run is a no-op, so what upstream measures is what loading
        consumed; a forward's activations come out of the 1 - gpu_memory_utilization
        headroom.

        Upstream turns the returned bytes into blocks by its pool's bytes per block,
        while Spyre gives every layer its own page per block (``allocate_spyre_kv_caches``),
        which costs more wherever groups would share a page. So the measured device bytes
        are converted into blocks at Spyre's cost per block, capped at the batch area in
        blocks, and handed back in upstream's units.
        """
        available = super().determine_available_memory()
        layout = _kv_block_layout(self.vllm_config, self.model_runner.get_kv_cache_spec())
        if layout is None:
            return available
        pool_bytes, spyre_bytes, batch_blocks = layout
        blocks = max(available, 0) // spyre_bytes
        if blocks > batch_blocks:
            logger.info(
                "Capping the KV cache at %d blocks (%s GiB on the device), what %d "
                "sequences at max_model_len=%d can use.",
                batch_blocks,
                format_gib(batch_blocks * spyre_bytes),
                self.scheduler_config.max_num_seqs,
                self.model_config.max_model_len,
            )
        elif available > 0:
            logger.warning(
                "The KV cache holds %d of the %d blocks that %d sequences at "
                "max_model_len=%d need; requests queue for pages instead. Raise "
                "--gpu-memory-utilization or lower --max-num-seqs / --max-model-len "
                "to fill the batch.",
                blocks,
                batch_blocks,
                self.scheduler_config.max_num_seqs,
                self.model_config.max_model_len,
            )
        if available <= 0:
            return available
        return min(blocks, batch_blocks) * pool_bytes

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
