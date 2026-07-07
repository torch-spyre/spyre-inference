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

# `import torch_spyre` is intentionally deferred to inside `init_device`.
# Importing it loads `libspyre_comms.so`, which captures
# `RANK` / `WORLD_SIZE` / `LOCAL_RANK` / `LOCAL_WORLD_SIZE` via
# `std::getenv` at dlopen time and caches them. Those env vars are only
# known per-worker, so they must be populated before the C library
# loads. `spyre_inference/__init__.py` sets
# `TORCH_DEVICE_BACKEND_AUTOLOAD=0` so torch's `[torch.backends]`
# autoload doesn't trigger the load at `import torch` time.

import vllm.v1.worker.cpu_worker as cpu_worker_module
from vllm.logger import init_logger
from vllm.v1.worker.cpu_worker import CPUWorker
from vllm.v1.worker.worker_base import CompilationTimes

from spyre_inference.custom_ops import register_all
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

logger = init_logger(__name__)


class TorchSpyreWorker(CPUWorker):
    """A worker class that executes the model on IBM's Spyre device.

    Inherits from CPUWorker but extends init_device to:
    - Create a TorchSpyreModelRunner with torch.device("spyre")
    """

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        # vllm.v1.worker.gpu_worker.Worker.load_model wraps weight loading in
        # a memory pool context that calls get_mem_allocator_instance(). That
        # only short-circuits to nullcontext() when current_platform.is_cpu()
        # returns True; our platform reports OOT (so custom-op forward_oot
        # dispatch works), so the upstream check falls through and raises.
        # Spyre weights live on-device, not in a host-side cumem allocator,
        # so a nullcontext is the correct behaviour here.
        return nullcontext()

    def init_device(self) -> None:
        # Populate the env vars that `libspyre_comms.so` reads at dlopen
        # time. `setdefault` leaves torchrun-supplied values intact.
        # DP>1 is rejected in TorchSpyrePlatform.check_and_update_config,
        # so parallel_config.world_size is the global rank count and
        # LOCAL_WORLD_SIZE == WORLD_SIZE on a single node. Revisit once
        # multi-node TP is supported.
        world_size = self.vllm_config.parallel_config.world_size
        os.environ.setdefault("RANK", str(self.rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(self.local_rank))
        os.environ.setdefault("LOCAL_WORLD_SIZE", str(world_size))

        # Trigger torch_spyre's autoload manually now that the env vars
        # are set. Autoload registers the `spyre` device and the
        # `spyreccl` distributed backend, and imports
        # `torch_spyre._C` (which loads `libspyre_comms.so`).
        import torch_spyre

        torch_spyre._autoload()

        # Pin this worker to its assigned card before the spyreccl
        # backend is constructed in `init_process_group`.
        torch.spyre.set_device(self.local_rank)

        # Register all the custom ops here when a worker is created.
        # This has to happen before the model is loaded, so that all the
        # layers will be swapped out with the custom implementations for spyre.
        register_all()

        # Patch the CPUModelRunner with the TorchSpyreModelRunner.
        # We pass the unindexed `torch.device("spyre")` because the
        # current device for this process is already pinned via
        # `set_device(local_rank)` above; tensors created on
        # `torch.device("spyre")` will land on that card.
        original = cpu_worker_module.CPUModelRunner
        cpu_worker_module.CPUModelRunner = lambda *a, **kw: TorchSpyreModelRunner(  # ty: ignore[invalid-assignment]
            self.vllm_config,
            torch.device("spyre"),
        )
        try:
            super().init_device()
        finally:
            cpu_worker_module.CPUModelRunner = original

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
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )
