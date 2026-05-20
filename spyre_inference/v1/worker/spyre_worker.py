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

import torch

# `import torch_spyre` triggers the autoload that calls
# `torch.utils.rename_privateuse1_backend("spyre")`, registers the
# `spyre` device module, and registers the `spyreccl` distributed
# backend with torch.distributed. This must run before vllm's
# `init_distributed_environment` calls `dist.is_backend_available` /
# `dist.init_process_group` with the platform's
# `dist_backend = "cpu:gloo,spyre:spyreccl"` string. The
# `# noqa: F401` is intentional — we only import for the side effect.
import torch_spyre  # noqa: F401

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.cpu_worker import CPUWorker
from vllm.v1.worker.worker_base import CompilationTimes
import vllm.v1.worker.cpu_worker as cpu_worker_module

from spyre_inference.custom_ops import register_all
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

logger = init_logger(__name__)


class TorchSpyreWorker(CPUWorker):
    """A worker class that executes the model on IBM's Spyre device.

    Inherits from CPUWorker but extends init_device to:
    - Create a TorchSpyreModelRunner with torch.device("spyre")
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker,
        )

        # Register all the custom ops here when a worker is created.
        # This has to happen before the model is loaded, so that all the
        # layers will be swapped out with the custom implementations for spyre.
        register_all()

    def init_device(self) -> None:
        # Pin this worker process to its assigned Spyre card before any
        # Spyre runtime / spyreccl backend construction occurs. The
        # spyreccl backend creator (in torch_spyre/__init__.py) calls
        # `torch.spyre._impl._lazy_init()` against `current_device()`,
        # so this must happen before `super().init_device()` runs the
        # `dist.init_process_group(...)` that materializes spyreccl.
        torch.spyre.set_device(self.local_rank)

        # `libspyre_comms.so::SpyreDeviceInfo` reads RANK / WORLD_SIZE /
        # LOCAL_RANK / LOCAL_WORLD_SIZE from the environment when the
        # spyreccl backend is constructed and throws
        # `EnvironmentVariableException` if any are unset. vllm's
        # MultiprocExecutor uses TCP-based init (`tcp://...`) and does
        # NOT set those env vars, so we populate them here before
        # `super().init_device()` runs init_process_group. When the
        # worker is launched via torchrun the env vars are already set
        # and `setdefault` leaves them alone.
        #
        # DP>1 is rejected in TorchSpyrePlatform.check_and_update_config,
        # so parallel_config.world_size (TP*PP*PCP) is the global rank
        # count here, and on a single node LOCAL_WORLD_SIZE == WORLD_SIZE.
        # Once multi-node TP is supported, derive LOCAL_WORLD_SIZE from
        # the node-local rank count.
        world_size = self.vllm_config.parallel_config.world_size
        os.environ.setdefault("RANK", str(self.rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(self.local_rank))
        os.environ.setdefault("LOCAL_WORLD_SIZE", str(world_size))

        # Patch the CPUModelRunner with the TorchSpyreModelRunner.
        # We pass the unindexed `torch.device("spyre")` because the
        # current device for this process is already pinned via
        # `set_device(local_rank)` above; tensors created on
        # `torch.device("spyre")` will land on that card.
        original = cpu_worker_module.CPUModelRunner
        cpu_worker_module.CPUModelRunner = lambda *a, **kw: TorchSpyreModelRunner(
            self.vllm_config,
            torch.device("spyre"),
        )
        try:
            # Invoke the upstream init_device method with the
            # CPUModelRunner patched. This wires up the distributed
            # environment via vllm's init_worker_distributed_environment,
            # which uses TorchSpyrePlatform.dist_backend
            # ("cpu:gloo,spyre:spyreccl") to build the world / TP groups.
            super().init_device()
        finally:
            cpu_worker_module.CPUModelRunner = original

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # FIXME: Work around for https://github.com/torch-spyre/torch-spyre/issues/1420
        # Ensure registration of Spyre decompositions before FX Graph tracing
        import torch._inductor.decomposition
        from torch_spyre._inductor.decompositions import spyre_decompositions  # ty: ignore[unresolved-import]

        for op, impl in spyre_decompositions.items():
            if "addm" in op.name():
                logger.warning(
                    "FIXME: Adding %s decomposition to work-around torch-spyre crash", op.name()
                )
                torch._inductor.decomposition.decompositions[op] = impl
        import time

        warmup_start_time = time.perf_counter()
        self.model_runner.warming_up_model()
        self.compilation_config.compilation_time = time.perf_counter() - warmup_start_time
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )
