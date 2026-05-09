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

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.cpu_worker import CPUWorker
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
        # Patch the CPUModelRunner with the TorchSpyreModelRunner
        original = cpu_worker_module.CPUModelRunner
        cpu_worker_module.CPUModelRunner = lambda *a, **kw: TorchSpyreModelRunner(
            self.vllm_config,
            torch.device("spyre"),
        )
        try:
            # We will invoke the upstream init_device method with the
            # CPUModelRunner patched. This will ensure that everything for the CPUWorker is setup,
            # but the spyre-specific model runner is instantiated instead.
            super().init_device()
        finally:
            cpu_worker_module.CPUModelRunner = original

    def compile_or_warm_up_model(self) -> float:
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
        return self.compilation_config.compilation_time
