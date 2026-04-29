"""A Torch Spyre worker class."""

import os
import sys
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.cpu_worker import CPUWorker
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.utils.torch_utils import set_random_seed
from vllm.platforms import CpuArchEnum, current_platform

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
        # Call the relevant parts from the CPUWorker upstream, but don't create the CPUModelRunner
        
        # Check whether critical libraries are loaded
        def check_preloaded_libs(name: str):
            ld_preload_list = os.environ.get("LD_PRELOAD", "")
            if name not in ld_preload_list:
                logger.warning(
                    "%s is not found in LD_PRELOAD. "
                    "For best performance, please follow the section "
                    "`set LD_PRELOAD` in "
                    "https://docs.vllm.ai/en/latest/getting_started/installation/cpu/ "
                    "to setup required pre-loaded libraries.",
                    name,
                )

        if sys.platform.startswith("linux"):
            check_preloaded_libs("libtcmalloc")
            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                check_preloaded_libs("libiomp")

        def skip_set_num_threads(x: int):
            logger.warning(
                "CPU backend doesn't allow to use "
                "`torch.set_num_threads` after the thread binding, skip it."
            )

        torch.set_num_threads = skip_set_num_threads

        # Note: unique identifier for creating allreduce shared memory
        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(":")[-1]
        # Initialize the distributed environment.
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct Spyre model runner with torch.device("spyre")
        self.model_runner = TorchSpyreModelRunner(
            self.vllm_config,
            torch.device("spyre"),
        )

    def compile_or_warm_up_model(self) -> float:
        # FIXME: Work around for https://github.com/torch-spyre/torch-spyre/issues/1420
        # Ensure registration of Spyre decompositions before FX Graph tracing
        import torch._inductor.decomposition
        from torch_spyre._inductor.decompositions import spyre_decompositions

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
