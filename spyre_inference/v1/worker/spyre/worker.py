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

"""MRV2: Spyre worker (vllm/v1/worker/gpu_worker.py:Worker).

``init_device`` is copied from ``TorchSpyreWorker.init_device`` rather than
calling ``super()``: the V1 parent builds a ``TorchSpyreModelRunner``, whose
``__init__`` monkey-patches a process-wide Triton kernel this path never
needs. ``test_worker.py`` guards the copy against drift.

No ``execute_model`` override: the stock ``Worker.execute_model`` already
calls ``runner.pool()`` once ``VLLM_USE_V2_MODEL_RUNNER`` is set.

``Worker.model_runner`` is nominally typed to V1's GPUModelRunner; upstream's
own gpu_worker.py carries the same ``# type: ignore`` for its V2 one.
"""

import os

import torch
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

from spyre_inference import envs
from spyre_inference.custom_ops import register_all
from spyre_inference.models import register_models
from spyre_inference.platform import _raise_dynamo_recompile_limits
from spyre_inference.v1.worker import compile_guard
from spyre_inference.v1.worker.spyre import device_shims, kernels
from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2
from spyre_inference.v1.worker.spyre_worker import (  # shared with V1; moves at the flip
    TorchSpyreWorker,
    _get_spyre_pcie_address,
    logger,
)


class SpyreWorkerV2(TorchSpyreWorker):
    """MRV2: vllm/v1/worker/gpu_worker.py:Worker, via TorchSpyreWorker."""

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

        device_shims.install()  # MRV2-only: torch.cuda Stream/Event placeholders
        kernels.bind_or_raise()  # MRV2-only: the kernel seam (DEFAULT_PATH_KERNELS)

        # Construct the model runner directly — no monkey-patching needed.
        self.model_runner = SpyreModelRunnerV2(  # ty: ignore[invalid-assignment]
            self.vllm_config,
            torch.device("spyre"),
        )

        logger.info("Using Spyre Model Runner V2")  # MRV2-only
