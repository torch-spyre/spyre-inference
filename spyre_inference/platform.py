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

import importlib.metadata
import multiprocessing
import os
import sys
from string import Template
from typing import TYPE_CHECKING

import torch

from spyre_inference import envs


# When running this plugin on a Mac, we assume it's for local development
# purposes. However, due to a compatibility issue with vLLM, which overrides
# the Triton module with a placeholder, vLLM may fail to load on macOS. To
# mitigate this issue, we can safely remove the Triton module (if imported)
# and rely on PyTorch to handle the absence of Triton, ensuring fine execution
# in eager mode.
if sys.platform.startswith("darwin"):
    if sys.modules.get("triton"):
        del sys.modules["triton"]

from vllm.logger import init_logger
from vllm.platforms import PlatformEnum
from vllm.platforms.cpu import CpuPlatform
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

if TYPE_CHECKING:
    # NB: We can't eagerly import many things from vllm since vllm.config
    # will import this file. These would lead to circular imports
    from vllm.config import VllmConfig
    from vllm.config.kernel import IrOpPriorityConfig
else:
    VllmConfig = None
    IrOpPriorityConfig = None

logger = init_logger(__name__)


class TorchSpyrePlatform(CpuPlatform):
    _enum = PlatformEnum.OOT

    # "spyre" device_name no longer worked due to https://github.com/vllm-project/vllm/pull/16464
    device_name: str = "cpu"
    device_type: str = "cpu"

    # Primary dispatch key for direct_register_custom_op. Kept as CPU
    # because some custom ops receive CPU-only tensors (e.g. rotary_embedding).
    # All ops are ALSO registered for PrivateUse1 (Spyre) via
    # register_spyre_dispatch() in each module's register() function,
    # so dispatch works regardless of tensor device.
    dispatch_key: str = "CPU"

    # Multi-backend init string consumed by both vllm's
    # `init_distributed_environment` and `torch.distributed.new_group`.
    # `gloo` handles CPU tensors (used by vllm's parallel-state cpu_group
    # and any host-side coordination); `spyreccl` handles Spyre tensors
    # for the device_group. See `torch_spyre._autoload` (registers
    # DISTRIBUTED_BACKEND_NAME via `dist.Backend.register_backend`).
    dist_backend: str = "cpu:gloo,spyre:spyreccl"

    # Register the PyTorch Native Attention implementation as the CUSTOM backend.
    # SPYRE_ATTN_IMPL=exp selects spyre_attn_exp.py; anything else uses spyre_attn.py.
    if envs.SPYRE_ATTN_IMPL == "exp":
        _backend_path = "spyre_inference.v1.attention.backends.spyre_attn_exp.SpyreAttentionBackend"
    else:
        _backend_path = "spyre_inference.v1.attention.backends.spyre_attn.SpyreAttentionBackend"

    register_backend(AttentionBackendEnum.CUSTOM, _backend_path)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "torch-spyre"

    @classmethod
    def log_server_boot(cls, vllm_config: VllmConfig) -> None:
        # Only log in main process (not in TP workers)
        if multiprocessing.current_process().name != "MainProcess":
            return

        # yapf: disable
        logo_template = Template(
            template="\n    ${red}▄█▀▀█▄${r}  ${orange}█▀▀▀█▄${r}  ${yellow}█   █${r}  ${green}█▀▀▀█▄${r}  ${blue}█▀▀▀▀${r}    ${w}█  █▄   █  █▀▀▀▀ █▀▀▀▀  █▀▀▀█▄ █▀▀▀▀  █▄   █  ▄█▀▀█▄ █▀▀▀▀${r}\n" # noqa: E501
            "    ${red}▀▀▄▄▄${r}   ${orange}█▄▄▄█▀${r}  ${yellow}▀▄ ▄▀${r}  ${green}█▄▄▄█▀${r}  ${blue}█▄▄▄${r}     ${w}█  █ █  █  █▄▄▄  █▄▄▄   █▄▄▄█▀ █▄▄▄   █ █  █  █      █▄▄▄${r}\n" # noqa: E501
            "         ${red}█${r}  ${orange}█${r}        ${yellow}▀█▀${r}   ${green}█ ▀█▄${r}   ${blue}█${r}        ${w}█  █  █ █  █     █      █ ▀█▄  █      █  █ █  █      █${r}\n" # noqa: E501
            "    ${red}▀▄▄▄█▀${r}  ${orange}█${r}         ${yellow}█${r}    ${green}█   ▀█${r}  ${blue}█▄▄▄▄${r}    ${w}█  █   ▀█  █     █▄▄▄▄  █   ▀█ █▄▄▄▄  █   ▀█  ▀█▄▄█▀ █▄▄▄▄${r}\n" # noqa: E501
            "\n    version ${w}%s${r}    model ${w}%s${r}\n"
        )
        # yapf: enable
        colors = {
            "w": "\033[97;1m",  # white
            "o": "\033[93m",  # orange
            "b": "\033[94m",  # blue
            "r": "\033[0m",  # reset
            "red": "\033[91m",  # red (rainbow start)
            "orange": "\033[38;5;208m",  # orange
            "yellow": "\033[93m",  # yellow
            "green": "\033[92m",  # green
            "blue": "\033[94m",  # blue (rainbow end)
        }

        message = logo_template.substitute(colors)

        version = importlib.metadata.version("spyre_inference")

        model_name = vllm_config.model_config.model if vllm_config.model_config else "N/A"

        print(message % (version, model_name), flush=True)

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: VllmConfig) -> None:
        """Set Spyre-specific config defaults before vLLM's defaulting logic."""
        from vllm.config import CompilationMode

        vllm_config.compilation_config.mode = CompilationMode.NONE

        # Force eager execution. torch.compile with the Spyre inductor
        # backend requires ALL graph tensors on Spyre, but our CPU fallback
        # ops (embedding, linear, rotary, attention) create intermediate
        # CPU tensors that the Spyre backend cannot codegen. Once all layers
        # run natively on Spyre, this can be removed to enable compilation.
        vllm_config.model_config.enforce_eager = True

        # In check_and_update_config we assert this must be float16 for spyre.
        # This must be set here as the default, otherwise all usage (including test fixtures) would
        # require setting the dtype.
        vllm_config.model_config.dtype = torch.float16

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # The base `CpuPlatform` returns `CpuCommunicator`, which delegates
        # to gloo collectives. With `dist_backend = "cpu:gloo,spyre:spyreccl"`
        # the device_group is bound to spyreccl, so we need a Spyre-aware
        # communicator that knows which collectives the comms library
        # actually implements (and falls back manually for the rest).
        # See `spyre_inference/distributed/spyre_communicator.py`.
        return "spyre_inference.distributed.spyre_communicator.SpyreCommunicator"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, *args, **kwargs) -> str:
        if selected_backend == AttentionBackendEnum.CUSTOM:
            return AttentionBackendEnum.CUSTOM.get_path()
        else:
            return super().get_attn_backend_cls(selected_backend, *args, **kwargs)

    @classmethod
    def get_default_ir_op_priority(cls, vllm_config: "VllmConfig") -> "IrOpPriorityConfig":
        from vllm.config.kernel import IrOpPriorityConfig

        return IrOpPriorityConfig.with_default(
            ["native"],
            rms_norm=["spyre", "native"],
            # rms_norm=["native"],
        )

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        cls.log_server_boot(vllm_config)

        # ---- compilation / custom ops ----
        # compilation_config = vllm_config.compilation_config

        # Check if the model dtype is different from float16,
        # which is only currently supported in torch-spyre
        if vllm_config.model_config.dtype != torch.float16:
            raise ValueError(
                f"The model dtype needs to be torch.float16 for spyre, "
                f"but was specified to be {vllm_config.model_config.dtype}"
            )

        parallel_config = vllm_config.parallel_config

        # Spyre does not currently support data parallelism. The worker's
        # WORLD_SIZE / RANK derivation in spyre_worker.init_device assumes a
        # single DP replica, and the spyre-comms global rank space has not
        # been validated for DP×TP configurations.
        if parallel_config.data_parallel_size > 1:
            raise ValueError(
                f"Spyre does not support data_parallel_size > 1 "
                f"(got {parallel_config.data_parallel_size})."
            )

        # ---- worker ----
        if parallel_config.worker_cls == "auto":
            # "auto" defaults to the CPUWorker as we inherit from the CpuPlatform
            # Override with TorchSpyreWorker for Spyre-specific functionality
            worker_class = "spyre_inference.v1.worker.spyre_worker.TorchSpyreWorker"
            logger.info("Loading worker from: %s", worker_class)
            parallel_config.worker_cls = worker_class

        # ---- scheduler ----
        scheduler_config = vllm_config.scheduler_config
        # default scheduler
        scheduler_class = "vllm.v1.core.sched.scheduler.Scheduler"
        # if a torch spyre specific scheduler class is needed it can be loaded with
        # scheduler_class = "spyre_inference.v1.core.scheduler.TorchSpyreScheduler"
        logger.info("Loading scheduler from: %s", scheduler_class)
        scheduler_config.scheduler_cls = scheduler_class

        # CPUWorker derives its KV-cache budget from host RAM, but on Spyre
        # the cache lives on-device — the host-RAM math is meaningless and
        # `gpu_memory_utilization * total_RAM` typically exceeds available
        # RAM on Spyre boxes, tripping CPUWorker.__init__'s preflight check.
        # Setting VLLM_CPU_KVCACHE_SPACE makes CpuPlatform.check_and_update_config
        # populate `cache_config.kv_cache_memory_bytes` below, which both
        # bypasses the preflight check and short-circuits the host-RSS math
        # in CPUWorker.determine_available_memory. Skip when the user has
        # explicitly supplied --kv-cache-memory-bytes so we don't clobber it.
        if vllm_config.cache_config.kv_cache_memory_bytes is None:
            os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "4")

        # call CpuPlatform.check_and_update_config()
        super().check_and_update_config(vllm_config)
