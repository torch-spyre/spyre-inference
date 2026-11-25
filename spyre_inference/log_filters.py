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

"""vLLM log records that are noise on Spyre. Each entry has a row in the vLLM upgrade
procedure (Step 2b), and test_log_filters.py fails when vLLM rewords its template."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LogFilter:
    logger: str
    template: str
    # Drops every record of the template when None.
    drop: Callable[[logging.LogRecord], bool] | None = None


def _platform_cast(record: logging.LogRecord) -> bool:
    args = record.args
    return (
        isinstance(args, tuple)
        and len(args) == 3
        and args[2] is torch.float16
        and args[1] in (torch.bfloat16, torch.float32)
    )


def _triton_unused(record: logging.LogRecord) -> bool:
    from spyre_inference import envs

    return envs.environment_variables["SPYRE_MRV2_KERNELS"]() != "triton"


FILTERS: tuple[LogFilter, ...] = (
    LogFilter(
        "vllm.model_executor.models.utils",
        "Attempted to load weight %s with dtype %s into parameter with dtype %s",
        _platform_cast,
    ),
    LogFilter(
        "vllm.config.kernel",
        "Final IR op priority after setting platform defaults: %s",
    ),
    LogFilter(
        "vllm.platforms.cpu",
        "Triton is not installed. triton-cpu is expected on x86 CPUs.",
        _triton_unused,
    ),
    LogFilter(
        "vllm.v1.worker.gpu.buffer_utils",
        "Pinned memory is not available on this platform; falling back "
        "to device memory for UVA buffers.",
    ),
    LogFilter(
        "vllm.config.vllm",
        "Inductor compilation was disabled by user settings, optimizations settings "
        "that are only active during inductor compilation will be ignored.",
    ),
    LogFilter(
        "vllm.utils.torch_utils",
        "OMP_NUM_THREADS=%s is set; leaving Torch threads at %d for serving. "
        "Multi-threaded torch CPU ops during serving can degrade performance through "
        "spin-wait contention and cgroup CPU-quota throttling.",
    ),
)


class _SpyreLogFilter(logging.Filter):
    def __init__(self, entries: list[LogFilter]) -> None:
        super().__init__()
        self._entries = entries

    def filter(self, record: logging.LogRecord) -> bool:
        for entry in self._entries:
            if record.msg == entry.template and (entry.drop is None or entry.drop(record)):
                return False
        return True


def install() -> None:
    """Attach the filters to their loggers; idempotent, run from ``register()``."""
    by_logger: dict[str, list[LogFilter]] = {}
    for entry in FILTERS:
        by_logger.setdefault(entry.logger, []).append(entry)
    for name, entries in by_logger.items():
        target = logging.getLogger(name)
        if not any(isinstance(f, _SpyreLogFilter) for f in target.filters):
            target.addFilter(_SpyreLogFilter(entries))
