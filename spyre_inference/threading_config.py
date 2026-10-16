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

"""CPU thread-pool sizing for containerized Spyre deployments.

CPU threading libraries (OpenMP, the BLAS backends, NumExpr) size their thread
pools from the host core count by default. In a CPU-limited container (e.g. a
Kubernetes pod whose quota is far below the node's core count) that
oversubscribes the container and causes contention that slows inference.
``configure_threading`` clamps these to the detected CPU budget.
"""

import math
import os

from vllm.logger import init_logger

from spyre_inference import envs

logger = init_logger(__name__)

THREADING_ENVS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]


def get_cpu_count(use_logical_cpus: bool = False) -> tuple[float | None, str]:
    """Resolution order: SPYRE_NUM_CPUS, cgroup v2 quota, psutil cores, os.cpu_count()."""
    if (num_cpu := envs.SPYRE_NUM_CPUS) > 0:
        return float(num_cpu), f"SPYRE_NUM_CPUS is set to {num_cpu}"

    cpu_count: float | None = None
    detection_message = ""

    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota_str, period_str = f.read().strip().split()
        if quota_str != "max":
            cpu_count = int(quota_str) / int(period_str)
            detection_message = f"Detected cgroup CPU limit of {cpu_count}"
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug("Error parsing /sys/fs/cgroup/cpu.max for CPU info", exc_info=e)

    if cpu_count is None:
        try:
            import psutil

            cpu_count = float(psutil.cpu_count(logical=use_logical_cpus))
            detection_message = (
                f"Detected {cpu_count} "
                f"{'logical' if use_logical_cpus else 'physical'} CPUs from psutil"
            )
        except ImportError:
            logger.info("Install psutil to count physical CPU cores")
        except Exception as e:
            logger.debug("Error using psutil to count CPUs", exc_info=e)

    if cpu_count is None and (n := os.cpu_count()) is not None:
        cpu_count = float(n)
        detection_message = f"Detected {cpu_count} CPUs from os.cpu_count()"

    return cpu_count, detection_message


def configure_threading(worker_count: int) -> None:
    """Clamp the CPU threading env vars to the per-worker CPU budget.

    Enabled by default; set SPYRE_UPDATE_THREAD_CONFIG=0 to only warn instead of
    overriding. vLLM already forces OMP_NUM_THREADS=1 for its own multi-worker
    case, so the main win here is the cgroup-aware count in single-worker
    deployments where the base image set the threading vars to the node's core count.
    """
    assert worker_count > 0
    env_map = {env: os.environ.get(env) for env in THREADING_ENVS}
    logger.info(
        "Initial threading configuration: %s",
        " ".join(f"{env}={value}" for env, value in env_map.items()),
    )

    cpu_count, detection_message = get_cpu_count()
    # math.ceil may sum to slightly more than cpu_count across workers.
    cpus_per_worker = math.ceil(cpu_count / worker_count) if cpu_count is not None else None

    thread_warning = (
        "Excessive threads may result in CPU contention; each worker process "
        "has its own thread pools. "
        if worker_count > 1
        else ""
    )
    failed_detection_message = (
        "Unable to detect available CPUs to validate threading configuration."
    )

    if envs.SPYRE_UPDATE_THREAD_CONFIG:
        if cpus_per_worker is None:
            raise RuntimeError(
                f"{failed_detection_message} Set SPYRE_NUM_CPUS, or set "
                "SPYRE_UPDATE_THREAD_CONFIG=0 and configure threading manually."
            )
        for env in THREADING_ENVS:
            os.environ[env] = str(cpus_per_worker)
        logger.info(
            "%s. Setting each threading configuration to %d for %d worker(s) "
            "(SPYRE_UPDATE_THREAD_CONFIG enabled).",
            detection_message,
            cpus_per_worker,
            worker_count,
        )
        return

    if cpus_per_worker is None:
        logger.info("%s %s", failed_detection_message, thread_warning)
        return

    def _float_or_0(s: str | None) -> float:
        try:
            return float(s)  # ty: ignore[invalid-argument-type]
        except (TypeError, ValueError):
            return 0.0

    oversubscribed = any(
        value is None or _float_or_0(value) > 1.2 * cpus_per_worker for value in env_map.values()
    )
    if oversubscribed:
        logger.warning(
            "%s %sRecommend setting each threading configuration to %d for %d "
            "worker(s). Set SPYRE_UPDATE_THREAD_CONFIG=1 to do this automatically.",
            detection_message,
            thread_warning,
            cpus_per_worker,
            worker_count,
        )
