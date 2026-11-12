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

"""Pin each worker to the CPUs of its Spyre card's NUMA node.

A worker's host work between collectives is short, so where the scheduler places its
threads shows up in TPOT, and CPUs near the card are faster than CPUs on another node.

The node must hold the worker's thread pools: OMP_NUM_THREADS, as `configure_threading`
set it, measured in the unit it was sized in. SPYRE_NUM_CPUS and the cgroup quota count
logical CPUs, so they are compared with the node's CPUs; without either, the pools were
sized from physical cores and are compared with the node's cores.

Everything the worker spawns afterwards inherits the mask, including Inductor's compile
workers during warmup. Inside a cgroup quota the quota is the tighter limit; on bare metal,
warmup compiles on the node's CPUs instead of the whole host's.

Leaves the affinity alone when the card reports no node (-1, as on s390x), when the host
has one node, when the engine runs in the caller's process, when the node is too small,
or when the process is already confined to fewer CPUs than the scheduler would give it:
`taskset`, `--numa-bind` and a cpuset are the user's choice.
"""

import contextlib
import math
import os
from pathlib import Path

from vllm.logger import init_logger

from spyre_inference.threading_config import get_cpu_count, runs_in_caller

logger = init_logger(__name__)

_SYSFS = Path("/sys")


def parse_cpu_list(text: str) -> set[int]:
    """`0-35,72-107` -> {0, ..., 35, 72, ..., 107}, the kernel's cpulist format."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def format_cpu_list(cpus: set[int]) -> str:
    """The inverse of parse_cpu_list, for logs."""
    runs: list[list[int]] = []
    for cpu in sorted(cpus):
        if runs and cpu == runs[-1][1] + 1:
            runs[-1][1] = cpu
        else:
            runs.append([cpu, cpu])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return None


def physical_cores(cpus: set[int], sysfs: Path = _SYSFS) -> int:
    """Distinct cores behind ``cpus``: SMT siblings share one. Counts each CPU as its own
    core when the topology can't be read."""
    cores = set()
    for cpu in cpus:
        topology = sysfs / f"devices/system/cpu/cpu{cpu}/topology"
        package = _read_int(topology / "physical_package_id")
        core = _read_int(topology / "core_id")
        cores.add((package, core) if package is not None and core is not None else ("cpu", cpu))
    return len(cores)


def numa_nodes(online: set[int], sysfs: Path = _SYSFS) -> dict[int, set[int]]:
    """Each NUMA node's online CPUs."""
    nodes: dict[int, set[int]] = {}
    for path in (sysfs / "devices/system/node").glob("node[0-9]*"):
        with contextlib.suppress(OSError, ValueError):
            if cpus := parse_cpu_list((path / "cpulist").read_text()) & online:
                nodes[int(path.name[len("node") :])] = cpus
    return nodes


def _set_process_affinity(cpus: set[int]) -> None:
    # sched_setaffinity moves one thread; repeat until no thread is left unpinned, in case
    # an unpinned one started another meanwhile.
    pinned: set[int] = set()
    while True:
        try:
            tids = {int(t) for t in os.listdir("/proc/self/task")} - pinned
        except OSError:
            os.sched_setaffinity(0, cpus)
            return
        if not tids:
            return
        for tid in tids:
            with contextlib.suppress(ProcessLookupError):  # the thread exited
                os.sched_setaffinity(tid, cpus)
        pinned |= tids


def pin_to_card_node(parallel_config, pcie_address: str, sysfs: Path = _SYSFS) -> None:
    """Confine this worker process to the CPUs of its card's NUMA node."""
    if not hasattr(os, "sched_setaffinity"):
        return
    if (
        runs_in_caller(parallel_config)
        or parallel_config.distributed_executor_backend == "external_launcher"
    ):
        # The worker is the caller's script, notebook, test run or torchrun rank, whose
        # mask would outlive the engine and pass to everything it spawns.
        logger.info("Not pinning: the engine runs in the caller's process.")
        return
    node = _read_int(sysfs / "bus/pci/devices" / pcie_address / "numa_node")
    if node is None or node < 0:
        logger.info("Not pinning: card %s reports no NUMA node.", pcie_address)
        return
    try:
        online = parse_cpu_list((sysfs / "devices/system/cpu/online").read_text())
    except (OSError, ValueError):
        return
    nodes = numa_nodes(online, sysfs)
    if len(nodes) < 2:
        logger.info("Not pinning: the host has one NUMA node with online CPUs.")
        return
    if node not in nodes:
        logger.info("Not pinning: card %s's NUMA node %d has no online CPUs.", pcie_address, node)
        return

    budget, source = get_cpu_count()
    threads_env = os.environ.get("OMP_NUM_THREADS", "")
    threads = (
        int(threads_env)
        if threads_env.isdigit() and int(threads_env) > 0
        else math.ceil((budget or len(online)) / parallel_config.world_size)
    )
    # Only the psutil fallback counts physical cores, so only there do the two counts differ.
    sized_in_cores = budget != get_cpu_count(use_logical_cpus=True)[0]
    capacity = physical_cores(nodes[node], sysfs) if sized_in_cores else len(nodes[node])
    if capacity < threads:
        logger.info(
            "Not pinning to card %s's NUMA node %d: its %d %s can't hold %d threads (%s).",
            pcie_address,
            node,
            capacity,
            "cores" if sized_in_cores else "CPUs",
            threads,
            source,
        )
        return

    allowed = os.sched_getaffinity(0)
    isolated: set[int] = set()
    with contextlib.suppress(OSError, ValueError):
        isolated = parse_cpu_list((sysfs / "devices/system/cpu/isolated").read_text())
    # isolcpus= narrows every task's default mask to the housekeeping CPUs.
    if allowed not in (online, online - isolated):
        logger.info(
            "Not pinning to NUMA node %d (CPUs %s): this process is already confined to CPUs %s.",
            node,
            format_cpu_list(nodes[node]),
            format_cpu_list(allowed),
        )
        return
    cpus = nodes[node] & allowed
    if not cpus:
        logger.info(
            "Not pinning: every CPU of card %s's NUMA node %d is isolated.", pcie_address, node
        )
        return

    _set_process_affinity(cpus)
    logger.info(
        "Pinned card %s's worker to NUMA node %d: CPUs %s.",
        pcie_address,
        node,
        format_cpu_list(cpus),
    )
