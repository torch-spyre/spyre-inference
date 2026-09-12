# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host resource sampler for benchmark runs.

Samples CPU and memory utilisation of all processes in the container in a
background thread while a benchmark runs.  Results are written as a single
*.host_resources.pytorch.json file per test in JSONEachRow format so that
ingest_vllm_benchmarks.py picks them up without any changes.

Cgroup-based sampling
---------------------
Both CPU and memory are read from the container's cgroup control files rather
than by iterating /proc/<pid> entries via psutil. This approach:

* Catches short-lived subprocesses that spawn and exit between samples —
  their CPU time is already baked into the monotonic ``usage_usec`` counter.
* Requires only 2 file reads per sample instead of O(N-processes) /proc reads,
  eliminating the TOCTOU races that occur when a process dies mid-iteration.

Cgroup path resolution
----------------------
The container's cgroup directory is resolved from ``/proc/self/mountinfo``
(the cgroup2 mount-root field).

``/proc/self/mountinfo`` field 4 (mount root) gives the real host-side path
where the cgroup2 fs is anchored, e.g. ``/system.slice/docker-abc.scope``.
Appending that to ``/sys/fs/cgroup`` gives the correct container-scoped
directory in all cases (bare host, cgroupns=private, cgroupns=host).

cgroup v2 files used
  cpu.stat          -> ``usage_usec`` (monotonic; delta / elapsed * 100)
  memory.current - memory.stat[inactive_file]

cgroup v1 fallback
  cpuacct/cpuacct.usage   -> nanoseconds (divide by 1000 to get usec equivalent)
  memory/memory.usage_in_bytes

Metrics emitted per test (one file, two phases)
-----------------------------------------------
  compile_host_cpu_pct_mean/_peak/_p99 - CPU % during model compile/warmup
  compile_host_mem_mb_mean/_peak/_p99  - cgroup memory charge (MiB) during compile/warmup
  host_cpu_pct_mean/_peak/_p99         - CPU % during steady-state inference
  host_mem_mb_mean/_peak/_p99          - cgroup memory charge (MiB) during steady-state inference

Pass the ``phase`` argument to summary() to select the prefix:
  compile_sampler.summary(phase="compile")  ->  compile_host_cpu_pct, compile_host_mem_mb
  bench_sampler.summary()                   ->  host_cpu_pct, host_mem_mb
Then merge both dicts and call write_resource_metrics() once.
"""

import json
import logging
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Default sampling interval in seconds.  Override via RESOURCE_SAMPLER_INTERVAL
# env var (e.g. set to 0.05 in unit tests to guarantee samples are collected during
# short-running test runs). Read at construction time (not module load) so
# monkeypatch.setenv works in tests.
_DEFAULT_INTERVAL = 1.0


# ---------------------------------------------------------------------------
# Cgroup helper utilities
# ---------------------------------------------------------------------------

def _cgroup_root() -> Path:
    """Return the cgroup2 directory that accounts for the current container.

    The correct approach is to read the cgroup2 entry from
    ``/proc/self/mountinfo``.

    ``/proc/self/mountinfo`` field 4 is the **mount root**: the path within
    the host cgroup hierarchy where the cgroup2 filesystem is rooted.  On a
    bare host this is ``/``.  Inside a cgroupns=private container it is the
    container's actual cgroup path on the host (e.g.
    ``/system.slice/docker-abc123.scope``), so
    ``/sys/fs/cgroup/<mount-root>`` is exactly the container's own cgroup
    directory regardless of namespace.

    Falls back to ``/sys/fs/cgroup`` (host root) if mountinfo is absent or
    unparsable.
    """
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            # Fields: mount-id parent-id major:minor mount-root mountpoint ...
            # The filesystem type is in the field after the " - " separator.
            # e.g.: "49 47 0:30 / /sys/fs/cgroup rw ... - cgroup2 cgroup2 rw ..."
            if " - cgroup2 " not in line:
                continue
            fields = line.split()
            mount_root = fields[3]  # field 4 (0-indexed: 3)
            return Path("/sys/fs/cgroup") / mount_root.lstrip("/")
    except (OSError, IndexError):
        pass
    return Path("/sys/fs/cgroup")


def _read_cgroup_cpu_usec(root: Path) -> int:
    """Return total CPU microseconds consumed by the cgroup.

    Tries cgroup v2 ``cpu.stat`` first, then cgroup v1 ``cpuacct.usage``
    (which is in nanoseconds and is converted to microseconds).
    Raises ``OSError`` if neither is readable.
    """
    v2 = root / "cpu.stat"
    if v2.exists():
        for line in v2.read_text().splitlines():
            if line.startswith("usage_usec "):
                return int(line.split()[1])

    # cgroup v1 fallback: cpuacct subsystem is mounted at a fixed system-wide
    # path, not relative to ``root``.  This means it returns the total CPU
    # time for the whole system rather than just the container's slice — acceptable
    # as a last-resort fallback on kernels that pre-date cgroup v2.
    # NOTE: RHEL 9+ uses cgroup v2 exclusively; this branch is never reached.
    v1 = Path("/sys/fs/cgroup/cpuacct/cpuacct.usage")
    if v1.exists():
        return int(v1.read_text().strip()) // 1000  # ns -> us

    raise OSError("cgroup cpu accounting files not found (tried v2 cpu.stat and v1 cpuacct.usage)")


def _read_cgroup_mem_mb(root: Path) -> float:
    """Return the working-set memory of the cgroup in MiB.

        working_set = memory.current - memory.stat[inactive_file]

    ``memory.current`` is the raw cgroup charge (anon + file cache + shm +
    kernel buffers).  ``inactive_file`` is the portion of the file-backed page
    cache that the kernel can evict under memory pressure (e.g. prefetched
    model weights that have already been consumed).  Subtracting it gives the
    memory the workload *actually needs* to keep running.

    Falls back to raw ``memory.current`` if ``memory.stat`` is absent or does
    not contain ``inactive_file`` (should not happen on cgroup v2).

    cgroup v1 fallback uses raw ``memory.usage_in_bytes`` (no inactive_file
    equivalent is subtracted).
    """
    v2_current = root / "memory.current"
    if v2_current.exists():
        current = int(v2_current.read_text().strip())
        inactive_file = 0
        v2_stat = root / "memory.stat"
        if v2_stat.exists():
            for line in v2_stat.read_text().splitlines():
                if line.startswith("inactive_file "):
                    inactive_file = int(line.split()[1])
                    break
        return max(0, current - inactive_file) / 1e6

    # NOTE: RHEL 9+ uses cgroup v2 exclusively; this branch is never reached.
    v1 = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if v1.exists():
        return int(v1.read_text().strip()) / 1e6

    raise OSError(
        "cgroup memory accounting files not found "
        "(tried v2 memory.current/memory.stat and v1 memory.usage_in_bytes)"
    )


class ContainerSampler:
    """Sample CPU utilisation and memory of the container via cgroup control files.

    Reading cgroup files is more robust than iterating /proc/<pid> entries:

    * ``usage_usec`` in ``cpu.stat`` is a monotonic counter that includes CPU
      time from processes that have already exited within the interval, so
      short-lived subprocesses are never missed.
    * ``memory.current`` counts all memory charged to the cgroup — anonymous
      pages, shared memory (tmpfs/shm), page-cache, and kernel buffers — reported
      as ``host_mem_mb``
    * Only 2 file reads per sample vs O(N-processes) /proc reads, with no
      TOCTOU races from processes dying mid-iteration.

    CPU% is the delta of ``usage_usec`` over the wall-clock interval,
    expressed as a percentage of one logical CPU.

        cpu_pct = (usage_usec[t2] - usage_usec[t1]) / elapsed_us * 100

    This value can exceed 100% when multiple cores are in use (e.g. 600%
    means ~6 cores fully utilised).  It is NOT normalised by CPU count.

    Usage::

        sampler = ContainerSampler()
        sampler.start()
        proc.wait()
        sampler.stop()
        write_resource_metrics("my_test", results_dir, sampler.summary())
    """

    def __init__(self, interval: float | None = None) -> None:
        if interval is None:
            interval = float(os.environ.get("RESOURCE_SAMPLER_INTERVAL", str(_DEFAULT_INTERVAL)))
        self._interval = interval
        self._cpu_samples: list[float] = []
        self._mem_samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._cgroup = _cgroup_root()

        # Establish the baseline CPU counter and wall-clock time so the first
        # real sample reflects the interval immediately after start().
        # cpu.stat is read before the timestamp for the same reason as in
        # _collect(): elapsed_us must cover the same window as delta_usec.
        self._last_cpu_usec: int = _read_cgroup_cpu_usec(self._cgroup)
        self._last_ts: float = time.monotonic()  # snapped immediately after read

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _collect(self) -> tuple[float, float]:
        """Read cpu.stat and memory.current and return instantaneous metrics.

        Returns (cpu_pct, mem_mb) where cpu_pct is the CPU usage since the
        last call, as a percentage of one logical CPU (can exceed 100% on
        multi-core systems).

        The wall-clock timestamp is snapped immediately after the cpu.stat
        read so that ``elapsed_us`` covers the same window as ``delta_usec``.
        Both are anchored to the cpu.stat read, not to the surrounding sleep,
        so sleep-duration jitter on a loaded system does not inflate
        ``elapsed_us`` and artificially deflate the reported CPU%.
        """
        cpu_usec = _read_cgroup_cpu_usec(self._cgroup)
        now = time.monotonic()
        mem_mb = _read_cgroup_mem_mb(self._cgroup)

        elapsed_us = (now - self._last_ts) * 1e6
        delta_usec = cpu_usec - self._last_cpu_usec
        self._last_cpu_usec = cpu_usec
        self._last_ts = now  # anchored to cpu.stat read, not to sleep wakeup

        cpu_pct = (delta_usec / elapsed_us * 100) if elapsed_us > 0 else 0.0
        return cpu_pct, mem_mb

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            cpu, mem = self._collect()
            self._cpu_samples.append(cpu)
            self._mem_samples.append(mem)

    def summary(self, phase: str = "") -> dict[str, dict[str, float]]:
        """Return {metric_name: {mean, peak, p99}} dicts. Empty if no samples.

        ``phase``, when set, is prepended to each metric name so that compile
        and bench results can be merged into a single file:
          compile_sampler.summary(phase="compile")  ->  compile_host_cpu_pct, ...
          bench_sampler.summary()                   ->  host_cpu_pct, ...
        """
        if not self._cpu_samples:
            return {}

        def _stats(samples: list[float]) -> dict[str, float]:
            s = sorted(samples)
            # p99: the value at the 99th percentile position.
            # For small sample counts this may equal peak; that's correct.
            p99_idx = min(len(s) - 1, max(0, int(len(s) * 0.99) - 1))
            return {
                "mean": statistics.mean(s),
                "peak": s[-1],
                "p99": s[p99_idx],
            }

        prefix = f"{phase}_" if phase else ""
        return {
            f"{prefix}host_cpu_pct": _stats(self._cpu_samples),
            f"{prefix}host_mem_mb": _stats(self._mem_samples),
        }


def write_resource_metrics(
    test_name: str,
    results_dir: Path,
    summary: dict[str, Any],
) -> None:
    """Write resource metrics as JSONEachRow into <test_name>.host_resources.pytorch.json.

    The format matches what ingest_vllm_benchmarks.py expects:
      {"benchmark": {"test_name": ...}, "metric": {"name": ..., "benchmark_values": [...]}}

    One line per (metric_name, stat_key) combination, e.g.:
      host_cpu_pct_mean, host_cpu_pct_peak, host_cpu_pct_p99, ...
    """
    if not summary:
        return

    out = results_dir / f"{test_name}.host_resources.pytorch.json"
    with open(out, "w") as f:
        for metric_name, stats in summary.items():
            for stat_key, value in stats.items():
                record = {
                    "benchmark": {"test_name": test_name},
                    "metric": {
                        "name": f"{metric_name}_{stat_key}",
                        "benchmark_values": [round(value, 3)],
                    },
                }
                f.write(json.dumps(record) + "\n")

    log.info("Wrote host resource metrics to %s", out.name)
