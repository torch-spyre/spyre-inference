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

"""Host resource sampler: samples cgroup CPU% and memory during benchmark runs.

Writes one *.host_resources.pytorch.json (JSONEachRow) per test, picked up by
ingest_vllm_benchmarks.py without changes.
"""

import json
import logging
import math
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Default interval; override via RESOURCE_SAMPLER_INTERVAL env var.
_DEFAULT_INTERVAL = 1.0


def _cgroup_root() -> Path:
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            # Fields: mount-id parent-id major:minor mount-root mountpoint ... - cgroup2 ...
            if " - cgroup2 " not in line:
                continue
            fields = line.split()
            mount_root = fields[3]  # field 4 (0-indexed: 3)
            return Path("/sys/fs/cgroup") / mount_root.lstrip("/")
    except (OSError, IndexError):
        pass
    return Path("/sys/fs/cgroup")


def _read_cgroup_cpu_usec(root: Path) -> int:
    v2 = root / "cpu.stat"
    if v2.exists():
        for line in v2.read_text().splitlines():
            if line.startswith("usage_usec "):
                return int(line.split()[1])

    # cgroup v1 fallback: cpuacct is system-wide, not container-scoped.
    # RHEL 9+ uses cgroup v2 exclusively; this branch is never reached.
    v1 = Path("/sys/fs/cgroup/cpuacct/cpuacct.usage")
    if v1.exists():
        return int(v1.read_text().strip()) // 1000  # ns -> us

    raise OSError("cgroup cpu accounting files not found (tried v2 cpu.stat and v1 cpuacct.usage)")


def _read_cgroup_mem_mb(root: Path) -> float:
    # working_set = memory.current - memory.stat[inactive_file]
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

    # RHEL 9+ uses cgroup v2 exclusively; this branch is never reached.
    v1 = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if v1.exists():
        return int(v1.read_text().strip()) / 1e6

    raise OSError(
        "cgroup memory accounting files not found "
        "(tried v2 memory.current/memory.stat and v1 memory.usage_in_bytes)"
    )


class ContainerSampler:
    """Background cgroup sampler. cpu_pct can exceed 100% (not normalised by CPU count)."""

    def __init__(self, interval: float | None = None) -> None:
        if interval is None:
            interval = float(os.environ.get("RESOURCE_SAMPLER_INTERVAL", str(_DEFAULT_INTERVAL)))
        self._interval = interval
        self._cpu_samples: list[float] = []
        self._mem_samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._cgroup = _cgroup_root()

        # Baseline so first sample reflects the interval immediately after start().
        self._last_cpu_usec: int = _read_cgroup_cpu_usec(self._cgroup)
        self._last_ts: float = time.monotonic()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _collect(self) -> tuple[float, float]:
        # Timestamp is anchored to the cpu.stat read, not to sleep wakeup,
        # so jitter doesn't inflate elapsed_us and deflate cpu_pct.
        cpu_usec = _read_cgroup_cpu_usec(self._cgroup)
        now = time.monotonic()
        mem_mb = _read_cgroup_mem_mb(self._cgroup)

        elapsed_us = (now - self._last_ts) * 1e6
        delta_usec = cpu_usec - self._last_cpu_usec
        self._last_cpu_usec = cpu_usec
        self._last_ts = now

        cpu_pct = (delta_usec / elapsed_us * 100) if elapsed_us > 0 else 0.0
        return cpu_pct, mem_mb

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                cpu, mem = self._collect()
            except OSError as exc:
                log.warning("cgroup read error (sample skipped): %s", exc)
                continue
            self._cpu_samples.append(cpu)
            self._mem_samples.append(mem)

    def summary(self, phase: str = "") -> dict[str, dict[str, float]]:
        """Return {metric: {mean, peak, p99}}; phase prefix for compile vs bench."""
        if not self._cpu_samples:
            return {}

        def _stats(samples: list[float]) -> dict[str, float]:
            s = sorted(samples)
            p99_idx = min(len(s) - 1, max(0, math.ceil(len(s) * 0.99) - 1))
            return {
                "mean": statistics.mean(s),
                "peak": s[-1],
                "p99": s[p99_idx],
            }

        prefix = f"{phase}_" if phase else ""
        return {
            f"{prefix}host_cpu_pct_total": _stats(self._cpu_samples),
            f"{prefix}host_mem_mb": _stats(self._mem_samples),
        }


def write_resource_metrics(
    test_name: str,
    results_dir: Path,
    summary: dict[str, Any],
    model: str = "",
) -> None:
    """Write summary as JSONEachRow into <test_name>.host_resources.pytorch.json."""
    if not summary:
        return

    out = results_dir / f"{test_name}.host_resources.pytorch.json"
    benchmark_meta: dict[str, Any] = {"test_name": test_name}
    if model:
        benchmark_meta["model"] = model
    with open(out, "w") as f:
        for metric_name, stats in summary.items():
            for stat_key, value in stats.items():
                record = {
                    "benchmark": benchmark_meta,
                    "metric": {
                        "name": f"{metric_name}_{stat_key}",
                        "benchmark_values": [round(value, 3)],
                    },
                }
                f.write(json.dumps(record) + "\n")

    log.info("Wrote host resource metrics to %s", out.name)
