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

Writes one *.host_resources.json (JSONEachRow) per test, ingested by
spyre_clickhouse_ingest.vllm (torch-spyre), which expects this suffix and record shape.
"""

import contextlib
import json
import logging
import math
import os
import statistics
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import psutil
except ImportError:  # the series then omits rss_b
    psutil = None

log = logging.getLogger(__name__)

# Default interval; override via RESOURCE_SAMPLER_INTERVAL env var.
_DEFAULT_INTERVAL = 1.0


def _resolve_cgroup(proc_cgroup: str, mountinfo: str) -> Path | None:
    """This process's own cgroup directory, from /proc/self/cgroup and /proc/self/mountinfo.

    The cgroup2 mount may expose a subtree (mount root != "/"), and the process may sit
    below the mount's top, so the directory is the 0:: path made relative to the mount root
    and joined onto the mountpoint. The mountpoint alone is the root cgroup, which is
    machine-wide and has no memory.current, unless the process has a private cgroup namespace.
    """
    own = next((ln[3:] for ln in proc_cgroup.splitlines() if ln.startswith("0::")), None)
    if own is None:
        return None
    for line in mountinfo.splitlines():
        # Fields: mount-id parent-id major:minor mount-root mountpoint ... - cgroup2 ...
        if " - cgroup2 " not in line:
            continue
        fields = line.split()
        mount_root, mountpoint = fields[3], fields[4]
        try:
            return Path(mountpoint) / PurePosixPath(own).relative_to(mount_root)
        except ValueError:
            # Not under the mounted subtree: the mount top is all that is visible.
            return Path(mountpoint)
    return None


def _cgroup_root() -> Path:
    try:
        root = _resolve_cgroup(
            Path("/proc/self/cgroup").read_text(), Path("/proc/self/mountinfo").read_text()
        )
    except (OSError, IndexError):
        root = None
    return root or Path("/sys/fs/cgroup")


def _read_cgroup_cpu_usec(root: Path) -> int:
    v2 = root / "cpu.stat"
    if v2.exists():
        for line in v2.read_text().splitlines():
            if line.startswith("usage_usec "):
                return int(line.split()[1])
    raise OSError(f"usage_usec not found in {v2}")


def _read_cgroup_throttle(root: Path) -> tuple[float, int]:
    """Cumulative (throttled seconds, throttled periods) from cpu.stat."""
    stat = dict(line.split() for line in (root / "cpu.stat").read_text().splitlines())
    return int(stat.get("throttled_usec", 0)) / 1e6, int(stat.get("nr_throttled", 0))


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
    raise OSError(f"memory.current not found under {root}")


class ContainerSampler:
    """Background cgroup sampler. cpu_pct can exceed 100% (not normalised by CPU count)."""

    def __init__(self, interval: float | None = None) -> None:
        if interval is None:
            interval = float(os.environ.get("RESOURCE_SAMPLER_INTERVAL", str(_DEFAULT_INTERVAL)))
        self._interval = interval
        self._cpu_samples: list[float] = []
        self._mem_samples: list[float] = []
        # One row per sample for bench_support: t_s (monotonic), cpu_cores, cgroup_b, and
        # rss_b / throttled_cum_s / nr_throttled where readable.
        self.series: list[dict[str, float]] = []
        self._proc = psutil.Process() if psutil else None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._cgroup = _cgroup_root()

        # Baseline so first sample reflects the interval immediately after start().
        self._last_cpu_usec: int = _read_cgroup_cpu_usec(self._cgroup)
        self._last_ts: float = time.monotonic()
        self.t0 = self._last_ts

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

    def _extras(self) -> dict[str, float]:
        out: dict[str, float] = {}
        with contextlib.suppress(OSError, ValueError):
            out["throttled_cum_s"], out["nr_throttled"] = _read_cgroup_throttle(self._cgroup)
        if self._proc:
            rss = 0
            try:
                procs = [self._proc, *self._proc.children(recursive=True)]
            except psutil.Error:
                return out
            for p in procs:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:  # exited between listing and reading
                    continue
            out["rss_b"] = rss
        return out

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                cpu, mem = self._collect()
            except Exception as exc:
                log.warning("cgroup read error (sample skipped): %s", exc)
                continue
            self._cpu_samples.append(cpu)
            self._mem_samples.append(mem)
            self.series.append(
                {
                    "t_s": time.monotonic(),
                    "cpu_cores": cpu / 100,
                    "cgroup_b": mem * 1e6,
                    **self._extras(),
                }
            )

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
    """Write summary as JSONEachRow into <test_name>.host_resources.json.

    Each line: {"test_name": "...", "metric": "host_cpu_pct_total_mean", "value": 42.1}
    Suffix is distinct from .pytorch.json so ingest uses a dedicated extractor.
    test_name is top-level so identity is never derived from the filename.
    """
    if not summary:
        return

    out = results_dir / f"{test_name}.host_resources.json"
    with open(out, "w") as f:
        for metric_name, stats in summary.items():
            for stat_key, value in stats.items():
                record: dict[str, Any] = {
                    "test_name": test_name,
                    "metric": f"{metric_name}_{stat_key}",
                    "value": round(value, 3),
                }
                if model:
                    record["model"] = model
                f.write(json.dumps(record) + "\n")

    log.info("Wrote host resource metrics to %s", out.name)
