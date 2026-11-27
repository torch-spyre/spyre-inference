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

"""Always-on, off-the-clock records for each benchmark run by run_vllm_benchmarks.py.

Per test, under ``<results-dir>/<test_name>/`` -- a subdirectory, so the ingest's
top-level ``*.json`` glob never sees these and no result row changes:

    env.json             what machine, which cards, which knobs, which image
    phases.json          how long each phase of the run took
    compile_summary.json the per-process torch-spyre summaries, merged

Every writer swallows its own errors: a missing record must never fail a
benchmark, and nothing here runs inside a timed section.

Host resource metrics are deliberately not written here. ``resource_sampler.py``
already writes ``<test_name>.host_resources.json`` and an ingest extractor
already reads it; a second shape under a near-identical name would be two
sources of one truth.
"""

import functools
import glob
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

ARTIFACT_JSON = "/home/senuser/spyre_artifact.json"
VFIO_DIR = "/dev/vfio"
IOMMU_GROUPS = "/sys/kernel/iommu_groups"
PCI_DEVICES = "/sys/bus/pci/devices"

# torch-spyre writes compile_summary.<pid>.json into the directory named by this
# variable (one file per process, so one per TP worker).
COMPILE_SUMMARY_VAR = "TORCH_SPYRE_COMPILE_SUMMARY_DIR"
# Its total_s and passes come from the frontend timing recorder, which that
# variable alone does not switch on.
TIMING_VAR = "TORCH_SPYRE_TIMING"


def _read(path: str, field: str, missing: dict) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError as e:
        missing[field] = f"{type(e).__name__}: {e}"
        return None


def _run(cmd: list[str], field: str, missing: dict, timeout: int = 10) -> str:
    if not shutil.which(cmd[0]):
        missing[field] = f"{cmd[0]} not installed"
        return ""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        missing[field] = f"{cmd[0]}: {type(e).__name__}"
        return ""
    if r.returncode:
        missing[field] = f"{cmd[0]} rc={r.returncode}: {r.stderr.strip()[:200]}"
    return r.stdout


def _cpu(missing: dict) -> dict:
    text = _read("/proc/cpuinfo", "cpu", missing) or ""
    first = dict(re.findall(r"^([^\t:]+?)\s*:\s*(.*)$", text.split("\n\n")[0], re.M))
    mhz = [float(m) for m in re.findall(r"^cpu MHz\s*:\s*([\d.]+)", text, re.M)]
    lscpu = dict(re.findall(r"^([^:]+?):\s+(.*)$", _run(["lscpu"], "lscpu", missing), re.M))
    model = first.get("model name") or first.get("cpu") or lscpu.get("Model name")
    if model is None:
        missing["cpu.model"] = "no model name in /proc/cpuinfo or lscpu"
    return {
        "model": model,
        "mhz": round(sum(mhz) / len(mhz), 1) if mhz else None,
        "threads": lscpu.get("CPU(s)"),
        "threads_per_core": lscpu.get("Thread(s) per core"),
        "cores_per_socket": lscpu.get("Core(s) per socket"),
        "sockets": lscpu.get("Socket(s)"),
        "numa_nodes": lscpu.get("NUMA node(s)"),
        "governor": _read(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "cpu.governor", missing
        ),
    }


def _spyre_class(dev: str) -> str | None:
    """The PCI class of `dev` if it is a Spyre card: IBM vendor 0x1014, class 0x12xx."""
    try:
        if Path(f"{dev}/vendor").read_text().strip() != "0x1014":
            return None
        cls = Path(f"{dev}/class").read_text().strip()
    except OSError:
        return None
    return cls if cls.startswith("0x1200") else None


def _cards(missing: dict) -> list[dict]:
    """The cards this process can actually open: one per numbered node under /dev/vfio.

    Not a /sys/bus/pci scan. sysfs is not namespaced, so inside a pod that lists every
    card on the machine rather than the one the device plugin assigned. What the pod
    holds are the IOMMU groups passed in as /dev/vfio/<group>, which is also the set
    SPYRE_DEVICES indexes.
    """
    try:
        groups = sorted((p.name for p in Path(VFIO_DIR).iterdir() if p.name.isdigit()), key=int)
    except OSError as e:
        missing["cards"] = f"{VFIO_DIR}: {type(e).__name__}: {e}"
        return []
    if not groups:
        missing["cards"] = f"no numbered node under {VFIO_DIR}: no card is assigned here"
    cards = []
    for group in groups:
        card: dict = {"iommu_group": group}
        for link in sorted(Path(f"{IOMMU_GROUPS}/{group}/devices").glob("*")):
            dev = f"{PCI_DEVICES}/{link.name}"
            cls = _spyre_class(dev)
            if cls is None:
                continue
            card |= {"pci": link.name, "vendor_id": "0x1014", "class": cls}
            for f in ("device", "subsystem_device", "current_link_speed", "current_link_width"):
                if os.path.exists(f"{dev}/{f}"):
                    card[f] = _read(f"{dev}/{f}", f"card.{link.name}.{f}", missing)
        if "pci" not in card:
            missing[f"card.{group}"] = f"no IBM accelerator under {IOMMU_GROUPS}/{group}/devices"
        cards.append(card)
    return cards


def _cards_on_host() -> int:
    """Cards the machine has, assigned to this container or not: a run that held one of
    twelve shared the machine with whatever used the other eleven."""
    return sum(_spyre_class(d) is not None for d in glob.glob(f"{PCI_DEVICES}/*"))


def _knobs(env) -> dict[str, str]:
    """The variables that change what the stack does, out of the env a benchmark will run with."""
    prefixes = (
        "AIU", "SPYRE_", "TORCH_SPYRE", "TORCH_SENDNN", "VLLM_", "FLEX_", "DXP_", "LX_",
        "SENLIB", "OMP_", "DT", "TORCHINDUCTOR",
    )  # fmt: skip
    return {k: v for k, v in sorted(env.items()) if k == "SENCORES" or k.startswith(prefixes)}


@functools.cache
def env_fingerprint() -> dict:
    """Host, CPU, cgroup limits, Spyre cards and artifact identity; never raises.

    An unreadable field is None with its reason under `_missing`. Cached, so nothing that
    varies per test belongs here: the host does not change between tests, and the first
    call is made before any timed section starts.
    """
    missing: dict[str, str] = {}
    try:
        meminfo = _read("/proc/meminfo", "mem_total_b", missing) or ""
        mem = re.search(r"MemTotal:\s+(\d+) kB", meminfo)
        osrel = dict(
            re.findall(
                r'^(\w+)="?([^"\n]*)"?$', _read("/etc/os-release", "os", missing) or "", re.M
            )
        )
        cards = _cards(missing)
        try:
            artifact = json.loads(Path(ARTIFACT_JSON).read_text())
        except (OSError, ValueError) as e:
            artifact = None
            missing["artifact"] = f"{type(e).__name__}: {e}"
        return {
            "host": os.environ.get("NODE_NAME") or socket.gethostname(),
            "pod": os.environ.get("HOSTNAME"),
            "arch": platform.machine(),
            "cpu": _cpu(missing),
            "cgroup": {
                "cpu_max": _read("/sys/fs/cgroup/cpu.max", "cgroup.cpu_max", missing),
                "memory_max": _read("/sys/fs/cgroup/memory.max", "cgroup.memory_max", missing),
            },
            "mem_total_b": int(mem.group(1)) * 1024 if mem else None,
            "kernel": platform.release(),
            "os": osrel.get("PRETTY_NAME"),
            "cards": cards,
            "card_kind": "/".join(sorted({c["device"] for c in cards if c.get("device")})) or None,
            "cards_on_host": _cards_on_host(),
            "image": os.environ.get("IMAGE_REF"),
            "artifact": artifact,
            "_missing": missing,
        }
    except Exception as e:
        return {"_missing": {**missing, "fingerprint": f"{type(e).__name__}: {e}"}}


def _write_json(path: Path, obj) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=1, sort_keys=True))
    except (OSError, TypeError) as e:
        log.warning("Could not write %s: %s", path.name, e)


class Recorder:
    """Collects one test's phases, then writes its records. Never raises."""

    def __init__(self, results_dir: Path, test_name: str, runtime_config: dict) -> None:
        self.dir = Path(results_dir) / test_name
        self._env = {**env_fingerprint(), "runtime_config": runtime_config}
        self._child_env = os.environ
        self._phases: list[dict] = []
        self._server_startup_sec: float | None = None
        self.compile_dir = self.dir / "compile"

    def child_env(self, env: dict) -> None:
        """Point the benchmarked process at this test's compile-summary directory, in place.

        setdefault, not assignment: an operator who exported either variable meant it.

        `env` is kept by reference, not copied, and the knobs are read off it at write()
        time: this process's own environment is not what a benchmark runs with -- the
        per-test `environment_variables:` block is merged into `env` alone.
        """
        env.setdefault(COMPILE_SUMMARY_VAR, str(self.compile_dir))
        env.setdefault(TIMING_VAR, "1")
        self._child_env = env

    def mark(self, name: str | None) -> None:
        """End the open phase and start `name` (None just ends it)."""
        now = time.monotonic()
        if self._phases and self._phases[-1]["t_end"] is None:
            self._phases[-1]["t_end"] = now
        if name:
            self._phases.append({"name": name, "t_start": now, "t_end": None})

    def set_server_startup(self, seconds: float | None) -> None:
        """Seconds from spawning the server to its first healthy response."""
        self._server_startup_sec = seconds

    def _write_compile_summary(self) -> None:
        """Merge the per-pid summaries (one per TP worker) into one, if the image wrote any."""
        merged = {"total_s": 0.0, "passes": {}, "cache": {"entries": 0, "hits": 0, "misses": 0}}
        files = sorted(self.compile_dir.glob("compile_summary.*.json"))
        for f in files:
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            merged["total_s"] += d.get("total_s") or 0.0
            for k, v in (d.get("passes") or {}).items():
                merged["passes"][k] = merged["passes"].get(k, 0.0) + v
            for k in merged["cache"]:
                merged["cache"][k] += (d.get("cache") or {}).get(k) or 0
        if files:
            merged["n_processes"] = len(files)
            _write_json(self.dir / "compile_summary.json", merged)

    def write(self) -> None:
        """Write env, phases and the merged compile summary; never raises."""
        self.mark(None)
        try:
            _write_json(self.dir / "env.json", {**self._env, "knobs": _knobs(self._child_env)})
            _write_json(
                self.dir / "phases.json",
                {
                    "phases": self._phases,
                    "server_startup_sec": self._server_startup_sec,
                    "clock": "monotonic",
                    "wall_offset_s": time.time() - time.monotonic(),
                },
            )
            self._write_compile_summary()
        except Exception as e:
            log.warning("Could not write bench records: %s", e)
