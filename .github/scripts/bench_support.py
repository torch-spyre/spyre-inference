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

"""Always-on, off-the-clock records for each benchmark run by run_vllm_benchmarks.py.

Per test, under `<results-dir>/<test_name>/` (a subdirectory, so the ingest's top-level
`*.json` glob never sees it): env.json, phases.json, host_resources.json +
host_series.jsonl and, for serve, outputs.json. Every writer swallows its own errors:
a missing record must never fail a benchmark.
"""

import functools
import glob
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

MAX_SERIES_POINTS = 2000
ARTIFACT_JSON = "/home/senuser/spyre_artifact.json"

# Short, fixed and greedy: the hash is only comparable across runs if these never change.
OUTPUT_PROMPTS = (
    "The capital of France is",
    "def fibonacci(n):",
    "Q: What is 17 times 23?\nA:",
    "Once upon a time, in a small village by the sea,",
)
OUTPUT_MAX_TOKENS = 16


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


def _cards(missing: dict) -> list[dict]:
    cards = []
    for dev in sorted(glob.glob("/sys/bus/pci/devices/*")):
        try:
            vendor = Path(f"{dev}/vendor").read_text().strip()
            cls = Path(f"{dev}/class").read_text().strip()
        except OSError:
            continue
        if vendor == "0x1014" and cls.startswith("0x1200"):
            card = {"pci": os.path.basename(dev), "vendor_id": vendor, "class": cls}
            for f in ("device", "subsystem_device", "current_link_speed", "current_link_width"):
                if os.path.exists(f"{dev}/{f}"):
                    card[f] = _read(f"{dev}/{f}", f"card.{card['pci']}.{f}", missing)
            cards.append(card)
    if not cards:
        missing["cards"] = "no IBM (0x1014) class-0x12xx PCI device visible in /sys/bus/pci"
    return cards


def _knobs() -> dict[str, str]:
    prefixes = (
        "SPYRE_", "TORCH_SPYRE", "VLLM_", "FLEX_", "DXP_", "LX_", "SENLIB", "OMP_", "DTLOG",
        "TORCHINDUCTOR",
    )  # fmt: skip
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if k in ("SENCORES", "DXP_LX_FRAC_AVAIL") or k.startswith(prefixes)
    }


@functools.cache
def env_fingerprint() -> dict:
    """Host, CPU, cgroup limits, Spyre cards, knobs and artifact identity; never raises.

    An unreadable field is None with its reason under `_missing`. Cached: the host does not
    change between tests, and the first call is made before any timed section starts.
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
            "image": os.environ.get("IMAGE_REF"),
            "knobs": _knobs(),
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


def _downsample(rows: list, limit: int) -> list:
    if len(rows) <= limit:
        return rows
    step = -(-len(rows) // limit)
    return rows[::step]


class Recorder:
    """Collects phases and sampler series for one test, then writes the shared-format files."""

    def __init__(self, results_dir: Path, test_name: str, runtime_config: dict) -> None:
        self.dir = results_dir / test_name
        self._env = {**env_fingerprint(), "runtime_config": runtime_config}
        self._phases: list[dict] = []
        self._samplers: list = []
        # torch-spyre writes compile_summary.<pid>.json here when the image supports it.
        self.compile_dir = self.dir / "compile"

    def child_env(self, env: dict) -> None:
        """Point the benchmarked process at this test's compile-summary directory, in place."""
        env.setdefault("TORCH_SPYRE_COMPILE_SUMMARY_DIR", str(self.compile_dir))

    def _write_compile_summary(self) -> None:
        """Merge per-pid summaries (one per TP worker) into compile_summary.json, if any."""
        merged = {"total_s": 0.0, "passes": {}, "cache": {"entries": 0, "hits": 0, "misses": 0}}
        files = sorted(self.compile_dir.glob("compile_summary.*.json"))
        for f in files:
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            merged["total_s"] += d.get("total_s", 0.0)
            for k, v in d.get("passes", {}).items():
                merged["passes"][k] = merged["passes"].get(k, 0.0) + v
            for k in merged["cache"]:
                merged["cache"][k] += d.get("cache", {}).get(k, 0)
        if files:
            _write_json(self.dir / "compile_summary.json", merged)

    def mark(self, name: str | None) -> None:
        """End the open phase and start `name` (None just ends it)."""
        now = time.monotonic()
        if self._phases and self._phases[-1]["t_end"] is None:
            self._phases[-1]["t_end"] = now
        if name:
            self._phases.append({"name": name, "t_start": now, "t_end": None})

    def add_sampler(self, sampler) -> None:
        self._samplers.append(sampler)

    def write(self) -> None:
        """Write env, phases and host resources; never raises."""
        self.mark(None)
        try:
            _write_json(self.dir / "env.json", self._env)
            _write_json(
                self.dir / "phases.json",
                {
                    "phases": self._phases,
                    "clock": "monotonic",
                    "wall_offset_s": time.time() - time.monotonic(),
                },
            )
            self._write_host_resources()
            self._write_compile_summary()
        except Exception as e:
            log.warning("Could not write bench records: %s", e)

    def _phase_of(self, t: float) -> str:
        for p in self._phases:
            if p["t_end"] is not None and p["t_start"] <= t <= p["t_end"]:
                return p["name"]
        return "other"

    def _write_host_resources(self) -> None:
        rows = []
        for s in self._samplers:
            prev = getattr(s, "t0", None)
            for r in getattr(s, "series", []):
                dt = r["t_s"] - prev if prev is not None else 0.0
                prev = r["t_s"]
                rows.append({**r, "dt_s": dt, "phase": self._phase_of(r["t_s"])})
        if not rows:
            return
        rows.sort(key=lambda r: r["t_s"])
        summary = {}
        for name in dict.fromkeys(r["phase"] for r in rows):
            ph = [r for r in rows if r["phase"] == name]
            cores = [r["cpu_cores"] for r in ph]
            the = [r["throttled_cum_s"] for r in ph if "throttled_cum_s" in r]
            nthr = [r["nr_throttled"] for r in ph if "nr_throttled" in r]
            summary[name] = {
                "cpu_s": round(sum(r["cpu_cores"] * r["dt_s"] for r in ph), 3),
                "cpu_cores_mean": round(statistics.mean(cores), 3),
                "cpu_cores_peak": round(max(cores), 3),
                "throttled_s": round(max(the) - min(the), 3) if the else None,
                "nr_throttled": int(max(nthr) - min(nthr)) if nthr else None,
                "rss_peak_b": max((r["rss_b"] for r in ph if "rss_b" in r), default=None),
                "cgroup_peak_b": max((r["cgroup_b"] for r in ph if "cgroup_b" in r), default=None),
                # Needs torch.spyre in the benchmarked process; the harness only sees subprocesses.
                "dev_allocated_peak_b": None,
            }
        series = [
            {
                "t_s": r["t_s"],
                "phase": r["phase"],
                "cpu_cores": r["cpu_cores"],
                "rss_b": r.get("rss_b"),
                "cgroup_b": r.get("cgroup_b"),
                "throttled_cum_s": r.get("throttled_cum_s"),
                "dev_allocated_b": None,
            }
            for r in _downsample(rows, MAX_SERIES_POINTS)
        ]
        _write_json(
            self.dir / "host_resources.json",
            {"phases": summary, "series_file": "host_series.jsonl", "samples": len(rows)},
        )
        with open(self.dir / "host_series.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in series)

    def write_outputs(self, outputs: dict | None) -> None:
        if outputs:
            _write_json(self.dir / "outputs.json", outputs)


def greedy_output_hashes(base_url: str, model: str, timeout: float = 120.0) -> dict | None:
    """Hash the greedy completion of each fixed prompt against a live server, or None.

    Run after the timed request phase. Hashes token ids when the server returns them
    (`return_token_ids`), else the completion text.
    """
    hashes, source = [], "token_ids"
    try:
        for prompt in OUTPUT_PROMPTS:
            body = json.dumps(
                {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": OUTPUT_MAX_TOKENS,
                    "temperature": 0,
                    "return_token_ids": True,
                }
            ).encode()
            req = urllib.request.Request(
                f"{base_url}/v1/completions", body, {"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                choice = json.load(resp)["choices"][0]
            ids = choice.get("token_ids")
            if ids is None:
                source, payload = "text", choice["text"]
            else:
                payload = json.dumps(ids)
            hashes.append(hashlib.sha256(payload.encode()).hexdigest())
    except Exception as e:
        log.warning("Output hash skipped: %s", e)
        return None
    return {"prompts": len(hashes), "token_sha256": hashes, "greedy": True, "hashed": source}
