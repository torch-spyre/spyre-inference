# Copyright 2026 The Torch-Spyre Authors.
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

"""
Script that uses the pastamachine module to convert a Helion kernel to a
Spyre-compiled callable, validate on CPU, execute on Spyre with profiling,
and compare results.

Sweeps over power-of-two vector sizes up to 8192 AND over different
helion.Config settings (block_sizes).  For each (size, config) combination
both profiling tables (sorted by cpu_time_total and by cuda_time_total) are
parsed to JSON.  All per-combination JSONs are stored in a timestamped
sub-folder.  The top-5 entries by Spyre time share are printed for every
combination.
"""

import torch
import helion
import helion.language as hl
import sys
import os
import json
import re
from datetime import datetime

from torch.profiler import profile, ProfilerActivity

# Add pastamachine and torch-spyre to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../new_stack/torch-spyre'))

import pastamachine

import math
import logging
# logging.getLogger("pastamachine").setLevel(logging.WARNING)
# logging.getLogger("pastamachine").setLevel(logging.DEBUG)


# ── Helpers ───────────────────────────────────────────────────────────────

def parse_profiler_table(table_str):
    """Parse a PyTorch profiler table string into a list of dicts."""
    lines = table_str.strip().splitlines()

    separator_indices = []
    for i, line in enumerate(lines):
        if set(line.strip()) <= {'-', ' '}:
            separator_indices.append(i)

    if len(separator_indices) < 2:
        return []

    header_line = lines[separator_indices[0] + 1]
    data_start = separator_indices[1] + 1

    sep = lines[separator_indices[0]]
    col_ranges = []
    start = None
    for i, ch in enumerate(sep):
        if ch == '-' and start is None:
            start = i
        elif ch != '-' and start is not None:
            col_ranges.append((start, i))
            start = None
    if start is not None:
        col_ranges.append((start, len(sep)))

    headers = []
    for s, e in col_ranges:
        headers.append(header_line[s:e].strip())

    rows = []
    for line in lines[data_start:]:
        if set(line.strip()) <= {'-', ' ', ''}:
            break
        if line.strip().startswith("Self") or line.strip().startswith("---"):
            break
        vals = {}
        for hdr, (s, e) in zip(headers, col_ranges):
            vals[hdr] = line[s:e].strip()
        if any(v for v in vals.values()):
            rows.append(vals)

    return rows


def parse_self_totals(table_str):
    """Extract 'Self ... time total' lines from the profiler table footer.

    Returns a dict like:
        {"Self CPU time total": "43.626ms", "Self SPYRE time total": "104.118us"}
    """
    totals = {}
    for line in table_str.strip().splitlines():
        line = line.strip()
        m = re.match(r'^(Self .+ time total):\s*(.+)$', line)
        if m:
            totals[m.group(1)] = m.group(2).strip()
    return totals


def parse_time_us(s):
    """Parse a time string like '1.234ms' or '567.890us' to microseconds."""
    m = re.search(r'([\d.]+)\s*(ms|us|s)', s)
    if not m:
        return 0.0
    val, unit = float(m.group(1)), m.group(2)
    if unit == 's':
        return val * 1_000_000
    elif unit == 'ms':
        return val * 1_000
    return val


def _is_device_col(col_name):
    """Check if a column name refers to the accelerator (AIU / SPYRE)."""
    low = col_name.lower()
    return "aiu" in low or "spyre" in low


def parse_percentage(s):
    """Extract a numeric percentage value from a string like '45.23%'."""
    m = re.search(r'([\d.]+)%', s)
    return float(m.group(1)) if m else 0.0


def config_label(cfg):
    """Return a short, filesystem-safe label for a helion.Config."""
    if cfg is None:
        return "default"
    bs = cfg.block_sizes
    return "bs" + "_".join(str(b) for b in bs)


def verify_sdsc_cores(meta, cfg):
    """Check that every SDSC kernel's numCoresUsed matches the expected
    core division derived from numWkSlicesPerDim.

    Returns a list of per-kernel result dicts with keys:
        operation, numCoresUsed, expected, status ("PASS" / "MISMATCH" / "SKIP")
    Raises AssertionError if any kernel has a MISMATCH.
    """
    summaries = pastamachine.summarize_meta(meta)
    dim_to_bs = getattr(meta, "tile_dim_to_block_value", {}) or {}
    results = []

    for info in summaries:
        op_name = info.get("operation", "")
        num_cores_used = info.get("numCoresUsed", "N/A")
        wk_slices = info.get("numWkSlicesPerDim", {})
        expected = math.prod(wk_slices.values()) if wk_slices else 1

        # If no config was given or this op wasn't affected, skip the check
        if cfg is None or not dim_to_bs:
            results.append({
                "operation": op_name,
                "numCoresUsed": num_cores_used,
                "expected": expected,
                "status": "SKIP (no config)",
            })
            continue

        matched = {
            name: dims for name, dims in dim_to_bs.items()
            if op_name and op_name in name
        }
        if not matched:
            results.append({
                "operation": op_name,
                "numCoresUsed": num_cores_used,
                "expected": expected,
                "status": "SKIP (not affected)",
            })
            continue

        if num_cores_used == "N/A":
            results.append({
                "operation": op_name,
                "numCoresUsed": num_cores_used,
                "expected": expected,
                "status": "SKIP (N/A)",
            })
            continue

        if num_cores_used == expected:
            status = "PASS"
        else:
            status = "MISMATCH"

        results.append({
            "operation": op_name,
            "numCoresUsed": num_cores_used,
            "expected": expected,
            "status": status,
        })

    return results


# ── Define a Helion kernel ──────────────────────────────────────────────────

@helion.kernel()
def inplace_add(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
):
    for tile in hl.tile(out.size()):
        out[tile] += a[tile] + b[tile]


# ── Configuration ─────────────────────────────────────────────────────────

SPYRE_DEVICE = torch.device("spyre")
SENCORES = os.getenv("SENCORES", "default")

# Powers of two up to 8192
VECTOR_SIZES = [2**k for k in range(5, 14)]  # 32, 64, ..., 8192

# Different helion.Config settings to sweep over.
# None means "use Helion's default config".
HELION_CONFIGS = [
    None,
    helion.Config(block_sizes=[16]),
    helion.Config(block_sizes=[32]),
    helion.Config(block_sizes=[64]),
    helion.Config(block_sizes=[128]),
    helion.Config(block_sizes=[256]),
]

# Create output folder: /home/ngl/results/<timestamp>_helion_to_spyre_v6/
script_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sweep_dir = os.path.join(
    "/home/ngl/results",
    f"{timestamp}_helion_to_spyre_v6",
)
os.makedirs(sweep_dir, exist_ok=True)
print(f"Sweep output folder: {sweep_dir}")
print(f"SENCORES={SENCORES}")
print(f"Vector sizes: {VECTOR_SIZES}")
print(f"Helion configs: {[config_label(c) for c in HELION_CONFIGS]}")

# ── Sweep ─────────────────────────────────────────────────────────────────

sweep_summary = []

for vec_size in VECTOR_SIZES:
    for cfg in HELION_CONFIGS:
        cfg_lbl = config_label(cfg)

        # Skip combinations where block_size > vec_size (would be trivial / invalid)
        if cfg is not None and cfg.block_sizes[0] > vec_size:
            print(f"\nSKIP: vec_size={vec_size}, config={cfg_lbl} (block_size > vec_size)")
            continue

        print(f"\n{'#'*70}")
        print(f"#  vector_size = {vec_size},  config = {cfg_lbl}")
        print(f"{'#'*70}")

        # ── Prepare tensors ───────────────────────────────────────────────
        a_spyre = torch.full([vec_size], 1.0, device=SPYRE_DEVICE)
        b_spyre = torch.full([vec_size], 2.0, device=SPYRE_DEVICE)
        out_spyre = torch.zeros([vec_size], device=SPYRE_DEVICE)

        # ── Transpile ─────────────────────────────────────────────────────
        transpile_kwargs = dict(
            do_verify_run_on_cpu=True,
            return_meta=True,
        )
        if cfg is not None:
            transpile_kwargs["config"] = cfg

        compiled_for_spyre, meta = pastamachine.compile_helion_to_spyre(
            inplace_add,
            (a_spyre, b_spyre, out_spyre),
            **transpile_kwargs,
        )

        print("\n--- SDSC analysis ---")
        pastamachine.print_meta_summary(meta)

        # ── Verify core counts ────────────────────────────────────────────
        core_results = verify_sdsc_cores(meta, cfg)
        for cr in core_results:
            print(f"  [core check] op={cr['operation']}: "
                  f"numCoresUsed={cr['numCoresUsed']}, "
                  f"expected={cr['expected']} → {cr['status']}")
        mismatches = [cr for cr in core_results if cr["status"] == "MISMATCH"]
        assert not mismatches, (
            f"SDSC core count MISMATCH for vec_size={vec_size}, config={cfg_lbl}: "
            f"{mismatches}"
        )

        # ── Compute core count ────────────────────────────────────────────
        MAX_CORES = 32
        if cfg is not None:
            block_size = cfg.block_sizes[0]
            num_cores = min(vec_size // block_size, MAX_CORES)
        else:
            block_size = None
            num_cores = None
            for cr in core_results:
                if cr['numCoresUsed'] != 'N/A':
                    num_cores = cr['numCoresUsed']
                    break

        # ── Snapshot inputs on CPU (before Spyre execution mutates out) ──
        cpu_inputs_snapshot = tuple(
            t.detach().cpu().clone() for t in (a_spyre, b_spyre, out_spyre)
        )

        # ── Profile ───────────────────────────────────────────────────────
        print("\n--- Executing with profiling ---")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            record_shapes=True,
        ) as prof:
            result = compiled_for_spyre(a_spyre, b_spyre, out_spyre)

        assert result is None
        spyre_output = out_spyre

        # ── Profiling tables ──────────────────────────────────────────────
        table_cpu = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
        table_spyre = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)

        # Display-friendly labels
        table_cpu_display = table_cpu.replace("CUDA", "AIU")
        table_spyre_display = table_spyre.replace("CUDA", "AIU")

        print("\n--- Profiling: sorted by CPU time ---")
        print(table_cpu_display)
        print("\n--- Profiling: sorted by Spyre (AIU) time ---")
        print(table_spyre_display)

        # ── Parse tables ──────────────────────────────────────────────────
        parsed_cpu = parse_profiler_table(table_cpu_display)
        parsed_spyre = parse_profiler_table(table_spyre_display)

        totals_cpu = parse_self_totals(table_cpu_display)
        totals_spyre = parse_self_totals(table_spyre_display)

        profiling_data = {
            "vector_size": vec_size,
            "config": cfg_lbl,
            "block_size": block_size,
            "num_cores": num_cores,
            "sencores": SENCORES,
            "self_totals_cpu_table": totals_cpu,
            "self_totals_spyre_table": totals_spyre,
            "sorted_by_cpu_time": parsed_cpu,
            "sorted_by_spyre_time": parsed_spyre,
        }

        # ── Save per-combination JSON ─────────────────────────────────────
        json_filename = f"profiling_v6_vecsize{vec_size}_cfg{cfg_lbl}_sencores{SENCORES}.json"
        json_path = os.path.join(sweep_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(profiling_data, f, indent=2)
        print(f"  Saved: {json_path}")

        # ── Compute Spyre time excl. memory ops ─────────────────────────
        MEMORY_OPS = {"Memset (Device)", "Memcpy (HtoD)"}
        self_spyre_key = None
        for col in (parsed_spyre[0].keys() if parsed_spyre else []):
            if col.startswith("Self") and _is_device_col(col) and "%" not in col:
                self_spyre_key = col
                break

        spyre_excl_mem_us = 0.0
        spyre_memset_us = 0.0
        if self_spyre_key and parsed_spyre:
            for row in parsed_spyre:
                name = row.get("Name", row.get(list(row.keys())[0], ""))
                t = parse_time_us(row.get(self_spyre_key, "0"))
                if name not in MEMORY_OPS:
                    spyre_excl_mem_us += t
                if name == "Memset (Device)":
                    spyre_memset_us += t

        # ── Collect summary row ───────────────────────────────────────────
        all_totals = {**totals_cpu, **totals_spyre}
        summary_row = {
            "vector_size": vec_size,
            "config": cfg_lbl,
            "block_size": block_size,
            "num_cores": num_cores,
            "sencores": SENCORES,
        }
        for k, v in all_totals.items():
            summary_row[k] = v
            summary_row[f"{k} (us)"] = parse_time_us(v)
        summary_row["Self SPYRE time excl. memory (us)"] = spyre_excl_mem_us
        summary_row["Self SPYRE Memset Device (us)"] = spyre_memset_us
        sweep_summary.append(summary_row)

        # ── Top-5 by Spyre share ─────────────────────────────────────────
        spyre_pct_col = None
        spyre_time_col = None
        for col in (parsed_spyre[0].keys() if parsed_spyre else []):
            if _is_device_col(col) and "%" in col:
                spyre_pct_col = col
            if _is_device_col(col) and "total" in col.lower() and "%" not in col:
                spyre_time_col = col

        sort_col = spyre_pct_col or spyre_time_col
        sort_key = parse_percentage if spyre_pct_col else parse_time_us

        if parsed_spyre and sort_col:
            sorted_entries = sorted(
                parsed_spyre,
                key=lambda r: sort_key(r.get(sort_col, "0")),
                reverse=True,
            )
            top5 = sorted_entries[:5]
            print(f"\n{'='*60}")
            print(f"  Top 5 by Spyre share (column: {sort_col})")
            print(f"{'='*60}")
            name_col = "Name" if "Name" in top5[0] else list(top5[0].keys())[0]
            for i, row in enumerate(top5, 1):
                print(f"  {i}. {row.get(name_col, 'N/A'):40s}  {row.get(sort_col, 'N/A')}")
            print(f"{'='*60}")
        else:
            print("\nWARNING: Could not identify an AIU/Spyre time column.")

        # ── Compare with expected (computed via graph_module on CPU) ─────
        meta.graph_module(*cpu_inputs_snapshot)
        expected = cpu_inputs_snapshot[2]  # out tensor after in-place execution
        spyre_output_cpu = spyre_output.cpu()
        if torch.allclose(spyre_output_cpu, expected, rtol=1e-5, atol=1e-5):
            print(f"Correctness:  vec_size={vec_size}, config={cfg_lbl}: PASSED")
        else:
            max_diff = torch.max(torch.abs(spyre_output_cpu - expected)).item()
            print(f"Correctness:  vec_size={vec_size}, config={cfg_lbl}: FAILED (max diff={max_diff})")

# ── Save sweep summary ────────────────────────────────────────────────────

summary_path = os.path.join(sweep_dir, "sweep_summary.json")
with open(summary_path, "w") as f:
    json.dump(sweep_summary, f, indent=2)
print(f"\nSweep summary saved to: {summary_path}")
print("\n=== Sweep Complete ===")
