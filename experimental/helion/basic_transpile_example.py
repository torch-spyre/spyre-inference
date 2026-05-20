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
Spyre-compiled callable, validate on CPU, execute on Spyre, and compare.
"""

import torch
import helion
import helion.language as hl
import sys
import os


os.environ["SPYRE_INDUCTOR_LOG"] = "1"
os.environ["SPYRE_INDUCTOR_LOG_LEVEL"] = "DEBUG"
os.environ["TORCH_SENDNN_LOG"] = "DEBUG"
# os.environ["DT_DEEPRT_VERBOSE"] = "1"
# os.environ["DTLOG_LEVEL"] = "debug"

# Add pastamachine and torch-spyre to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../new_stack/torch-spyre'))

import pastamachine

import logging
# logging.getLogger("pastamachine").setLevel(logging.WARNING)  # suppress info
logging.getLogger("pastamachine").setLevel(logging.DEBUG)


# ── Define a Helion kernel ──────────────────────────────────────────────────

@helion.kernel()
def inplace_add(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
):
    for tile in hl.tile(out.size()):
        out[tile] += a[tile] + b[tile]

# ── Prepare Spyre inputs ───────────────────────────────────────────────────

SPYRE_DEVICE = torch.device("spyre")
# VECTOR_SIZE = 128
VECTOR_SIZE = 512
a_spyre = torch.full([VECTOR_SIZE], 1.0, device=SPYRE_DEVICE)
b_spyre = torch.full([VECTOR_SIZE], 2.0, device=SPYRE_DEVICE)
out_spyre = torch.zeros([VECTOR_SIZE], device=SPYRE_DEVICE)

# ── Transpile: Helion → FX graph → (CPU verify) → Spyre compile ───────────

# Use helion.Config to control core division: block_size=32 with
# VECTOR_SIZE=512 → 512/32 = 16 cores.
config = helion.Config(block_sizes=[32])

compiled_for_spyre, meta = pastamachine.compile_helion_to_spyre(
    inplace_add,
    (a_spyre, b_spyre, out_spyre),
    do_verify_run_on_cpu=True,
    return_meta=True,
    config=config,
)

# ── Print SDSC summary ───────────────────────────────────────────────────

print("\n--- SDSC analysis ---")
pastamachine.print_meta_summary(meta)

# ── Execute on Spyre ───────────────────────────────────────────────────────

print("\n--- Executing Spyre Compiled Function ---")
result = compiled_for_spyre(a_spyre, b_spyre, out_spyre)
print(f"Spyre execution complete")
print(f"Result type: {type(result)}")

# we know it is in-place
assert result is None
spyre_output = out_spyre

print(f"Spyre result shape: {spyre_output.shape}")
print(f"Spyre result device: {spyre_output.device}")
print(f"Spyre result dtype: {spyre_output.dtype}")
print(spyre_output.cpu())

# ── Compare with expected ──────────────────────────────────────────────────

print("\n--- Comparing Spyre vs Expected ---")
expected = torch.full([VECTOR_SIZE], 3.0, device="cpu")
spyre_output_cpu = spyre_output.cpu()

max_diff = torch.max(torch.abs(spyre_output_cpu - expected)).item()
mean_diff = torch.mean(torch.abs(spyre_output_cpu - expected)).item()
print(f"Max absolute difference:  {max_diff}")
print(f"Mean absolute difference: {mean_diff}")

if torch.allclose(spyre_output_cpu, expected, rtol=1e-5, atol=1e-5):
    print("Spyre vs Expected: PASSED (results match within tolerance)")
else:
    print("Spyre vs Expected: FAILED (results differ beyond tolerance)")
    print(f"  Spyre[:10]    = {spyre_output_cpu[:10]}")
    print(f"  Expected[:10] = {expected[:10]}")

print("\n=== Execution Complete ===")
