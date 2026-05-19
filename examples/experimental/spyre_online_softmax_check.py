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
Experimental example for Spyre Inference project.
This code is for demonstration and testing purposes.

Fused Tiled Attention with Online Softmax — Spyre Operation Support Test

This test verifies which operations required by the online softmax algorithm
are supported on Spyre device. The implementation uses FlashAttention-style
fused tiled attention that combines:
    1. QK^T (query-key multiplication)
    2. Online softmax (numerically stable, tiled)
    3. PV (probability-value multiplication)

Online Softmax Algorithm (per query tile):
    Initialize: output = 0, max = -inf, sum = 0
    For each KV tile:
        1. scores = Q_tile @ K_tile^T
        2. tile_max = max(scores)
        3. new_max = max(max, tile_max)        ← needs element-wise maximum
        4. Rescale: output *= exp(max - new_max), sum *= exp(max - new_max)
        5. Update: max = new_max
        6. tile_probs = exp(scores - max)
        7. Accumulate: output += tile_probs @ V_tile, sum += sum(tile_probs)
    Normalize: output = output / sum

OPERATION DETECTION
    The detection probe runs torch.maximum() on Spyre once and inspects the
    error. If the probe surfaces an InductorError ("UnimplementedOp"), the
    hybrid path runs torch.maximum on CPU via host round-trips.

    Manual override via environment variable (optional):
        SPYRE_ONLINE_SOFTMAX_ALLOW_CPU_FALLBACK=0/1

KNOWN SPYRE BUGS THIS SCRIPT WORKS AROUND
    1. torch.maximum() — recognized by the Spyre compiler but fails inductor
       codegen with 'UnimplementedOp'. Hybrid path performs the op on CPU.
    2. Seq-dim narrowed-view reads — slicing a Spyre tensor along the seq
       dimension at a non-zero offset (e.g. q[:, 32:64, :]) produces wrong
       data when consumed by bmm. .clone() and .contiguous() on the device
       do NOT repair it — they faithfully copy the corrupted source. The
       only known fix is to slice on host, then transfer (see _seq_slice).
       Symmetric to the destination-side bug already worked around in
       torch_spyre/ops/eager.py:_is_narrowed_view (spyre__copy_from).
"""

import os

if os.environ.get("SPYRE_DEBUG"):
    os.environ.setdefault("SPYRE_INDUCTOR_LOG", "1")
    os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "DEBUG")
    os.environ.setdefault("TORCH_SENDNN_LOG", "DEBUG")

import traceback

import torch
from torch.spyre import SpyreTensorLayout, get_device_dtype  # type: ignore[import-not-found]


DEVICE = torch.device("spyre")

# ---------------------------------------------------------------------------
# Operation detection
# ---------------------------------------------------------------------------

print("=" * 60)
print("Online Softmax — Spyre Operation Support Test")
print("=" * 60)
print("\nDetecting Spyre operation support...")


def _probe_torch_maximum() -> bool:
    """Return True iff torch.maximum() works on Spyre."""
    a = torch.tensor([[1.0, 2.0]], dtype=torch.float16, device=DEVICE)
    b = torch.tensor([[1.5, 1.0]], dtype=torch.float16, device=DEVICE)
    try:
        torch.maximum(a, b).cpu()
        print("  ✅ torch.maximum() — Supported")
        return True
    except Exception as e:
        print(f"  torch.maximum() — Not supported: {type(e).__name__}")
        return False


MAXIMUM_SUPPORTED = _probe_torch_maximum()

ALLOW_CPU_FALLBACK = not MAXIMUM_SUPPORTED
override = os.environ.get("SPYRE_ONLINE_SOFTMAX_ALLOW_CPU_FALLBACK")
if override is not None:
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}
    val = override.strip().lower()
    if val in truthy:
        ALLOW_CPU_FALLBACK = True
    elif val in falsy:
        ALLOW_CPU_FALLBACK = False
    else:
        raise ValueError(
            f"SPYRE_ONLINE_SOFTMAX_ALLOW_CPU_FALLBACK={override!r}; "
            f"expected one of {sorted(truthy | falsy)}"
        )
    print(f"   (Overridden by env: ALLOW_CPU_FALLBACK={ALLOW_CPU_FALLBACK})")

print(
    f"\nMode: "
    f"{'Hybrid (CPU fallback for torch.maximum)' if ALLOW_CPU_FALLBACK else 'Native Spyre only'}"
)
print("=" * 60)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_HEADS = 8
HEAD_SIZE = 128
SEQ_LEN_Q = 64
SEQ_LEN_KV = 256

Q_TILE_SIZE = 32
KV_TILE_SIZE = 64

SCALE = 1.0 / (HEAD_SIZE**0.5)

# fp16 tolerance for end-to-end attention vs fp32 reference
FP16_TOL = 1e-2

# ---------------------------------------------------------------------------
# Spyre workarounds
# ---------------------------------------------------------------------------


def _seq_slice(t: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Slice a tensor along dim 1 (sequence dim).

    Workaround for a Spyre seq-dim narrowed-view read bug. See module
    docstring "Known Spyre bugs". On Spyre we slice on host, then transfer.
    """
    if t.device.type == "spyre":
        return t.cpu()[:, lo:hi, :].contiguous().to(t.device)
    return t[:, lo:hi, :]


def _elemwise_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch.maximum with optional CPU fallback.

    The hybrid path round-trips a/b through CPU because the Spyre inductor
    lowering for torch.maximum fails codegen with 'UnimplementedOp'.
    """
    if ALLOW_CPU_FALLBACK and a.device.type != "cpu":
        return torch.maximum(a.cpu(), b.cpu()).to(a.device)
    return torch.maximum(a, b)


# ---------------------------------------------------------------------------
# Fused Tiled Attention
# ---------------------------------------------------------------------------


def fused_tiled_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
) -> torch.Tensor:
    """FlashAttention-style fused tiled attention with online softmax.

    Single implementation; `_elemwise_max` and `_seq_slice` adapt to the
    target device. KV tiles are sliced once and reused across Q tiles to
    avoid redundant host round-trips on Spyre.
    """
    num_heads, seq_len_q, head_size = query.shape
    _, seq_len_kv, _ = key.shape

    num_q_tiles = (seq_len_q + q_tile_size - 1) // q_tile_size
    num_kv_tiles = (seq_len_kv + kv_tile_size - 1) // kv_tile_size

    # Pre-slice KV tiles once: independent of the Q-tile loop, so on Spyre
    # this saves (num_q_tiles - 1) host round-trips per KV tile per tensor.
    kv_tiles = []
    for kv_idx in range(num_kv_tiles):
        kv_start = kv_idx * kv_tile_size
        kv_end = min(kv_start + kv_tile_size, seq_len_kv)
        kv_tiles.append(
            (
                _seq_slice(key, kv_start, kv_end),
                _seq_slice(value, kv_start, kv_end),
            )
        )

    output = torch.zeros_like(query)

    for q_idx in range(num_q_tiles):
        q_start = q_idx * q_tile_size
        q_end = min(q_start + q_tile_size, seq_len_q)
        q_tile = _seq_slice(query, q_start, q_end)
        q_tile_len = q_end - q_start

        tile_max = None  # initialized on the first KV iteration
        tile_sum = torch.zeros(
            (num_heads, q_tile_len, 1),
            dtype=query.dtype,
            device=query.device,
        )
        tile_output = torch.zeros(
            (num_heads, q_tile_len, head_size),
            dtype=query.dtype,
            device=query.device,
        )

        for kv_idx, (k_tile, v_tile) in enumerate(kv_tiles):
            scores = torch.bmm(q_tile, k_tile.transpose(-2, -1)) * scale
            scores_max = scores.max(dim=-1, keepdim=True)[0]

            if kv_idx == 0:
                # First tile: skip the rescale step. Avoids -inf arithmetic
                # and one set of multiplies on tile_output / tile_sum.
                tile_max = scores_max
                tile_probs = torch.exp(scores - tile_max)
                tile_output = torch.bmm(tile_probs, v_tile)
                tile_sum = tile_probs.sum(dim=-1, keepdim=True)
            else:
                old_max = tile_max
                tile_max = _elemwise_max(tile_max, scores_max)
                rescale = torch.exp(old_max - tile_max)
                tile_output = tile_output * rescale
                tile_sum = tile_sum * rescale
                tile_probs = torch.exp(scores - tile_max)
                tile_output = tile_output + torch.bmm(tile_probs, v_tile)
                tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)

        output[:, q_start:q_end, :] = tile_output / tile_sum

    return output


def standard_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference attention that materializes the full attention matrix."""
    scores = torch.bmm(query, key.transpose(-2, -1)) * scale
    probs = torch.nn.functional.softmax(scores, dim=-1)
    return torch.bmm(probs, value)


# ---------------------------------------------------------------------------
# CPU validation
# ---------------------------------------------------------------------------

print("\nCreating test inputs:")
print(f"  Query: [{NUM_HEADS}, {SEQ_LEN_Q}, {HEAD_SIZE}]")
print(f"  Key:   [{NUM_HEADS}, {SEQ_LEN_KV}, {HEAD_SIZE}]")
print(f"  Value: [{NUM_HEADS}, {SEQ_LEN_KV}, {HEAD_SIZE}]")

torch.manual_seed(42)
query_cpu = torch.randn(NUM_HEADS, SEQ_LEN_Q, HEAD_SIZE, dtype=torch.float32)
key_cpu = torch.randn(NUM_HEADS, SEQ_LEN_KV, HEAD_SIZE, dtype=torch.float32)
value_cpu = torch.randn(NUM_HEADS, SEQ_LEN_KV, HEAD_SIZE, dtype=torch.float32)

print("\nComputing reference attention (standard unfused, CPU)...")
reference_output = standard_attention(query_cpu, key_cpu, value_cpu, SCALE)

print(f"Computing fused tiled attention (Q_tile={Q_TILE_SIZE}, KV_tile={KV_TILE_SIZE}, CPU)...")
fused_output_cpu = fused_tiled_attention(
    query_cpu,
    key_cpu,
    value_cpu,
    SCALE,
    Q_TILE_SIZE,
    KV_TILE_SIZE,
)

max_diff_cpu = torch.max(torch.abs(reference_output - fused_output_cpu)).item()
mean_diff_cpu = torch.mean(torch.abs(reference_output - fused_output_cpu)).item()
print("\nCPU Validation:")
print(f"  Max difference:  {max_diff_cpu:.2e}")
print(f"  Mean difference: {mean_diff_cpu:.2e}")
print(
    "  Matches reference!"
    if max_diff_cpu < 1e-5
    else f"  Differs from reference (max_diff={max_diff_cpu:.2e})"
)

# ---------------------------------------------------------------------------
# Spyre run
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Testing on Spyre Device")
print("=" * 60)

query_fp16 = query_cpu.to(dtype=torch.float16)
key_fp16 = key_cpu.to(dtype=torch.float16)
value_fp16 = value_cpu.to(dtype=torch.float16)


def create_qkv_layout(num_heads: int, seq_len: int, head_size: int):
    """Stickified layout on the head_size dimension.

    `dim_map` is only passed when the installed SpyreTensorLayout accepts
    it. SpyreTensorLayout is a pybind11 class so `inspect.signature` can't
    introspect it; instead we capability-probe by trying with dim_map and
    falling back on TypeError.
    """
    common = dict(
        device_size=[num_heads, seq_len, head_size // 64, 64],
        stride_map=[seq_len * head_size, head_size, 64, 1],
        device_dtype=get_device_dtype(torch.float16),
    )
    try:
        return SpyreTensorLayout(dim_map=[0, 1, 2, 2], **common)
    except TypeError:
        return SpyreTensorLayout(**common)


print("\nCreating Spyre tensor layouts and transferring inputs...")
query_layout = create_qkv_layout(NUM_HEADS, SEQ_LEN_Q, HEAD_SIZE)
key_layout = create_qkv_layout(NUM_HEADS, SEQ_LEN_KV, HEAD_SIZE)
value_layout = create_qkv_layout(NUM_HEADS, SEQ_LEN_KV, HEAD_SIZE)

query_spyre = query_fp16.to(DEVICE, device_layout=query_layout)
key_spyre = key_fp16.to(DEVICE, device_layout=key_layout)
value_spyre = value_fp16.to(DEVICE, device_layout=value_layout)

print(f"  Query: {query_spyre.shape}, device={query_spyre.device}")
print(f"  Key:   {key_spyre.shape}, device={key_spyre.device}")
print(f"  Value: {value_spyre.shape}, device={value_spyre.device}")

print(
    f"\nComputing fused tiled attention on Spyre "
    f"({'CPU fallback' if ALLOW_CPU_FALLBACK else 'native Spyre'} for torch.maximum)..."
)
try:
    fused_output_spyre = fused_tiled_attention(
        query_spyre,
        key_spyre,
        value_spyre,
        SCALE,
        Q_TILE_SIZE,
        KV_TILE_SIZE,
    )
    fused_output_spyre_cpu = fused_output_spyre.cpu().to(dtype=torch.float32)

    max_diff_spyre = torch.max(torch.abs(reference_output - fused_output_spyre_cpu)).item()
    mean_diff_spyre = torch.mean(torch.abs(reference_output - fused_output_spyre_cpu)).item()
    print("\nSpyre Validation:")
    print(f"  Max difference:  {max_diff_spyre:.2e}")
    print(f"  Mean difference: {mean_diff_spyre:.2e}")
    if max_diff_spyre < FP16_TOL:
        print(f"  Matches reference within fp16 tolerance ({FP16_TOL:.0e})")
    else:
        print(f"  Differs from reference (max_diff={max_diff_spyre:.2e})")
        print(f"    Reference sample: {reference_output[0, 0, :5]}")
        print(f"    Spyre sample:     {fused_output_spyre_cpu[0, 0, :5]}")
except Exception:
    print("  Error during Spyre computation:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
print("Spyre operation support summary:")
print("  Supported: bmm, mul, add, sub, div, exp, sum(dim=-1), max(dim=-1)")
print("  Blocker:   torch.maximum (UnimplementedOp at codegen)")
print("  Blocker:   seq-dim narrowed-view reads on Spyre (bmm yields garbage)")
print("                workaround: slice on host then transfer (_seq_slice)")
