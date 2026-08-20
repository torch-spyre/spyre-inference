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

"""Estimate the working-set size of one iteration of the online-softmax inner
loop in SpyreAttentionImpl (_create_compilable_page_attn).

The per-iteration working set is the sum of bytes of the live tensors the
compiler must keep resident to evaluate the score/prob/accumulate chain. The
coarse-tile hint maps tiles across corelets, so the residency budget is the
per-corelet scratchpad and the working set of interest is that of ONE tile.
When a tile exceeds the per-corelet budget the compiler spills to HBM, adding
IO. Tiling the KV-head dim (tile_kv_heads = N) splits num_kv_heads into N tiles
(one per corelet, up to num_corelets), shrinking every tensor whose leading dim
is num_kv_heads by 1/N. The scratchpad size and corelet count are supplied by
the caller (Spyre: 2 MiB/corelet, 64 corelets).

This is a pure arithmetic model (no device, no torch). For each requested
tiling it reports two per-corelet numbers:
  - peak: all tensors live at once during one iteration (the scratchpad-fit
    check),
  - carry_over: only the online-softmax accumulators (tile_output, tile_max,
    tile_sum) that must survive to the next iteration, i.e. what we want to
    keep resident in scratchpad between iterations.
It also reports how many corelets the tiling occupies.

Usage:
    python scripts/attn_working_set.py \\
        --head-size 128 --num-query-heads 32 --num-kv-heads 8 \\
        --block-size 128 --padded-query-len 32 \\
        --tile-kv-heads 1,2,4,8 --scratchpad-kib 2048 --num-corelets 64
"""

import argparse
import math

_DTYPE_BYTES = 2  # float16


def _fmt(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} GiB"


def _shapes(hkv, qpk, pql, block_size, head_size):
    return {
        "S": (hkv, qpk, pql, block_size),  # scores / tile_probs
        "R": (hkv, qpk, pql, 1),  # per-row reductions (max/sum/rescale)
        "O": (hkv, qpk, pql, head_size),  # tile_output
        "KV": (hkv, 1, block_size, head_size),  # k_page_4d / v_page_4d (post-permute)
        "Q": (hkv, qpk, pql, head_size),  # q tile
        "M": (pql, block_size),  # mask_tile
        "A": (hkv, qpk, 1, block_size),  # alibi_bias_tile
    }


def carried_tensors(
    hkv: int, qpk: int, pql: int, block_size: int, head_size: int
) -> dict[str, tuple[int, ...]]:
    """State that must survive to the NEXT iteration (the online-softmax carry).

    Only the running accumulators persist across the loop: tile_output, tile_max,
    tile_sum. Everything else (q tile, k/v page, scores, probs, mask, new_max,
    rescale) is transient and freed at the iteration boundary. This is the set
    we want to keep resident in scratchpad between iterations.
    """
    s = _shapes(hkv, qpk, pql, block_size, head_size)
    return {"tile_output": s["O"], "tile_max": s["R"], "tile_sum": s["R"]}


def peak_tensors(
    hkv: int,
    qpk: int,
    pql: int,
    block_size: int,
    head_size: int,
    has_alibi: bool,
) -> dict[str, tuple[int, ...]]:
    """Tensors live simultaneously at the peak of one iteration (i > 0 branch).

    Carried state (tile_output/tile_max/tile_sum) plus the transients needed to
    produce the next accumulator values. tile_max is shown once (the rebind
    `tile_max = new_max` frees the old buffer); new_max/rescale are the extra
    live reductions at the peak. Shapes mirror _create_compilable_page_attn.
    """
    s = _shapes(hkv, qpk, pql, block_size, head_size)
    t: dict[str, tuple[int, ...]] = {
        # carried in
        "tile_output": s["O"],
        "tile_max": s["R"],
        "tile_sum": s["R"],
        # transient
        "q": s["Q"],
        "k_page_4d": s["KV"],
        "v_page_4d": s["KV"],
        "mask_tile": s["M"],
        "scores": s["S"],
        "scores_max": s["R"],
        "tile_probs": s["S"],
        "new_max": s["R"],
        "rescale": s["R"],
    }
    if has_alibi:
        t["alibi_bias_tile"] = s["A"]
    return t


def working_set_bytes(tensors: dict[str, tuple[int, ...]]) -> int:
    return sum(math.prod(shape) * _DTYPE_BYTES for shape in tensors.values())


def report(
    head_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    block_size: int,
    padded_query_len: int,
    tile_counts: list[int],
    has_alibi: bool,
    scratchpad_bytes: int | None,
    num_corelets: int,
) -> None:
    qpk = num_query_heads // num_kv_heads
    print(
        f"shape: head_size={head_size} num_query_heads={num_query_heads} "
        f"num_kv_heads={num_kv_heads} (qpk={qpk}) block_size={block_size} "
        f"padded_query_len={padded_query_len} alibi={has_alibi} dtype=fp16"
    )
    if scratchpad_bytes is not None:
        print(
            f"scratchpad budget: {_fmt(scratchpad_bytes)} per corelet "
            f"({num_corelets} corelets)"
        )
    print()

    for n in tile_counts:
        if num_kv_heads % n != 0:
            print(f"tile_kv_heads={n}: SKIP (does not divide num_kv_heads={num_kv_heads})")
            continue
        hkv = num_kv_heads // n
        # Working set of ONE tile = what one corelet must hold.
        peak = working_set_bytes(
            peak_tensors(hkv, qpk, padded_query_len, block_size, head_size, has_alibi)
        )
        carry = working_set_bytes(
            carried_tensors(hkv, qpk, padded_query_len, block_size, head_size)
        )
        line = (
            f"tile_kv_heads={n}: hkv_per_tile={hkv}  "
            f"peak={_fmt(peak)}  carry_over={_fmt(carry)}"
        )
        if scratchpad_bytes is not None:
            fits = "FITS" if peak <= scratchpad_bytes else "SPILLS"
            line += (
                f"  (peak {peak / scratchpad_bytes * 100:.1f}%, "
                f"carry {carry / scratchpad_bytes * 100:.1f}% of budget, {fits})"
            )
        line += f"  corelets_used={min(n, num_corelets)}/{num_corelets}"
        print(line)

    print("\nper-tensor breakdown (tile_kv_heads=1, peak / i>0 branch):")
    base = peak_tensors(num_kv_heads, qpk, padded_query_len, block_size, head_size, has_alibi)
    carried_names = set(carried_tensors(num_kv_heads, qpk, padded_query_len, block_size, head_size))
    for name, shape in sorted(base.items(), key=lambda kv: -math.prod(kv[1])):
        b = math.prod(shape) * _DTYPE_BYTES
        tag = "carry" if name in carried_names else "transient"
        print(f"  {name:16s} {str(shape):28s} {_fmt(b):>10s}  [{tag}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--num-query-heads", type=int, default=32)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--padded-query-len", type=int, default=32)
    ap.add_argument(
        "--tile-kv-heads",
        type=str,
        default="1,2,4,8",
        help="comma-separated tile_kv_heads values to compare",
    )
    ap.add_argument("--alibi", action="store_true", help="include ALiBi bias tile")
    ap.add_argument(
        "--scratchpad-kib",
        type=int,
        required=True,
        help="per-corelet scratchpad budget in KiB (e.g. Spyre: 2048)",
    )
    ap.add_argument(
        "--num-corelets",
        type=int,
        required=True,
        help="number of corelets tiles can map across (e.g. Spyre: 64)",
    )
    args = ap.parse_args()

    tile_counts = [int(x) for x in args.tile_kv_heads.split(",") if x]
    scratchpad_bytes = args.scratchpad_kib * 1024 if args.scratchpad_kib else None
    report(
        head_size=args.head_size,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        block_size=args.block_size,
        padded_query_len=args.padded_query_len,
        tile_counts=tile_counts,
        has_alibi=args.alibi,
        scratchpad_bytes=scratchpad_bytes,
        num_corelets=args.num_corelets,
    )


if __name__ == "__main__":
    main()
