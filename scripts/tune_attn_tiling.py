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

"""Tune the KV-head coarse-tile count (tile_kv_heads) for the Spyre online-softmax
attention kernel, using torch profiling as the measurement.

For one attention shape it sweeps tile_kv_heads over the divisors of num_kv_heads,
runs the kernel under torch.profiler, and ranks candidates by TOTAL self device
(SPYRE / PrivateUse1) time — memory ops (Memcpy / Memset / restickify) INCLUDED,
because keeping the gathered page resident is precisely about avoiding on-device
round-trips, so the ranking must count them. The memory-op share is reported
separately as a diagnostic: a tiling that spills the page shows a larger share.
Every candidate is correctness-gated against a CPU reference before it can win.
The winning {"tile_kv_heads": N} is written to the shape-keyed JSON consumed by
SpyreAttentionImpl (see spyre_attn._get_attn_tile_config), and a timestamped
per-run JSON with all candidates + printed tables is saved for inspection.

Run on the Spyre pod:
    cd /home/ngl/helion-experiments/spyre-inference \\
        && /opt/spyre-inference/bin/python scripts/tune_attn_tiling.py \\
        --head-size 128 --num-query-heads 32 --num-kv-heads 8 --block-size 128 \\
        --context-loop-iterations 4
"""

import argparse
import contextlib
import json
import os
import sys
from datetime import datetime

import torch
from torch.profiler import ProfilerActivity, profile

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.utils.torch_utils import set_random_seed

from spyre_inference.custom_ops.utils import convert

from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyreAttentionMetadataBuilder,
    SpyrePagedKVCache,
    slot_major_kv_layout,
    _TILE_CONFIG_DIR,
    _attn_tile_config_filename,
)


def _to_cache_device(cache_cpu, device):
    """Move a KV cache to device, pinning the slot-major layout on Spyre exactly
    as the model runner allocates it (#551). Without it the index_copy_ scatter
    in reshape_and_cache silently writes the wrong rows (torch-spyre#3705)."""
    if device.type != "spyre":
        return cache_cpu.to(device)
    num_blocks, block_size, num_kv_heads, head_size = cache_cpu.shape
    return cache_cpu.to(
        device,
        device_layout=slot_major_kv_layout(
            num_blocks * block_size, num_kv_heads, head_size, cache_cpu.dtype
        ),
    )


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _build_inputs(
    *,
    head_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    block_size: int,
    context_loop_iterations: int,
    query_len: int,
    device: str,
    seed: int = 0,
):
    """Build a single-sequence attention input + metadata + CPU reference.

    Mirrors the setup in tests/test_spyre_attn.py::_run_spyre_attn_test for one
    sequence. Populates the KV cache directly (no reshape_and_cache) so the
    profiled forward measures the online-softmax path.

    context_loop_iterations is the online-softmax loop trip count (number of KV
    blocks the kernel iterates). The KV length is derived as
    context_loop_iterations * block_size (full blocks), so the caller names the
    loop count directly instead of reasoning about ceil(kv_len / block_size).
    """
    from vllm.config import get_current_vllm_config

    dtype = torch.float16
    cache_num_blocks = 256  # physical page pool, unrelated to the loop count
    set_random_seed(seed)
    torch.set_default_device("cpu")

    # Full-blocks workload: the loop runs exactly context_loop_iterations times.
    kv_len = context_loop_iterations * block_size

    scale = head_size**-0.5
    num_queries_per_kv = num_query_heads // num_kv_heads

    query = torch.randn(query_len, num_query_heads, head_size, dtype=dtype)
    key = torch.randn(query_len, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(query_len, num_kv_heads, head_size, dtype=dtype)
    k_pages_cpu = torch.zeros(cache_num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    v_pages_cpu = torch.zeros(cache_num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)

    max_num_blocks_per_seq = (kv_len + block_size - 1) // block_size
    block_table = torch.randint(
        0, cache_num_blocks, (1, max_num_blocks_per_seq), dtype=torch.int32
    )

    historical_len = kv_len - query_len
    # Historical context: pre-filled directly into the pages.
    for token_idx in range(historical_len):
        blk = block_table[0, token_idx // block_size].item()
        off = token_idx % block_size
        k_pages_cpu[blk][off] = torch.randn(num_kv_heads, head_size, dtype=dtype)
        v_pages_cpu[blk][off] = torch.randn(num_kv_heads, head_size, dtype=dtype)
    # Query-token KV: written into pages here (for the reference) and passed to
    # forward() as key/value so reshape_and_cache writes the same slots.
    slot_mapping = []
    for token_idx in range(historical_len, kv_len):
        blk = block_table[0, token_idx // block_size].item()
        off = token_idx % block_size
        k_pages_cpu[blk][off] = key[token_idx - historical_len]
        v_pages_cpu[blk][off] = value[token_idx - historical_len]
        slot_mapping.append(blk * block_size + off)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)

    seq_lens = torch.tensor([kv_len], dtype=torch.int32)
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)

    vllm_config = get_current_vllm_config()
    from unittest.mock import Mock

    vllm_config.model_config.get_num_attention_heads = Mock(return_value=num_query_heads)
    vllm_config.model_config.get_num_kv_heads = Mock(return_value=num_kv_heads)

    kv_cache_spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )
    builder = SpyreAttentionMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layers.0.self_attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        num_reqs=1,
        num_actual_tokens=query_len,
        max_query_len=query_len,
        max_seq_len=kv_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=True,
    )
    attn_metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

    cache_device = torch.device(device)
    k_pages = _to_cache_device(k_pages_cpu, cache_device)
    v_pages = _to_cache_device(v_pages_cpu, cache_device)

    ref = _ref_attn(
        query, k_pages_cpu, v_pages_cpu, query_len, kv_len, block_table, block_size, scale
    )

    return {
        "query": query,
        "key": key,
        "value": value,
        "k_pages": k_pages,
        "v_pages": v_pages,
        "attn_metadata": attn_metadata,
        "scale": scale,
        "num_queries_per_kv": num_queries_per_kv,
        "cache_device": cache_device,
        "ref": ref,
    }


def _ref_attn(query, key_cache, value_cache, query_len, kv_len, block_table, block_size, scale):
    """Minimal single-sequence, full-causal reference (no alibi/soft-cap/window)."""
    block_indices = block_table.cpu().numpy()[0, : (kv_len + block_size - 1) // block_size]
    k = torch.cat([key_cache[i] for i in block_indices], dim=0)[:kv_len]
    v = torch.cat([value_cache[i] for i in block_indices], dim=0)[:kv_len]
    q = query * scale
    if q.shape[1] != k.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k = torch.repeat_interleave(k, rep, dim=1)
        v = torch.repeat_interleave(v, rep, dim=1)
    attn = torch.einsum("qhd,khd->hqk", q, k).float()
    mask = torch.triu(torch.ones(query_len, kv_len), diagonal=kv_len - query_len + 1).bool()
    attn.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1).to(v.dtype)
    return torch.einsum("hqk,khd->qhd", attn, v)


@torch.inference_mode()
def _run_once(inputs, tile_kv_heads, head_size, num_query_heads, num_kv_heads):
    impl = SpyreAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=inputs["scale"],
        num_kv_heads=num_kv_heads,
        tile_kv_heads=tile_kv_heads,
    )
    output = torch.empty_like(inputs["query"]).to(inputs["cache_device"])
    kv_cache = SpyrePagedKVCache(k_pages=inputs["k_pages"], v_pages=inputs["v_pages"])
    device = inputs["cache_device"]
    # Post-#551 reshape_and_cache scatters via on-device index_copy_, so q/k/v
    # must be on the pages' device (asserted in _reshape_and_cache).
    impl.forward(
        layer=None,
        query=convert(inputs["query"], device),
        key=convert(inputs["key"], device),
        value=convert(inputs["value"], device),
        kv_cache=kv_cache,
        attn_metadata=inputs["attn_metadata"],
        output=output,
    )
    return output


def _assert_device_profiler_active(prof) -> None:
    """Abort unless the probe profile carries real Spyre device events.

    The AIUPTI backend is compiled into torch-spyre only when built with
    USE_SPYRE_PROFILER=1 (spyre-inference pins it to "0" by default); without it
    the PrivateUse1 slot is a silent no-op — the trace has CPU rows but zero
    device events, so self_device_time_total is 0 everywhere and ranking would be
    on noise. Detect that here rather than emit a meaningless config.
    See docs/user_guide/kineto_profiling.md sec 1.1/1.3/1.4.
    """
    total = sum(getattr(ka, "self_device_time_total", 0.0) for ka in prof.key_averages())
    if total <= 0.0:
        raise SystemExit(
            "No Spyre device events in the probe profile (self_device_time_total==0 "
            "for every op). The AIUPTI profiler backend is not active, so ranking "
            "would be on noise.\n"
            "Rebuild torch-spyre with USE_SPYRE_PROFILER=1 (see "
            "docs/user_guide/kineto_profiling.md sec 1.3), verify with "
            "`ldd <torch_spyre/_C.so> | grep libaiupti`, or pass "
            "--allow-empty-device-profile to override (not recommended)."
        )


def _device_times_us(prof) -> tuple[float, float]:
    """Return (total_self_device_us, memory_op_self_device_us).

    Total includes memory ops on purpose: keeping the gathered page resident is
    exactly about avoiding on-device Memcpy/restickify round-trips, so the
    ranking metric must count them. The memory-op sum is returned separately as
    a diagnostic — a tiling that spills the page shows a larger memory share.
    Memory ops matched by substring (restickify, memcpy, memset) so device-side
    relayout copies count even when the exact op name varies.
    """
    total = 0.0
    mem = 0.0
    for ka in prof.key_averages():
        t = getattr(ka, "self_device_time_total", 0.0)
        total += t
        name = ka.key.lower()
        if any(m in name for m in ("memcpy", "memset", "restickify", "stickif")):
            mem += t
    return total, mem


@contextlib.contextmanager
def _spyre_vllm_config():
    """Establish a Spyre vLLM config context for standalone (non-pytest) use.

    Mirrors the default_vllm_config pytest fixture in
    tests/plugin/spyre_testing_plugin/pytest_plugin.py so that
    get_current_vllm_config() (called from SpyreAttentionMetadataBuilder /
    _build_inputs) resolves outside of pytest.
    """
    from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig
    from vllm.forward_context import set_forward_context
    from vllm.platforms import PlatformEnum, current_platform
    from spyre_inference.custom_ops import register_all

    current_platform._enum = PlatformEnum.OOT
    register_all()
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
        model_config=ModelConfig(dtype=torch.float16),
    )
    with set_current_vllm_config(config), set_forward_context(None, config):
        yield


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--num-query-heads", type=int, default=32)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument(
        "--context-loop-iterations",
        type=int,
        default=4,
        help="online-softmax loop trip count = number of KV blocks the kernel "
        "iterates; KV length is context_loop_iterations * block_size. Must be "
        ">= 2 so the accumulation path (i>0) is exercised (default 4).",
    )
    ap.add_argument("--query-len", type=int, default=1, help="1 = decode")
    ap.add_argument("--device", default="spyre")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="profiled forward() calls per candidate; the device-time metric is "
        "averaged over them to reduce jitter (default 10)",
    )
    ap.add_argument("--atol", type=float, default=0.3)
    ap.add_argument("--rtol", type=float, default=0.2)
    ap.add_argument(
        "--candidates",
        type=str,
        default="",
        help="comma-separated tile_kv_heads values; default = all divisors of num_kv_heads",
    )
    ap.add_argument(
        "--allow-empty-device-profile",
        action="store_true",
        help="skip the AIUPTI device-profiler startup check (ranks on noise if the "
        "backend is inactive; not recommended)",
    )
    args = ap.parse_args()

    # Establish the vLLM config context for the whole run. Entered manually (not
    # a `with` block) to avoid re-indenting the body; the script ends with
    # os._exit(0), which bypasses context cleanup anyway.
    _cfg_ctx = _spyre_vllm_config()
    _cfg_ctx.__enter__()

    candidates = (
        [int(x) for x in args.candidates.split(",") if x]
        if args.candidates
        else _divisors(args.num_kv_heads)
    )

    inputs = _build_inputs(
        head_size=args.head_size,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        block_size=args.block_size,
        context_loop_iterations=args.context_loop_iterations,
        query_len=args.query_len,
        device=args.device,
    )
    ref = inputs["ref"]

    # Startup guard: confirm the AIUPTI device profiler is active before ranking.
    # Probe with tile_kv_heads=1 (the no-op baseline) after a warmup so compile
    # events don't pollute the check.
    if not args.allow_empty_device_profile:
        _run_once(inputs, 1, args.head_size, args.num_query_heads, args.num_kv_heads)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        ) as probe:
            _run_once(inputs, 1, args.head_size, args.num_query_heads, args.num_kv_heads)
        _assert_device_profiler_active(probe)

    results = []
    for tile_kv_heads in candidates:
        if args.num_kv_heads % tile_kv_heads != 0:
            print(f"skip tile_kv_heads={tile_kv_heads} (does not divide num_kv_heads)")
            continue

        # Correctness gate.
        out = _run_once(inputs, tile_kv_heads, args.head_size, args.num_query_heads, args.num_kv_heads)
        max_diff = (out.to("cpu") - ref).abs().max().item()
        correct = torch.allclose(out.to("cpu"), ref, atol=args.atol, rtol=args.rtol)
        if not correct:
            print(f"tile_kv_heads={tile_kv_heads}: INCORRECT (max_diff={max_diff:.4g}); excluded")
            results.append({"tile_kv_heads": tile_kv_heads, "correct": False, "max_diff": max_diff})
            continue

        # Warmup, then profile args.iterations forwards and average (the
        # profiler accumulates events across all calls in the block, so the
        # summed device time is divided by the iteration count).
        for _ in range(args.warmup):
            _run_once(inputs, tile_kv_heads, args.head_size, args.num_query_heads, args.num_kv_heads)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            record_shapes=True,
        ) as prof:
            for _ in range(args.iterations):
                _run_once(
                    inputs, tile_kv_heads, args.head_size, args.num_query_heads, args.num_kv_heads
                )

        total_us, total_mem_us = _device_times_us(prof)
        metric = total_us / args.iterations
        mem_us = total_mem_us / args.iterations
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20).replace(
            "CUDA", "AIU"
        )
        mem_share = (mem_us / metric * 100.0) if metric else 0.0
        print(
            f"\ntile_kv_heads={tile_kv_heads}: device_time_total={metric:.3f}us/iter  "
            f"memory_ops={mem_us:.3f}us ({mem_share:.1f}%)  max_diff={max_diff:.4g}"
        )
        print(table)
        results.append(
            {
                "tile_kv_heads": tile_kv_heads,
                "correct": True,
                "max_diff": max_diff,
                "device_time_total_us": metric,
                "device_time_memory_us": mem_us,
                "table": table,
            }
        )

    ranked = sorted(
        [r for r in results if r.get("correct")],
        key=lambda r: r["device_time_total_us"],
    )
    if not ranked:
        print("No correct candidate; not writing a config.")
        sys.exit(1)

    winner = ranked[0]
    print(
        f"\nWinner: tile_kv_heads={winner['tile_kv_heads']} "
        f"({winner['device_time_total_us']:.3f}us total device time, "
        f"{winner['device_time_memory_us']:.3f}us in memory ops)"
    )

    os.makedirs(_TILE_CONFIG_DIR, exist_ok=True)
    fname = _attn_tile_config_filename(
        args.head_size,
        args.num_kv_heads,
        args.num_query_heads // args.num_kv_heads,
        args.block_size,
    )
    cfg_path = os.path.join(_TILE_CONFIG_DIR, fname)
    with open(cfg_path, "w") as f:
        json.dump({"tile_kv_heads": winner["tile_kv_heads"]}, f, indent=2)
    print(f"Wrote {cfg_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(__file__), "..", "tuning_runs")
    os.makedirs(run_dir, exist_ok=True)
    run_path = os.path.join(
        run_dir, f"tune_attn_{fname.replace('.json', '')}_{ts}.json"
    )
    with open(run_path, "w") as f:
        json.dump(
            {
                "shape": {
                    "head_size": args.head_size,
                    "num_query_heads": args.num_query_heads,
                    "num_kv_heads": args.num_kv_heads,
                    "block_size": args.block_size,
                    "context_loop_iterations": args.context_loop_iterations,
                    "query_len": args.query_len,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                },
                "winner": winner["tile_kv_heads"],
                "candidates": results,
            },
            f,
            indent=2,
        )
    print(f"Wrote {run_path}")

    os._exit(0)  # avoid TimestampCalibrator abort at teardown (see profile example)


if __name__ == "__main__":
    main()
