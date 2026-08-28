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

"""Tune the head-axis coarse-tile counts (tile_kv_heads, tile_q_heads) for the
Spyre online-softmax attention kernel by torch profiling.

Sweeps tile_kv_heads for one attention shape and ranks candidates by total self
device time (memory ops included, since tiling trades page IO for on-chip
transients). Each candidate is correctness-gated against the untiled device
baseline; the winner is written to the shape-keyed JSON that SpyreAttentionImpl
loads (spyre_attn._get_attn_tile_config).

tile_kv_heads only takes effect at padded_query_len >= KV_HEAD_TILE_THRESHOLD, so
sweep with --query-len in the gated window.

    python scripts/tune_attn_tiling.py --query-len 1024 --candidates 1,2,4
"""

import argparse
import contextlib
import json
import logging
import os
import re
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
    """Move a KV cache to device, pinning the slot-major layout on Spyre as the
    model runner does (#551); without it reshape_and_cache scatters wrong rows
    (torch-spyre#3705)."""
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
    batch_size: int = 1,
    seed: int = 0,
):
    """Build a batch_size-sequence attention input + metadata + CPU reference.

    All sequences share query_len/kv_len so the compiled kernel is reused across
    the per-seq forward() loop. The KV cache is populated directly (no
    reshape_and_cache) so the profiled forward measures the online-softmax path.
    context_loop_iterations is the KV-block loop trip count; kv_len =
    context_loop_iterations * block_size.
    """
    from vllm.config import get_current_vllm_config

    dtype = torch.float16
    cache_num_blocks = 256  # physical page pool, unrelated to the loop count
    set_random_seed(seed)
    torch.set_default_device("cpu")

    kv_len = context_loop_iterations * block_size

    scale = head_size**-0.5
    num_queries_per_kv = num_query_heads // num_kv_heads

    k_pages_cpu = torch.zeros(cache_num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    v_pages_cpu = torch.zeros(cache_num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    max_num_blocks_per_seq = (kv_len + block_size - 1) // block_size
    historical_len = kv_len - query_len

    queries: list[torch.Tensor] = []
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    block_tables_rows: list[torch.Tensor] = []
    slot_mapping: list[int] = []
    refs: list[torch.Tensor] = []

    for _s in range(batch_size):
        q_s = torch.randn(query_len, num_query_heads, head_size, dtype=dtype)
        k_s = torch.randn(query_len, num_kv_heads, head_size, dtype=dtype)
        v_s = torch.randn(query_len, num_kv_heads, head_size, dtype=dtype)
        block_table_s = torch.randint(
            0, cache_num_blocks, (1, max_num_blocks_per_seq), dtype=torch.int32
        )
        # Historical context: pre-filled directly into the pages.
        for token_idx in range(historical_len):
            blk = block_table_s[0, token_idx // block_size].item()
            off = token_idx % block_size
            k_pages_cpu[blk][off] = torch.randn(num_kv_heads, head_size, dtype=dtype)
            v_pages_cpu[blk][off] = torch.randn(num_kv_heads, head_size, dtype=dtype)
        # Query-token KV: written into pages (for the ref) and passed to forward().
        for token_idx in range(historical_len, kv_len):
            blk = block_table_s[0, token_idx // block_size].item()
            off = token_idx % block_size
            k_pages_cpu[blk][off] = k_s[token_idx - historical_len]
            v_pages_cpu[blk][off] = v_s[token_idx - historical_len]
            slot_mapping.append(blk * block_size + off)
        refs.append(
            _ref_attn(
                q_s, k_pages_cpu, v_pages_cpu, query_len, kv_len, block_table_s, block_size, scale
            )
        )
        queries.append(q_s)
        keys.append(k_s)
        values.append(v_s)
        block_tables_rows.append(block_table_s[0])

    query = torch.cat(queries, dim=0)
    key = torch.cat(keys, dim=0)
    value = torch.cat(values, dim=0)
    block_table = torch.stack(block_tables_rows, dim=0)  # [batch_size, blocks_per_seq]
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)
    ref = torch.cat(refs, dim=0)  # [batch_size*query_len, num_heads, head_size]

    seq_lens = torch.tensor([kv_len] * batch_size, dtype=torch.int32)
    query_start_loc = torch.arange(0, (batch_size + 1) * query_len, query_len, dtype=torch.int32)

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
        num_reqs=batch_size,
        num_actual_tokens=batch_size * query_len,
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

    # Convert q/k/v to the pages' device ONCE here so the profiled forward()
    # excludes the per-call H2D transfer (which would otherwise dominate).
    query_dev = convert(query, cache_device)
    key_dev = convert(key, cache_device)
    value_dev = convert(value, cache_device)

    return {
        "query": query_dev,
        "key": key_dev,
        "value": value_dev,
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


def _make_impl(inputs, tile_kv_heads, tile_q_heads, head_size, num_query_heads, num_kv_heads):
    """Build one SpyreAttentionImpl for a candidate, reused across warmup and
    profiled iterations so compile cost stays out of the ranked metric."""
    return SpyreAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=inputs["scale"],
        num_kv_heads=num_kv_heads,
        tile_kv_heads=tile_kv_heads,
        tile_q_heads=tile_q_heads,
    )


@torch.inference_mode()
def _run_once(impl, inputs):
    output = torch.empty_like(inputs["query"]).to(inputs["cache_device"])
    kv_cache = SpyrePagedKVCache(k_pages=inputs["k_pages"], v_pages=inputs["v_pages"])
    # q/k/v are already on device; the metadata device mirrors are populated by
    # the first forward() and reused, so warmup takes them out of the profile.
    impl.forward(
        layer=None,
        query=inputs["query"],
        key=inputs["key"],
        value=inputs["value"],
        kv_cache=kv_cache,
        attn_metadata=inputs["attn_metadata"],
        output=output,
    )
    return output


@torch.inference_mode()
def _verify_loopspec(
    inputs, tile_kv_heads, tile_q_heads, head_size, num_query_heads, num_kv_heads
) -> bool:
    """Confirm the coarse-tile hint emitted a tiled loop.

    Compiles forward() under run_and_get_code and checks the generated source for
    `LoopSpec(count=sympify('N'))` per tiled head dim — the structural signal the
    hint took effect, independent of (IO-dominated) timing. A dropped hint emits
    no such LoopSpec.
    """
    import torch._dynamo
    import torch._inductor.config
    from torch._inductor.utils import run_and_get_code

    # Both compile caches must be defeated or a later candidate reuses an earlier
    # artifact and reports its tile count: the count reaches the compiler through
    # the spyre_hint side-channel, not the FX graph cache key.
    prev_disable = torch._inductor.config.force_disable_caches
    torch._inductor.config.force_disable_caches = True
    torch._dynamo.reset()
    try:
        impl = SpyreAttentionImpl(
            num_heads=num_query_heads,
            head_size=head_size,
            scale=inputs["scale"],
            num_kv_heads=num_kv_heads,
            tile_kv_heads=tile_kv_heads,
            tile_q_heads=tile_q_heads,
        )
        # Force the attention kernel to compile regardless of the vLLM config mode
        # (STOCK_TORCH_COMPILE), so run_and_get_code captures inductor source.
        impl._compile_attn = True
        output = torch.empty_like(inputs["query"]).to(inputs["cache_device"])
        kv_cache = SpyrePagedKVCache(k_pages=inputs["k_pages"], v_pages=inputs["v_pages"])
        _, source_codes = run_and_get_code(
            impl.forward,
            None,
            inputs["query"],
            inputs["key"],
            inputs["value"],
            kv_cache,
            inputs["attn_metadata"],
            output,
        )
    finally:
        torch._inductor.config.force_disable_caches = prev_disable

    src = "\n".join(source_codes)
    checks = []
    if tile_kv_heads > 1:
        checks.append(("kv_head", tile_kv_heads))
    if tile_q_heads > 1:
        checks.append(("qpk", tile_q_heads))
    all_found = "LoopSpec(" in src and len(checks) > 0
    parts = []
    for name, count in checks:
        needle = f"count=sympify('{count}')"
        hit = needle in src
        all_found &= hit
        parts.append(f"{name}÷{count}={'ok' if hit else 'MISS'}")
    print(
        f"[verify] kv_head={tile_kv_heads} qpk={tile_q_heads}: "
        f"{'FOUND' if all_found else 'MISSING'} [{', '.join(parts)}] "
        f"({len(source_codes)} source module(s))"
    )
    return all_found


class _LxPinningCapture(logging.Handler):
    """Capture the allocator's per-buffer LX-vs-HBM residency decisions.

    torch-spyre logs one line per op at DEBUG: `lx_pinning: <buf> (<op>) -> <reason>`.
    `reason == "lx"` means the buffer stays in the LX scratchpad; anything else
    means it round-trips HBM.
    """

    _RE = re.compile(r"lx_pinning:\s*(\S+)\s*\(([^)]*)\)\s*(?:->|→)\s*(.+)")

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[tuple[str, str, str]] = []

    def emit(self, record):
        m = self._RE.search(record.getMessage())
        if m:
            self.records.append((m.group(1), m.group(2).strip(), m.group(3).strip()))


@torch.inference_mode()
def _verify_residency(
    inputs, tile_kv_heads, tile_q_heads, head_size, num_query_heads, num_kv_heads
) -> bool:
    """Report which attention buffers are LX-resident vs written back to HBM.

    Compiles the tiled kernel while capturing the allocator's `lx_pinning:`
    decisions. Returns True if any softmax transient stayed LX (the win); the K/V
    page HBM round-trip is expected under the both-pages hoist and does not fail.
    """
    import torch._dynamo
    import torch._inductor.config
    from torch._inductor.utils import run_and_get_code

    # torch-spyre's logging must be initialized and the allocator logger set to
    # DEBUG BEFORE the compile, else its lazy re-config drops the first candidate's
    # records.
    try:
        from torch_spyre import logging_config as _spyre_log_cfg

        _spyre_log_cfg.initialize()
        _spyre_log_cfg.configure_python_logging()
        _spyre_log_cfg.set_log_level("spyre.inductor.scratchpad.allocator", "DEBUG")
    except Exception:
        pass

    alloc_logger = logging.getLogger("spyre.inductor.scratchpad.allocator")
    prev_level = alloc_logger.level
    alloc_logger.setLevel(logging.DEBUG)
    capture = _LxPinningCapture()
    alloc_logger.addHandler(capture)

    prev_disable = torch._inductor.config.force_disable_caches
    torch._inductor.config.force_disable_caches = True
    torch._dynamo.reset()
    try:
        impl = SpyreAttentionImpl(
            num_heads=num_query_heads,
            head_size=head_size,
            scale=inputs["scale"],
            num_kv_heads=num_kv_heads,
            tile_kv_heads=tile_kv_heads,
            tile_q_heads=tile_q_heads,
        )
        impl._compile_attn = True
        output = torch.empty_like(inputs["query"]).to(inputs["cache_device"])
        kv_cache = SpyrePagedKVCache(k_pages=inputs["k_pages"], v_pages=inputs["v_pages"])
        run_and_get_code(
            impl.forward,
            None,
            inputs["query"],
            inputs["key"],
            inputs["value"],
            kv_cache,
            inputs["attn_metadata"],
            output,
        )
    finally:
        torch._inductor.config.force_disable_caches = prev_disable
        alloc_logger.removeHandler(capture)
        alloc_logger.setLevel(prev_level)

    # The reshape_and_cache kernels also emit lx_pinning lines; the attention
    # kernel is the one carrying coarse_tile_* buffers.
    lx = [r for r in capture.records if r[2] == "lx"]
    hbm = [r for r in capture.records if r[2] != "lx"]
    # Page reads: hoisted K/V gather slices feeding the matmuls.
    page_hbm = [r for r in hbm if "read_copy" in r[0] and ("arg" in r[0] or "buf" in r[0])]
    transient_lx = [
        r for r in lx if r[1] in ("mul", "add", "sum", "div", "sub", "exp", "maximum", "amax")
    ]

    print(
        f"\n[residency] kv_head={tile_kv_heads} qpk={tile_q_heads}: "
        f"{len(lx)} buffer(s) LX-resident, {len(hbm)} in HBM"
    )
    print(
        f"  softmax transients LX-resident: {len(transient_lx)} "
        f"({'OK — on-chip' if transient_lx else 'NONE — tiling bought no residency'})"
    )
    print(
        f"  K/V page reads in HBM: {len(page_hbm)} "
        f"(expected: the both-pages hoist round-trips DRAM by design)"
    )
    if hbm:
        # Group HBM reasons for a compact diagnostic.
        from collections import Counter

        reasons = Counter(r[2] for r in hbm)
        for reason, n in reasons.most_common():
            print(f"    HBM×{n}: {reason}")
    return bool(transient_lx)


def _assert_device_profiler_active(prof) -> None:
    """Abort unless the probe profile carries real Spyre device events.

    The AIUPTI backend is compiled in only with USE_SPYRE_PROFILER=1; without it
    self_device_time_total is 0 everywhere and ranking would be on noise. See
    docs/user_guide/kineto_profiling.md.
    """
    total = sum(getattr(ka, "self_device_time_total", 0.0) for ka in prof.key_averages())
    if total <= 0.0:
        raise SystemExit(
            "No Spyre device events in the probe profile (self_device_time_total==0). "
            "The AIUPTI profiler backend is not active, so ranking would be on noise. "
            "Rebuild torch-spyre with USE_SPYRE_PROFILER=1 (see "
            "docs/user_guide/kineto_profiling.md) or pass --allow-empty-device-profile."
        )


def _device_times_us(prof) -> tuple[float, float]:
    """Return (total_self_device_us, memory_op_self_device_us).

    Total includes memory ops on purpose: tiling is about avoiding on-device
    round-trips, so the ranking metric must count them. The memory sum is a
    diagnostic. Matched by substring so relayout copies count as their name varies.
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
    """Establish a Spyre vLLM config context for standalone (non-pytest) use, so
    get_current_vllm_config() resolves outside pytest."""
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
        default=0,
        help="KV-block loop trip count; kv_len = context_loop_iterations * "
        "block_size. Must be >= 2. Default (0) derives a value so kv_len > "
        "query_len (one block of history).",
    )
    ap.add_argument(
        "--query-len",
        type=int,
        default=512,
        help="padded query length. tile_kv_heads only takes effect at >= KV_HEAD_TILE_THRESHOLD.",
    )
    ap.add_argument("--device", default="spyre")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="sequences per forward(), all sharing --query-len (default 1)",
    )
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="profiled forward() calls per candidate, averaged (default 10)",
    )
    ap.add_argument("--atol", type=float, default=0.3)
    ap.add_argument("--rtol", type=float, default=0.2)
    ap.add_argument(
        "--candidates",
        type=str,
        default="",
        help="comma-separated tile_kv_heads values to sweep; default = all "
        "divisors of --num-kv-heads",
    )
    ap.add_argument(
        "--tile-q-heads",
        type=int,
        default=1,
        help="fixed tile_q_heads (num_queries_per_kv split) held constant across "
        "the tile_kv_heads sweep; default 1 (kv_head only)",
    )
    ap.add_argument(
        "--allow-empty-device-profile",
        action="store_true",
        help="skip the AIUPTI device-profiler startup check (ranks on noise if the "
        "backend is inactive; not recommended)",
    )
    ap.add_argument(
        "--verify-loopspec",
        action="store_true",
        help="verify each tile>1 candidate emits a LoopSpec(count=sympify('N')) in "
        "the generated source (structural check the hint took effect), then exit "
        "without profiling/ranking",
    )
    ap.add_argument(
        "--verify-residency",
        action="store_true",
        help="report which attention buffers are LX-resident vs HBM (the "
        "allocator's lx_pinning decisions), then exit",
    )
    args = ap.parse_args()

    # Default loop count adds one block of prefill history over the padded query
    # length (kv_len > query_len). An explicit value overrides this.
    if args.context_loop_iterations <= 0:
        query_blocks = (args.query_len + args.block_size - 1) // args.block_size
        args.context_loop_iterations = max(2, query_blocks + 1)
    if args.context_loop_iterations * args.block_size < args.query_len:
        raise SystemExit(
            f"--context-loop-iterations={args.context_loop_iterations} gives "
            f"kv_len={args.context_loop_iterations * args.block_size} < "
            f"query_len={args.query_len}; increase it so kv_len >= query_len"
        )

    # Enter the vLLM config context for the whole run (torn down by os._exit).
    _cfg_ctx = _spyre_vllm_config()
    _cfg_ctx.__enter__()

    num_queries_per_kv = args.num_query_heads // args.num_kv_heads
    tile_q_heads = args.tile_q_heads
    if num_queries_per_kv % tile_q_heads != 0:
        raise SystemExit(
            f"--tile-q-heads={tile_q_heads} must divide num_queries_per_kv={num_queries_per_kv}"
        )

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
        batch_size=args.batch_size,
    )
    ref = inputs["ref"]

    # Verify-only modes: structural loopspec check or residency report, then exit.
    def _tileable(tile_kv_heads: int) -> bool:
        if args.num_kv_heads % tile_kv_heads != 0:
            return False
        if tile_kv_heads > 1 and args.num_kv_heads // tile_kv_heads < 2:
            return False
        return tile_kv_heads > 1 or tile_q_heads > 1

    if args.verify_loopspec or args.verify_residency:
        verify_fn = _verify_residency if args.verify_residency else _verify_loopspec
        all_ok = True
        for tile_kv_heads in candidates:
            if not _tileable(tile_kv_heads):
                continue
            all_ok &= verify_fn(
                inputs,
                tile_kv_heads,
                tile_q_heads,
                args.head_size,
                args.num_query_heads,
                args.num_kv_heads,
            )
        # os._exit skips atexit flush, so drain stdout first.
        sys.stdout.flush()
        os._exit(0 if all_ok else 1)

    # Startup guard: confirm the AIUPTI device profiler is active before ranking.
    if not args.allow_empty_device_profile:
        probe_impl = _make_impl(
            inputs, 1, 1, args.head_size, args.num_query_heads, args.num_kv_heads
        )
        _run_once(probe_impl, inputs)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        ) as probe:
            _run_once(probe_impl, inputs)
        _assert_device_profiler_active(probe)

    results = []

    # In-family baseline: the untiled kernel output on device. A correct tile runs
    # the same fp16 device math regrouped, so it must match this tightly. The CPU
    # ref drifts with sequence length (fp16 accumulation), so it is only a coarse
    # sanity floor.
    baseline_impl = _make_impl(
        inputs, 1, 1, args.head_size, args.num_query_heads, args.num_kv_heads
    )
    baseline = _run_once(baseline_impl, inputs).to("cpu")
    ref_max_diff = (baseline - ref).abs().max().item()
    ref_outliers = int(((baseline - ref).abs() > args.atol * 2).sum().item())
    print(
        f"untiled baseline vs CPU ref: max_diff={ref_max_diff:.4g}, "
        f"outliers(>{args.atol * 2:.2g})={ref_outliers}/{baseline.numel()} "
        f"(fp16 noise; tiles are gated against this untiled baseline, not the ref)"
    )

    for tile_kv_heads in candidates:
        if args.num_kv_heads % tile_kv_heads != 0:
            print(
                f"skip tile_kv_heads={tile_kv_heads} "
                f"(does not divide num_kv_heads={args.num_kv_heads})"
            )
            continue
        # The backend clamps counts that leave a per-tile extent < 2 (see
        # _clamp_tile_count), so timing them would record a count that never runs.
        if tile_kv_heads > 1 and args.num_kv_heads // tile_kv_heads < 2:
            print(f"skip tile_kv_heads={tile_kv_heads} (per-tile extent < 2)")
            continue

        # One impl per candidate, reused across the gate, warmup, and profiled
        # iterations so compile cost stays out of the ranked metric.
        impl = _make_impl(
            inputs,
            tile_kv_heads,
            tile_q_heads,
            args.head_size,
            args.num_query_heads,
            args.num_kv_heads,
        )

        # Correctness gate on the outlier FRACTION, not max_diff: a correct tile
        # differs from the untiled baseline only by fp16 regrouping noise (a few
        # outliers), while a broken tile diverges on a large fraction of elements.
        out = _run_once(impl, inputs).to("cpu")
        adiff = (out - baseline).abs()
        max_diff = adiff.max().item()
        outlier_thresh = max(1.0, args.atol * 4)
        outlier_frac = (adiff > outlier_thresh).float().mean().item()
        correct = outlier_frac <= 1e-3
        if not correct:
            print(
                f"tile_kv_heads={tile_kv_heads}: INCORRECT vs untiled baseline "
                f"(max_diff={max_diff:.4g}, {outlier_frac * 100:.3f}% of elems "
                f">{outlier_thresh:.2g} — structural divergence); excluded"
            )
            results.append({"tile_kv_heads": tile_kv_heads, "correct": False, "max_diff": max_diff})
            continue

        # Warmup, then profile args.iterations forwards and average.
        for _ in range(args.warmup):
            _run_once(impl, inputs)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            record_shapes=True,
        ) as prof:
            for _ in range(args.iterations):
                _run_once(impl, inputs)

        total_us, total_mem_us = _device_times_us(prof)
        metric = total_us / args.iterations
        mem_us = total_mem_us / args.iterations
        table = (
            prof.key_averages()
            .table(sort_by="cuda_time_total", row_limit=20)
            .replace("CUDA", "AIU")
        )
        mem_share = (mem_us / metric * 100.0) if metric else 0.0
        print(
            f"\ntile_kv_heads={tile_kv_heads} tile_q_heads={tile_q_heads}: "
            f"device_time_total={metric:.3f}us/iter  "
            f"memory_ops={mem_us:.3f}us ({mem_share:.1f}%)  max_diff={max_diff:.4g}"
        )
        print(table)
        results.append(
            {
                "tile_kv_heads": tile_kv_heads,
                "tile_q_heads": tile_q_heads,
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
        f"\nWinner: tile_kv_heads={winner['tile_kv_heads']} tile_q_heads={tile_q_heads} "
        f"({winner['device_time_total_us']:.3f}us total device time, "
        f"{winner['device_time_memory_us']:.3f}us in memory ops)"
    )

    os.makedirs(_TILE_CONFIG_DIR, exist_ok=True)
    fname = _attn_tile_config_filename(
        args.head_size,
        args.num_kv_heads,
        num_queries_per_kv,
        args.block_size,
    )
    cfg_path = os.path.join(_TILE_CONFIG_DIR, fname)
    with open(cfg_path, "w") as f:
        json.dump(
            {"tile_kv_heads": winner["tile_kv_heads"], "tile_q_heads": tile_q_heads},
            f,
            indent=2,
        )
    print(f"Wrote {cfg_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(__file__), "..", "tuning_runs")
    os.makedirs(run_dir, exist_ok=True)
    run_path = os.path.join(run_dir, f"tune_attn_{fname.replace('.json', '')}_{ts}.json")
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
                    "batch_size": args.batch_size,
                    "tile_q_heads": tile_q_heads,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                },
                "winner": {
                    "tile_kv_heads": winner["tile_kv_heads"],
                    "tile_q_heads": tile_q_heads,
                },
                "candidates": results,
            },
            f,
            indent=2,
        )
    print(f"Wrote {run_path}")

    sys.stdout.flush()
    os._exit(0)  # avoid TimestampCalibrator abort at teardown (see profile example)


if __name__ == "__main__":
    main()
