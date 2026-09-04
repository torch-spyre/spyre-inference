#!/usr/bin/env python3
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

"""Micro-benchmark for the Spyre attention kernel via the torch profiler. See README.

    SPYRE_ATTN_PROFILING=1 .venv/bin/python3 scripts/microbench/spyre_attn_microbench.py \
        --config scripts/microbench/configs/granite33_8b_bs64.json
"""

import argparse
import contextlib
import gc
import itertools
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# _ATTN_PROFILING is read at import time in spyre_attn, so set it before import.
os.environ["SPYRE_ATTN_PROFILING"] = "1"
os.environ.setdefault("VLLM_PLUGINS", "spyre_inference")
for _k, _v in (
    ("RANK", "0"),
    ("LOCAL_RANK", "0"),
    ("WORLD_SIZE", "1"),
    ("LOCAL_WORLD_SIZE", "1"),
    ("MASTER_ADDR", "127.0.0.1"),
    ("MASTER_PORT", "29500"),
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

SPANS = {
    "online_softmax": "spyre_attn::online_softmax",
    "forward": "spyre_attn::forward",
    "reshape_and_cache": "spyre_attn::reshape_and_cache",
}
MEMORY_OP_MARKERS = ("memcpy", "memset", "restickify", "stickif")

# Spyre requires float16 (platform.py raises otherwise).
DTYPE = torch.float16


VARIANT_REGISTRY: dict[str, dict] = {}


def register_variant(name, impl_label, compiled=True, available_fn=None):
    VARIANT_REGISTRY[name] = {
        "impl_label": impl_label,
        "compiled": compiled,
        "available": available_fn or (lambda: True),
    }


register_variant("online_softmax_compiled", "Implementation.SPYRE_ONLINE_SOFTMAX", True)
register_variant("online_softmax_eager", "Implementation.SPYRE_ONLINE_SOFTMAX_EAGER", False)


@contextlib.contextmanager
def spyre_vllm_config(compiled: bool):
    """Establish a Spyre vLLM config context for standalone (non-pytest) use."""
    from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig, CompilationMode
    from vllm.forward_context import set_forward_context
    from vllm.platforms import PlatformEnum, current_platform

    from spyre_inference.custom_ops import register_all

    current_platform._enum = PlatformEnum.OOT
    register_all()
    mode = CompilationMode.STOCK_TORCH_COMPILE if compiled else CompilationMode.NONE
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"], mode=mode),
        model_config=ModelConfig(dtype=DTYPE),
    )
    with set_current_vllm_config(config), set_forward_context(None, config):
        yield


def ref_attn(query, key_cache, value_cache, query_lens, kv_lens, block_tables, block_size, scale):
    """Full-causal varlen reference, no alibi/soft-cap/window (Granite shape)."""
    block_tables_np = block_tables.cpu().numpy()
    outputs = []
    start = 0
    for i, (query_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        q = query[start : start + query_len] * scale
        idx = block_tables_np[i, : (kv_len + block_size - 1) // block_size]
        k = torch.cat([key_cache[j] for j in idx], dim=0)[:kv_len]
        v = torch.cat([value_cache[j] for j in idx], dim=0)[:kv_len]
        if q.shape[1] != k.shape[1]:
            rep = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, rep, dim=1)
            v = torch.repeat_interleave(v, rep, dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(torch.ones(query_len, kv_len), diagonal=kv_len - query_len + 1).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        outputs.append(torch.einsum("hqk,khd->qhd", attn, v))
        start += query_len
    return torch.cat(outputs, dim=0)


def build_metadata(
    num_query_heads,
    num_kv_heads,
    head_size,
    block_size,
    seq_lens,
    query_start_loc,
    block_table,
    slot_mapping,
):
    """Drive the real SpyreAttentionMetadataBuilder."""
    from unittest.mock import Mock

    from vllm.config import get_current_vllm_config
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import AttentionSpec

    from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadataBuilder

    vllm_config = get_current_vllm_config()
    vllm_config.model_config.get_num_attention_heads = Mock(return_value=num_query_heads)
    vllm_config.model_config.get_num_kv_heads = Mock(return_value=num_kv_heads)

    builder = SpyreAttentionMetadataBuilder(
        kv_cache_spec=AttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=DTYPE,
        ),
        layer_names=["layers.0.self_attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        num_reqs=len(seq_lens),
        num_actual_tokens=int(query_start_loc[-1].item()),
        max_query_len=int((query_start_loc[1:] - query_start_loc[:-1]).max().item()),
        max_seq_len=int(seq_lens.max().item()),
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=True,
    )
    return builder.build(common_prefix_len=0, common_attn_metadata=common)


def _fused_qkv_kv_views(query, key, value, device):
    """K/V as the backend receives them: strided views of a fused QKV on device."""
    from spyre_inference.custom_ops.utils import convert

    num_tokens = query.shape[0]
    slabs = [t.reshape(num_tokens, -1) for t in (query, key, value)]
    qkv = convert(torch.cat(slabs, dim=-1), device)
    _, k_view, v_view = qkv.split([s.shape[-1] for s in slabs], dim=-1)
    return (
        k_view.view(num_tokens, key.shape[1], key.shape[2]),
        v_view.view(num_tokens, value.shape[1], value.shape[2]),
    )


def build_inputs_from_requests(
    query_lens,
    seq_lens,
    num_query_heads,
    num_kv_heads,
    head_size,
    block_size,
    num_blocks,
    device,
    seed=0,
    kv_layout="plain",
):
    """Varlen multi-sequence inputs from explicit per-request lengths."""
    from vllm.utils.torch_utils import set_random_seed

    from spyre_inference.custom_ops.utils import convert
    from spyre_inference.v1.attention.backends.spyre_attn import slot_major_kv_layout

    assert len(query_lens) == len(seq_lens)
    for ql, sl in zip(query_lens, seq_lens):
        assert sl >= ql >= 1, f"require seq_len >= query_len >= 1, got {sl} < {ql}"

    torch.set_default_device("cpu")
    set_random_seed(seed)

    num_seqs = len(query_lens)
    max_kv = max(seq_lens)
    blocks_per_seq = (max_kv + block_size - 1) // block_size
    if num_blocks < blocks_per_seq:
        return None  # cache too small for this shape

    scale = head_size**-0.5
    total_q = sum(query_lens)

    query = torch.randn(total_q, num_query_heads, head_size, dtype=DTYPE)
    key = torch.randn(total_q, num_kv_heads, head_size, dtype=DTYPE)
    value = torch.randn(total_q, num_kv_heads, head_size, dtype=DTYPE)
    k_pages_cpu = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=DTYPE)
    v_pages_cpu = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=DTYPE)

    block_tables = torch.randint(0, num_blocks, (num_seqs, blocks_per_seq), dtype=torch.int32)

    slot_mapping = []
    hist_k, hist_v, hist_slots = [], [], []
    q_off = 0
    for s in range(num_seqs):
        ql, kvl = query_lens[s], seq_lens[s]
        hist = kvl - ql
        if hist > 0:
            hk = torch.randn(hist, num_kv_heads, head_size, dtype=DTYPE)
            hv = torch.randn(hist, num_kv_heads, head_size, dtype=DTYPE)
            for t in range(hist):
                blk = block_tables[s, t // block_size].item()
                k_pages_cpu[blk][t % block_size] = hk[t]
                v_pages_cpu[blk][t % block_size] = hv[t]
                hist_slots.append(blk * block_size + t % block_size)
            hist_k.append(hk)
            hist_v.append(hv)
        for t in range(hist, kvl):
            blk = block_tables[s, t // block_size].item()
            off = t % block_size
            k_pages_cpu[blk][off] = key[q_off + t - hist]
            v_pages_cpu[blk][off] = value[q_off + t - hist]
            slot_mapping.append(blk * block_size + off)
        q_off += ql
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)

    attn_metadata = build_metadata(
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor([0] + list(query_lens), dtype=torch.int32).cumsum(0, dtype=torch.int32),
        block_tables,
        slot_mapping,
    )

    cache_device = torch.device(device)

    def to_device(cache):
        # plain: host-populated cache, plain transfer. Matches
        # tests/attention/test_spyre_attn.py, correct at these shapes.
        # slot_major / slot_major_devfill: numerically WRONG, kept only to
        # reproduce the finding (see README). _reshape_and_cache views pages as
        # [-1, H, D] and relies on the slot-outermost device layout, which
        # convert() does not reproduce for a host tensor.
        if cache_device.type != "spyre" or kv_layout == "plain":
            return cache.to(cache_device)
        nb, bsz, h, d = cache.shape
        layout = slot_major_kv_layout(nb * bsz, h, d, cache.dtype)
        if kv_layout == "slot_major":
            return cache.to(cache_device, device_layout=layout)
        return torch.zeros_like(cache).to(cache_device, device_layout=layout)

    k_pages, v_pages = to_device(k_pages_cpu), to_device(v_pages_cpu)

    if kv_layout == "slot_major_devfill" and cache_device.type == "spyre" and hist_slots:
        # Same index_copy_ the kernel uses, on the same [-1, H, D] view.
        slots_dev = torch.tensor(hist_slots, dtype=torch.int64).to(cache_device)
        hk_dev = convert(torch.cat(hist_k), cache_device)
        hv_dev = convert(torch.cat(hist_v), cache_device)
        view = (-1, num_kv_heads, head_size)
        k_pages.view(view).index_copy_(0, slots_dev, hk_dev)
        v_pages.view(view).index_copy_(0, slots_dev, hv_dev)
    key_dev, value_dev = _fused_qkv_kv_views(query, key, value, cache_device)

    return {
        "query_dev": convert(query, cache_device),
        "key_dev": key_dev,
        "value_dev": value_dev,
        "k_pages": k_pages,
        "v_pages": v_pages,
        "k_pages_cpu": k_pages_cpu,
        "v_pages_cpu": v_pages_cpu,
        "query_cpu": query,
        "block_tables": block_tables,
        "attn_metadata": attn_metadata,
        "scale": scale,
        "cache_device": cache_device,
        "query_lens": list(query_lens),
        "seq_lens": list(seq_lens),
        "total_query_tokens": total_q,
    }


def grid_to_requests(
    batch_size, seqlen, decode_share, partial_prefill_share, prompt_pattern, block_size
):
    """Lower a Cartesian grid point onto explicit (query_lens, seq_lens)."""
    cycle = itertools.cycle(prompt_pattern)
    init_lens = [max(1, int(np.ceil(seqlen * next(cycle)))) for _ in range(batch_size)]

    decode_seqs = int(np.ceil(batch_size * decode_share))
    prefill_seqs = batch_size - decode_seqs
    partial_seqs = int(np.ceil(prefill_seqs * partial_prefill_share))

    half = itertools.cycle([p * 0.5 for p in prompt_pattern])
    partial_ctx = []
    for length in init_lens:
        raw = int(np.ceil(length // block_size * next(half))) * block_size
        partial_ctx.append(max(0, raw - block_size) if raw >= length else raw)

    query_lens = (
        [1] * decode_seqs
        + [init_lens[i] - partial_ctx[i] for i in range(decode_seqs, decode_seqs + partial_seqs)]
        + init_lens[decode_seqs + partial_seqs :]
    )
    seq_lens = (
        list(init_lens[:decode_seqs])
        + init_lens[decode_seqs : decode_seqs + partial_seqs]
        + init_lens[decode_seqs + partial_seqs :]
    )
    # Decode requests need at least one historical token to be a real decode.
    query_lens = [max(1, q) for q in query_lens]
    seq_lens = [max(seq_lens[i], query_lens[i]) for i in range(batch_size)]
    return query_lens, seq_lens


def assert_device_profiler_active(prof):
    """Abort unless the probe profile carries real Spyre device events."""
    total = sum(getattr(e, "self_device_time_total", 0.0) or 0.0 for e in prof.events())
    if total <= 0.0:
        raise SystemExit(
            "No Spyre device events in the probe profile. The AIUPTI backend is "
            "inactive, so measurements would be noise.\n"
            "Rebuild torch-spyre with USE_SPYRE_PROFILER=1 (see "
            "docs/user_guide/kineto_profiling.md sec 1.3) and verify with "
            "`ldd <torch_spyre/_C.so> | grep libaiupti`."
        )


def assert_span_present(prof, span):
    """Abort unless the requested record_function span is in the trace."""
    if not any(e.name == span for e in prof.events()):
        raise SystemExit(
            f"record_function span '{span}' absent from the trace. "
            "SPYRE_ATTN_PROFILING must be '1' *before* spyre_attn is imported "
            "(spyre_attn.py reads _ATTN_PROFILING at import time)."
        )


def span_device_times(prof, span=SPANS["online_softmax"]):
    """Device time (us) attributed to `span`, and its memory-op subtotal.

    Kineto does not propagate device time to record_function parents and AIUPTI
    emits no correlation ids, so attribute each kernel to the innermost span
    containing its *start* (dispatch is async, so containment would drop kernels
    that finish after the span closes).
    """
    events = list(prof.events())
    spans = sorted(
        ((e.name, e.time_range) for e in events if e.name.startswith("spyre_attn::")),
        key=lambda item: item[1].end - item[1].start,  # innermost first
    )
    outer = span == SPANS["forward"]
    # Union of every window for this span; spans is innermost-first, so the
    # first match would pick the narrowest forward window and drop the rest.
    windows = [tr for n, tr in spans if n == span]
    total = mem = 0.0
    for e in events:
        dev = getattr(e, "self_device_time_total", 0.0) or 0.0
        if dev <= 0:
            continue
        start = e.time_range.start
        if outer:
            # forward encloses the leaf spans, so credit every kernel starting
            # inside any forward window rather than to the innermost child.
            if any(w.start <= start <= w.end for w in windows):
                total += dev
                if any(m in e.name.lower() for m in MEMORY_OP_MARKERS):
                    mem += dev
            continue
        owner = next((n for n, tr in spans if tr.start <= start <= tr.end), None)
        if owner != span:
            continue
        total += dev
        if any(m in e.name.lower() for m in MEMORY_OP_MARKERS):
            mem += dev
    return total, mem


def span_cpu_time_us(prof, span=SPANS["online_softmax"]):
    for e in prof.events():
        if e.name == span:
            return e.time_range.elapsed_us()
    return float("nan")


def make_forward(inputs, num_query_heads, num_kv_heads, head_size):
    from spyre_inference.v1.attention.backends.spyre_attn import (
        SpyreAttentionImpl,
        SpyrePagedKVCache,
    )

    impl = SpyreAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=inputs["scale"],
        num_kv_heads=num_kv_heads,
    )
    # Allocate from the host query then transfer, as test and production do:
    # empty_like on an on-device tensor gives a layout the kernel's write does
    # not match, and the readback is then garbage.
    output = torch.empty_like(inputs["query_cpu"]).to(inputs["cache_device"])
    kv_cache = SpyrePagedKVCache(k_pages=inputs["k_pages"], v_pages=inputs["v_pages"])

    @torch.inference_mode()
    def run():
        impl.forward(
            layer=None,
            query=inputs["query_dev"],
            key=inputs["key_dev"],
            value=inputs["value_dev"],
            kv_cache=kv_cache,
            attn_metadata=inputs["attn_metadata"],
            output=output,
        )
        return output

    return run, output


def measure(run, iterations, span=SPANS["online_softmax"]):
    """Profile each forward in its own short window.

    The AIUPTI backend has a fixed trace-buffer pool and stops capturing once
    full (kineto_profiling.md §4.5), so one long window truncates the timeline.
    """
    dev_us, mem_us, cpu_us = [], [], []
    for _ in range(iterations):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
            run()
        total, mem = span_device_times(prof, span)
        dev_us.append(total)
        mem_us.append(mem)
        cpu_us.append(span_cpu_time_us(prof, span))
    return dev_us, mem_us, cpu_us


def run_config(entry, variant, cfg, records, csv_path, block_size=None):
    """Build, gate for correctness, and measure one (shape, variant) pair."""
    meta = VARIANT_REGISTRY[variant]
    num_q, num_kv = cfg["num_query_heads"], cfg["num_kv_heads"]
    head_size = cfg["head_size"]
    block_size = block_size or cfg["block_size"]

    query_lens, seq_lens = entry["query_lens"], entry["seq_lens"]
    name = entry["name"]
    max_kv = max(seq_lens)
    num_blocks = max(cfg["num_blocks"], (max_kv + block_size - 1) // block_size)

    span = SPANS[cfg.get("span", "online_softmax")]
    print(
        f"  {variant:26} bs={block_size:<4} {name:24} nreqs={len(query_lens)} "
        f"q={sum(query_lens)} kv={max_kv}",
        flush=True,
    )

    row = {
        "capture_name": name,
        "capture_type": entry["capture_type"],
        "num_reqs": len(query_lens),
        "total_query_tokens": sum(query_lens),
        "capture_max_query_len": max(query_lens),
        "max_seq_len": max_kv,
        "num_decode_tokens": sum(1 for q in query_lens if q == 1),
        "num_query_heads": num_q,
        "num_kv_heads": num_kv,
        "head_size": head_size,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "kv_layout": cfg.get("kv_layout", "plain"),
        "num_kv_blocks_iterated": (max_kv + block_size - 1) // block_size,
        "dtype": str(DTYPE),
        "implementation": meta["impl_label"],
        "variant": variant,
        "benchmark_mode": "BenchmarkMode.SPYRE_PROFILER",
        "span": span,
        "model": cfg.get("model", ""),
        "ms": float("nan"),
        "min_ms": float("nan"),
        "max_ms": float("nan"),
        "device_time_memory_us": float("nan"),
        "memory_share_pct": float("nan"),
        "cpu_time_ms": float("nan"),
        "allclose_pass": False,
        "max_abs_diff": float("nan"),
        "num_outliers": -1,
        "fallback_clean": True,
        "error": "",
    }
    row.update(entry.get("extra_cols", {}))

    inputs = None
    try:
        inputs = build_inputs_from_requests(
            query_lens,
            seq_lens,
            num_q,
            num_kv,
            head_size,
            block_size,
            num_blocks,
            cfg.get("device", "spyre"),
            seed=cfg.get("seed", 0),
            kv_layout=cfg.get("kv_layout", "plain"),
        )
        if inputs is None:
            row["error"] = "insufficient blocks"
            print("    -> skipped (insufficient blocks)", flush=True)
            records.append(row)
            return

        run, output = make_forward(inputs, num_q, num_kv, head_size)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run()  # first call: compile + populate metadata device mirrors
            row["fallback_clean"] = not any("fallback" in str(w.message).lower() for w in caught)

        # Correctness gate before timing, same semantics as
        # tests/attention/test_spyre_attn.py. The reference reads KV pages back
        # off the device, not the host copy: under slot-major the on-device
        # contents are what the kernel actually read.
        atol, rtol = cfg.get("atol", 0.3), cfg.get("rtol", 0.2)
        max_outliers = cfg.get("max_outliers", 5)
        got = output.to("cpu").float()
        ref = ref_attn(
            inputs["query_cpu"],
            inputs["k_pages"].to("cpu"),
            inputs["v_pages"].to("cpu"),
            inputs["query_lens"],
            inputs["seq_lens"],
            inputs["block_tables"],
            block_size,
            inputs["scale"],
        ).float()
        diff = (got - ref).abs()
        n_outliers = int((diff > atol + rtol * ref.abs()).sum().item())
        row["max_abs_diff"] = diff.max().item()
        row["num_outliers"] = n_outliers
        row["allclose_pass"] = n_outliers <= max_outliers
        if not row["allclose_pass"]:
            print(
                f"    -> CORRECTNESS FAIL (max_diff={row['max_abs_diff']:.4g}, "
                f"outliers={n_outliers}/{diff.numel()})",
                flush=True,
            )

        for _ in range(cfg.get("warmup", 2)):
            run()

        dev_us, mem_us, cpu_us = measure(run, cfg.get("iterations", 10), span)
        if dev_us and max(dev_us) > 0:
            row["ms"] = float(np.median(dev_us)) / 1000.0
            row["min_ms"] = float(np.min(dev_us)) / 1000.0
            row["max_ms"] = float(np.max(dev_us)) / 1000.0
            row["device_time_memory_us"] = float(np.median(mem_us))
            row["memory_share_pct"] = (
                100.0 * np.median(mem_us) / np.median(dev_us) if np.median(dev_us) else 0.0
            )
            row["cpu_time_ms"] = float(np.median(cpu_us)) / 1000.0
            print(
                f"    -> {row['ms'] * 1000:.1f}us device "
                f"(min={row['min_ms'] * 1000:.1f}, max={row['max_ms'] * 1000:.1f}) "
                f"mem={row['memory_share_pct']:.1f}%  cpu={row['cpu_time_ms']:.2f}ms  "
                f"max_diff={row['max_abs_diff']:.3g}",
                flush=True,
            )
        else:
            row["error"] = "no device time attributed to span"
            print("    -> no device time attributed", flush=True)

    except Exception as exc:  # keep the sweep alive; record the failure
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        print(f"    -> ERROR: {row['error']}", flush=True)
        if cfg.get("stop_on_failure"):
            raise
    finally:
        records.append(row)
        if csv_path is not None:
            pd.DataFrame(records).to_csv(csv_path, sep="\t", index=False)
        # Spyre's VFIO driver keeps DMA regions mapped until storage is
        # released; accumulating them across configs exhausts the table.
        del inputs
        gc.collect()


def entries_from_config(cfg):
    """Build the shape list from either request-list or grid mode."""
    entries = []

    for e in cfg.get("capture_batches", []):
        query_lens = _expand(e, "query_lens")
        seq_lens = _expand(e, "seq_lens")
        entries.append(
            {
                "name": e.get("name", f"nreqs{len(query_lens)}"),
                "capture_type": e.get("capture_type") or _infer_type(query_lens),
                "query_lens": query_lens,
                "seq_lens": seq_lens,
                "extra_cols": e.get("extra_cols", {}),
            }
        )

    grid = cfg.get("grid")
    if grid:
        patterns = grid.get("prompt_patterns", [[1.0]])
        for pattern in patterns:
            for bs in grid["batch_sizes"]:
                for seqlen in grid["sequence_lengths"]:
                    for ds in grid["decode_shares"]:
                        pps = grid.get("partial_prefill_share", 0.0)
                        ql, sl = grid_to_requests(bs, seqlen, ds, pps, pattern, cfg["block_size"])
                        entries.append(
                            {
                                "name": f"grid_bs{bs}_sl{seqlen}_ds{ds}",
                                "capture_type": _infer_type(ql),
                                "query_lens": ql,
                                "seq_lens": sl,
                                "extra_cols": {
                                    "seqlen": seqlen,
                                    "decode_share": ds,
                                    "partial_prefill_share": pps,
                                    "prompt_pattern": str(pattern),
                                    "realistic_prompt_mode": len(pattern) > 1,
                                    "gqa_mode": cfg["num_query_heads"] != cfg["num_kv_heads"],
                                },
                            }
                        )
    return entries


def _expand(entry, key):
    if key in entry:
        return [int(x) for x in entry[key]]
    rle = entry.get(f"{key}_rle")
    if rle is None:
        raise ValueError(f"entry needs '{key}' or '{key}_rle': {entry}")
    out = []
    for value, count in rle:
        out.extend([int(value)] * int(count))
    return out


def _infer_type(query_lens):
    if all(q == 1 for q in query_lens):
        return "decode"
    if all(q > 1 for q in query_lens):
        return "prefill"
    return "mixed"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default="./microbench_results")
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--kv-layout",
        choices=["plain", "slot_major", "slot_major_devfill"],
        default=None,
        help="KV page device layout. 'plain' (default) is correct for a "
        "host-populated cache. 'slot_major_devfill' matches the "
        "worker: zeroed slot-major alloc, history written on device. "
        "'slot_major' pins the worker layout on a host-populated "
        "cache and is numerically wrong; kept to reproduce that.",
    )
    ap.add_argument(
        "--span",
        choices=sorted(SPANS),
        default=None,
        help="record_function scope to attribute device time to (default: online_softmax)",
    )
    ap.add_argument("--stop-on-failure", action="store_true")
    ap.add_argument("--no-output", action="store_true")
    ap.add_argument("--allow-empty-device-profile", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    for key, val in (
        ("iterations", args.iterations),
        ("warmup", args.warmup),
        ("device", args.device),
        ("span", args.span),
        ("kv_layout", args.kv_layout),
    ):
        if val is not None:
            cfg[key] = val
    if args.variants:
        cfg["variants"] = args.variants
    cfg["stop_on_failure"] = args.stop_on_failure
    cfg.setdefault("device", "spyre")

    variants = [v for v in cfg["variants"] if VARIANT_REGISTRY[v]["available"]()]
    if not variants:
        raise SystemExit("no available variants")
    # One vLLM config context per process, so compiled and eager cannot be
    # mixed in a single run.
    compiled_modes = {VARIANT_REGISTRY[v]["compiled"] for v in variants}
    if len(compiled_modes) > 1:
        raise SystemExit(
            "compiled and eager variants need separate runs (the compilation mode "
            "is fixed for the process). Re-run with --variants one at a time."
        )

    entries = entries_from_config(cfg)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_dir) / cfg.get("run_label", "run") / stamp
    if not args.no_output:
        out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = None if args.no_output else out_dir / "spyre_attn_microbench.csv"

    print(f"\nSpyre attention micro-benchmark  [{stamp}]")
    print(f"  model      : {cfg.get('model', 'n/a')}")
    print(
        f"  shape      : q_heads={cfg['num_query_heads']} kv_heads={cfg['num_kv_heads']} "
        f"head_size={cfg['head_size']} "
        f"block_size={cfg.get('block_sizes') or cfg['block_size']} dtype={DTYPE}"
    )
    print(f"  span       : {SPANS[cfg.get('span', 'online_softmax')]}")
    print(f"  variants   : {variants}")
    print(f"  shapes     : {len(entries)}")
    print(f"  iterations : {cfg.get('iterations', 10)} (warmup {cfg.get('warmup', 2)})")
    print(f"  output     : {out_dir if not args.no_output else 'none'}\n", flush=True)

    records = []
    with spyre_vllm_config(compiled=next(iter(compiled_modes))):
        import torch._dynamo

        # The kernel specializes per (num_blocks, aligned_max_query_len), so a
        # sweep legitimately blows through the default recompile limit.
        torch._dynamo.config.accumulated_recompile_limit = 4096
        torch._dynamo.config.recompile_limit = 4096

        from spyre_inference.v1.attention.backends import spyre_attn as sa

        if not sa._ATTN_PROFILING:
            raise SystemExit("SPYRE_ATTN_PROFILING was not honoured; span will be absent.")

        # Startup guard: probe on the smallest shape.
        probe = entries[0]
        probe_inputs = build_inputs_from_requests(
            probe["query_lens"],
            probe["seq_lens"],
            cfg["num_query_heads"],
            cfg["num_kv_heads"],
            cfg["head_size"],
            cfg["block_size"],
            cfg["num_blocks"],
            cfg["device"],
        )
        probe_run, _ = make_forward(
            probe_inputs, cfg["num_query_heads"], cfg["num_kv_heads"], cfg["head_size"]
        )
        probe_run()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
            probe_run()
        sel_span = SPANS[cfg.get("span", "online_softmax")]
        assert_span_present(prof, sel_span)
        if not args.allow_empty_device_profile:
            assert_device_profiler_active(prof)
        total, _ = span_device_times(prof, sel_span)
        print(f"  [guard] profiler active; '{sel_span}' device time = {total:.1f}us\n", flush=True)
        del probe_inputs, probe_run
        gc.collect()

        block_sizes = cfg.get("block_sizes") or [cfg["block_size"]]
        for variant in variants:
            for bsz in block_sizes:
                for entry in entries:
                    run_config(entry, variant, cfg, records, csv_path, block_size=bsz)

    df = pd.DataFrame(records)
    if not args.no_output:
        df.to_csv(csv_path, sep="\t", index=False)
        df.to_csv(out_dir / "spyre_attn_microbench_final.csv", sep="\t", index=False)
        print(f"\nDone. {len(df)} measurements -> {out_dir}")
    ok = df[df["ms"].notna()]
    print(
        f"  measured {len(ok)}/{len(df)}; correctness pass "
        f"{int(df['allclose_pass'].sum())}/{len(df)}"
    )

    sys.stdout.flush()
    sys.stderr.flush()
    # TimestampCalibrator aborts in its destructor at teardown (sec 4.4).
    os._exit(0)


if __name__ == "__main__":
    main()
