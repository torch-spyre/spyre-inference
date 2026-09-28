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
        --config scripts/microbench/configs/granite33_8b_e2e_ref.json
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
    # Emitted here, not by spyre_attn: Inductor fuses production's KV write into the
    # block graph's RMSNorm+QKV kernel, so it has no span of its own. See README.
    "reshape_and_cache": "spyre_attn::reshape_and_cache",
    "layer": "spyre_attn::layer",
}
OUTER_SPANS = (SPANS["forward"], SPANS["layer"])
MEMORY_OP_MARKERS = ("memcpy", "memset", "restickify", "stickif", "copy_from_d2d")

# Spyre requires float16 (platform.py raises otherwise).
DTYPE = torch.float16

# check_and_update_config caps max_num_batched_tokens here for decoder models, so a
# longer batch is unschedulable at any max_model_len.
_MAX_BATCHED_TOKENS = 512


VARIANT_REGISTRY: dict[str, dict] = {}


def register_variant(name, impl_label, compiled=True, batched=False, available_fn=None):
    VARIANT_REGISTRY[name] = {
        "impl_label": impl_label,
        "compiled": compiled,
        "batched": batched,
        "available": available_fn or (lambda: True),
    }


register_variant("online_softmax_compiled", "Implementation.SPYRE_ONLINE_SOFTMAX", True)
register_variant("online_softmax_eager", "Implementation.SPYRE_ONLINE_SOFTMAX_EAGER", False)
register_variant(
    "batched_decode_compiled", "Implementation.SPYRE_BATCHED_DECODE", True, batched=True
)


@contextlib.contextmanager
def spyre_vllm_config(compiled: bool, block_size: int, limits: dict):
    """Establish a Spyre vLLM config context for standalone (non-pytest) use.

    ``limits`` comes from ``derive_lattice``: defaulted, the padding the kernel
    sees would be an accident of whichever model ``ModelConfig()`` resolves.
    ``block_size`` must reach ``cache_config``, which the metadata builder asserts
    its kv_cache_spec against.
    """
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.config.compilation import CompilationConfig, CompilationMode
    from vllm.forward_context import set_forward_context
    from vllm.platforms import PlatformEnum, current_platform

    from spyre_inference.custom_ops import register_all

    current_platform._enum = PlatformEnum.OOT
    register_all()
    mode = CompilationMode.STOCK_TORCH_COMPILE if compiled else CompilationMode.NONE
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(block_size=block_size),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=limits["max_model_len"],
            max_num_batched_tokens=limits["max_num_batched_tokens"],
            max_num_seqs=limits["max_num_seqs"],
        ),
        compilation_config=CompilationConfig(custom_ops=["all"], mode=mode),
        model_config=ModelConfig(dtype=DTYPE, max_model_len=limits["max_model_len"]),
    )
    with set_current_vllm_config(config), set_forward_context(None, config):
        yield config


def ref_attn(
    query,
    key_cache,
    value_cache,
    query_lens,
    kv_lens,
    block_tables,
    block_size,
    scale,
    sliding_window=None,
):
    """Causal varlen reference, optionally restricted to a sliding window."""
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
        if sliding_window is not None:
            q_pos = torch.arange(query_len) + kv_len - query_len
            k_pos = torch.arange(kv_len)
            mask |= k_pos.unsqueeze(0) < (q_pos - sliding_window + 1).unsqueeze(1)
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
    sliding_window=None,
):
    """Drive the real SpyreAttentionMetadataBuilder; return it alongside its metadata.

    The builder comes back so callers can ask it what a shape should have been
    rather than restating its formulas.
    """
    from unittest.mock import Mock

    from vllm.config import get_current_vllm_config
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec

    from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadataBuilder

    vllm_config = get_current_vllm_config()
    vllm_config.model_config.get_num_attention_heads = Mock(return_value=num_query_heads)
    vllm_config.model_config.get_num_kv_heads = Mock(return_value=num_kv_heads)

    spec_cls = FullAttentionSpec if sliding_window is not None else AttentionSpec
    spec_kwargs = {"sliding_window": sliding_window} if sliding_window is not None else {}
    builder = SpyreAttentionMetadataBuilder(
        kv_cache_spec=spec_cls(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=DTYPE,
            **spec_kwargs,
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
    return builder, builder.build(common_prefix_len=0, common_attn_metadata=common)


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


def _head_major_pages(k_cpu, v_cpu, device):
    """The zeroed pages the worker allocates, in the head-major device layout."""
    from spyre_inference.v1.attention.ops.layout import head_major_kv_layout

    nb, bsz, kv, d = k_cpu.shape
    if device.type != "spyre":
        return (
            k_cpu.permute(0, 2, 1, 3).contiguous().to(device),
            v_cpu.permute(0, 2, 1, 3).contiguous().to(device),
        )
    layout = head_major_kv_layout(nb * kv, bsz, d, k_cpu.dtype)
    shape = (nb, kv, bsz, d)
    return (
        torch.zeros(shape, dtype=k_cpu.dtype).to(device, device_layout=layout),
        torch.zeros(shape, dtype=v_cpu.dtype).to(device, device_layout=layout),
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
    kv_layout="slot_major_devfill",
    attn_kv_layout="token_major",
    sliding_window=None,
):
    """Varlen multi-sequence inputs from explicit per-request lengths."""
    from vllm.utils.torch_utils import set_random_seed

    from spyre_inference.custom_ops.utils import convert
    from spyre_inference.v1.attention.ops.layout import slot_major_kv_layout

    assert len(query_lens) == len(seq_lens)
    for ql, sl in zip(query_lens, seq_lens):
        assert sl >= ql >= 1, f"require seq_len >= query_len >= 1, got {sl} < {ql}"

    torch.set_default_device("cpu")
    set_random_seed(seed)

    num_seqs = len(query_lens)
    # max_model_len wide, as vLLM's is, not as wide as this shape needs: the builder
    # pads each block count onto a bucket and the kernel gathers every padded page,
    # so a table sized to the real count leaves the padding with no page id to read.
    from vllm.config import get_current_vllm_config

    max_model_len = get_current_vllm_config().model_config.max_model_len
    blocks_per_seq = max((max_model_len + block_size - 1) // block_size, 1)
    if num_blocks < num_seqs * blocks_per_seq:
        return None  # cache too small to give every sequence its own pages

    scale = head_size**-0.5
    total_q = sum(query_lens)

    query = torch.randn(total_q, num_query_heads, head_size, dtype=DTYPE)
    key = torch.randn(total_q, num_kv_heads, head_size, dtype=DTYPE)
    value = torch.randn(total_q, num_kv_heads, head_size, dtype=DTYPE)
    k_pages_cpu = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=DTYPE)
    v_pages_cpu = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=DTYPE)

    # Sample without replacement: an aliased page would let one sequence
    # overwrite another's KV and shrink the set of pages actually gathered.
    block_tables = torch.randperm(num_blocks, dtype=torch.int32)[: num_seqs * blocks_per_seq].view(
        num_seqs, blocks_per_seq
    )

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

    builder, attn_metadata = build_metadata(
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor([0] + list(query_lens), dtype=torch.int32).cumsum(0, dtype=torch.int32),
        block_tables,
        slot_mapping,
        sliding_window,
    )

    cache_device = torch.device(device)

    def to_device(cache):
        # plain: host-populated cache, plain transfer. Matches
        # tests/attention/test_spyre_attn.py, correct at these shapes.
        # _reshape_and_cache views pages as [-1, H, D] and requires the
        # slot-outermost device layout, which host conversion cannot reproduce.
        # slot_major is therefore wrong; slot_major_devfill uses production's
        # device-side history write.
        if cache_device.type != "spyre" or kv_layout == "plain":
            return cache.to(cache_device)
        nb, bsz, h, d = cache.shape
        if kv_layout == "page_major":
            from torch_spyre._C import SpyreTensorLayout

            # device axes: [page, kv_head, D-stick, token, D-element]
            return cache.to(
                cache_device,
                device_layout=SpyreTensorLayout(
                    [nb, bsz, h, d],
                    [bsz * h * d, h * d, d, 1],
                    cache.dtype,
                    [1, 0, 2, 3],
                ),
            )
        layout = slot_major_kv_layout(nb * bsz, h, d, cache.dtype)
        if kv_layout == "slot_major":
            return cache.to(cache_device, device_layout=layout)
        return torch.zeros_like(cache).to(cache_device, device_layout=layout)

    if attn_kv_layout == "head_major":
        # kv_layout is a token-major device-layout knob and does not apply here.
        k_pages, v_pages = _head_major_pages(k_pages_cpu, v_pages_cpu, cache_device)
    else:
        k_pages, v_pages = to_device(k_pages_cpu), to_device(v_pages_cpu)

    # Eager index_copy_ takes an int64 index, which falls back to CPU and lands the rows in
    # the wrong place, so make_forward writes the history through the impl's compiled store.
    devfill = cache_device.type == "spyre" and bool(hist_slots)
    hist_seed = None
    if devfill and (attn_kv_layout == "head_major" or kv_layout == "slot_major_devfill"):
        hist_seed = {
            "key": convert(torch.cat(hist_k), cache_device),
            "value": convert(torch.cat(hist_v), cache_device),
            "slots_cpu": torch.tensor(hist_slots, dtype=torch.int64),
        }
    key_dev, value_dev = _fused_qkv_kv_views(query, key, value, cache_device)

    return {
        "query_dev": convert(query, cache_device),
        "key_dev": key_dev,
        "value_dev": value_dev,
        "slots_dev": convert(slot_mapping, cache_device),
        "slot_mapping_cpu": slot_mapping,
        "hist_seed": hist_seed,
        "k_pages": k_pages,
        "v_pages": v_pages,
        "k_pages_cpu": k_pages_cpu,
        "v_pages_cpu": v_pages_cpu,
        "query_cpu": query,
        "block_tables": block_tables,
        "attn_metadata": attn_metadata,
        "builder": builder,
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


def span_device_times(prof, span=SPANS["layer"]):
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
    outer = span in OUTER_SPANS
    # Union of every window for this span; spans is innermost-first, so the
    # first match would pick the narrowest forward window and drop the rest.
    windows = [tr for n, tr in spans if n == span]
    total = mem = 0.0
    n_compute = 0
    by_op: dict[str, list[float]] = {}

    def credit(e, dev):
        nonlocal total, mem, n_compute
        total += dev
        if any(m in e.name.lower() for m in MEMORY_OP_MARKERS):
            mem += dev
        else:
            n_compute += 1
        slot = by_op.setdefault(e.name, [0.0, 0])
        slot[0] += dev
        slot[1] += 1

    for e in events:
        dev = getattr(e, "self_device_time_total", 0.0) or 0.0
        if dev <= 0:
            continue
        start = e.time_range.start
        if outer:
            # forward encloses the leaf spans, so credit every kernel starting
            # inside any forward window rather than to the innermost child.
            if any(w.start <= start <= w.end for w in windows):
                credit(e, dev)
            continue
        owner = next((n for n, tr in spans if tr.start <= start <= tr.end), None)
        if owner != span:
            continue
        credit(e, dev)
    return total, mem, n_compute, by_op


def span_cpu_time_us(prof, span=SPANS["layer"]):
    for e in prof.events():
        if e.name == span:
            return e.time_range.elapsed_us()
    return float("nan")


def make_forward(
    inputs,
    num_query_heads,
    num_kv_heads,
    head_size,
    kv_write=False,
    attn_kv_layout="token_major",
):
    """One emulated attention layer, in the order ``attn_layer`` runs it.

    The query is staged here because ``attn_layer`` stages it in the block graph,
    outside the attention span: passing a non-staging query instead makes
    ``pre_staged`` False and silently moves both copies inside the measured span.
    """
    from torch_spyre.streams import synchronize

    from spyre_inference.v1.attention.backends.spyre_attn import (
        SpyreAttentionImpl,
        SpyrePagedKVCache,
    )

    impl_cls = SpyreAttentionImpl
    if attn_kv_layout == "head_major":
        from spyre_inference.v1.attention.backends.spyre_head_major_attn import (
            SpyreHeadMajorAttentionImpl,
        )

        impl_cls = SpyreHeadMajorAttentionImpl

    impl = impl_cls(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=inputs["scale"],
        num_kv_heads=num_kv_heads,
    )
    device = inputs["cache_device"]
    # Allocate from the host query then transfer, as test and production do:
    # empty_like on an on-device tensor gives a layout the kernel's write does
    # not match, and the readback is then garbage.
    output = torch.empty_like(inputs["query_cpu"]).to(device)
    kv_cache = SpyrePagedKVCache(k_pages=inputs["k_pages"], v_pages=inputs["v_pages"])
    q_staging, out_staging = impl.staging_buffers(device)
    rows = inputs["total_query_tokens"]
    slots_dev = impl.kv_write_index(inputs["slot_mapping_cpu"], device)

    seed = inputs.get("hist_seed")
    if seed is not None:
        impl.do_kv_cache_update(
            None,
            seed["key"],
            seed["value"],
            kv_cache,
            impl.kv_write_index(seed["slots_cpu"], device),
        )

    @torch.inference_mode()
    def run():
        with torch.profiler.record_function(SPANS["layer"]):
            if kv_write:
                with torch.profiler.record_function(SPANS["reshape_and_cache"]):
                    impl.do_kv_cache_update(
                        None, inputs["key_dev"], inputs["value_dev"], kv_cache, slots_dev
                    )
            q_staging[:rows] = inputs["query_dev"]
            impl.forward(
                layer=None,
                query=q_staging,
                key=inputs["key_dev"],
                value=inputs["value_dev"],
                kv_cache=kv_cache,
                attn_metadata=inputs["attn_metadata"],
                output=out_staging,
            )
            output.copy_(out_staging[:rows])
            # The device queue lags the CPU by about one kernel, so a loop's last
            # kernel starts just after the enclosing span closes and start-based
            # attribution drops it. Only this span is sync-terminated, hence exact.
            synchronize(device)
        return output

    return run, output, impl


def measure(run, iterations, span=SPANS["layer"]):
    """Profile each forward in its own short window.

    The AIUPTI backend has a fixed trace-buffer pool and stops capturing once
    full (kineto_profiling.md §4.5), so one long window truncates the timeline.
    """
    dev_us, mem_us, cpu_us, n_kernels, per_op = [], [], [], [], []
    for _ in range(iterations):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
            run()
        total, mem, n_compute, by_op = span_device_times(prof, span)
        dev_us.append(total)
        mem_us.append(mem)
        n_kernels.append(n_compute)
        per_op.append(by_op)
        cpu_us.append(span_cpu_time_us(prof, span))
    # The window whose total is the reported median, so the breakdown adds up to `ms`.
    median_at = int(np.argsort(dev_us)[len(dev_us) // 2]) if dev_us else 0
    return dev_us, mem_us, cpu_us, n_kernels, per_op[median_at] if per_op else {}


def unreachable_reason(query_lens) -> str:
    """Why the engine could never schedule this shape, or "" if it could.

    Against the platform cap, not the derived limits: those come from the shapes, so
    they can never rule one out.
    """
    if sum(query_lens) > _MAX_BATCHED_TOKENS:
        return (
            f"batch of {sum(query_lens)} tokens above the platform's "
            f"max_num_batched_tokens cap of {_MAX_BATCHED_TOKENS}; the scheduler "
            "cannot pack this many into one step, so production chunks it"
        )
    return ""


def record_padding(row, attn_metadata, query_lens, seq_lens, block_size, builder=None):
    """Record the shape the kernel got, and flag it when that is not the one asked for."""
    # A mask stack holds one [aligned_query_len, block_size] mask tile per active block,
    # so shape[0] is the number of blocks the kernel iterated for that sequence.
    realized_blocks = [int(m.shape[0]) for m in attn_metadata.attention_mask_stacks]
    realized_query = list(attn_metadata.aligned_query_lens)
    row["num_kv_blocks_iterated"] = max(realized_blocks)
    row["padded_query_len"] = max(realized_query)

    declared_blocks = [(s + block_size - 1) // block_size for s in seq_lens]
    declared_query = [max(1, q) for q in query_lens]
    if builder is not None and builder.sliding_window is not None:
        # A window drops out-of-window blocks and pads back to its bucket maximum,
        # so the identity does not hold; only the per-sequence bound is meaningful.
        # The bound comes from the builder rather than being restated here, so the
        # check cannot drift from the padding it verifies. It is asked for the same
        # bucket build() used -- the padded block count, not the sequence's own.
        for s, (realized, aligned) in enumerate(zip(realized_blocks, realized_query)):
            bound = builder._max_active_blocks(builder._pad_num_blocks(declared_blocks[s]), aligned)
            if realized > bound:
                row["error"] = (
                    f"windowed active blocks {realized} exceed the bucket bound {bound} "
                    f"for sequence {s}"
                )
                print(f"    -> {row['error']}", flush=True)
                return
        if realized_query != declared_query:
            row["error"] = f"query padding is not the identity: {declared_query}->{realized_query}"
            print(f"    -> {row['error']}", flush=True)
        return
    if realized_blocks != declared_blocks or realized_query != declared_query:
        row["error"] = (
            f"padding is not the identity: blocks {declared_blocks}->{realized_blocks}, "
            f"query {declared_query}->{realized_query}"
        )
        print(f"    -> {row['error']}", flush=True)


def record_attn_path(row, impl, attn_metadata, batched_variant):
    """Record which read path the impl took, and flag a declined batched gate."""
    row["num_decode_seqs"] = attn_metadata.num_decode_seqs
    row["padded_num_seqs"] = (
        -1 if attn_metadata.padded_num_seqs is None else attn_metadata.padded_num_seqs
    )
    row["decode_uniformity"] = attn_metadata.decode_uniformity
    row["blocks_per_chunk"] = (
        -1 if attn_metadata.blocks_per_chunk is None else attn_metadata.blocks_per_chunk
    )
    batched = impl._batched_decode_preconditions_met(attn_metadata)
    if not batched:
        row["attn_path"] = "per_seq"
        if batched_variant:
            row["error"] = "batched decode gate declined; measured the per-seq loop instead"
            print(f"    -> {row['error']}", flush=True)
        return
    all_decode = attn_metadata.num_decode_seqs == attn_metadata.num_seqs
    row["attn_path"] = "batched" if all_decode else "batched+per_seq"


def expected_kernels(row, span):
    """Compute kernels the span should contain, to catch a short attribution.

    The loop runs one kernel per sequence; the batched path collapses the decode
    prefix's into one. Staging copies count as memory ops, so they are excluded.
    Head-major's decode kernel emits its store as a second kernel per call, where
    token-major and head-major's batched-GQA prefill kernel both fuse it into the
    attention one.
    """
    num_seqs, num_decode = row["num_reqs"], row["num_decode_seqs"]
    if row["attn_path"].startswith("batched"):
        attn = 1 + (num_seqs - num_decode)
    else:
        attn = num_seqs
    if row.get("attn_kv_layout") == "head_major" and row["padded_query_len"] == 1:
        attn *= 2
    write = 1 if row["kv_write"] else 0
    if span == SPANS["reshape_and_cache"]:
        return write
    if span == SPANS["layer"]:
        return attn + write
    return attn  # forward / online_softmax


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

    attn_kv_layout = cfg.get("attn_kv_layout", "token_major")
    span = SPANS[cfg.get("span", "layer")]
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
        "kv_layout": cfg.get("kv_layout", "slot_major_devfill"),
        "attn_kv_layout": attn_kv_layout,
        "num_kv_blocks_iterated": -1,
        "padded_query_len": -1,
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
        "ref_abs_max": float("nan"),
        "num_outliers": -1,
        "fallback_clean": True,
        "attn_path": "",
        "num_decode_seqs": -1,
        "padded_num_seqs": -1,
        "decode_uniformity": float("nan"),
        "blocks_per_chunk": -1,
        "kv_write": bool(cfg.get("kv_write", False)),
        "late_compile": False,
        "kernels_attributed": -1,
        "kernels_expected": -1,
        "error": "",
    }
    row.update(entry.get("extra_cols", {}))

    inputs = None
    try:
        unreachable = unreachable_reason(query_lens)
        if unreachable:
            row["error"] = unreachable
            print(f"    -> skipped ({unreachable})", flush=True)
            records.append(row)
            return

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
            kv_layout=cfg.get("kv_layout", "slot_major_devfill"),
            attn_kv_layout=attn_kv_layout,
            sliding_window=cfg.get("sliding_window"),
        )
        if inputs is None:
            from vllm.config import get_current_vllm_config

            max_model_len = get_current_vllm_config().model_config.max_model_len
            needed = len(query_lens) * ((max_model_len + block_size - 1) // block_size)
            row["error"] = f"insufficient blocks (need num_blocks >= {needed})"
            print(f"    -> skipped ({row['error']})", flush=True)
            records.append(row)
            return

        record_padding(
            row,
            inputs["attn_metadata"],
            query_lens,
            seq_lens,
            block_size,
            inputs["builder"],
        )

        run, output, impl = make_forward(
            inputs,
            num_q,
            num_kv,
            head_size,
            kv_write=row["kv_write"],
            attn_kv_layout=attn_kv_layout,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run()  # first call: compile + populate metadata device mirrors
            row["fallback_clean"] = not any("fallback" in str(w.message).lower() for w in caught)

        record_attn_path(row, impl, inputs["attn_metadata"], meta["batched"])

        # Correctness gate before timing, same semantics as
        # tests/attention/test_spyre_attn.py. The reference reads KV pages back
        # off the device, not the host copy: under slot-major the on-device
        # contents are what the kernel actually read.
        atol, rtol = cfg.get("atol", 0.3), cfg.get("rtol", 0.2)
        max_outliers = cfg.get("max_outliers", 5)
        got = output.to("cpu").float()
        k_ref, v_ref = inputs["k_pages"].to("cpu"), inputs["v_pages"].to("cpu")
        if attn_kv_layout == "head_major":
            k_ref, v_ref = k_ref.permute(0, 2, 1, 3), v_ref.permute(0, 2, 1, 3)
        ref = ref_attn(
            inputs["query_cpu"],
            k_ref,
            v_ref,
            inputs["query_lens"],
            inputs["seq_lens"],
            inputs["block_tables"],
            block_size,
            inputs["scale"],
            cfg.get("sliding_window"),
        ).float()
        diff = (got - ref).abs()
        n_outliers = int((diff > atol + rtol * ref.abs()).sum().item())
        row["max_abs_diff"] = diff.max().item()
        row["ref_abs_max"] = ref.abs().max().item()
        if row["ref_abs_max"] == 0.0:
            print(
                "    -> WARNING: reference is all zeros, so this comparison is vacuous "
                "(the cache the kernel read holds no data)",
                flush=True,
            )
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

        # A compile landing in a measured window puts Inductor time in the number.
        from torch._dynamo.utils import counters

        graphs_before = counters["stats"]["unique_graphs"]
        dev_us, mem_us, cpu_us, n_kernels, by_op = measure(run, cfg.get("iterations", 10), span)
        row["late_compile"] = counters["stats"]["unique_graphs"] != graphs_before
        if row["late_compile"]:
            print("    -> WARNING: a kernel compiled inside a measured window", flush=True)
        row["kernels_attributed"] = int(np.median(n_kernels))
        row["kernels_expected"] = expected_kernels(row, span)
        if row["kernels_attributed"] != row["kernels_expected"]:
            row["error"] = (
                f"attributed {row['kernels_attributed']} compute kernels, expected "
                f"{row['kernels_expected']}; the span under-counts (use --span layer)"
            )
            print(f"    -> {row['error']}", flush=True)
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
                f"max_diff={row['max_abs_diff']:.3g} ref_max={row['ref_abs_max']:.3g}",
                flush=True,
            )
            if cfg.get("top_ops"):
                ops = sorted(by_op.items(), key=lambda kv: -kv[1][0])
                print(f"    -- top device ops in '{span}' (median window):", flush=True)
                for op_name, (op_us, op_n) in ops[: cfg["top_ops"]]:
                    print(
                        f"       {op_us:9.1f}us  n={op_n:<4} "
                        f"{100.0 * op_us / max(row['ms'] * 1000, 1e-9):5.1f}%  {op_name}",
                        flush=True,
                    )
                if csv_path is not None:
                    pd.DataFrame(
                        [
                            {
                                "capture_name": name,
                                "attn_kv_layout": attn_kv_layout,
                                "max_seq_len": max_kv,
                                "op": op_name,
                                "device_us": op_us,
                                "count": op_n,
                            }
                            for op_name, (op_us, op_n) in ops
                        ]
                    ).to_csv(
                        Path(csv_path).with_name(f"ops_{attn_kv_layout}_{name}_{max_kv}.tsv"),
                        sep="\t",
                        index=False,
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


def derive_lattice(entries):
    """The bucket lattice that makes the builder's padding the identity.

    One bucket per declared length, so ``find_*_bucket`` returns it unchanged. The kv
    ladder stays in tokens rather than block-aligned: ``num_blocks_buckets`` derives
    from it through the builder's own ``ceil(len / block_size)``, so one lattice holds
    for a whole ``block_sizes`` sweep, which ``envs`` caching on first read requires.
    """
    kv = sorted({s for e in entries for s in e["seq_lens"]})
    # max_num_batched_tokens is pinned, not derived: staging_rows is it plus one and
    # those buffers are the kernel's query and output arguments, so a decode-only shape
    # list would otherwise measure a narrower gather than production's. It joins the
    # ladder because _resolve_buckets appends a limit its ladder tops out below.
    # 1 is the decode rung, reached by a config that declares no wider query at all.
    query = sorted({q for e in entries for q in e["query_lens"]} | {1, _MAX_BATCHED_TOKENS})
    num_seqs = sorted({len(e["query_lens"]) for e in entries})
    return {
        "max_model_len": kv[-1],
        "max_num_batched_tokens": _MAX_BATCHED_TOKENS,
        "max_num_seqs": num_seqs[-1],
        "SPYRE_ATTN_KV_BUCKETS": ",".join(map(str, kv)),
        "SPYRE_ATTN_QUERY_BUCKETS": ",".join(map(str, query)),
        "SPYRE_ATTN_NUM_SEQS_BUCKETS": ",".join(map(str, num_seqs)),
    }


def run_startup_guard(cfg, entries, block_size, span, args):
    """Abort unless the probe profile carries device events and the chosen span."""
    probe = entries[0]
    num_blocks = max(cfg["num_blocks"], (max(probe["seq_lens"]) + block_size - 1) // block_size)
    probe_inputs = build_inputs_from_requests(
        probe["query_lens"],
        probe["seq_lens"],
        cfg["num_query_heads"],
        cfg["num_kv_heads"],
        cfg["head_size"],
        block_size,
        num_blocks,
        cfg["device"],
        kv_layout=cfg.get("kv_layout", "slot_major_devfill"),
        attn_kv_layout=cfg.get("attn_kv_layout", "token_major"),
        sliding_window=cfg.get("sliding_window"),
    )
    probe_run, _, _ = make_forward(
        probe_inputs,
        cfg["num_query_heads"],
        cfg["num_kv_heads"],
        cfg["head_size"],
        kv_write=bool(cfg.get("kv_write", False)),
        attn_kv_layout=cfg.get("attn_kv_layout", "token_major"),
    )
    probe_run()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        probe_run()
    assert_span_present(prof, span)
    if not args.allow_empty_device_profile:
        assert_device_profiler_active(prof)
    total, _, _, _ = span_device_times(prof, span)
    print(f"  [guard] profiler active; '{span}' device time = {total:.1f}us\n", flush=True)
    del probe_inputs, probe_run
    gc.collect()


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
        choices=["plain", "slot_major", "slot_major_devfill", "page_major"],
        default=None,
        help="KV page device layout. 'slot_major_devfill' (default) matches the "
        "worker: zeroed slot-major alloc, history written on device. 'plain' is "
        "correct for a host-populated cache. 'slot_major' pins the worker layout "
        "on a host-populated cache and is numerically wrong. 'page_major' is an "
        "index_select layout/SDSC probe.",
    )
    ap.add_argument(
        "--attn-kv-layout",
        choices=["token_major", "head_major"],
        default=None,
        help="KV cache decomposition the backend reads (SPYRE_ATTN_KV_LAYOUT). "
        "'head_major' (default) stores a page as [KV, block_size, head_size] and selects "
        "the head-major backend, the only one whose batched decode kernel runs under the "
        "default tiled walk.",
    )
    ap.add_argument(
        "--span",
        choices=sorted(SPANS),
        default=None,
        help="record_function scope to attribute device time to. Default 'layer' is the "
        "whole emulated attention layer, and the only span that closes after a device "
        "sync, so it is the only one whose total cannot silently drop a kernel.",
    )
    ap.add_argument(
        "--kv-write",
        action="store_true",
        help="also scatter K/V into the cache each iteration, as attn_layer does, "
        "so the reshape_and_cache and layer spans have something to measure",
    )
    ap.add_argument(
        "--top-ops",
        type=int,
        default=0,
        help="print the N device ops holding the most time in the measured span, and "
        "write the full per-op breakdown of that window beside the CSV",
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
        ("attn_kv_layout", args.attn_kv_layout),
        ("kv_write", args.kv_write or None),
    ):
        if val is not None:
            cfg[key] = val
    retired = [
        k
        for k in (
            "max_model_len",
            "max_num_batched_tokens",
            "max_num_seqs",
            "attn_kv_buckets",
            "attn_query_buckets",
            "attn_num_seqs_buckets",
        )
        if k in cfg
    ]
    if retired:
        raise SystemExit(
            f"{args.config} sets {retired}, which no longer configure anything: the "
            "lattice is derived from the shapes. Declare the shape you want measured; "
            "to model a coarser deployment ladder, declare the padded length it "
            "rounds to."
        )
    if args.variants:
        cfg["variants"] = args.variants
    cfg["stop_on_failure"] = args.stop_on_failure
    cfg["top_ops"] = args.top_ops
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
    # SPYRE_BATCHED_DECODE is cached on first read and sets the builder's
    # reorder_batch_threshold, so it cannot vary within a process either.
    batched_modes = {VARIANT_REGISTRY[v]["batched"] for v in variants}
    if len(batched_modes) > 1:
        raise SystemExit(
            "batched and per-seq decode variants need separate runs "
            "(SPYRE_BATCHED_DECODE is process-wide). Re-run with --variants one at a time."
        )
    os.environ["SPYRE_BATCHED_DECODE"] = "1" if next(iter(batched_modes)) else "0"
    # Selects the backend via the platform, and is cached on first envs read like the rest.
    attn_kv_layout = cfg.setdefault("attn_kv_layout", "head_major")
    os.environ["SPYRE_ATTN_KV_LAYOUT"] = attn_kv_layout
    from spyre_inference import envs

    if (
        attn_kv_layout == "token_major"
        and next(iter(batched_modes))
        and envs.SPYRE_ATTN_FOR_EACH_TILE
    ):
        raise SystemExit(
            "token-major batched decode is declined under the tiled walk; run the batched "
            "variant on head_major, or set SPYRE_ATTN_FOR_EACH_TILE=0."
        )

    entries = entries_from_config(cfg)
    limits = derive_lattice(entries)
    # envs caches on first read, so this has to land before any bucketer is built.
    for key, value in limits.items():
        if key.startswith("SPYRE_"):
            os.environ[key] = value
    sel_span = SPANS[cfg.get("span", "layer")]
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
    print(f"  span       : {sel_span}")
    print(f"  kv layout  : {attn_kv_layout}")
    print(f"  kv write   : {bool(cfg.get('kv_write', False))}")
    print(f"  variants   : {variants}")
    print(f"  shapes     : {len(entries)}")
    print(f"  kv buckets : {limits['SPYRE_ATTN_KV_BUCKETS']}")
    print(f"  q buckets  : {limits['SPYRE_ATTN_QUERY_BUCKETS']}")
    print(f"  seq buckets: {limits['SPYRE_ATTN_NUM_SEQS_BUCKETS']}")
    print(f"  iterations : {cfg.get('iterations', 10)} (warmup {cfg.get('warmup', 2)})")
    print(f"  output     : {out_dir if not args.no_output else 'none'}\n", flush=True)

    records = []
    guarded = False
    block_sizes = cfg.get("block_sizes") or [cfg["block_size"]]
    # The builder asserts its kv_cache_spec against cache_config.block_size, which is
    # fixed per config, so a block-size sweep needs one config context per size.
    for bsz in block_sizes:
        with spyre_vllm_config(next(iter(compiled_modes)), bsz, limits) as vllm_config:
            import torch._dynamo

            # The kernel specializes per (num_blocks, aligned_max_query_len), so a
            # sweep legitimately blows through the default recompile limit.
            torch._dynamo.config.accumulated_recompile_limit = 4096
            torch._dynamo.config.recompile_limit = 4096

            from spyre_inference.v1.attention.backends import spyre_attn as sa

            if not sa._ATTN_PROFILING:
                raise SystemExit("SPYRE_ATTN_PROFILING was not honoured; span will be absent.")
            print(
                f"  limits     : max_model_len={vllm_config.model_config.max_model_len} "
                f"max_num_batched_tokens="
                f"{vllm_config.scheduler_config.max_num_batched_tokens} "
                f"max_num_seqs={vllm_config.scheduler_config.max_num_seqs}",
                flush=True,
            )

            if not guarded:
                run_startup_guard(cfg, entries, bsz, sel_span, args)
                guarded = True

            for variant in variants:
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
