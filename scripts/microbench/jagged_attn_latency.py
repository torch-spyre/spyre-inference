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

"""Compare warmed head-major and jagged attention without a device profiler."""

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import warnings
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCENARIOS = ("decode1", "decode4", "prefill512", "mixed512", "jagged69")


def request_lengths(scenario, context):
    if scenario == "decode1":
        return [1], [context]
    if scenario == "decode4":
        return [1] * 4, [
            context,
            7 * context // 8 - 13,
            3 * context // 4 + 7,
            5 * context // 8 + 17,
        ]
    if scenario == "prefill512":
        return [512], [context]
    if scenario == "mixed512":
        return [1, 511], [context, 3 * context // 4 + 17]
    if scenario == "jagged69":
        return [1, 65, 3], [context, 3 * context // 4 + 17, context // 2 + 33]
    raise ValueError(scenario)


def reference_attention(query, keys, values, block_table, query_lens, seq_lens):
    """Float32 dense causal attention, independent of either backend's schedule."""
    import torch

    _, kv_heads, block_size, head_size = keys.shape
    groups = query.shape[1] // kv_heads
    output = torch.empty((sum(query_lens), *query.shape[1:]), dtype=torch.float32)
    offset = 0
    for seq, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens, strict=True)):
        page_ids = block_table[seq, : (seq_len + block_size - 1) // block_size].long()
        positions = torch.arange(seq_len)
        for kv_head in range(kv_heads):
            k = keys[page_ids, kv_head].reshape(-1, head_size)[:seq_len].float()
            v = values[page_ids, kv_head].reshape(-1, head_size)[:seq_len].float()
            heads = slice(kv_head * groups, (kv_head + 1) * groups)
            for start in range(0, query_len, 64):
                end = min(start + 64, query_len)
                q = query[offset + start : offset + end, heads].float().transpose(0, 1)
                scores = (q @ k.T) * head_size**-0.5
                q_positions = seq_len - query_len + torch.arange(start, end)
                scores.masked_fill_(positions[None, :] > q_positions[:, None], float("-inf"))
                output[offset + start : offset + end, heads] = (scores.softmax(-1) @ v).transpose(
                    0, 1
                )
        offset += query_len
    return output


def check_output(actual, expected):
    import torch

    actual = actual.float()
    error = actual - expected
    finite = bool(torch.isfinite(actual).all())
    relative_l2 = float(error.norm() / expected.norm()) if finite else None
    outliers = error.abs() > 0.002 + 0.02 * expected.abs()
    outlier_count = int(outliers.sum())
    result = {
        "finite": finite,
        "max_abs_error": float(error.abs().max()) if finite else None,
        "relative_l2_error": relative_l2,
        "atol": 0.002,
        "rtol": 0.02,
        "max_relative_l2": 0.02,
        "elementwise_outliers": outlier_count,
        "allowed_outliers": 5,
        "outlier_atol": 0.004,
        "outlier_rtol": 0.04,
    }
    try:
        assert finite, "output contains nonfinite values"
        # Follow the existing attention tests' bounded-outlier policy, while
        # retaining a tighter base tolerance and an independent global check.
        if outlier_count > 5:
            torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)
        elif outlier_count:
            torch.testing.assert_close(actual[outliers], expected[outliers], atol=0.004, rtol=0.04)
        assert relative_l2 is not None and relative_l2 <= 0.02, result
    except AssertionError as exc:
        result.update(passed=False, error=str(exc))
    else:
        result["passed"] = True
    return result


def percentile(samples, quantile):
    ordered = sorted(samples)
    index = (len(ordered) - 1) * quantile
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def save_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_manifest(module):
    root = Path(module.__file__).resolve().parent.parent
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"])
    sources = sorted(Path(module.__file__).resolve().parent.rglob("*.py"))
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    return {
        "root": str(root),
        "commit": commit,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "python_sources_sha256": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384, 32768]
    )
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=930)
    parser.add_argument("--reference-device-values", action="store_true")
    parser.add_argument(
        "--jagged-schedule",
        choices=("serving", "flat", "nested", "page_parallel", "split", "mixed"),
        default="serving",
    )
    parser.add_argument(
        "--mixed-decode-schedule", choices=("page_parallel", "split"), default="page_parallel"
    )
    parser.add_argument(
        "--split-pages-per-partition",
        type=int,
        choices=(1, 2, 4, 8, 16, 32),
        default=1,
        help="Maximum serial pages per partial state in two-kernel decode",
    )
    parser.add_argument("--jagged-parallel-entries", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument(
        "--output-buffer",
        choices=("staging", "model"),
        default="staging",
        help="Output buffer: paired staging or model token buckets for pure decode",
    )
    parser.add_argument(
        "--jagged-query-tile-size",
        type=int,
        choices=(0, 1, 64, 128, 256, 512),
        default=0,
        help="Legacy tile width; 0 selects automatically. Serving always uses mixed widths",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output_buffer == "model" and any(
        s not in ("decode1", "decode4") for s in args.scenarios
    ):
        parser.error("model output buffers require pure decode; mixed/prefill use paired staging")
    if args.jagged_schedule == "flat" and args.jagged_query_tile_size == 1:
        parser.error("one-row query tiles require nested or page_parallel attention")
    if args.jagged_schedule in ("page_parallel", "split") and (
        args.jagged_query_tile_size not in (0, 1)
        or any(s not in ("decode1", "decode4") for s in args.scenarios)
    ):
        parser.error("page_parallel/split require decode scenarios and query tile width 0 or 1")
    if not (args.warmup >= 2 and args.samples >= 3):
        parser.error("at least two warmups and three samples are required")
    if min(args.contexts) < 1024 or max(args.contexts) > args.max_model_len:
        parser.error("contexts must be between 1024 and max-model-len")
    if args.num_heads % args.num_kv_heads or args.head_size % 64 or args.block_size % 64:
        parser.error("heads must divide by KV heads; head and block sizes must be stick aligned")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    os.environ["SPYRE_NUM_CPUS"] = "8"
    os.environ["SPYRE_ATTN_PROFILING"] = "0"
    os.environ.setdefault("VLLM_PLUGINS", "spyre_inference")
    os.environ.setdefault("DXP_LOOP_UNROLL", "0")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "8"
    for name, value in (
        ("RANK", "0"),
        ("LOCAL_RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_WORLD_SIZE", "1"),
    ):
        os.environ.setdefault(name, value)

    import torch
    import torch_spyre
    from torch._dynamo.utils import counters
    from torch_spyre.ops.fallbacks import FallbackWarning
    from vllm.config import CompilationMode
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import AttentionSpec

    import spyre_inference
    from spyre_inference import envs
    from spyre_inference.custom_ops import register_all
    from spyre_inference.custom_ops.utils import convert, row_outermost_layout
    from spyre_inference.threading_config import configure_threading
    from spyre_inference.v1.attention.backends import spyre_attn, spyre_head_major_attn
    from spyre_inference.v1.attention.jagged_plan import (
        build_jagged_decode_plan,
        build_jagged_mixed_plan,
        build_jagged_plan,
        build_jagged_tile_plan,
        partition_jagged_decode_plan,
    )
    from spyre_inference.v1.attention.ops.jagged_decode_attn import jagged_decode_attn_kernel
    from spyre_inference.v1.attention.ops.jagged_mixed_attn import jagged_join_outputs
    from spyre_inference.v1.attention.ops.jagged_split_decode import (
        jagged_decode_merge_kernel,
        jagged_decode_partials_kernel,
    )
    from spyre_inference.v1.attention.ops.jagged_tile_attn import jagged_tile_attn_kernel
    from spyre_inference.v1.attention.ops.layout import head_major_kv_layout

    torch_spyre._autoload()
    register_all()
    configure_threading(1)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    dependencies = subprocess.check_output(["ldd", torch_spyre._C.__file__], text=True)
    if "libaiupti" in dependencies:
        raise RuntimeError("latency requires torch-spyre built with USE_SPYRE_PROFILER=0")
    if not envs.SPYRE_ATTN_FOR_EACH_TILE or not envs.SPYRE_BATCHED_DECODE:
        raise RuntimeError("the comparison requires for_each_tile and the default batched decode")
    if any(
        (
            envs.SPYRE_ATTN_KV_BUCKETS,
            envs.SPYRE_ATTN_QUERY_BUCKETS,
            envs.SPYRE_ATTN_NUM_SEQS_BUCKETS,
        )
    ):
        raise RuntimeError("unset attention bucket overrides to compare the default padding policy")
    if envs.SPYRE_ATTN_MAX_CORES:
        raise RuntimeError("unset SPYRE_ATTN_MAX_CORES to preserve default core caps")

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type="generate",
            max_model_len=args.max_model_len,
            dtype=torch.float16,
            model_arch_config=SimpleNamespace(per_layer_overrides=None),
            get_num_attention_heads=lambda _: args.num_heads,
            get_num_kv_heads=lambda _: args.num_kv_heads,
        ),
        cache_config=SimpleNamespace(block_size=args.block_size),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512, max_num_seqs=4),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.STOCK_TORCH_COMPILE, static_forward_context={}
        ),
    )
    report = {
        "started_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": {**vars(args), "output": str(args.output)},
        "metric": "synchronized warmed backend forward wall latency, milliseconds",
        "excluded": [
            "KV insertion",
            "CPU metadata build",
            "metadata transfers",
            "query staging",
            "compilation",
        ],
        "sources": {
            "spyre_inference": source_manifest(spyre_inference),
            "torch_spyre": source_manifest(torch_spyre),
        },
        "native_extension": {
            "path": torch_spyre._C.__file__,
            "sha256": hashlib.sha256(Path(torch_spyre._C.__file__).read_bytes()).hexdigest(),
            "ldd": dependencies,
        },
        "torch_version": torch.__version__,
        "python": sys.executable,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("SPYRE_", "TORCH_SPYRE_", "DXP_", "SEN", "DEEPTOOLS_", "RUNTIME_"))
            or k
            in (
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
                "TORCHINDUCTOR_CACHE_DIR",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "rows": [],
    }
    save_report(args.output, report)
    device = torch.device("spyre")
    torch.spyre.set_device(0)
    torch.spyre.synchronize()
    spec = AttentionSpec(
        block_size=args.block_size,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        dtype=torch.float16,
    )
    generator = torch.Generator().manual_seed(args.seed)
    pages_per_seq = (args.max_model_len + args.block_size - 1) // args.block_size
    num_pages = 4 * pages_per_seq + 1
    table = torch.randperm(num_pages - 1, generator=generator, dtype=torch.int32).reshape(
        4, pages_per_seq
    )
    shape = (num_pages, args.num_kv_heads, args.block_size, args.head_size)
    print(f"Allocating shared KV cache {shape}, 513 query staging rows", flush=True)
    keys = torch.randn(shape, generator=generator, dtype=torch.float16)
    values = torch.randn(shape, generator=generator, dtype=torch.float16)
    query = torch.randn(
        (513, args.num_heads, args.head_size), generator=generator, dtype=torch.float16
    )
    report["cache_shape"] = list(shape)
    report["query_shape"] = list(query.shape)
    cache_layout = head_major_kv_layout(
        num_pages * args.num_kv_heads, args.block_size, args.head_size, torch.float16
    )

    with ExitStack() as stack:
        plan_builder = {
            "serving": build_jagged_mixed_plan,
            "flat": build_jagged_plan,
            "nested": build_jagged_tile_plan,
            "page_parallel": build_jagged_decode_plan,
            "split": build_jagged_decode_plan,
            "mixed": build_jagged_mixed_plan,
        }[args.jagged_schedule]

        def build_plan(*inputs, **kwargs):
            kwargs.pop("max_parallel_entries", None)
            if args.jagged_schedule == "serving":
                return plan_builder(
                    *inputs, **kwargs, max_parallel_entries=args.jagged_parallel_entries
                )
            if args.jagged_schedule == "flat":
                kwargs.pop("workspace", None)
            width = args.jagged_query_tile_size
            if args.jagged_schedule == "mixed":
                plan = plan_builder(
                    *inputs,
                    **kwargs,
                    query_tile_size=width,
                    max_parallel_entries=args.jagged_parallel_entries,
                )
                if args.mixed_decode_schedule == "split":
                    plan = replace(
                        plan,
                        groups=tuple(
                            partition_jagged_decode_plan(group, args.split_pages_per_partition)
                            if group.page_indices.ndim == 4
                            else group
                            for group in plan.groups
                        ),
                    )
                return plan
            if not width:
                lengths = (inputs[0][1:] - inputs[0][:-1]).tolist()
                if max(lengths, default=0) <= 1:
                    width = 64 if args.jagged_schedule == "flat" else 1
                else:
                    best_rows = None
                    for candidate in (64, 128, 256, 512):
                        if candidate >= kwargs["query_capacity"]:
                            break
                        tiles = sum((length + candidate - 1) // candidate for length in lengths)
                        padded_rows = candidate * (1 << (max(1, tiles) - 1).bit_length())
                        if best_rows is None or padded_rows <= best_rows:
                            width, best_rows = candidate, padded_rows
            if args.jagged_schedule in ("split", "page_parallel"):
                kwargs["max_parallel_entries"] = args.jagged_parallel_entries
            plan = plan_builder(*inputs, **kwargs, query_tile_size=width)
            if args.jagged_schedule == "split":
                plan = partition_jagged_decode_plan(plan, args.split_pages_per_partition)
            return plan

        stack.enter_context(patch.object(spyre_attn, "build_jagged_mixed_plan", build_plan))
        if args.jagged_schedule in ("split", "mixed"):
            partials_compiled = torch.compile(
                jagged_decode_partials_kernel, dynamic=False, fullgraph=True
            )
            merge_compiled = torch.compile(
                jagged_decode_merge_kernel, dynamic=False, fullgraph=True
            )

            def split_decode(query, k, v, q_ids, out_ids, p_ids, bounds, offsets, scale, **kwargs):
                partials = partials_compiled(
                    query,
                    k,
                    v,
                    q_ids,
                    p_ids,
                    bounds,
                    offsets,
                    scale,
                    reduce_pages=args.split_pages_per_partition > 1,
                    **kwargs,
                )
                return merge_compiled(*partials, out_ids, query.shape[0], offsets.shape[-1])

            if args.jagged_schedule == "mixed":
                decode_compiled = torch.compile(
                    jagged_decode_attn_kernel, dynamic=False, fullgraph=True
                )
                prefill_compiled = torch.compile(
                    jagged_tile_attn_kernel, dynamic=False, fullgraph=True
                )
                join_compiled = torch.compile(jagged_join_outputs, dynamic=False, fullgraph=True)

                def mixed_attention(query, k, v, *tables_and_scale, **kwargs):
                    tables, scale = tables_and_scale[:-1], tables_and_scale[-1]
                    outputs = []
                    for offset in range(0, len(tables), 5):
                        group = tables[offset : offset + 5]
                        if group[2].ndim == 4:
                            kernel = (
                                split_decode
                                if args.mixed_decode_schedule == "split"
                                else decode_compiled
                            )
                        else:
                            kernel = prefill_compiled
                        outputs.append(kernel(query, k, v, *group, scale, **kwargs))
                    return (
                        outputs[0]
                        if len(outputs) == 1
                        else join_compiled(tuple(outputs), query.shape[0])
                    )

                selected_kernel = mixed_attention
            else:
                selected_kernel = split_decode
            stack.enter_context(patch.object(spyre_attn, "_jagged_attn_compiled", selected_kernel))
        elif args.jagged_schedule not in ("flat", "serving"):
            kernel = (
                jagged_tile_attn_kernel
                if args.jagged_schedule == "nested"
                else jagged_decode_attn_kernel
            )
            stack.enter_context(
                patch.object(
                    spyre_attn,
                    "_jagged_attn_compiled",
                    torch.compile(kernel, dynamic=False, fullgraph=True),
                )
            )
        stack.enter_context(patch.object(spyre_attn, "get_current_vllm_config", lambda: config))
        stack.enter_context(
            patch.object(spyre_head_major_attn, "get_current_vllm_config", lambda: config)
        )
        stack.enter_context(torch.inference_mode())
        stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("error", FallbackWarning)
        stack.enter_context(
            torch._dynamo.config.patch(capture_scalar_outputs=True, recompile_limit=64)
        )
        cache = spyre_attn.SpyrePagedKVCache(
            convert(keys, device, device_layout=cache_layout),
            convert(values, device, device_layout=cache_layout),
        )
        query_device = convert(
            query, device, device_layout=row_outermost_layout(query.shape, query.dtype)
        )
        if args.reference_device_values:
            realized = (query_device.cpu(), cache.k_pages.cpu(), cache.v_pages.cpu())
            report["input_roundtrip"] = {
                name: {
                    "max_abs_error": float((actual - original).abs().max()),
                    "changed_elements": int((actual != original).sum()),
                    "total_elements": original.numel(),
                }
                for name, original, actual in zip(
                    ("query", "keys", "values"), (query, keys, values), realized, strict=True
                )
            }
            print(f"INPUT_ROUNDTRIP {report['input_roundtrip']}", flush=True)
            query, keys, values = realized
        legs = {}
        for name in ("baseline", "jagged"):
            with patch.object(envs, "SPYRE_JAGGED_ATTENTION", name == "jagged"):
                builder = spyre_attn.SpyreAttentionMetadataBuilder(
                    spec, [], config, torch.device("cpu")
                )
                impl = spyre_head_major_attn.SpyreHeadMajorAttentionImpl(
                    num_heads=args.num_heads,
                    num_kv_heads=args.num_kv_heads,
                    head_size=args.head_size,
                    scale=args.head_size**-0.5,
                )
                if name == "jagged" and args.jagged_schedule != "serving":
                    impl._jagged_direct_output = False
                    impl.staging_output_rows = impl.staging_rows
            q_staging, out_staging = impl.staging_buffers(device)
            q_staging.copy_(query_device)
            legs[name] = (builder, impl, q_staging, out_staging)
        torch.spyre.synchronize()
        report["buckets"] = {
            "query": builder.attn_bucketer.query_buckets,
            "blocks": builder.attn_bucketer.num_blocks_buckets,
            "sequences": builder.attn_bucketer.num_seqs_buckets,
        }

        for context in args.contexts:
            for scenario in args.scenarios:
                q_lens, seq_lens = request_lengths(scenario, context)
                starts = torch.tensor(
                    [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32
                )
                lengths = torch.tensor(seq_lens, dtype=torch.int32)
                slots = torch.cat(
                    [
                        table[s, positions // args.block_size].long() * args.block_size
                        + positions % args.block_size
                        for s, (ql, sl) in enumerate(zip(q_lens, seq_lens, strict=True))
                        for positions in [torch.arange(sl - ql, sl)]
                    ]
                )
                common = CommonAttentionMetadata(
                    query_start_loc=starts,
                    query_start_loc_cpu=starts,
                    seq_lens=lengths,
                    num_reqs=len(q_lens),
                    num_actual_tokens=sum(q_lens),
                    max_query_len=max(q_lens),
                    max_seq_len=max(seq_lens),
                    block_table_tensor=table[: len(q_lens)],
                    slot_mapping=slots,
                    causal=True,
                )
                print(
                    f"CASE {scenario} context={context} queries={q_lens} lengths={seq_lens}",
                    flush=True,
                )
                expected = reference_attention(query, keys, values, table, q_lens, seq_lens)
                active = {}
                for name, (builder, impl, q_staging, out_staging) in legs.items():
                    if args.output_buffer == "model":
                        count = sum(q_lens)
                        rows = 1 << (count - 1).bit_length()
                        output_shape = (rows, args.num_heads, args.head_size)
                        out_staging = convert(
                            torch.zeros(output_shape, dtype=query.dtype),
                            device,
                            device_layout=row_outermost_layout(output_shape, query.dtype),
                        )
                    row = {
                        "scenario": scenario,
                        "context": context,
                        "query_lens": q_lens,
                        "seq_lens": seq_lens,
                        "backend": name,
                        "output_rows": out_staging.shape[0],
                        "samples_ms": [],
                    }
                    report["rows"].append(row)
                    before = time.perf_counter()
                    with patch.object(envs, "SPYRE_JAGGED_ATTENTION", name == "jagged"):
                        metadata = builder.build(0, common)
                    row["metadata_build_ms"] = (time.perf_counter() - before) * 1000
                    if name == "jagged" and args.jagged_schedule in ("mixed", "serving"):
                        plan = metadata.jagged_plan
                        group_details = []
                        for group in plan.groups:
                            if group.page_indices.ndim == 4:
                                groups, chunks, entries, _ = group.page_indices.shape
                                group_details.append(
                                    {
                                        "kind": args.mixed_decode_schedule,
                                        "query_tile_size": 1,
                                        "work_capacity": groups * chunks * entries,
                                        "page_loop_iterations": groups * chunks,
                                    }
                                )
                            else:
                                group_details.append(
                                    {
                                        "kind": "nested",
                                        "query_tile_size": group.query_indices.shape[1],
                                        "work_capacity": group.page_indices.shape[0]
                                        * group.page_indices.shape[1],
                                        "page_loop_iterations": group.page_indices.shape[0]
                                        * group.page_indices.shape[1],
                                    }
                                )
                        row.update(
                            dispatch=" + ".join(group["kind"] for group in group_details),
                            groups=group_details,
                            work_items=sum(group.num_work_items for group in plan.groups),
                            work_capacity=sum(group["work_capacity"] for group in group_details),
                        )
                    elif name == "jagged":
                        plan = metadata.jagged_plan
                        row["dispatch"] = {
                            "flat": "jagged_page_attn_kernel",
                            "nested": "jagged_tile_attn_kernel",
                            "page_parallel": "jagged_decode_attn_kernel",
                            "split": "jagged_decode_partials_kernel + jagged_decode_merge_kernel",
                        }[args.jagged_schedule]
                        row["work_items"] = plan.num_work_items
                        row["work_capacity"] = plan.query_indices.shape[0]
                        row["query_tile_size"] = plan.output_indices.shape[1]
                        if args.jagged_schedule in ("page_parallel", "split"):
                            tasks, chunks, entries, _ = plan.page_indices.shape
                            groups = plan.output_indices.shape[0]
                            queries_per_group = plan.output_indices.shape[1]
                            slots = entries // queries_per_group
                            row.update(
                                query_tile_size=1,
                                query_tile_capacity=groups * queries_per_group,
                                query_group_capacity=groups,
                                queries_per_group=queries_per_group,
                                page_capacity=tasks * chunks * slots // groups,
                                page_chunk_capacity=chunks,
                                parallel_page_slots=slots,
                                parallel_entries=entries,
                                page_loop_iterations=tasks * chunks,
                                work_capacity=tasks * chunks * entries,
                            )
                            if args.jagged_schedule == "split":
                                pages_per_partial = (
                                    chunks if args.split_pages_per_partition > 1 else 1
                                )
                                row["pages_per_partial"] = pages_per_partial
                                row["partial_task_capacity"] = tasks
                                row["partial_state_logical_bytes"] = (
                                    tasks
                                    * chunks
                                    // pages_per_partial
                                    * entries
                                    * args.num_heads
                                    * (args.head_size + 2)
                                    * 2
                                )
                        elif args.jagged_schedule == "nested":
                            row["query_tile_capacity"] = plan.query_indices.shape[0]
                            row["page_capacity"] = plan.page_indices.shape[1]
                            row["work_capacity"] *= row["page_capacity"]
                    else:
                        batched = impl._batched_decode_preconditions_met(metadata, num_pages)
                        remaining_queries = metadata.aligned_query_lens[
                            metadata.num_decode_seqs if batched else 0 :
                        ]
                        dispatch = ["batched_decode_head_major_kernel"] if batched else []
                        dispatch.extend(
                            "page_attn_head_major_prefill_kernel"
                            if ql > 1
                            else "page_attn_head_major_decode_kernel"
                            for ql in remaining_queries
                        )
                        row["dispatch"] = " + ".join(dispatch)
                        row["aligned_query_lens"] = metadata.aligned_query_lens
                        row["padded_num_blocks"] = metadata.padded_num_blocks
                        row["padded_num_seqs"] = metadata.padded_num_seqs
                        row["blocks_per_chunk"] = metadata.blocks_per_chunk
                    print(
                        f"WARM {name} {scenario} {context} dispatch={row['dispatch']} "
                        f"work={row.get('work_capacity')}",
                        flush=True,
                    )
                    save_report(args.output, report)
                    graphs = counters["stats"]["unique_graphs"]
                    before = time.perf_counter()
                    for _ in range(args.warmup):
                        impl.forward(None, q_staging, None, None, cache, metadata, out_staging)
                        torch.spyre.synchronize()
                    row["warmup_seconds"] = time.perf_counter() - before
                    row["warmup_graphs"] = counters["stats"]["unique_graphs"] - graphs
                    row["correctness"] = check_output(out_staging.cpu()[: sum(q_lens)], expected)
                    print(
                        f"READY {name} warmup={row['warmup_seconds']:.2f}s "
                        f"correctness={row['correctness']}",
                        flush=True,
                    )
                    if row["correctness"]["passed"]:
                        active[name] = (impl, q_staging, out_staging, metadata, row)
                    save_report(args.output, report)

                before_graphs = counters["stats"]["unique_graphs"]
                gc.collect()
                gc.disable()
                try:
                    for iteration in range(args.samples):
                        order = list(active) if iteration % 2 == 0 else list(reversed(active))
                        for name in order:
                            impl, q_staging, out_staging, metadata, row = active[name]
                            torch.spyre.synchronize()
                            before = time.perf_counter_ns()
                            impl.forward(None, q_staging, None, None, cache, metadata, out_staging)
                            torch.spyre.synchronize()
                            row["samples_ms"].append((time.perf_counter_ns() - before) / 1e6)
                finally:
                    gc.enable()
                    save_report(args.output, report)
                graph_delta = counters["stats"]["unique_graphs"] - before_graphs
                for name, (_, _, out_staging, _, row) in active.items():
                    row["measured_new_graphs"] = graph_delta
                    row["post_timing_correctness"] = check_output(
                        out_staging.cpu()[: sum(q_lens)], expected
                    )
                    row["valid_timing"] = (
                        graph_delta == 0 and row["post_timing_correctness"]["passed"]
                    )
                    if row["valid_timing"]:
                        samples = row["samples_ms"]
                        row.update(
                            median_ms=statistics.median(samples),
                            p90_ms=percentile(samples, 0.9),
                            min_ms=min(samples),
                            max_ms=max(samples),
                            spread=max(samples) / min(samples),
                        )
                        print(
                            f"RESULT {scenario} {context} {name}: "
                            f"median={row['median_ms']:.3f}ms p90={row['p90_ms']:.3f}ms "
                            f"new_graphs={graph_delta}",
                            flush=True,
                        )
                save_report(args.output, report)
                if graph_delta:
                    raise RuntimeError("unexpected compilation during the measured window")
        torch.spyre.synchronize()
    report["finished_utc"] = datetime.now(UTC).isoformat()
    save_report(args.output, report)


if __name__ == "__main__":
    main()
