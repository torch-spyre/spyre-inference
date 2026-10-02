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

"""Compare complete decoder steps using the real vLLM model and attention builders."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from jagged_attn_latency import SCENARIOS, request_lengths

HISTORY_TEXTS = (
    "A computer stores information in memory and follows instructions to process it. "
    "Programs divide larger tasks into smaller operations. Testing checks that these "
    "operations produce the expected results for different inputs.\n",
    "Water moves through rivers, lakes, oceans, and the atmosphere. Sunlight causes "
    "water to evaporate, and clouds form as the air cools. Rain and snow return water "
    "to the land, where plants and animals depend on it.\n",
    "A library collects books on many subjects. Readers can search the catalogue, "
    "borrow a book, and return it after reading. Libraries also provide quiet places "
    "to study and help people find reliable information.\n",
    "A garden changes with the seasons. Seeds need suitable soil, water, and light "
    "to grow. Gardeners observe the plants, remove weeds, and harvest vegetables "
    "when they are ready. Each season offers a chance to learn.\n",
)


def _summary(samples):
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "min_ms": min(samples),
        "max_ms": max(samples),
        "spread": max(samples) / min(samples),
        "samples_ms": samples,
    }


def _worker_info(worker):
    import torch
    import torch_spyre

    import spyre_inference
    from spyre_inference.v1.attention import jagged_plan
    from spyre_inference.v1.attention.backends import spyre_attn
    from spyre_inference.v1.worker import spyre_model_runner

    dependencies = subprocess.check_output(["ldd", torch_spyre._C.__file__], text=True)
    if "libaiupti" in dependencies:
        raise RuntimeError("latency requires torch-spyre built without profiler instrumentation")
    runner = worker.model_runner
    caches = runner._spyre_kv_caches
    sources = {}
    for module in (torch_spyre, spyre_inference, jagged_plan, spyre_attn, spyre_model_runner):
        path = Path(module.__file__).resolve()
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "rank": worker.rank,
        "torch_version": torch.__version__,
        "sources": sources,
        "torch_num_threads": torch.get_num_threads(),
        "compile_sizes": runner.compilation_config.compile_sizes,
        "num_attention_layers": len(caches),
        "num_attention_groups": len({id(b) for b in runner._attn_metadata_builders().values()}),
        "kv_cache_shapes": {name: list(cache[0].shape) for name, cache in caches.items()},
        "logical_kv_bytes": sum(t.numel() * t.element_size() for c in caches.values() for t in c),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "SPYRE_DEVICES",
                "SPYRE_NUM_CPUS",
                "SPYRE_JAGGED_ATTENTION",
                "SPYRE_JAGGED_PARALLEL_ENTRIES",
                "SPYRE_ATTN_RECORD",
                "SPYRE_COMPILE_GRANULARITY",
                "SPYRE_COMPILE_GUARD",
                "DXP_LOOP_UNROLL",
                "SPYRE_KERNEL_CACHE",
                "TORCHINDUCTOR_CACHE_DIR",
            )
        },
    }


def _replay_case(worker, spec):
    import numpy as np
    import torch
    from torch._dynamo.utils import counters
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backend import CommonAttentionMetadata

    from spyre_inference.v1.worker import compile_guard
    from spyre_inference.v1.worker.spyre_model_runner import _set_spyre_compilation_settings

    runner = worker.model_runner
    config = runner.vllm_config
    block_size = config.cache_config.block_size
    query_lens, seq_lens = spec["query_lens"], spec["seq_lens"]
    num_tokens = sum(query_lens)
    num_reqs = len(query_lens)
    padded_rows = runner.spyre_shape_bucketer.find_bucket(num_tokens)
    assert padded_rows is not None
    blocks_per_seq = (config.model_config.max_model_len + block_size - 1) // block_size
    page_rows = spec.get("page_rows", num_reqs)
    pages = torch.arange(1, page_rows * blocks_per_seq + 1, dtype=torch.int32)
    # A deterministic permutation exercises indirect gathers without touching null page 0.
    pages = pages.flip(0).reshape(page_rows, blocks_per_seq)[:num_reqs].contiguous()
    assert max(int(pages.max()), 0) < min(c[0].shape[0] for c in runner._spyre_kv_caches.values())
    positions = torch.zeros(padded_rows, dtype=torch.int64)
    positions[:num_tokens] = torch.cat(
        [torch.arange(sl - ql, sl) for ql, sl in zip(query_lens, seq_lens, strict=True)]
    )
    input_ids = torch.zeros(padded_rows, dtype=torch.int32)
    if "input_ids" in spec:
        input_ids[:num_tokens] = torch.tensor(spec["input_ids"], dtype=torch.int32)
    else:
        input_ids[:num_tokens] = (torch.arange(num_tokens, dtype=torch.int32) * 17 + 42) % 10000
    starts_list = [0]
    for length in query_lens:
        starts_list.append(starts_list[-1] + length)
    logits_rows = torch.tensor(starts_list[1:], dtype=torch.int64) - 1
    layer_builders = runner._attn_metadata_builders()
    builder_groups = {}
    for name, builder in layer_builders.items():
        builder_groups.setdefault(id(builder), (builder, []))[1].append(name)

    def step():
        begin = time.perf_counter_ns()
        starts = torch.tensor(starts_list, dtype=torch.int32)
        lengths = torch.tensor(seq_lens, dtype=torch.int32)
        slots = torch.zeros(padded_rows, dtype=torch.int64)
        for seq, (lo, hi) in enumerate(zip(starts_list[:-1], starts_list[1:], strict=True)):
            p = positions[lo:hi]
            slots[lo:hi] = pages[seq, p // block_size].long() * block_size + p % block_size
        common = CommonAttentionMetadata(
            query_start_loc=starts,
            query_start_loc_cpu=starts,
            seq_lens=lengths,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max(query_lens),
            max_seq_len=max(seq_lens),
            block_table_tensor=pages,
            slot_mapping=slots,
            causal=True,
        )
        metadata = {}
        for builder, names in builder_groups.values():
            built = builder.build(0, common)
            metadata.update((name, built) for name in names)
        metadata_done = time.perf_counter_ns()
        with set_forward_context(
            metadata,
            config,
            num_tokens=padded_rows,
            slot_mapping={name: slots for name in layer_builders},
        ):
            hidden = runner.model(input_ids=input_ids, positions=positions)
            logits = runner.model.compute_logits(hidden[logits_rows])
        torch.spyre.synchronize()
        end = time.perf_counter_ns()
        return logits, (metadata_done - begin) / 1e6, (end - begin) / 1e6

    with (
        torch.inference_mode(),
        set_current_vllm_config(config),
        _set_spyre_compilation_settings(config),
        torch._dynamo.config.patch(capture_scalar_outputs=True),
    ):
        torch.spyre.synchronize()
        warm_started = time.monotonic()
        if spec["explicit_warmup"]:
            with compile_guard.allow_compile("explicit replay shape warmup"):
                for _ in range(spec["warmup"]):
                    step()
        else:
            # Recording claims to cover these shapes; the armed guard must stay active.
            for _ in range(spec["warmup"]):
                step()
        warm_seconds = time.monotonic() - warm_started
        before = counters["stats"]["unique_graphs"]
        metadata_samples, total_samples = [], []
        compile_guard.arm(compile_guard.CompileGuardLevel.ERROR)
        first_logits = None
        for _ in range(spec["samples"]):
            logits, metadata_ms, total_ms = step()
            metadata_samples.append(metadata_ms)
            total_samples.append(total_ms)
            if first_logits is None:
                first_logits = logits.clone()
        new_graphs = counters["stats"]["unique_graphs"] - before
        if new_graphs:
            raise RuntimeError(f"{new_graphs} graphs compiled inside the measured window")
        values = logits.float().cpu()
        finite = bool(torch.isfinite(values).all())
        repeat_error = float((values - first_logits.float().cpu()).abs().max())
        if not finite or repeat_error != 0:
            raise AssertionError(
                f"invalid replay logits: finite={finite}, repeat_error={repeat_error}"
            )
        path = Path(spec["artifact"]).with_suffix(f".rank{worker.rank}.npz")
        np.savez(
            path, logits=values.numpy(), input_ids=input_ids.numpy(), positions=positions.numpy()
        )
    return {
        "rank": worker.rank,
        "scenario": spec["scenario"],
        "context": spec["context"],
        "query_lens": query_lens,
        "seq_lens": seq_lens,
        "padded_model_rows": padded_rows,
        "warmup_seconds": warm_seconds,
        "new_graphs": new_graphs,
        "metadata_and_slot_publication": _summary(metadata_samples),
        "complete_step": _summary(total_samples),
        "logits_finite": finite,
        "repeat_max_abs_error": repeat_error,
        "greedy_tokens": values.argmax(-1).tolist(),
        "logits_artifact": str(path),
    }


def _prefill_history_chunk(worker, spec):
    import torch
    from torch._dynamo.utils import counters
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backend import CommonAttentionMetadata

    from spyre_inference.v1.worker import compile_guard
    from spyre_inference.v1.worker.spyre_model_runner import _set_spyre_compilation_settings

    runner = worker.model_runner
    config = runner.vllm_config
    block_size = config.cache_config.block_size
    n = len(spec["input_ids"])
    padded_rows = runner.spyre_shape_bucketer.find_bucket(n)
    assert padded_rows is not None
    page_rows = config.scheduler_config.max_num_seqs
    blocks_per_seq = (config.model_config.max_model_len + block_size - 1) // block_size
    pages = torch.arange(1, page_rows * blocks_per_seq + 1, dtype=torch.int32)
    pages = pages.flip(0).reshape(page_rows, blocks_per_seq)
    pages = pages[spec["request"] : spec["request"] + 1].contiguous()
    assert int(pages.max()) < min(c[0].shape[0] for c in runner._spyre_kv_caches.values())
    positions = torch.zeros(padded_rows, dtype=torch.int64)
    positions[:n] = torch.arange(spec["start"], spec["start"] + n)
    input_ids = torch.zeros(padded_rows, dtype=torch.int32)
    input_ids[:n] = torch.tensor(spec["input_ids"], dtype=torch.int32)
    slots = torch.zeros(padded_rows, dtype=torch.int64)
    slots[:n] = (
        pages[0, positions[:n] // block_size].long() * block_size + positions[:n] % block_size
    )
    starts = torch.tensor([0, n], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor([spec["start"] + n], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=n,
        max_query_len=n,
        max_seq_len=spec["start"] + n,
        block_table_tensor=pages,
        slot_mapping=slots,
        causal=True,
    )
    layer_builders = runner._attn_metadata_builders()
    built_groups, metadata = {}, {}
    with (
        torch.inference_mode(),
        set_current_vllm_config(config),
        _set_spyre_compilation_settings(config),
        torch._dynamo.config.patch(capture_scalar_outputs=True),
        compile_guard.allow_compile("prefill replay history"),
    ):
        begin = time.perf_counter_ns()
        before = counters["stats"]["unique_graphs"]
        for name, builder in layer_builders.items():
            if id(builder) not in built_groups:
                built_groups[id(builder)] = builder.build(0, common)
            metadata[name] = built_groups[id(builder)]
        with set_forward_context(
            metadata,
            config,
            num_tokens=padded_rows,
            slot_mapping={name: slots for name in layer_builders},
        ):
            runner.model(input_ids=input_ids, positions=positions)
        torch.spyre.synchronize()
    return {
        "rank": worker.rank,
        "elapsed_ms": (time.perf_counter_ns() - begin) / 1e6,
        "new_graphs": counters["stats"]["unique_graphs"] - before,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ibm-granite/granite-3.3-8b-instruct")
    parser.add_argument("--backend", choices=("baseline", "jagged"), required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=[1024, 32768])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=["decode1", "decode4", "mixed512", "jagged69"],
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--kv-cache-gib", type=float, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--record-attention", action="store_true")
    parser.add_argument(
        "--prefill-history",
        action="store_true",
        help="populate KV with model-prefilled repeated text before timing (reported separately)",
    )
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--generation-smoke", action="store_true")
    parser.add_argument("--generation-iters", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 1 or args.generation_iters < 1:
        parser.error("sample, warmup, and generation iteration counts must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    cases = []
    if not args.generation_only:
        for context in args.contexts:
            if not 1024 <= context <= args.max_model_len:
                parser.error("contexts must be between 1024 and max-model-len")
            for scenario in args.scenarios:
                ql, sl = request_lengths(scenario, context)
                if len(ql) > args.max_num_seqs or sum(ql) > args.max_num_batched_tokens:
                    parser.error(f"case {scenario} exceeds configured sequence/token budget")
                cases.append(dict(scenario=scenario, context=context, query_lens=ql, seq_lens=sl))
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "SPYRE_NUM_CPUS": "8",
        "SPYRE_ATTN_PROFILING": "0",
        "SPYRE_JAGGED_ATTENTION": str(int(args.backend == "jagged")),
        "SPYRE_ATTN_RECORD": str(int(args.record_attention)),
        "SPYRE_COMPILE_GUARD": "error" if args.record_attention else "off",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "300",
        "HF_HUB_OFFLINE": "1",
        # Only this local LLM instance receives our benchmark worker callables.
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    }.items():
        os.environ[name] = value
    os.environ.setdefault("VLLM_PLUGINS", "spyre_inference")
    os.environ.setdefault("SPYRE_JAGGED_PARALLEL_ENTRIES", "64")
    os.environ.setdefault("DXP_LOOP_UNROLL", "0")

    # Worker rank environment must be established before loading torch-spyre/comms.
    from vllm import LLM, SamplingParams

    compile_sizes = sorted(
        {1, args.max_num_seqs, args.max_num_batched_tokens}
        | {1 << i for i in range(args.max_num_seqs.bit_length()) if 1 << i <= args.max_num_seqs}
    )
    report = {
        "started_utc": datetime.now(UTC).isoformat(),
        "scope": __doc__,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compile_sizes": compile_sizes,
        "rows": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    start = time.monotonic()
    llm = LLM(
        model=args.model,
        enforce_eager=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        compilation_config={"compile_sizes": compile_sizes},
        enable_prefix_caching=False,
        kv_cache_memory_bytes=int(args.kv_cache_gib * 1024**3),
        seed=1234,
    )
    report["initialization_seconds"] = time.monotonic() - start
    report["workers"] = llm.collective_rpc(_worker_info, timeout=300)
    save()
    history_tokens = []
    history_filled = [0] * args.max_num_seqs
    if args.prefill_history and cases:
        tokenizer = llm.get_tokenizer()
        lengths = [0] * args.max_num_seqs
        for case in cases:
            for request, length in enumerate(case["seq_lens"]):
                rounded = (
                    (length + args.max_num_batched_tokens - 1)
                    // args.max_num_batched_tokens
                    * args.max_num_batched_tokens
                )
                lengths[request] = max(lengths[request], min(rounded, args.max_model_len))
        for request, length in enumerate(lengths):
            pattern = tokenizer.encode(
                HISTORY_TEXTS[request % len(HISTORY_TEXTS)], add_special_tokens=False
            )
            history_tokens.append(
                (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]
            )
        report["history"] = {
            "mode": "model-prefilled repeated text",
            "texts": HISTORY_TEXTS,
            "lengths": lengths,
            "tokens_sha256": hashlib.sha256(json.dumps(history_tokens).encode()).hexdigest(),
            "chunks": [],
            "elapsed_seconds": 0.0,
        }
        save()
    for case in cases:
        if history_tokens:
            for request, length in enumerate(case["seq_lens"]):
                for start in range(history_filled[request], length, args.max_num_batched_tokens):
                    chunk = history_tokens[request][start : start + args.max_num_batched_tokens]
                    history_start = time.monotonic()
                    rows = llm.collective_rpc(
                        _prefill_history_chunk,
                        timeout=300,
                        args=({"request": request, "start": start, "input_ids": chunk},),
                    )
                    report["history"]["elapsed_seconds"] += time.monotonic() - history_start
                    history_filled[request] = start + len(chunk)
                    report["history"]["chunks"].append(
                        {
                            "request": request,
                            "start": start,
                            "num_tokens": len(chunk),
                            "workers": rows,
                        }
                    )
                    save()
                    print(
                        f"HISTORY {args.backend} request={request} "
                        f"tokens={history_filled[request]}",
                        flush=True,
                    )
        print(f"CASE {args.backend} {case}", flush=True)
        spec = {
            **case,
            "warmup": args.warmup,
            "samples": args.samples,
            "explicit_warmup": not args.record_attention,
            "artifact": str(args.output.with_suffix("")) + f"-{case['scenario']}-{case['context']}",
        }
        if history_tokens:
            spec["page_rows"] = args.max_num_seqs
            spec["input_ids"] = [
                token
                for request, (ql, sl) in enumerate(
                    zip(case["query_lens"], case["seq_lens"], strict=True)
                )
                for token in history_tokens[request][sl - ql : sl]
            ]
        rows = llm.collective_rpc(_replay_case, timeout=300, args=(spec,))
        report["rows"].extend(rows)
        save()
        for row in rows:
            print(
                f"RESULT {args.backend} {case['scenario']} {case['context']} rank={row['rank']} "
                f"total={row['complete_step']['median_ms']:.3f}ms "
                f"metadata={row['metadata_and_slot_publication']['median_ms']:.3f}ms "
                f"new_graphs={row['new_graphs']}",
                flush=True,
            )
    if args.generation_smoke or args.generation_only:
        prompts = ["What are IBMs main businesses?", "The capital of France is"][
            : args.max_num_seqs
        ]
        sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
        results, samples = [], []
        for iteration in range(args.generation_iters + 2):
            start = time.perf_counter_ns()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            tokens = [output.outputs[0].token_ids for output in outputs]
            if results and tokens != results[0]:
                raise AssertionError("generation is not deterministic across repeated requests")
            results.append(tokens)
            if iteration >= 2:
                samples.append(elapsed)
        report["generation"] = {
            "prompts": prompts,
            "output_tokens": 16,
            "token_ids": results[-1],
            "texts": [output.outputs[0].text for output in outputs],
            "whole_generation": _summary(samples),
        }
        save()
        print(f"GENERATION {args.backend} {report['generation']}", flush=True)
    report["completed_utc"] = datetime.now(UTC).isoformat()
    save()


if __name__ == "__main__":
    main()
