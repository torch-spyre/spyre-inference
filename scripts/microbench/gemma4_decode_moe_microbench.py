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

"""Measure the compiled Gemma 4 decode MoE path at power-of-two batch sizes."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("SPYRE_NUM_CPUS", "8")
os.environ.setdefault("VLLM_PLUGINS", "spyre_inference")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.autograd.profiler import DeviceType  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

EXPERTS, HIDDEN, INTER, TOP_K = 128, 2816, 704, 8
DTYPE = torch.float16
SPAN = "gemma4_decode_moe"


def _device_layer(seed: int) -> tuple[SimpleNamespace, dict[str, torch.Tensor]]:
    from torch_spyre._C import get_elem_in_stick
    from torch_spyre._inductor import config as spyre_config

    from spyre_inference import envs, moe
    from spyre_inference.moe import (
        SpyreMoERecipe,
        _derive_moe_chunks,
        _down_chunk_pool_alias,
        _route_reduce_dtype,
        _to_spyre_expert_weight,
    )

    stick = get_elem_in_stick(DTYPE)
    chunks = _derive_moe_chunks(
        HIDDEN,
        INTER,
        TOP_K,
        stick,
        torch.empty((), dtype=DTYPE).element_size(),
        spyre_config.sencores,
        envs.SPYRE_MOE_CHUNKS,
    )
    if chunks is None:
        raise RuntimeError("the selected Gemma 4 MoE shape has no safe gathered chunk layout")
    generator = torch.Generator().manual_seed(seed)
    gate = torch.randn(EXPERTS, HIDDEN, INTER, dtype=DTYPE, generator=generator).mul_(0.01)
    up = torch.randn(EXPERTS, HIDDEN, INTER, dtype=DTYPE, generator=generator).mul_(0.01)
    down = torch.randn(EXPERTS, INTER, HIDDEN, dtype=DTYPE, generator=generator).mul_(0.01)
    down.mul_(torch.rand(EXPERTS, dtype=DTYPE, generator=generator)[:, None, None] + 0.5)

    gate_device = _to_spyre_expert_weight(gate, ())
    up_device = _to_spyre_expert_weight(up, ())
    # Keep one persistent down-weight allocation and expose decode chunks as a view.
    down_device = _to_spyre_expert_weight(down, (), kernel_order=True)
    gate_alias = moe._chunk_pool_alias(gate_device, chunks)
    up_alias = moe._chunk_pool_alias(up_device, chunks)
    down_alias = _down_chunk_pool_alias(down_device, chunks)

    layer = SimpleNamespace(
        spyre_moe_recipe=SpyreMoERecipe("gelu_tanh", "full_softmax"),
        spyre_moe_gate=gate_device,
        spyre_moe_up=up_device,
        spyre_moe_down=down_device,
        spyre_moe_down_alias=down_alias,
        spyre_moe_gate_alias=gate_alias,
        spyre_moe_up_alias=up_alias,
        spyre_moe_stick=stick,
        spyre_moe_route_dtype=_route_reduce_dtype(EXPERTS, DTYPE),
        spyre_moe_regions={},
        top_k=TOP_K,
        spyre_moe_chunks=chunks,
    )
    return layer, {"gate": gate, "up": up, "down": down}


def _dense_reference(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    probs = torch.softmax(router_logits.float(), dim=-1).to(router_logits.dtype)
    selected, indices = torch.topk(probs.float(), TOP_K, dim=-1)
    route_weights = selected / selected.sum(-1, keepdim=True)
    output = torch.zeros_like(x, dtype=torch.float32)
    for token in range(x.shape[0]):
        row = x[token : token + 1].float()
        for slot in range(TOP_K):
            expert = int(indices[token, slot])
            gate = F.gelu(row @ weights["gate"][expert].float(), approximate="tanh")
            up = row @ weights["up"][expert].float()
            output[token] += ((gate * up) @ weights["down"][expert].float()).squeeze(0) * float(
                route_weights[token, slot]
            )
    return output


def _profile_once(
    region: Any, args: tuple[torch.Tensor, ...]
) -> tuple[float, list[dict[str, float | str]]]:
    span_us: float | None = None
    with (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            record_shapes=False,
            acc_events=True,
        ) as prof,
        torch.profiler.record_function(SPAN),
    ):
        region(*args)
        torch.spyre.synchronize()

    for event in prof.events():
        if event.name == SPAN:
            span_us = event.time_range.elapsed_us()
            break
    if span_us is None:
        raise RuntimeError(f"profiler did not record {SPAN!r}")

    result = prof.profiler.kineto_results
    if result is None:
        raise RuntimeError("profiler did not produce Kineto results")
    device_events: list[dict[str, float | str]] = []
    for event in result.events():
        try:
            name = event.name()
        except UnicodeDecodeError:
            name = "<undecodable>"
        if event.device_type() == DeviceType.CPU:
            continue
        duration_us = event.duration_ns() / 1000
        if duration_us > 0:
            device_events.append({"name": name, "duration_us": duration_us})
    if not device_events:
        raise RuntimeError("no Spyre device events; check that torch-spyre includes AIUPTI")
    return span_us / 1000, device_events


def _stats(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
    }


def _run_batch(
    batch_size: int,
    layer: SimpleNamespace,
    weights: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from torch_spyre._inductor import config as spyre_config
    from torch_spyre.ops.fallbacks import FallbackWarning

    from spyre_inference.moe import _gathered_tokens

    generator = torch.Generator().manual_seed(args.seed + batch_size)
    x = torch.randn(batch_size, HIDDEN, dtype=DTYPE, generator=generator).mul_(0.05)
    router_logits = torch.randn(batch_size, EXPERTS, dtype=DTYPE, generator=generator)
    x_device, logits_device = x.to("spyre"), router_logits.to("spyre")
    device_args = (layer, x_device, logits_device)
    expected = _dense_reference(x, router_logits, weights)
    region = torch.compile(_gathered_tokens, backend="inductor", fullgraph=True, dynamic=False)

    with (
        spyre_config.patch({"frontend_pool_allocation": True}),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always", FallbackWarning)
        compile_start = time.perf_counter()
        actual = region(*device_args)
        torch.spyre.synchronize()
        compile_seconds = time.perf_counter() - compile_start
        torch.testing.assert_close(actual.cpu().float(), expected, atol=2e-2, rtol=2e-2)

        for _ in range(args.warmup_iters):
            region(*device_args)
            torch.spyre.synchronize()

        wall_ms = []
        for _ in range(args.iterations):
            torch.spyre.synchronize()
            start = time.perf_counter_ns()
            region(*device_args)
            torch.spyre.synchronize()
            wall_ms.append((time.perf_counter_ns() - start) / 1_000_000)

        profile_span_ms = []
        all_device_event_sum_ms = []
        compute_device_event_sum_ms = []
        memory_device_event_sum_ms = []
        device_event_counts = []
        profiled_device_events = []
        for profile_iteration in range(args.profile_iters):
            span_ms, events = _profile_once(region, device_args)
            memory_us = sum(
                float(event["duration_us"])
                for event in events
                if any(marker in str(event["name"]).lower() for marker in ("memcpy", "memset"))
            )
            all_us = sum(float(event["duration_us"]) for event in events)
            profile_span_ms.append(span_ms)
            all_device_event_sum_ms.append(all_us / 1000)
            compute_device_event_sum_ms.append((all_us - memory_us) / 1000)
            memory_device_event_sum_ms.append(memory_us / 1000)
            device_event_counts.append(len(events))
            profiled_device_events.append({"iteration": profile_iteration + 1, "events": events})

    fallbacks = [str(item.message) for item in caught if issubclass(item.category, FallbackWarning)]
    if fallbacks:
        raise RuntimeError(f"batch {batch_size} MoE decode fell back to CPU: {fallbacks}")

    return {
        "batch_size": batch_size,
        "shape": {
            "experts": EXPERTS,
            "hidden": HIDDEN,
            "intermediate": INTER,
            "top_k": TOP_K,
            "dtype": str(DTYPE),
        },
        "compile_and_first_run_s": compile_seconds,
        "warmup_iters": args.warmup_iters,
        "iterations": args.iterations,
        "latency_ms": _stats(wall_ms),
        "profile_iters": args.profile_iters,
        "profiled_cpu_span_ms": _stats(profile_span_ms),
        "all_device_event_sum_ms": _stats(all_device_event_sum_ms),
        "compute_device_event_sum_ms": _stats(compute_device_event_sum_ms),
        "memcpy_memset_event_sum_ms": _stats(memory_device_event_sum_ms),
        "device_event_counts": device_event_counts,
        "profiled_device_events": profiled_device_events,
        "fallback_clean": True,
        "correctness": "matches dense CPU reference",
    }


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="incremental JSON results file")
    args = parser.parse_args()
    if not args.batch_sizes or any(n < 1 or n & (n - 1) for n in args.batch_sizes):
        parser.error("--batch-sizes must contain positive powers of two")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes must not contain duplicates")
    if args.iterations < 1 or args.profile_iters < 1 or args.warmup_iters < 0:
        parser.error("iterations/profile-iters must be positive and warmup-iters non-negative")

    torch.set_num_threads(int(os.environ["SPYRE_NUM_CPUS"]))
    layer, weights = _device_layer(args.seed)
    payload: dict[str, Any] = {
        "model": "Gemma 4 MoE decode layer",
        "shape": {
            "experts": EXPERTS,
            "hidden": HIDDEN,
            "intermediate": INTER,
            "top_k": TOP_K,
            "dtype": str(DTYPE),
        },
        "measurements": [],
    }
    for batch_size in args.batch_sizes:
        result = _run_batch(batch_size, layer, weights, args)
        payload["measurements"].append(result)
        if args.output:
            _write_output(args.output, payload)
        print(
            json.dumps(
                {
                    "batch_size": batch_size,
                    "latency_median_ms": result["latency_ms"]["median_ms"],
                    "compute_event_sum_median_ms": result["compute_device_event_sum_ms"][
                        "median_ms"
                    ],
                    "memcpy_memset_sum_median_ms": result["memcpy_memset_event_sum_ms"][
                        "median_ms"
                    ],
                    "device_event_counts": result["device_event_counts"],
                    "fallback_clean": result["fallback_clean"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
