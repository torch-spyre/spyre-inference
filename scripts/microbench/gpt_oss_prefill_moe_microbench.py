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

"""Time gpt-oss prefill MoE: one expert per trip against a group of experts per trip.

This is the traced prefill path (``apply_monolithic`` while compiling). Group 1 is what
gpt-oss runs today. Group 4 is what Gemma-4 uses for 128 to 512 tokens. Faster only if
``grouped_over_single`` is below 1. Decode is not measured; 1 to 8 tokens already run
only the selected experts.

``120b-tp4`` is 128 experts and intermediate 720. ``20b-tp1`` is 32 experts and
intermediate 2880. Hidden is 2880 and top-k is 4 in both.

.. code-block:: bash

    SPYRE_NUM_CPUS=8 uv run --no-sync python scripts/microbench/gpt_oss_prefill_moe_microbench.py \\
        --output /tmp/gpt-oss-prefill-moe.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import warnings
from pathlib import Path

os.environ.setdefault("SPYRE_NUM_CPUS", "8")
os.environ.setdefault("VLLM_PLUGINS", "spyre_inference")

import torch  # noqa: E402

HIDDEN, TOP_K = 2880, 4
DTYPE = torch.float16
LIMIT, ALPHA = 7.0, 1.702
SHAPES = {
    "120b-tp4": (128, 720),
    "120b-tp8": (128, 360),
    "20b-tp1": (32, 2880),
}


class _Layer(torch.nn.Module):
    def __init__(self, experts: int, inter: int, recipe: object) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)

        def randn(*shape: int, scale: float) -> torch.Tensor:
            return (torch.randn(*shape, generator=generator) * scale).to(DTYPE)

        self.w13_weight = torch.nn.Parameter(
            randn(experts, 2 * inter, HIDDEN, scale=0.02), requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(
            randn(experts, HIDDEN, inter, scale=0.02), requires_grad=False
        )
        self.w13_bias = torch.nn.Parameter(
            randn(experts, 2 * inter, scale=0.5), requires_grad=False
        )
        self.w2_bias = torch.nn.Parameter(randn(experts, HIDDEN, scale=0.5), requires_grad=False)
        self.top_k = TOP_K
        self.spyre_moe_recipe = recipe


def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(-LIMIT, LIMIT)
    return (up + 1) * gate * torch.sigmoid(gate * ALPHA)


def _token_reference(x: torch.Tensor, logits: torch.Tensor, layer: _Layer) -> torch.Tensor:
    """One token, the gpt-oss mix: softmax over the top-4 logits, then the interleaved FFN."""
    weights, indices = torch.topk(logits[0].float(), TOP_K)
    weights = torch.softmax(weights, dim=-1)
    row = x[0].float()
    output = torch.zeros(HIDDEN, dtype=torch.float32)
    w13 = layer.w13_weight.detach()
    w2 = layer.w2_weight.detach()
    for slot in range(TOP_K):
        expert = int(indices[slot])
        gate_up = w13[expert].float() @ row + layer.w13_bias[expert].float()
        hidden = w2[expert].float() @ _swiglu(gate_up) + layer.w2_bias[expert].float()
        output += hidden * float(weights[slot])
    return output


def _compile_prefill(layer: _Layer):
    from spyre_inference.moe import SpyreUnquantizedFusedMoEMethod

    def prefill(x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        # ``* 1`` keeps the inputs in the graph, as a compiled block would.
        return SpyreUnquantizedFusedMoEMethod.apply_monolithic(None, layer, x * 1, logits * 1)

    return torch.compile(prefill, backend="inductor", fullgraph=True, dynamic=False)


def _time(fn, x: torch.Tensor, logits: torch.Tensor, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn(x, logits)
        torch.spyre.synchronize()
    samples = []
    for _ in range(iterations):
        torch.spyre.synchronize()
        start = time.perf_counter_ns()
        fn(x, logits)
        torch.spyre.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(samples)


def _run_shape(name: str, experts: int, inter: int, args: argparse.Namespace) -> list[dict]:
    from torch_spyre._inductor import config as spyre_config
    from torch_spyre.ops.fallbacks import FallbackWarning

    from spyre_inference.moe import SpyreMoERecipe, _prepare_layer

    cores = spyre_config.sencores
    for group in args.groups:
        if group > 1 and (cores % group or experts % group):
            raise SystemExit(f"group {group} must divide {cores} cores and {experts} experts")

    base = SpyreMoERecipe("swigluoai", "topk_softmax")
    layer = _Layer(experts, inter, base)
    # Keep the checkpoint tensors for the reference. ``_prepare_layer`` drops the parameters.
    w13 = layer.w13_weight.detach().clone()
    w2 = layer.w2_weight.detach().clone()
    w13_bias = layer.w13_bias.detach().clone()
    w2_bias = layer.w2_bias.detach().clone()
    _prepare_layer(layer)
    layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
    layer.w13_bias = torch.nn.Parameter(w13_bias, requires_grad=False)
    layer.w2_bias = torch.nn.Parameter(w2_bias, requires_grad=False)

    rows = []
    for tokens in args.tokens:
        generator = torch.Generator().manual_seed(args.seed + tokens + inter)
        x = (torch.randn(tokens, HIDDEN, generator=generator) * 0.05).to(DTYPE)
        logits = torch.randn(tokens, experts, generator=generator).to(DTYPE)
        x_device, logits_device = x.to("spyre"), logits.to("spyre")
        expected = _token_reference(x, logits, layer)
        medians: dict[int, float] = {}
        baseline: torch.Tensor | None = None
        for group in args.groups:
            layer.spyre_moe_recipe = SpyreMoERecipe(
                "swigluoai",
                "topk_softmax",
                expert_group=group,
                expert_group_tokens=(tokens, tokens) if group > 1 else None,
            )
            compiled = _compile_prefill(layer)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", FallbackWarning)
                started = time.perf_counter()
                actual = compiled(x_device, logits_device)
                torch.spyre.synchronize()
                compile_s = time.perf_counter() - started
            fallbacks = [
                str(item.message) for item in caught if issubclass(item.category, FallbackWarning)
            ]
            if fallbacks:
                raise RuntimeError(f"{name} {tokens} tokens group {group} fell back: {fallbacks}")
            actual_cpu = actual.cpu().float()
            if group == args.groups[0]:
                torch.testing.assert_close(actual_cpu[0], expected, atol=3e-2, rtol=3e-2)
                baseline = actual_cpu
            elif baseline is not None:
                torch.testing.assert_close(actual_cpu, baseline, atol=3e-2, rtol=3e-2)
            medians[group] = _time(compiled, x_device, logits_device, args.warmup, args.iterations)
            single = medians[args.groups[0]]
            row = {
                "shape": name,
                "experts": experts,
                "intermediate": inter,
                "tokens": tokens,
                "group": group,
                "cores": cores,
                "median_ms": medians[group],
                "grouped_over_single": medians[group] / single if single else None,
                "compile_s": round(compile_s, 2),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=sorted(SHAPES), default="120b-tp4")
    parser.add_argument("--tokens", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--groups", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(n < 1 or n % 64 for n in args.tokens):
        parser.error("--tokens must be positive multiples of the fp16 stick (64)")
    if not args.groups or args.groups[0] != 1 or any(g < 1 for g in args.groups):
        parser.error("--groups must start with 1")

    torch.set_num_threads(int(os.environ["SPYRE_NUM_CPUS"]))
    experts, inter = SHAPES[args.shape]
    rows = _run_shape(args.shape, experts, inter, args)
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
