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

"""Latency of an hf-adapters Gemma-4 generate, comparable to `vllm bench latency`.

ttft.py (spyre-inference#1102) generalized to --output-len > 1 and to tensor parallelism:
launched by torchrun (WORLD_SIZE > 1), the model loads with tp_plan="auto", as hf-adapters'
multicard smoke test does. With --output-len 1 it is the TTFT workload's hf arm above TP1,
where ttft.py cannot shard. Times whole generate calls like ttft.py, and reads hf-adapters'
own per-token timing (first token, then one entry per decode step; each step copies its
logits to the host, so every entry is device-complete) from the same calls.
"""

import argparse
import contextlib
import io
import json
import os
import re
import statistics
import sys
import time

import torch
import torch_spyre  # noqa: F401
from hf_adapters import AutoSpyreModelForCausalLM

p = argparse.ArgumentParser()
p.add_argument("--model", default="/models/google/gemma-4-26B-A4B")
p.add_argument("--input-len", type=int, default=1984)
p.add_argument("--output-len", type=int, default=1)
p.add_argument("--prefill-chunk", type=int, default=512)
p.add_argument("--warmup", type=int, default=2)
p.add_argument("--iters", type=int, default=5)
p.add_argument("--output-json")
a = p.parse_args()

world = int(os.environ.get("WORLD_SIZE", "1"))
rank = int(os.environ.get("RANK", "0"))
PER_TOKEN = re.compile(r"Per-token: ([\d.,\s]+) ms")

torch.set_num_threads(8)  # as ttft.py
model = AutoSpyreModelForCausalLM.from_pretrained(
    a.model, dtype=torch.float16, tp_plan="auto" if world > 1 else None
)
vocab = model.config.get_text_config().vocab_size
torch.manual_seed(0)  # tensor parallelism needs the same prompt on every rank
ids = torch.randint(10, vocab, (1, a.input_len))


def generate() -> tuple[float, list[float]]:
    printed = io.StringIO()
    torch.spyre.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), contextlib.redirect_stdout(printed):
        # min_new_tokens suppresses EOS, so every call emits exactly --output-len tokens
        # (vllm bench latency ignores EOS).
        model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=a.output_len,
            min_new_tokens=a.output_len,
            do_sample=False,
            prefill_chunk_size=a.prefill_chunk,
            timing=True,
        )
    torch.spyre.synchronize()
    elapsed = time.perf_counter() - t0
    m = PER_TOKEN.search(printed.getvalue())
    return elapsed, [float(t) / 1000 for t in m.group(1).split(",")] if m else []


for _ in range(a.warmup):  # the first call compiles
    generate()
runs = [generate() for _ in range(a.iters)]

if rank == 0:
    latencies = [r[0] for r in runs]
    first = [r[1][0] for r in runs if r[1]]
    steps = [s for r in runs for s in r[1][1:]]
    lat = sorted(latencies)
    line = (
        f"input_len={a.input_len} output_len={a.output_len} tp={world} latency median "
        f"{lat[len(lat) // 2]:.3f} s (min {lat[0]:.3f}, max {lat[-1]:.3f})"
    )
    if first:
        line += f"; first token median {statistics.median(first):.3f} s"
    if steps:
        line += (
            f"; decode step mean {statistics.mean(steps) * 1000:.2f} ms "
            f"(median {statistics.median(steps) * 1000:.2f})"
        )
    print(line)
    if a.output_json:
        with open(a.output_json, "w") as f:
            json.dump(
                {
                    "input_len": a.input_len,
                    "output_len": a.output_len,
                    "tp": world,
                    "latencies": latencies,
                    "avg_latency": statistics.mean(latencies),
                    "per_token": [r[1] for r in runs],
                },
                f,
                indent=2,
            )
if world > 1:
    sys.stdout.flush()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # Skips interpreter teardown and the libsenlib-dd2.so destructor abort that hf-adapters'
    # multicard script documents for torchrun.
    os._exit(0)
