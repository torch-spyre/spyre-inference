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

"""TTFT of an hf-adapters Gemma-4 prefill, comparable to `vllm bench latency --output-len 1`."""

import argparse
import time

import torch
import torch_spyre  # noqa: F401
from hf_adapters import AutoSpyreModelForCausalLM

p = argparse.ArgumentParser()
p.add_argument("--model", default="/models/google/gemma-4-26B-A4B")
p.add_argument("--input-len", type=int, default=1984)
p.add_argument("--prefill-chunk", type=int, default=512)
p.add_argument("--warmup", type=int, default=2)
p.add_argument("--iters", type=int, default=5)
a = p.parse_args()

torch.set_num_threads(
    8
)  # hf-adapters prefill is host-thread sensitive; the default (cores / 2) is much slower
model = AutoSpyreModelForCausalLM.from_pretrained(a.model, dtype=torch.float16)
vocab = model.config.get_text_config().vocab_size
ids = torch.randint(10, vocab, (1, a.input_len))


def ttft():
    torch.spyre.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=1,
            do_sample=False,
            prefill_chunk_size=a.prefill_chunk,
        )
    torch.spyre.synchronize()
    return time.perf_counter() - t0


for _ in range(a.warmup):  # the first call compiles
    ttft()
times = sorted(ttft() for _ in range(a.iters))
print(
    f"input_len={a.input_len} TTFT median {times[len(times) // 2]:.3f} s "
    f"(min {times[0]:.3f}, max {times[-1]:.3f})"
)
