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

"""New-stack profile run — matches docs/getting_started/kineto_profiling.md §2.1.

Usage:
    source ./setup_profile_env.sh          # activates venv + exports env vars
    python -u profile_spyre_inference.py

Output:
    logs/<hostname>_<pid>.<ts>.pt.trace.json  (Chrome/Perfetto format)
"""

import os

# external_launcher reads these from env
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

import torch
from torch.profiler import ProfilerActivity, profile
from vllm import LLM, SamplingParams
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.config import AttentionConfig

llm = LLM(
    model="ibm-granite/granite-3.3-8b-instruct",
    dtype="float16",
    max_model_len=32,
    max_num_seqs=1,
    num_gpu_blocks_override=64,
    attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
    distributed_executor_backend="external_launcher",  # worker in-process
)

os.makedirs("logs/", exist_ok=True)

prompts = ["What do you know about Zurich?"]
samplings = [SamplingParams(max_tokens=4, temperature=0.0)]

# Warmup
for _ in range(2):
    llm.generate(prompts, samplings)

# Profiled generate
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("logs/"),
    record_shapes=True,
    acc_events=True,
) as prof:
    outputs = llm.generate(prompts, samplings)

# Optional terminal summary
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20).replace("CUDA", "AIU"))

os._exit(0)  # avoids TimestampCalibrator abort at teardown
