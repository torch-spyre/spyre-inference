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

"""Central registry for spyre-inference environment variables.

Every ``SPYRE_*`` knob lives in ``environment_variables`` below. Read one as a
module attribute (``spyre_inference.envs.SPYRE_NUM_CPUS``); the value is
computed from the environment on first access and then cached, so set the env
var before the value is first read. ``docs/user_guide/env_vars.md`` renders the
table below directly from this file, so keep each entry's comment accurate.
"""

import os
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    # Declarations so type checkers and IDEs resolve the lazily-provided
    # module attributes. Keep in sync with ``environment_variables``.
    SPYRE_DEVICES: str | None = None
    SPYRE_COMPILE_GRANULARITY: str = "block"
    SPYRE_COMPILE_GUARD: str = "off"
    SPYRE_ATTN_PROFILING: bool = False
    SPYRE_ATTN_RECORD: bool = True
    SPYRE_ATTN_KV_BUCKETS: str | None = None
    SPYRE_ATTN_QUERY_BUCKETS: str | None = None
    SPYRE_ATTN_NUM_SEQS_BUCKETS: str | None = None
    SPYRE_ATTN_KV_LAYOUT: str = "token_major"
    SPYRE_ATTN_MAX_CORES: int = 0
    SPYRE_KV_MAJOR_GATHER_2D: bool = False
    SPYRE_KV_MAJOR_CHUNK_SIZE: int = 8
    SPYRE_KV_MAJOR_WORK_DIV_HINTS: bool = False
    SPYRE_BATCHED_DECODE: bool = True
    SPYRE_KERNEL_CACHE: bool = False
    SPYRE_MAX_NUM_PARTIAL_PREFILLS: int = 1
    SPYRE_NUM_CPUS: int = 0
    SPYRE_UPDATE_THREAD_CONFIG: bool = True

_cache: dict[str, Any] = {}


# --8<-- [start:env-vars-definition]
environment_variables: dict[str, Callable[[], Any]] = {
    # Comma-separated Spyre device indices to run on, mapping each tensor-parallel
    # local rank to a physical card. Unset lets the runtime pick the default device(s).
    "SPYRE_DEVICES": lambda: os.getenv("SPYRE_DEVICES"),
    # Granularity of the decoder's torch.compile graph:
    #  - "block": compile one transformer block at a time (default)
    #  - "model": compile the whole model as a single graph
    "SPYRE_COMPILE_GRANULARITY": lambda: os.getenv("SPYRE_COMPILE_GRANULARITY") or "block",
    # What to do when a model block, attention kernel or the lm_head compiles *after*
    # warmup, which costs a full Inductor compile mid-request:
    #  - "off": nothing (default)
    #  - "warn": log each distinct violation
    #  - "error": raise. Use in CI to keep a warmup-coverage regression from landing,
    #    bearing in mind it catches most but not all: torch runs its compile-start
    #    callbacks only when a process-wide pending counter goes 0 -> 1, so a compile
    #    starting while another is in flight goes unreported.
    # torch-spyre compiles every eager aten op, so those compiles continue for the
    # whole run; they are never reported.
    "SPYRE_COMPILE_GUARD": lambda: os.getenv("SPYRE_COMPILE_GUARD") or "off",
    # When "1", wrap attention forward/softmax in torch.profiler.record_function
    # spans for kineto trace capture. Off by default: profiled runs are not
    # wall-clock comparable.
    "SPYRE_ATTN_PROFILING": lambda: bool(int(os.getenv("SPYRE_ATTN_PROFILING", "0"))),
    # When "1" (default), pre-compile every attention variant the run can need during
    # warmup, so no request pays an Inductor compile mid-serving. "0" falls back to
    # compiling each variant lazily on first use.
    "SPYRE_ATTN_RECORD": lambda: bool(int(os.getenv("SPYRE_ATTN_RECORD", "1"))),
    # Comma-separated kv_len buckets to record, unset uses the default buckets of
    # powers of two from block_size up to max_model_len.
    "SPYRE_ATTN_KV_BUCKETS": lambda: os.getenv("SPYRE_ATTN_KV_BUCKETS"),
    # Comma-separated query_len buckets to record, unset uses the default buckets
    # [1] + multiples of min(512, max_num_batched_tokens) up to max_num_batched_tokens.
    "SPYRE_ATTN_QUERY_BUCKETS": lambda: os.getenv("SPYRE_ATTN_QUERY_BUCKETS"),
    # Comma-separated num_seqs buckets for the batched decode kernel, unset uses the
    # default buckets of powers of two from 4 up to max_num_seqs.
    "SPYRE_ATTN_NUM_SEQS_BUCKETS": lambda: os.getenv("SPYRE_ATTN_NUM_SEQS_BUCKETS"),
    # Which KV cache layout the decoder attention backend uses, within a page:
    #  - "token_major": [num_blocks, block_size, num_kv_heads, head_size] (default)
    #  - "head_major":  [num_blocks, num_kv_heads, block_size, head_size], which drops
    #    the per-page permute the kernels do before the matmuls
    "SPYRE_ATTN_KV_LAYOUT": lambda: os.getenv("SPYRE_ATTN_KV_LAYOUT") or "token_major",
    # Core cap for the attention compile only, leaving the rest of the model on all 32.
    # "0" (default) lets the LX path pick its own cap and leaves the others uncapped.
    "SPYRE_ATTN_MAX_CORES": lambda: int(os.getenv("SPYRE_ATTN_MAX_CORES", "0")),
    # When "1", head-major batched decode stores cache rows KV-major and gathers them
    # through a [entries * KV, 1] index tensor. Experimental; off by default.
    "SPYRE_KV_MAJOR_GATHER_2D": lambda: bool(int(os.getenv("SPYRE_KV_MAJOR_GATHER_2D", "0"))),
    # Blocks per KV-major batched-decode chunk. Experimental path only.
    "SPYRE_KV_MAJOR_CHUNK_SIZE": lambda: int(os.getenv("SPYRE_KV_MAJOR_CHUNK_SIZE", "8")),
    # When "1", force KV-major batched decode's BMMs to share KV-head and entry ownership.
    "SPYRE_KV_MAJOR_WORK_DIV_HINTS": lambda: bool(
        int(os.getenv("SPYRE_KV_MAJOR_WORK_DIV_HINTS", "0"))
    ),
    # When "1" (default), enables the batched multi-sequence decode kernel for
    # batches of at least _MIN_BATCHED_SEQS sequences; smaller batches take the
    # per-seq loop either way. "0" forces the loop for all batch sizes.
    "SPYRE_BATCHED_DECODE": lambda: bool(int(os.getenv("SPYRE_BATCHED_DECODE", "1"))),
    # When "1", reuse compiled Spyre kernels across processes by caching them on
    # disk. Off by default. TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 disables the cache
    # even when this flag is enabled.
    "SPYRE_KERNEL_CACHE": lambda: os.getenv("SPYRE_KERNEL_CACHE", "0") == "1",
    # Maximum number of sequences allowed to prefill in the same batch. "1" (default)
    # serialises prefills, so a batch spends the whole token budget on one prompt
    # instead of topping itself up with a short chunk of the next. Any non-positive
    # value removes the cap, as does a pooling runner, which never decodes.
    "SPYRE_MAX_NUM_PARTIAL_PREFILLS": lambda: int(os.getenv("SPYRE_MAX_NUM_PARTIAL_PREFILLS", "1")),
    # CPU budget used to size thread pools. "0" (default) auto-detects the budget
    # (cgroup CPU quota, then physical core count).
    "SPYRE_NUM_CPUS": lambda: int(os.getenv("SPYRE_NUM_CPUS", "0")),
    # When "1" (default), clamp the CPU threading env vars (OMP_NUM_THREADS and
    # friends) to the detected budget to avoid thread oversubscription in
    # CPU-limited containers. Set to "0" to leave them untouched and only warn.
    "SPYRE_UPDATE_THREAD_CONFIG": lambda: bool(int(os.getenv("SPYRE_UPDATE_THREAD_CONFIG", "1"))),
}
# --8<-- [end:env-vars-definition]


def __getattr__(name: str) -> Any:
    if name in _cache:
        return _cache[name]
    if name in environment_variables:
        value = environment_variables[name]()
        _cache[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables.keys())


def clear_env_cache() -> None:
    """Drop cached values so the next access re-reads the environment."""
    _cache.clear()
