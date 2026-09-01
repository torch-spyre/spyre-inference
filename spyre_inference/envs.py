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
    SPYRE_ATTN_PROFILING: bool = False
    SPYRE_BUCKETED_DECODE: bool = False
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
    # When "1", wrap attention forward/softmax in torch.profiler.record_function
    # spans for kineto trace capture. Off by default: profiled runs are not
    # wall-clock comparable.
    "SPYRE_ATTN_PROFILING": lambda: bool(int(os.getenv("SPYRE_ATTN_PROFILING", "0"))),
    # When "1", enables the bucketed multi-sequence decode kernel. Off by default
    # pending performance characterisation at small batch sizes (num_seqs <= 4).
    # Re-enable to measure the path or to restore it after calibration.
    "SPYRE_BUCKETED_DECODE": lambda: bool(int(os.getenv("SPYRE_BUCKETED_DECODE", "0"))),
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
