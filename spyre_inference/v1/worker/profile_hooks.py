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

"""The phase timeline and engine limits beside a profiled trace.

vLLM already names every step in the trace: ``Worker.execute_model`` wraps each one
in ``execute_context_<n>(<tokens>)_generation_<n>(<tokens>)``, which carries the
prefill/decode split and the token counts. TTFT and ITL follow from that sequence --
the first step with context tokens is the prompt, each one after it is a token -- so
nothing here labels steps.

What a trace does not say is what the process was doing before the first step. The
Spyre worker marks the boundaries it owns (``weight_load``, ``compile_warmup``,
``ready``, ``profiled``, ``post``) to ``<dir>/phases.<pid>.jsonl`` with wall-clock
``t``, so several processes merge by time, and writes the engine's resolved limits to
``<dir>/engine.<pid>.json``.

``<dir>`` is the run's ``torch_profiler_dir``, so the sidecars land beside the trace
they annotate; ``SPYRE_PROFILE_DIR`` overrides it, which also records a timeline for a
run that is not being traced. With neither, every call here is a no-op.
"""

import contextlib
import json
import os
import threading
import time

from spyre_inference import envs

_state: dict = {"phase": None, "dir": None}
_lock = threading.Lock()


def configure(vllm_config) -> None:
    """Resolve where this process's sidecars go, and write the engine's limits there."""
    profiler_config = getattr(vllm_config, "profiler_config", None)
    out_dir = envs.SPYRE_PROFILE_DIR or getattr(profiler_config, "torch_profiler_dir", None)
    with _lock:
        _state["dir"] = out_dir or None
    if not out_dir:
        return
    # Instrumentation must never take the engine down, and the config shape is vLLM's.
    with contextlib.suppress(Exception):
        sched, model = vllm_config.scheduler_config, vllm_config.model_config
        limits = {
            "max_num_batched_tokens": sched.max_num_batched_tokens,
            "max_num_seqs": sched.max_num_seqs,
            "max_model_len": model.max_model_len,
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"engine.{os.getpid()}.json"), "w") as f:
            json.dump(limits, f)


def mark(phase: str) -> None:
    """Record a phase change in this process; repeats of the current phase are dropped."""
    with _lock:
        out_dir = _state["dir"]
        if not out_dir or _state["phase"] == phase:
            return
        _state["phase"] = phase
        with contextlib.suppress(OSError):
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"phases.{os.getpid()}.jsonl"), "a") as f:
                f.write(json.dumps({"t": time.time(), "phase": phase, "pid": os.getpid()}) + "\n")
