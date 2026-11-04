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

"""Step regions, phase markers and device-memory samples for profiled benchmark runs.

Installed in every vLLM process by the general plugin when ``SPYRE_PROFILE_DIR`` is set.
While a torch profiler runs in the EngineCore process (the worker is in-process at TP1),
each step with unfinished requests is a ``ttft`` region (requests were waiting) or
``itl_NNNN``, the names spyre-perf-suite's analysers key on, and holds one
``spyre::<prefill|decode|mixed>:<num_tokens>`` region opened once scheduling has decided the
batch. TP>1 workers trace in their own processes and carry only vLLM's per-step
``execute_context_*`` annotation. The engine's resolved limits go to
``<dir>/engine.<pid>.json``, phase changes to ``<dir>/phases.<pid>.jsonl`` and,
with ``SPYRE_PROFILE_MEMORY_INTERVAL``, device memory to ``<dir>/devmem.<pid>.jsonl``; the
last two carry wall-clock ``t`` so the processes merge by time.
"""

import contextlib
import json
import os
import threading
import time

import torch

from spyre_inference import envs

_state: dict = {"phase": None, "profiling": False, "seen_profile": False, "itl": 0}
_lock = threading.Lock()


def mark(phase: str) -> None:
    """Record a phase change in this process; repeats of the current phase are dropped."""
    out_dir = envs.SPYRE_PROFILE_DIR
    if not out_dir:
        return
    with _lock:
        if _state["phase"] == phase:
            return
        _state["phase"] = phase
        with contextlib.suppress(OSError):
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"phases.{os.getpid()}.jsonl"), "a") as f:
                f.write(json.dumps({"t": time.time(), "phase": phase, "pid": os.getpid()}) + "\n")


def step_tag(scheduler_output) -> tuple[str, int] | None:
    """(kind, scheduled tokens) of one step, or None when it schedules nothing."""
    tokens = scheduler_output.total_num_scheduled_tokens
    if not tokens:
        return None
    new = {r.req_id for r in scheduler_output.scheduled_new_reqs}
    cached = scheduler_output.scheduled_cached_reqs
    ctx = gen = 0
    for req_id in scheduler_output.num_scheduled_tokens:
        if req_id in new or cached.is_context_phase(req_id):
            ctx += 1
        else:
            gen += 1
    return ("mixed" if ctx and gen else "prefill" if ctx else "decode"), tokens


def _profiling() -> bool:
    return torch.autograd.profiler._is_profiler_enabled


def _wrap_schedule(engine) -> None:
    schedule = engine.scheduler.schedule

    def wrapped(*args, **kwargs):
        out = schedule(*args, **kwargs)
        tag = step_tag(out)
        if tag:
            mark(tag[0] if _state["profiling"] else "post" if _state["seen_profile"] else "warmup")
            if engine._spyre_spans and _profiling():
                span = torch.profiler.record_function(f"spyre::{tag[0]}:{tag[1]}")
                span.__enter__()
                engine._spyre_spans.append(span)
        return out

    engine.scheduler.schedule = wrapped


def _write_engine_limits(engine) -> None:
    """The limits the engine resolved (platform defaults included), once per process."""
    out_dir = envs.SPYRE_PROFILE_DIR
    if not out_dir:
        return
    with contextlib.suppress(Exception):
        sched, model = engine.vllm_config.scheduler_config, engine.vllm_config.model_config
        limits = {
            "max_num_batched_tokens": sched.max_num_batched_tokens,
            "max_num_seqs": sched.max_num_seqs,
            "max_model_len": model.max_model_len,
        }
        path = os.path.join(out_dir, f"engine.{os.getpid()}.json")
        with open(path, "w") as f:
            json.dump(limits, f)


def _wrap_step(step):
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, "_spyre_spans"):
            self._spyre_spans = []
            _wrap_schedule(self)
            _write_engine_limits(self)
        sched = self.scheduler
        if _profiling() and sched.get_num_unfinished_requests():
            if len(sched.waiting):
                name = "ttft"
            else:
                _state["itl"] += 1
                name = f"itl_{_state['itl']:04d}"
            span = torch.profiler.record_function(name)
            span.__enter__()
            self._spyre_spans.append(span)
        try:
            return step(self, *args, **kwargs)
        finally:
            while self._spyre_spans:
                self._spyre_spans.pop().__exit__(None, None, None)

    return wrapped


def _wrap_profile(profile):
    def wrapped(self, is_start: bool = True, *args, **kwargs):
        _state["profiling"] = is_start
        _state["seen_profile"] = True
        _state["itl"] = 0
        mark("profiled" if is_start else "post")
        return profile(self, is_start, *args, **kwargs)

    return wrapped


def _sample_device_memory(out_dir: str, interval: float, device: int | None) -> None:
    mem = torch.accelerator
    path = os.path.join(out_dir, f"devmem.{os.getpid()}.jsonl")
    with open(path, "a", buffering=1) as f:
        while True:
            try:
                free, total = mem.get_memory_info(device)
                row = {
                    "t": time.time(),
                    "allocated": mem.memory_allocated(device),
                    "max_allocated": mem.max_memory_allocated(device),
                    "reserved": mem.memory_reserved(device),
                    "free": free,
                    "total": total,
                }
            except Exception as e:  # device not up yet, or no allocator stats
                row = {"t": time.time(), "err": repr(e)[:120]}
            f.write(json.dumps(row) + "\n")
            time.sleep(interval)


def _wrap_warmup(warmup):
    def wrapped(self, *args, **kwargs):
        mark("compile_warmup")
        try:
            return warmup(self, *args, **kwargs)
        finally:
            mark("ready")

    wrapped._spyre_marked = True
    return wrapped


def _wrap_load_model(load_model):
    def wrapped(self, *args, **kwargs):
        interval, out_dir = envs.SPYRE_PROFILE_MEMORY_INTERVAL, envs.SPYRE_PROFILE_DIR
        if out_dir and interval > 0 and not _state.get("sampler"):
            # The current device is per thread: pass this worker's card explicitly.
            device = None
            with contextlib.suppress(Exception):
                device = torch.accelerator.current_device_index()
            _state["sampler"] = threading.Thread(
                target=_sample_device_memory, args=(out_dir, interval, device), daemon=True
            )
            _state["sampler"].start()
        # Patched here rather than at install, which also runs in the API server process.
        cls = type(self)
        if not getattr(cls.compile_or_warm_up_model, "_spyre_marked", False):
            cls.compile_or_warm_up_model = _wrap_warmup(cls.compile_or_warm_up_model)
        mark("weight_load")
        return load_model(self, *args, **kwargs)

    return wrapped


def install() -> None:
    """Patch EngineCore and the worker base once per process."""
    if _state.get("installed"):
        return
    _state["installed"] = True
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.worker.gpu_worker import Worker

    EngineCore.step = _wrap_step(EngineCore.step)
    EngineCore.step_with_batch_queue = _wrap_step(EngineCore.step_with_batch_queue)
    EngineCore.profile = _wrap_profile(EngineCore.profile)
    Worker.load_model = _wrap_load_model(Worker.load_model)
