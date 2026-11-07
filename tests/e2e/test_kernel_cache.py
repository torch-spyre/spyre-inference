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

"""Persistent Spyre kernel cache (SPYRE_KERNEL_CACHE) tests"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_device_count
from spyre_testing_plugin.vfio_reaper import wait_until_card_free

# Each run is a fresh Python process: torch-spyre reads SPYRE_KERNEL_CACHE once,
# at import time.
pytestmark = pytest.mark.uses_subprocess

_MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
_PROMPT = "What are IBMs main businesses?"

_RUN_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig

    model, tp, prompt, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        enforce_eager=False,
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=8,
        compilation_config=CompilationConfig(compile_sizes=[1, 8]),
    )
    out = llm.generate(prompt, SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False)
    completion = out[0].outputs[0]
    with open(out_path, "w") as f:
        json.dump({"text": completion.text, "token_ids": list(completion.token_ids)}, f)
    llm.llm_engine.engine_core.shutdown(timeout=60)
    """
)


# One run takes 2-4 min. Bounding each run lets a hung child be killed and the
# cards released before pytest-timeout ends the whole process.
_RUN_TIMEOUT_S = 600

# torch-spyre logs this when it cannot compute a cache key and compiles uncached.
_CACHE_DISABLED_MSG = "Kernel cache disabled"


def _cache_entries(cache_root: Path) -> set[str]:
    if not cache_root.is_dir():
        return set()
    return {
        p.name
        for p in cache_root.iterdir()
        if p.is_dir()
        and ".tmp." not in p.name
        and not p.name.endswith(".tmp")
        and p.name != "failed"
    }


def _run(
    tmp_path: Path,
    name: str,
    tp: int,
    kernel_cache: bool,
) -> tuple[dict, str]:
    """Run one generation in a fresh process; return its output and its log."""
    env = dict(os.environ)
    # Inherited cache switches would stop the warm run reusing the cold run's graph.
    for var in (
        "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
        "TORCHINDUCTOR_FX_GRAPH_CACHE",
        "TORCHINDUCTOR_AUTOGRAD_CACHE",
        "VLLM_DISABLE_COMPILE_CACHE",
        "TORCH_SPYRE_CACHE_SCAN",
        "SPYRE_ATTN_FOR_EACH_TILE",
    ):
        env.pop(var, None)
    env.update(
        SPYRE_KERNEL_CACHE="1" if kernel_cache else "0",
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"),
        VLLM_CACHE_ROOT=str(tmp_path / "vllm"),
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="36000",
    )
    out_path = tmp_path / f"{name}.json"
    log_path = tmp_path / f"{name}.log"
    timed_out = False
    try:
        with open(log_path, "w") as log:
            # A new session lets a timeout kill the EngineCore and TP workers too.
            proc = subprocess.Popen(
                [sys.executable, "-c", _RUN_SCRIPT, _MODEL, str(tp), _PROMPT, str(out_path)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=_RUN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    finally:
        freed = wait_until_card_free(exclude_pids={os.getpid()}, timeout=60)
    log_text = log_path.read_text()
    assert not timed_out, (
        f"{name} run timed out after {_RUN_TIMEOUT_S}s; log tail:\n{log_text[-5000:]}"
    )
    assert proc.returncode == 0, (
        f"{name} run exited with {proc.returncode}; log tail:\n{log_text[-5000:]}"
    )
    assert freed, f"Spyre devices were not released after the {name} run"
    return json.loads(out_path.read_text()), log_text


_TP2 = [
    pytest.mark.distributed,
    pytest.mark.skipif(spyre_device_count() < 2, reason="needs >=2 Spyre cards for TP=2"),
]


@pytest.mark.timeout(3 * (_RUN_TIMEOUT_S + 60) + 120)
@pytest.mark.parametrize(
    "tp",
    [
        pytest.param(1, id="tp1-shared", marks=pytest.mark.e2e_smoke),
        pytest.param(2, id="tp2-shared", marks=[*_TP2, pytest.mark.e2e_smoke]),
    ],
)
def test_kernel_cache(tmp_path: Path, tp: int) -> None:
    """Cache off, cold cache, warm cache: same output, and the warm run adds no entries."""
    cache_root = tmp_path / "inductor" / "inductor-spyre-cache"

    disabled, _ = _run(tmp_path, "disabled", tp, kernel_cache=False)
    assert not _cache_entries(cache_root), "cache populated with SPYRE_KERNEL_CACHE=0"

    cold, _ = _run(tmp_path, "cold", tp, kernel_cache=True)
    cold_entries = _cache_entries(cache_root)
    assert cold_entries, "cold run did not populate the kernel cache"
    assert not (cache_root / "failed").exists(), "cold run left failed compiles"
    assert cold == disabled, "cold-cache output differs from the uncached output"

    warm, warm_log = _run(tmp_path, "warm", tp, kernel_cache=True)
    # A run that bypasses the cache also adds no entries, so check it was consulted.
    assert _CACHE_DISABLED_MSG not in warm_log, "warm run did not use the kernel cache"
    assert warm == cold, "warm-cache output differs from the cold-cache output"

    new_entries = _cache_entries(cache_root) - cold_entries
    assert not new_entries, (
        f"warm run compiled {len(new_entries)} new kernels on top of the "
        f"{len(cold_entries)} cached ones"
    )
