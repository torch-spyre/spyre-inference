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
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_device_count
from spyre_testing_plugin.vfio_reaper import wait_until_card_free

# Each run is a fresh Python process: torch-spyre reads SPYRE_KERNEL_CACHE and
# spyre-inference reads SPYRE_ATTN_FOR_EACH_TILE once, at import time.
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


class CacheGrewError(AssertionError):
    """The populated-cache run added entries; the only failure the xfail accepts."""


def _cache_entries(cache_root: Path) -> set[str]:
    if not cache_root.is_dir():
        return set()
    return {
        p.name
        for p in cache_root.iterdir()
        if p.is_dir() and ".tmp." not in p.name and p.name != "failed"
    }


def _run(tmp_path: Path, name: str, tp: int, for_each_tile: bool, kernel_cache: bool) -> dict:
    env = dict(os.environ)
    env.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
    env.update(
        SPYRE_KERNEL_CACHE="1" if kernel_cache else "0",
        SPYRE_ATTN_FOR_EACH_TILE="1" if for_each_tile else "0",
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"),
        VLLM_CACHE_ROOT=str(tmp_path / "vllm"),
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="36000",
    )
    out_path = tmp_path / f"{name}.json"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUN_SCRIPT, _MODEL, str(tp), _PROMPT, str(out_path)],
            env=env,
            check=False,
        )
    finally:
        freed = wait_until_card_free(exclude_pids={os.getpid()}, timeout=60)
    assert proc.returncode == 0, f"{name} run exited with {proc.returncode}"
    assert freed, f"Spyre devices were not released after the {name} run"
    return json.loads(out_path.read_text())


@pytest.mark.timeout(3 * 1800)
@pytest.mark.parametrize(
    "for_each_tile",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                raises=CacheGrewError,
                reason=(
                    "with SPYRE_ATTN_FOR_EACH_TILE=1 some cache keys are not stable "
                    "across processes, so the populated-cache run compiles part of the "
                    "kernels again"
                ),
            ),
        ),
    ],
    ids=["for_each_tile_off", "for_each_tile_on"],
)
@pytest.mark.parametrize(
    "tp",
    [
        1,
        pytest.param(
            4,
            marks=[
                pytest.mark.distributed,
                pytest.mark.skipif(
                    spyre_device_count() < 4, reason="needs >=4 Spyre cards for TP=4"
                ),
            ],
        ),
    ],
    ids=["tp1", "tp4"],
)
def test_kernel_cache(tmp_path: Path, tp: int, for_each_tile: bool) -> None:
    """Cache off, cold cache, warm cache: same output, and the warm run adds no entries."""
    cache_root = tmp_path / "inductor" / "inductor-spyre-cache"

    disabled = _run(tmp_path, "disabled", tp, for_each_tile, kernel_cache=False)
    assert not _cache_entries(cache_root), "cache populated with SPYRE_KERNEL_CACHE=0"

    cold = _run(tmp_path, "cold", tp, for_each_tile, kernel_cache=True)
    cold_entries = _cache_entries(cache_root)
    assert cold_entries, "cold run did not populate the kernel cache"
    assert not (cache_root / "failed").exists(), "cold run left failed compiles"

    warm = _run(tmp_path, "warm", tp, for_each_tile, kernel_cache=True)
    assert cold == disabled, "cold-cache output differs from the uncached output"
    assert warm == disabled, "warm-cache output differs from the uncached output"

    new_entries = _cache_entries(cache_root) - cold_entries
    if new_entries:
        raise CacheGrewError(
            f"warm run compiled {len(new_entries)} new kernels on top of the "
            f"{len(cold_entries)} cached ones"
        )
