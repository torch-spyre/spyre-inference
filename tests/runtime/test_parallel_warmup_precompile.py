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

"""The pool around the warm-up's first pass: its guards and its reporting (no device).

The V2 runner's pool pass itself is tested in tests/mrv2/test_warmup.py."""

from __future__ import annotations

import contextlib
import os
import sys
import types

import pytest
import torch

from spyre_inference import envs
from spyre_inference.v1.worker import parallel_precompile

BUCKETS = [1, 4, 16, 512]


class _Pool:
    """What fork_inner_compiles yields: counts one submitted compile per cold dummy run."""

    def __init__(self):
        self.submitted = self.ok = self.failed = self.time_limited = 0


def _runner(monkeypatch, *, cold, workers="3", helper=True, time_limited=0):
    monkeypatch.setenv("SPYRE_PARALLEL_WARMUP_PRECOMPILE", workers)
    envs.clear_env_cache()
    pools, checks, resets, in_pool = [], [], [], []

    @contextlib.contextmanager
    def fork_inner_compiles(n, timeout_s, placeholders):
        assert placeholders is True
        pool = _Pool()
        pool.workers = n
        pools.append(pool)
        in_pool.append(True)
        try:
            yield pool
        finally:
            in_pool.pop()
        pool.ok = pool.submitted
        pool.time_limited = time_limited

    def second_pass_check(expected):
        check = types.SimpleNamespace(expected=expected, finished=False)
        check.finish = lambda: setattr(check, "finished", True)
        checks.append(check)
        return check

    if helper:
        module = types.SimpleNamespace(
            fork_inner_compiles=fork_inner_compiles, SecondPassCheck=second_pass_check
        )
        monkeypatch.setitem(sys.modules, "torch_spyre._inductor.parallel_precompile", module)
    else:
        monkeypatch.setitem(sys.modules, "torch_spyre._inductor.parallel_precompile", None)
    monkeypatch.setattr(torch._dynamo, "reset", lambda: resets.append(True))
    from spyre_inference import moe

    def warm_opaque_regions(model, sizes, device):
        runs.append(("pool" if in_pool else "loop", f"moe={sorted(sizes)}"))

    monkeypatch.setattr(moe, "warm_opaque_regions", warm_opaque_regions)

    runs: list[tuple[str, int | str]] = []

    def dummy_run(size):
        runs.append(("pool" if in_pool else "loop", size))
        if cold and in_pool:
            pools[-1].submitted += 1

    def trace(pool) -> bool:
        for size in sorted(BUCKETS, reverse=True):
            dummy_run(size)
            if pool.submitted == 0:
                return True
        return False

    def warm_regions() -> None:
        moe.warm_opaque_regions(None, BUCKETS, None)

    def warming_up_model():
        check = None
        if envs.SPYRE_PARALLEL_WARMUP_PRECOMPILE > 0:
            check = parallel_precompile.precompile_in_workers(trace, warm_regions)
        for size in sorted(BUCKETS, reverse=True):
            dummy_run(size)
        if check is not None:
            check.finish()

    runner = types.SimpleNamespace(warming_up_model=warming_up_model)
    return runner, runs, pools, checks, resets


def test_workers_stay_within_the_torch_thread_budget(monkeypatch):
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    runner, _, pools, _, _ = _runner(monkeypatch, cold=True, workers="64")
    runner.warming_up_model()

    assert pools[0].workers == 2


def test_off_by_default(monkeypatch):
    runner, runs, pools, _, resets = _runner(monkeypatch, cold=True, workers="0")
    runner.warming_up_model()

    assert pools == [] and resets == []
    assert [s for _, s in runs] == [512, 16, 4, 1]


def test_a_torch_spyre_without_the_helper_warns_and_compiles_in_order(monkeypatch, caplog):
    runner, runs, pools, _, resets = _runner(monkeypatch, cold=True, helper=False)
    with caplog.at_level("WARNING"):
        runner.warming_up_model()

    assert pools == [] and resets == []
    assert [s for _, s in runs] == [512, 16, 4, 1]
    assert any("fork_inner_compiles" in r.message for r in caplog.records)


def test_caching_precompile_keeps_the_workers_off(monkeypatch, caplog):
    runner, runs, pools, _, resets = _runner(monkeypatch, cold=True)
    with torch._dynamo.config.patch(caching_precompile=True), caplog.at_level("INFO"):
        runner.warming_up_model()

    assert pools == [] and resets == []
    assert [s for _, s in runs] == [512, 16, 4, 1]
    assert any("TORCH_CACHING_PRECOMPILE" in r.message for r in caplog.records)


def test_solves_that_stop_at_the_time_limit_are_reported(monkeypatch, caplog):
    runner, _, _, _, _ = _runner(monkeypatch, cold=True, time_limited=2)
    with caplog.at_level("WARNING"):
        runner.warming_up_model()

    assert any("2 of 4 graphs stopped LX planning" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ("workers", "tp", "preset", "expected"),
    [
        ("3", 2, None, "0"),  # the workers at TP > 1: context switching off
        ("3", 1, None, None),  # one rank: unchanged
        ("0", 4, None, None),  # workers off: unchanged
        ("3", 4, "1", "1"),  # an explicit value wins
    ],
)
def test_lx_context_switching_is_off_only_for_the_workers_at_tp(
    monkeypatch, workers, tp, preset, expected
):
    monkeypatch.setenv("SPYRE_PARALLEL_WARMUP_PRECOMPILE", workers)
    if preset is None:
        monkeypatch.delenv("ENABLE_LX_CONTEXT_SWITCHING", raising=False)
    else:
        monkeypatch.setenv("ENABLE_LX_CONTEXT_SWITCHING", preset)
    envs.clear_env_cache()
    parallel_precompile.keep_collectives_out_of_lx(tp)

    assert os.environ.get("ENABLE_LX_CONTEXT_SWITCHING") == expected
