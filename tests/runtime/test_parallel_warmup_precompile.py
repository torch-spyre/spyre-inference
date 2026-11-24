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

"""The warm-up buckets precompile in forked workers before the normal pass (no device)."""

from __future__ import annotations

import contextlib
import os
import sys
import types

import pytest
import torch
from vllm.config import CompilationMode

from spyre_inference import envs
from spyre_inference.v1.worker import parallel_precompile
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

BUCKETS = [1, 4, 16, 512]


class _Bucketer:
    def __init__(self, bucket_sizes):
        self.bucket_sizes = bucket_sizes
        self.warmed_up = False

    def mark_warmed_up(self):
        self.warmed_up = True


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
        assert model is runner.model and device == runner._spyre_device
        runs.append(("pool" if in_pool else "loop", f"moe={sorted(sizes)}"))

    monkeypatch.setattr(moe, "warm_opaque_regions", warm_opaque_regions)

    compilation_config = types.SimpleNamespace(
        compile_sizes=list(BUCKETS),
        inductor_compile_config={},
        static_forward_context={},
        mode=CompilationMode.NONE,
    )
    runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
    runner.model_config = types.SimpleNamespace(runner_type="generate", is_encoder_decoder=False)
    runner.supports_mm_inputs = False
    runner.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(enforce_eager=False),
        compilation_config=compilation_config,
    )
    runner.compilation_config = compilation_config
    runner._spyre_device = torch.device("cpu")
    runner.model = torch.nn.Module()
    runner.spyre_shape_bucketer = _Bucketer(list(BUCKETS))
    runner.max_num_reqs = 16
    runner.num_spec_tokens = 0
    runs: list[tuple[str, int]] = []

    def dummy_run(size, *args, **kwargs):
        runs.append(("pool" if in_pool else "loop", size))
        if cold and in_pool:
            pools[-1].submitted += 1
        hidden = torch.zeros(size, 8, dtype=torch.float16)
        return hidden, hidden[:16]

    def dummy_sampler_run(hidden_states):
        runs.append(("pool" if in_pool else "loop", f"rows={len(hidden_states)}"))
        return torch.tensor([])

    runner._dummy_run = dummy_run
    runner._dummy_sampler_run = dummy_sampler_run
    return runner, runs, pools, checks, resets


def test_a_cold_start_traces_every_bucket_in_the_pool_then_runs_the_normal_pass(monkeypatch):
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)  # the worker count is min(3, threads)
    runner, runs, pools, checks, resets = _runner(monkeypatch, cold=True)
    runner.warming_up_model()

    # The pool pass runs what the normal pass runs, every bucket then the sampler rows, and
    # ends with the experts the opaque MoE op hides from the stand-in blocks.
    one_pass = [512, 16, 4, 1, "rows=16", "rows=4", "rows=1"]
    moe = [("pool", "moe=[1, 4, 16, 512]")]
    assert runs == [("pool", s) for s in one_pass] + moe + [("loop", s) for s in one_pass]
    assert pools[0].workers == 3 and pools[0].submitted == 4
    assert resets == [True]  # the stand-in graphs are forgotten before the normal pass
    assert checks[0].expected == 4 and checks[0].finished
    assert runner.spyre_shape_bucketer.warmed_up


def test_a_restart_with_warm_caches_stops_after_the_first_bucket(monkeypatch):
    runner, runs, pools, checks, resets = _runner(monkeypatch, cold=False)
    runner.warming_up_model()

    one_pass = [512, 16, 4, 1, "rows=16", "rows=4", "rows=1"]
    assert runs == [("pool", 512)] + [("loop", s) for s in one_pass]  # no MoE warm-up either
    assert resets == [] and checks == []  # the first bucket's real graphs stay


def test_workers_stay_within_the_torch_thread_budget(monkeypatch):
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    runner, _, pools, _, _ = _runner(monkeypatch, cold=True, workers="64")
    runner.warming_up_model()

    assert pools[0].workers == 2


def test_off_by_default(monkeypatch):
    runner, runs, pools, _, resets = _runner(monkeypatch, cold=True, workers="0")
    runner.warming_up_model()

    assert pools == [] and resets == []
    assert [s for _, s in runs] == [512, 16, 4, 1, "rows=16", "rows=4", "rows=1"]


def test_a_torch_spyre_without_the_helper_warns_and_compiles_in_order(monkeypatch, caplog):
    runner, runs, pools, _, resets = _runner(monkeypatch, cold=True, helper=False)
    with caplog.at_level("WARNING"):
        runner.warming_up_model()

    assert pools == [] and resets == []
    assert [s for _, s in runs] == [512, 16, 4, 1, "rows=16", "rows=4", "rows=1"]
    assert any("fork_inner_compiles" in r.message for r in caplog.records)


def test_caching_precompile_keeps_the_workers_off(monkeypatch, caplog):
    runner, runs, pools, _, resets = _runner(monkeypatch, cold=True)
    with torch._dynamo.config.patch(caching_precompile=True), caplog.at_level("INFO"):
        runner.warming_up_model()

    assert pools == [] and resets == []
    assert [s for _, s in runs] == [512, 16, 4, 1, "rows=16", "rows=4", "rows=1"]
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
