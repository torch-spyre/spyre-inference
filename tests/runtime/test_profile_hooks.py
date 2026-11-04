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

"""Engine-step regions and phase markers of profile_hooks, on stand-in engine objects (no card)."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from spyre_inference import envs
from spyre_inference.v1.worker import profile_hooks


class _Cached:
    def __init__(self, ctx):
        self.ctx = set(ctx)

    def is_context_phase(self, req_id):
        return req_id in self.ctx


def _out(new=(), ctx=(), gen=(), tokens=None):
    reqs = [*new, *ctx, *gen]
    per = tokens or {r: 1 for r in reqs}
    return SimpleNamespace(
        total_num_scheduled_tokens=sum(per[r] for r in reqs),
        scheduled_new_reqs=[SimpleNamespace(req_id=r) for r in new],
        scheduled_cached_reqs=_Cached(ctx),
        num_scheduled_tokens={r: per[r] for r in reqs},
    )


@pytest.mark.parametrize(
    "out, tag",
    [
        (_out(new=["a"], tokens={"a": 128}), ("prefill", 128)),
        (_out(ctx=["a"], tokens={"a": 64}), ("prefill", 64)),
        (_out(gen=["a", "b"]), ("decode", 2)),
        (_out(new=["a"], gen=["b"], tokens={"a": 8, "b": 1}), ("mixed", 9)),
        (_out(), None),
    ],
)
def test_step_tag(out, tag):
    assert profile_hooks.step_tag(out) == tag


@pytest.fixture
def profile_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SPYRE_PROFILE_DIR", str(tmp_path))
    envs.clear_env_cache()
    monkeypatch.setattr(
        profile_hooks, "_state", {"phase": None, "profiling": False, "seen_profile": False}
    )
    yield tmp_path
    envs.clear_env_cache()


class _Engine:
    """EngineCore's step shape: schedule, run, return. A step with waiting requests admits one."""

    def __init__(self, outputs, waiting):
        self._outputs = outputs
        self.scheduler = SimpleNamespace(
            schedule=self._schedule, waiting=list(range(waiting)),
            get_num_unfinished_requests=lambda: 1,
        )  # fmt: skip

    def _schedule(self, *a):
        if self.scheduler.waiting:
            self.scheduler.waiting.pop()
        return self._outputs.pop(0)

    def step(self):
        self.scheduler.schedule()
        torch.ones(2) + 1
        return {}, True

    def profile(self, is_start=True, *a, **k):
        pass


def test_steps_get_regions_only_while_profiling(profile_dir):
    eng = _Engine(
        [_out(new=["w"], tokens={"w": 8}), _out(new=["a"], tokens={"a": 16}), _out(gen=["a"])],
        waiting=2,
    )
    step = profile_hooks._wrap_step(_Engine.step)
    prof_fn = profile_hooks._wrap_profile(_Engine.profile)

    step(eng)  # warmup, no profiler running
    prof_fn(eng, True)
    with profile(activities=[ProfilerActivity.CPU]) as p:
        step(eng)
        step(eng)
    prof_fn(eng, False)

    names = [e.name for e in p.events() if e.name.startswith(("spyre::", "ttft", "itl_"))]
    assert names.count("ttft") == 1 and names.count("itl_0001") == 1
    assert {"spyre::prefill:16", "spyre::decode:1"} <= set(names)
    assert not eng._spyre_spans

    (marks,) = profile_dir.glob("phases.*.jsonl")
    phases = [json.loads(x)["phase"] for x in marks.read_text().splitlines()]
    assert phases == ["warmup", "profiled", "prefill", "decode", "post"]


def test_mark_is_inert_without_a_profile_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("SPYRE_PROFILE_DIR", raising=False)
    envs.clear_env_cache()
    profile_hooks.mark("decode")
    assert not list(tmp_path.iterdir())
