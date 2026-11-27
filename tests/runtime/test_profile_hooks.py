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

"""The phase timeline and engine limits of profile_hooks, and the worker that marks them."""

import json
from types import SimpleNamespace

import pytest
from vllm.v1.worker.gpu_worker import Worker

from spyre_inference import envs
from spyre_inference.v1.worker import profile_hooks
from spyre_inference.v1.worker.spyre.worker import SpyreWorkerV2


def _vllm_config(trace_dir=None):
    return SimpleNamespace(
        profiler_config=None if trace_dir is None else SimpleNamespace(
            torch_profiler_dir=str(trace_dir)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256, max_num_seqs=4),
        model_config=SimpleNamespace(max_model_len=512),
    )  # fmt: skip


@pytest.fixture
def unconfigured(monkeypatch):
    monkeypatch.delenv("SPYRE_PROFILE_DIR", raising=False)
    monkeypatch.setattr(profile_hooks, "_state", {"phase": None, "dir": None})
    envs.clear_env_cache()
    yield
    envs.clear_env_cache()


def _phases(out_dir):
    (marks,) = out_dir.glob("phases.*.jsonl")
    return [json.loads(line)["phase"] for line in marks.read_text().splitlines()]


def test_sidecars_land_beside_the_trace(tmp_path, unconfigured):
    """With no override, the timeline annotates the trace it sits next to."""
    profile_hooks.configure(_vllm_config(trace_dir=tmp_path))
    profile_hooks.mark("weight_load")

    assert _phases(tmp_path) == ["weight_load"]
    (limits,) = tmp_path.glob("engine.*.json")
    assert json.loads(limits.read_text()) == {
        "max_num_batched_tokens": 256,
        "max_num_seqs": 4,
        "max_model_len": 512,
    }


def test_profile_dir_overrides_the_trace_dir(tmp_path, monkeypatch, unconfigured):
    """The env var also records a timeline for a run that is not being traced."""
    override, trace_dir = tmp_path / "override", tmp_path / "trace"
    monkeypatch.setenv("SPYRE_PROFILE_DIR", str(override))
    envs.clear_env_cache()

    profile_hooks.configure(_vllm_config(trace_dir=trace_dir))
    profile_hooks.mark("ready")

    assert _phases(override) == ["ready"]
    assert not trace_dir.exists()


def test_nothing_is_written_without_a_profiler(tmp_path, unconfigured):
    profile_hooks.configure(_vllm_config(trace_dir=None))
    profile_hooks.mark("weight_load")
    assert not list(tmp_path.iterdir())


def test_repeated_phase_marks_are_dropped(tmp_path, unconfigured):
    profile_hooks.configure(_vllm_config(trace_dir=tmp_path))
    profile_hooks.mark("decode")
    profile_hooks.mark("decode")
    profile_hooks.mark("prefill")
    assert _phases(tmp_path) == ["decode", "prefill"]


def test_the_worker_marks_the_boundaries_it_owns(tmp_path, monkeypatch, unconfigured):
    """Spyre's own worker marks the phases, so no vLLM internals are patched."""
    monkeypatch.setattr(Worker, "load_model", lambda self, *a, **k: None)
    monkeypatch.setattr(Worker, "profile", lambda self, *a, **k: None)
    worker = SpyreWorkerV2.__new__(SpyreWorkerV2)
    worker.vllm_config = _vllm_config(trace_dir=tmp_path)

    worker.load_model()
    worker.profile(is_start=True)
    worker.profile(is_start=False)

    assert _phases(tmp_path) == ["weight_load", "profiled", "post"]


def test_a_profile_that_cannot_start_is_not_marked(tmp_path, monkeypatch, unconfigured):
    """vLLM raises without a --profiler-config, and an unstarted profile is not a phase."""

    def refuse(self, *args, **kwargs):
        raise RuntimeError("Profiling is not enabled.")

    monkeypatch.setattr(Worker, "profile", refuse)
    worker = SpyreWorkerV2.__new__(SpyreWorkerV2)
    worker.vllm_config = _vllm_config(trace_dir=tmp_path)
    profile_hooks.configure(worker.vllm_config)

    with pytest.raises(RuntimeError):
        worker.profile(is_start=True)
    assert not list(tmp_path.glob("phases.*.jsonl"))
