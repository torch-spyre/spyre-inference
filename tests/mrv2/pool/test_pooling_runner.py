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

"""SpyrePoolingRunner.pool: host cursor, grid-direct CLS/MEAN, unpad otherwise."""

import types

import numpy as np
import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.late_interaction_runner import LateInteractionRunner
from vllm.v1.pool.metadata import PoolingStates

from spyre_inference.v1.pool import warmup
from spyre_inference.v1.pool.spyre_pooler import SpyreCLSPool, SpyreLastPool
from spyre_inference.v1.worker.spyre.pool import pooling_runner
from spyre_inference.v1.worker.spyre.pool.pooling_runner import SpyrePoolingRunner

_HIDDEN = 4


class _Head(nn.Module):
    """SequencePooler calls its head with (pooled, pooling_metadata)."""

    def forward(self, pooled, pooling_metadata):
        return pooled


def _runner(
    pooling: nn.Module, grid: tuple[int, int, list[int]] | None, device: str = "cpu"
) -> SpyrePoolingRunner:
    """The pooler path on ``device``: "cpu" checks the row logic without a card."""
    runner = SpyrePoolingRunner.__new__(SpyrePoolingRunner)
    runner.model = types.SimpleNamespace(pooler=SequencePooler(pooling=pooling, head=_Head()))
    runner.model_config = None
    runner.max_num_reqs = 8
    runner.pooling_on_spyre = True
    runner.spyre_device = torch.device(device)
    plan = types.SimpleNamespace(grid=grid) if grid is not None else None
    runner.encoder_plan_source = lambda: plan  # ty: ignore[invalid-assignment]
    runner.late_interaction_runner = LateInteractionRunner()
    runner.prompt_token_ids = {}
    return runner


def _batch(runner: SpyrePoolingRunner, lens: list[int], done: list[int] | None = None):
    """An InputBatch of len(lens) requests; ``done`` is each one's computed length."""
    done = done if done is not None else lens
    for i in range(len(lens)):
        runner.pooling_params[i] = PoolingParams(task="embed")
        runner.pooling_states[i] = PoolingStates()
    batch = types.SimpleNamespace(
        req_ids=[f"r{i}" for i in range(len(lens))],
        num_reqs=len(lens),
        idx_mapping_np=np.arange(len(lens)),
        num_scheduled_tokens=np.array(lens, dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor(done, dtype=torch.int32),
    )
    req_states = types.SimpleNamespace(
        prompt_len=types.SimpleNamespace(np=np.array(lens, dtype=np.int32))
    )
    return batch, req_states


def _with_state(runner: SpyrePoolingRunner) -> SpyrePoolingRunner:
    runner.pooling_params = {}
    runner.pooling_states = {}
    return runner


def test_grid_cls_reads_each_lanes_first_row() -> None:
    extent, query_lens = 8, [3, 1, 2]
    grid = (extent, 3, query_lens)
    runner = _with_state(_runner(SpyreCLSPool(), grid))
    batch, req_states = _batch(runner, query_lens)
    hidden = torch.arange(3 * extent, dtype=torch.float32).unsqueeze(1).expand(-1, _HIDDEN)

    pooled, finished = runner.pool(hidden, batch, req_states)

    assert finished == [True, True, True]
    assert torch.stack(list(pooled))[:, 0].tolist() == [0, 8, 16]


def test_ragged_cls_reads_packed_rows() -> None:
    query_lens = [3, 1, 2]
    runner = _with_state(_runner(SpyreCLSPool(), None))
    batch, req_states = _batch(runner, query_lens)
    hidden = torch.arange(sum(query_lens), dtype=torch.float32).unsqueeze(1).expand(-1, _HIDDEN)

    pooled, _ = runner.pool(hidden, batch, req_states)

    assert torch.stack(list(pooled))[:, 0].tolist() == [0, 3, 4]


def test_grid_last_unpads_before_pooling() -> None:
    """LAST can't read the grid, so the grid is re-compacted to packed order first."""
    extent, query_lens = 8, [3, 1, 2]
    grid = (extent, 3, query_lens)
    runner = _with_state(_runner(SpyreLastPool(), grid))
    batch, req_states = _batch(runner, query_lens)
    hidden = torch.arange(3 * extent, dtype=torch.float32).unsqueeze(1).expand(-1, _HIDDEN)

    pooled, _ = runner.pool(hidden, batch, req_states)

    # Last real row of each lane: 2, 8, 17.
    assert torch.stack(list(pooled))[:, 0].tolist() == [2, 8, 17]


def test_finished_mask_comes_from_the_cursor() -> None:
    """A request whose computed length is short of its prompt hasn't finished."""
    query_lens = [3, 2]
    runner = _with_state(_runner(SpyreLastPool(), None))
    batch, req_states = _batch(runner, query_lens, done=[3, 1])

    _, finished = runner.pool(torch.zeros(sum(query_lens), _HIDDEN), batch, req_states)

    assert finished == [True, False]


def test_host_pooling_unpads_the_grid_then_defers_to_upstream(monkeypatch) -> None:
    """FP32 heads pool on the host through upstream's own pool()."""
    extent, query_lens = 4, [2, 1]
    runner = _with_state(_runner(SpyreCLSPool(), (extent, 2, query_lens)))
    runner.pooling_on_spyre = False
    seen = []
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.pool.pooling_runner.PoolingRunner.pool",
        lambda self, hidden, batch, req_states: seen.append(hidden) or ("out", [True]),
    )
    hidden = torch.arange(2 * extent, dtype=torch.float32).unsqueeze(1)

    assert runner.pool(hidden, *_batch(runner, query_lens)) == ("out", [True])
    # Packed rows 0, 1 (lane 0) and 4 (lane 1), padded to the grid's row count.
    assert seen[0][:3, 0].tolist() == [0, 1, 4]


def test_supported_tasks_see_through_the_wrapper(monkeypatch) -> None:
    """is_pooling_model is an isinstance check, which the wrapper fails."""
    bare = object()
    wrapper = types.SimpleNamespace(_model=types.SimpleNamespace(_orig_mod=bare))
    seen = []
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.pool.pooling_runner.PoolingRunner.get_supported_tasks",
        staticmethod(lambda model: seen.append(model) or []),
    )

    SpyrePoolingRunner.get_supported_tasks(wrapper)  # ty: ignore[invalid-argument-type]

    assert seen == [bare]


def test_runner_reports_pooling_tasks_through_the_wrapper(monkeypatch) -> None:
    """Upstream's get_supported_tasks asks the base PoolingRunner about the wrapper."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2

    monkeypatch.setattr(GPUModelRunner, "get_supported_tasks", lambda self: ("generate",))
    monkeypatch.setattr(
        SpyrePoolingRunner, "get_supported_tasks", staticmethod(lambda model: ["embed"])
    )
    runner = SpyreModelRunnerV2.__new__(SpyreModelRunnerV2)
    runner.model = object()

    runner.is_pooling_model = True
    assert runner.get_supported_tasks() == ("generate", "embed")
    runner.is_pooling_model = False
    assert runner.get_supported_tasks() == ("generate",)


def test_dummy_run_on_spyre_uses_the_shared_dummy_metadata(monkeypatch) -> None:
    calls = []
    real = warmup.dummy_pooling_metadata

    def recording(pooler, model_config, task, num_tokens, max_num_reqs, device):
        calls.append((task, num_tokens, max_num_reqs, device))
        return real(pooler, types.SimpleNamespace(), task, num_tokens, max_num_reqs, device)

    monkeypatch.setattr(pooling_runner, "dummy_pooling_metadata", recording)
    monkeypatch.setattr(PoolingParams, "verify", lambda self, model_config: None)
    runner = _with_state(_runner(SpyreCLSPool(), None))

    pooled = runner._dummy_pooler_run_task(torch.zeros(12, _HIDDEN), "embed")

    assert calls == [("embed", 12, 8, torch.device("cpu"))]
    assert len(pooled) == 8, "min(num_tokens, max_num_reqs) dummy requests"


def test_dummy_run_on_the_host_defers_to_upstream(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.pool.pooling_runner.PoolingRunner._dummy_pooler_run_task",
        lambda self, hidden, task: seen.append((hidden.device.type, task)) or "out",
    )
    runner = _with_state(_runner(SpyreCLSPool(), None))
    runner.pooling_on_spyre = False

    assert runner._dummy_pooler_run_task(torch.zeros(4, _HIDDEN), "embed") == "out"
    assert seen == [("cpu", "embed")]


@pytest.mark.spyre_device
def test_grid_cls_on_spyre_matches_the_host() -> None:
    """The device path: a CPU body is moved to Spyre, then CLS reads the grid there."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    extent, query_lens = 8, [3, 1, 2]
    grid = (extent, 3, query_lens)
    torch.manual_seed(0)
    hidden = torch.randn(3 * extent, 64, dtype=torch.float16)
    runner = _with_state(_runner(SpyreCLSPool(), grid, device="spyre"))
    batch, req_states = _batch(runner, query_lens)

    pooled, finished = runner.pool(hidden, batch, req_states)

    assert finished == [True, True, True]
    assert all(t.device.type == "spyre" for t in pooled)
    torch.testing.assert_close(
        torch.stack([t.cpu() for t in pooled]), hidden[[0, extent, 2 * extent]]
    )
