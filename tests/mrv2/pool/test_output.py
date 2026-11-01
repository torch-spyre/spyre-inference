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

"""pool/output.py: the pooler output reaches the host synchronously, per request."""

import types

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.v1.pool.spyre_pooler import select_rows
from spyre_inference.v1.worker.spyre import device_shims
from spyre_inference.v1.worker.spyre.pool import output


def test_copy_keeps_only_finished_requests() -> None:
    pooled = torch.arange(6, dtype=torch.float16).view(3, 2)

    host = output.copy_pooler_output_to_cpu(pooled, [True, False, True])

    assert host[1] is None
    torch.testing.assert_close(host[0], pooled[0])
    torch.testing.assert_close(host[2], pooled[2])


def test_copy_accepts_a_per_request_list() -> None:
    pooled = [torch.zeros(3, 2), torch.ones(1, 2)]

    host = output.copy_pooler_output_to_cpu(pooled, [True, True])

    assert [tuple(t.shape) for t in host] == [(3, 2), (1, 2)]


def _runner(pooler_output, finished_mask, num_reqs=2):
    calls = []
    state = types.SimpleNamespace(
        input_batch=types.SimpleNamespace(
            req_ids=[f"r{i}" for i in range(num_reqs)], num_reqs=num_reqs
        ),
        hidden_states=torch.zeros(2, 3),
        finished_req_ids=set(),
        ec_connector_output=None,
    )
    runner = types.SimpleNamespace(
        execute_model_state=state,
        kv_connector=types.SimpleNamespace(post_forward=lambda ids: "kv-out"),
        pooling_runner=types.SimpleNamespace(
            pool=lambda hidden, batch, req_states: (pooler_output, finished_mask)
        ),
        req_states=None,
        postprocess_num_computed_tokens=lambda batch: calls.append(batch),
    )
    return runner, calls


def test_pool_sync_returns_host_output_directly(monkeypatch) -> None:
    fenced = []
    monkeypatch.setattr(device_shims, "spyre_fence", lambda: fenced.append(True))
    runner, calls = _runner(torch.ones(2, 3), [True, False])

    result = output.pool_sync(runner)  # ty: ignore[invalid-argument-type]

    assert result.req_id_to_index == {"r0": 0, "r1": 1}
    assert result.kv_connector_output == "kv-out"
    assert result.pooler_output[1] is None
    assert result.pooler_output[0].device.type == "cpu"
    assert runner.execute_model_state is None
    assert fenced == [True] and len(calls) == 1


def test_pool_sync_with_nothing_finished_reports_none_per_request(monkeypatch) -> None:
    monkeypatch.setattr(device_shims, "spyre_fence", lambda: None)
    runner, _ = _runner(torch.ones(2, 3), [False, False])

    result = output.pool_sync(runner)  # ty: ignore[invalid-argument-type]

    assert result.pooler_output == [None, None]


def test_pool_sync_after_a_failed_step_returns_none() -> None:
    runner, _ = _runner(None, [])
    runner.execute_model_state = None

    assert output.pool_sync(runner) is None  # ty: ignore[invalid-argument-type]


@pytest.mark.spyre_device
def test_pooled_spyre_tensor_matches_the_cpu_result() -> None:
    """Light hardware test: gather rows of a [4, 64] Spyre tensor and bring them back."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    torch.manual_seed(0)
    hidden = torch.randn(4, 64, dtype=torch.float16)
    rows = torch.tensor([0, 2])

    pooled = select_rows(hidden.to("spyre"), rows)
    host = output.copy_pooler_output_to_cpu(pooled, [True, True])

    assert all(t.device.type == "cpu" for t in host)
    torch.testing.assert_close(torch.stack(host), torch.index_select(hidden, 0, rows))
