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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CompilationMode

from spyre_inference.v1.worker import spyre_model_runner
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner


@pytest.mark.parametrize("zeros_only", [False, True])
@pytest.mark.parametrize("num_reqs", [1, 4])
def test_host_drafts_and_context_limit_clear_do_not_need_cuda(zeros_only, num_reqs):
    runner = SimpleNamespace(
        _draft_token_ids=torch.tensor([[3, 5, 7]] * num_reqs, dtype=torch.int32),
        input_batch=SimpleNamespace(req_ids=[str(i) for i in range(num_reqs)]),
        use_async_scheduling=False,
    )
    TorchSpyreModelRunner._copy_draft_token_ids_to_cpu(runner, None, zeros_only=zeros_only)
    assert runner._draft_token_ids == ([[0, 0, 0]] if zeros_only else [[3, 5, 7]]) * num_reqs
    assert runner._draft_token_req_ids == runner.input_batch.req_ids
    assert runner._draft_token_req_ids is not runner.input_batch.req_ids
    assert runner.prev_num_spec_tokens == 3


def test_zero_drafts_do_not_read_device_tensor_and_rows_are_independent():
    runner = SimpleNamespace(
        _draft_token_ids=torch.empty(4, 3, device="meta", dtype=torch.int32),
        input_batch=SimpleNamespace(req_ids=["a", "b", "c", "d"]),
        use_async_scheduling=False,
    )
    TorchSpyreModelRunner._copy_draft_token_ids_to_cpu(runner, None, zeros_only=True)
    assert runner._draft_token_ids == [[0, 0, 0]] * 4
    runner._draft_token_ids[0][0] = 1
    assert runner._draft_token_ids[1:] == [[0, 0, 0]] * 3


def test_host_draft_bookkeeping_rejects_async_scheduling():
    runner = SimpleNamespace(use_async_scheduling=True)
    with pytest.raises(AssertionError, match="async scheduling is unsupported"):
        TorchSpyreModelRunner._copy_draft_token_ids_to_cpu(runner, None)


@pytest.fixture()
def recording_runner(monkeypatch):
    monkeypatch.setenv("SPYRE_ATTN_RECORD", "1")
    compilation_config = SimpleNamespace(
        mode=CompilationMode.STOCK_TORCH_COMPILE,
        static_forward_context={},
        inductor_compile_config={},
    )
    runner = SimpleNamespace(
        compilation_config=compilation_config,
        vllm_config=SimpleNamespace(compilation_config=compilation_config),
        _spyre_kv_caches={"target": object(), "draft.0": object(), "draft.1": object()},
        attn_groups=[[SimpleNamespace(layer_names=["target"])]],
        _attn_metadata_builders=lambda: {},
        speculative_config=object(),
        drafter=SimpleNamespace(
            model=SimpleNamespace(
                get_draft_kv_cache_layer_names=lambda: ["draft.0", "draft.1"],
                get_draft_attn_causal=lambda: [False, True],
            )
        ),
    )
    record = MagicMock()
    monkeypatch.setattr(spyre_model_runner, "record_attention_graphs", record)
    return runner, record


@pytest.mark.parametrize("with_draft", [False, True])
def test_attention_warmup_reads_causality_from_draft_model(recording_runner, with_draft):
    runner, record = recording_runner
    if not with_draft:
        runner.speculative_config = None
        runner._spyre_kv_caches = {"target": object()}
    TorchSpyreModelRunner._record_attention_graphs(runner)
    expected = (
        {"target": True, "draft.0": False, "draft.1": True} if with_draft else {"target": True}
    )
    assert record.call_args.kwargs["causal_by_layer"] == expected


def test_attention_warmup_requires_draft_model_causality(recording_runner):
    runner, record = recording_runner
    del runner.drafter.model.get_draft_attn_causal
    with pytest.raises(AttributeError, match="get_draft_attn_causal"):
        TorchSpyreModelRunner._record_attention_graphs(runner)
    record.assert_not_called()


def test_attention_warmup_rejects_misaligned_draft_causality(recording_runner):
    runner, record = recording_runner
    runner.drafter.model.get_draft_attn_causal = lambda: [False]
    with pytest.raises(ValueError, match="zip"):
        TorchSpyreModelRunner._record_attention_graphs(runner)
    record.assert_not_called()
