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

import numpy as np
import pytest
import torch
from vllm.config import CompilationMode
from vllm.v1.outputs import SamplerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from spyre_inference.v1.worker import spyre_model_runner
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner


@pytest.fixture()
def cpu_sampling_runner():
    runner = object.__new__(TorchSpyreModelRunner)
    generators = {i: torch.Generator().manual_seed(37 + i) for i in range(2)}
    runner.input_batch = SimpleNamespace(
        num_reqs=3,
        req_ids=["prefill", "decode", "unseeded_prefill"],
        req_id_to_index={"prefill": 0, "decode": 1, "unseeded_prefill": 2},
        generators=generators,
        sampling_metadata=SimpleNamespace(generators=generators),
        update_async_output_token_ids=MagicMock(),
        vocab_size=10,
        num_tokens_no_spec=np.array([2, 2, 2]),
        token_ids_cpu=np.zeros((3, 8), dtype=np.int64),
        is_token_ids=np.zeros((3, 8), dtype=bool),
    )
    runner.discard_request_mask = SimpleNamespace(np=np.array([True, False, True, True]))
    runner.use_async_scheduling = False
    runner.max_model_len = 8
    runner.requests = {
        req_id: SimpleNamespace(output_token_ids=[]) for req_id in runner.input_batch.req_ids
    }
    return runner


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_discarded_prefill_does_not_advance_cpu_rng(cpu_sampling_runner, speculative, raises):
    runner = cpu_sampling_runner
    generators = runner.input_batch.generators
    original_states = {i: gen.get_state() for i, gen in generators.items()}
    result = object()
    num_draws = 17 if speculative else 1

    def sample(*args, **kwargs):
        for gen in runner.input_batch.sampling_metadata.generators.values():
            torch.rand(num_draws, generator=gen)
        if raises:
            raise RuntimeError("sampling failed")
        return result

    runner.sampler = MagicMock(side_effect=sample)
    runner.rejection_sampler = MagicMock(side_effect=sample)
    runner._get_spec_decode_draft_probs = MagicMock(return_value=None)
    metadata = object() if speculative else None
    if raises:
        with pytest.raises(RuntimeError, match="sampling failed"):
            runner._sample(None, metadata)
    else:
        assert runner._sample(None, metadata) is result
    assert runner.sampler.call_count == int(not speculative)
    assert runner.rejection_sampler.call_count == int(speculative)
    assert torch.equal(generators[0].get_state(), original_states[0])
    assert not torch.equal(generators[1].get_state(), original_states[1])
    assert runner.input_batch.generators is generators
    assert runner.input_batch.sampling_metadata.generators is generators

    # Completing prefill must start at the seed's first draw; live requests
    # continue from their next draw rather than repeating the previous one.
    runner.discard_request_mask.np[:3] = False
    for i, gen in generators.items():
        reference = torch.Generator()
        reference.set_state(original_states[i])
        if i == 1:
            torch.rand(num_draws, generator=reference)
        torch.testing.assert_close(
            torch.rand(num_draws, generator=gen),
            torch.rand(num_draws, generator=reference),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("speculative", [False, True])
def test_cpu_bookkeeping_discards_prefill_without_cuda_offsets(cpu_sampling_runner, speculative):
    runner = cpu_sampling_runner
    generators = runner.input_batch.generators
    original_states = {i: gen.get_state() for i, gen in generators.items()}
    runner._to_list = lambda tokens: tokens.tolist()
    runner._get_prompt_logprobs_dict = MagicMock(return_value={})
    token_ids = [[5, 9, -1], [6, 7, -1], [7, 9, -1]] if speculative else [[5], [6], [7]]
    output = SamplerOutput(sampled_token_ids=torch.tensor(token_ids), logprobs_tensors=None)
    scheduler = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(runner.input_batch.req_ids, 1))
    result = runner._bookkeeping_sync(scheduler, output, None, torch.empty(3, 1), 3)
    expected = [6, 7] if speculative else [6]
    assert result[3] == [[], expected, []]
    assert result[5] == runner.input_batch.req_ids
    assert result[5] is not runner.input_batch.req_ids
    assert runner.requests["decode"].output_token_ids == expected
    assert runner.requests["prefill"].output_token_ids == []
    assert runner.requests["unseeded_prefill"].output_token_ids == []
    assert runner.input_batch.num_tokens_no_spec.tolist() == [2, 2 + len(expected), 2]
    assert runner.input_batch.token_ids_cpu[1, 2 : 2 + len(expected)].tolist() == expected
    assert runner.input_batch.generators is generators
    assert runner.input_batch.sampling_metadata.generators is generators
    for i, gen in generators.items():
        assert torch.equal(gen.get_state(), original_states[i])


def test_cpu_bookkeeping_restores_generator_mapping_on_error(cpu_sampling_runner, monkeypatch):
    runner = cpu_sampling_runner
    generators = runner.input_batch.generators

    def fail(self, *args):
        assert self.input_batch.generators == {1: generators[1]}
        assert self.input_batch.sampling_metadata.generators is generators
        raise RuntimeError("bookkeeping failed")

    monkeypatch.setattr(GPUModelRunner, "_bookkeeping_sync", fail)
    with pytest.raises(RuntimeError, match="bookkeeping failed"):
        runner._bookkeeping_sync(None, None, None, None, 0)
    assert runner.input_batch.generators is generators


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
