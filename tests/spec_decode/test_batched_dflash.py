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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from spyre_inference.v1.spec_decode import dflash
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner


class RecordingModel:
    def __init__(self):
        self.context = []
        self.forwards = []
        self.anchors = []

    def combine_hidden_states(self, states):
        return states

    def precompute_and_store_context_kv(self, *args):
        self.context.append(args)

    def __call__(self, *, input_ids, positions):
        self.forwards.append((input_ids.clone(), positions.clone()))
        return torch.zeros(input_ids.numel(), 128)

    def propose_block(self, hidden, anchors):
        self.anchors.append(anchors.tolist())
        return [[int(anchor)] * 15 for anchor in anchors]


@pytest.fixture
def proposer(monkeypatch):
    metadata = []
    model = RecordingModel()
    monkeypatch.setattr(dflash, "set_forward_context", lambda *args, **kwargs: nullcontext())

    def build(common):
        metadata.append(common)
        return None, {}

    return SimpleNamespace(
        model=model,
        metadata=metadata,
        spyre_device=torch.device("cpu"),
        draft_block_size=16,
        block_size=128,
        max_model_len=512,
        max_batch_size=4,
        parallel_drafting_token_id=42,
        _context_warmed_rows=set(),
        _warmed_draft_batches=set(),
        build_per_group_and_layer_attn_metadata=build,
        vllm_config=None,
        _draft_attn_layer_names={"draft"},
    )


def test_mixed_acceptance_prefill_and_finished_request_are_mapped_independently(proposer):
    starts = torch.tensor([0, 4, 7, 12, 14])
    positions = torch.cat(
        (torch.arange(10, 14), torch.arange(20, 23), torch.arange(126, 131), torch.arange(40, 42))
    )
    pages = torch.tensor([[4, 1, 3, 2], [8, 7, 6, 5], [12, 9, 10, 11], [15, 16, 14, 13]])
    common = SimpleNamespace(
        query_start_loc_cpu=starts, slot_mapping=torch.arange(100, 114), block_table_tensor=pages
    )
    features = torch.randn(16, 128)
    result = dflash.SpyreDFlashProposer.propose_spyre(
        proposer,
        [[17, 19], [], [23], [31]],
        [features],
        positions,
        SimpleNamespace(num_draft_tokens=[3, 0, 4, 0]),
        common,
        skip_proposal=[False, False, True, False],
    )
    assert result == [[19] * 15, [], [], [31] * 15]
    assert proposer.model.context[0][2].tolist() == [
        100,
        101,
        0,
        0,
        104,
        105,
        106,
        107,
        0,
        0,
        0,
        0,
        112,
        113,
        0,
        0,
    ]
    assert len(proposer.model.forwards) == 1
    assert proposer.model.anchors == [[19, 31]]
    metadata = proposer.metadata[0]
    assert metadata.query_start_loc_cpu.tolist() == [0, 16, 32]
    assert metadata.seq_lens.tolist() == [28, 58]
    assert not metadata.causal
    torch.testing.assert_close(metadata.block_table_tensor, pages[[0, 3]])
    assert metadata.slot_mapping.tolist() == list(range(4 * 128 + 12, 4 * 128 + 28)) + list(
        range(15 * 128 + 42, 15 * 128 + 58)
    )


@pytest.mark.parametrize("over_limit", [False, True])
def test_four_requests_share_one_forward_and_only_active_requests_are_padded(proposer, over_limit):
    positions = torch.cat(
        (
            torch.arange(10, 26),
            torch.tensor([500 if over_limit else 100]),
            torch.arange(120, 136),
            torch.tensor([31]),
        )
    )
    common = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 16, 17, 33, 34]),
        slot_mapping=torch.arange(200, 234),
        block_table_tensor=torch.arange(1, 17).view(4, 4),
    )
    result = dflash.SpyreDFlashProposer.propose_spyre(
        proposer,
        [[11], [22], [33] * 5, [44]],
        [torch.randn(64, 128)],
        positions,
        SimpleNamespace(num_draft_tokens=[15, 0, 15, 0]),
        common,
    )
    assert result == [[11] * 15, [] if over_limit else [22] * 15, [33] * 15, [44] * 15]
    assert len(proposer.model.forwards) == 1
    metadata = proposer.metadata[0]
    assert metadata.num_reqs == (3 if over_limit else 4)
    assert metadata.num_actual_tokens == (48 if over_limit else 64)
    ids, query_positions = proposer.model.forwards[0]
    assert len(ids) == len(query_positions) == 64
    if over_limit:
        assert proposer.model.anchors == [[11, 33, 44, 0]]
        assert metadata.slot_mapping[48:].tolist() == [0] * 16
        assert query_positions[48:].tolist() == [0] * 16
    else:
        assert proposer.model.anchors == [[11, 22, 33, 44]]
        assert metadata.seq_lens.tolist() == [27, 117, 141, 48]


def test_all_prefilling_requests_commit_context_without_draft_forward(proposer):
    common = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 2, 5, 9, 10]), slot_mapping=torch.arange(100, 110)
    )
    result = dflash.SpyreDFlashProposer.propose_spyre(
        proposer,
        [[], [], [], []],
        [torch.randn(16, 128)],
        torch.arange(10),
        None,
        common,
    )
    assert result == [[], [], [], []]
    assert not proposer.model.forwards
    assert proposer.model.context[0][2].tolist() == list(range(100, 110)) + [0] * 6


def test_output_limits_follow_request_order_after_batch_reordering():
    seen = []
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(use_dflash=lambda: True),
        input_batch=SimpleNamespace(req_ids=["c", "a", "d", "b"]),
        requests={
            name: SimpleNamespace(
                output_token_ids=[1] * count, sampling_params=SimpleNamespace(max_tokens=limit)
            )
            for name, count, limit in (("a", 5, 5), ("b", 3, 5), ("c", 9, None), ("d", 7, 5))
        },
        drafter=SimpleNamespace(propose_spyre=lambda *args, **kwargs: seen.append(kwargs)),
        _get_positions=lambda count: torch.arange(count),
        effective_drafter_max_model_len=512,
    )
    TorchSpyreModelRunner.propose_draft_token_ids(
        runner,
        SimpleNamespace(total_num_scheduled_tokens=4),
        [[1]] * 4,
        SimpleNamespace(all_greedy=True),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert seen == [{"skip_proposal": [False, True, True, False], "max_model_len": 512}]


def test_context_limit_check_reaches_per_request_dflash_handling():
    runner = SimpleNamespace(speculative_config=SimpleNamespace(use_dflash=lambda: True))
    assert TorchSpyreModelRunner._input_fits_in_drafter(runner, SimpleNamespace(max_seq_len=512))
    assert not TorchSpyreModelRunner._input_fits_in_drafter(runner, None)


@pytest.mark.parametrize(
    "capacity,shapes",
    [
        (1, [16]),
        (2, [32, 16]),
        (3, [64, 32, 16]),
        (4, [64, 32, 16]),
        (5, [128, 64, 32, 16]),
        (8, [128, 64, 32, 16]),
    ],
)
def test_warmup_covers_shrinking_and_padded_proposals_once(proposer, monkeypatch, capacity, shapes):
    proposer.max_batch_size = capacity
    proposer.dflash_config = {"target_layer_ids": [1]}
    proposer.hidden_size = 128
    proposer.dtype = torch.float16
    published = []
    monkeypatch.setattr(dflash.attn_layer, "publish_null_slots", published.append)
    dflash.SpyreDFlashProposer.dummy_run(proposer, 64)
    dflash.SpyreDFlashProposer.dummy_run(proposer, 16)
    dflash.SpyreDFlashProposer.dummy_run(proposer, 64)
    assert [len(ids) for ids, _ in proposer.model.forwards] == published == shapes
    assert [len(context[0]) for context in proposer.model.context] == [64, 16]
    for context, positions, slots in proposer.model.context:
        assert positions.shape == slots.shape == (context.shape[0],)
        assert slots.dtype == torch.int64
        assert not slots.any()


def test_proposal_rejects_requests_beyond_configured_capacity(proposer):
    proposer.max_batch_size = 2
    with pytest.raises(AssertionError, match="one to 2 active requests"):
        dflash.SpyreDFlashProposer.propose_spyre(proposer, [[1]] * 3, None, None, None, None)
