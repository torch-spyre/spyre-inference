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

from spyre_inference.models.qwen3_dflash import SpyreDFlashQwen3Model
from spyre_inference.v1.spec_decode import dflash
from spyre_inference.v1.spec_decode.dflash import SpyreDFlashProposer
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner


@pytest.mark.parametrize(
    "output_count,max_tokens,skip",
    [
        (0, 1, False),
        (1, 1, True),
        (23, 24, False),
        (24, 24, True),
        (29, 24, True),
        (24, None, False),
    ],
)
def test_output_limit_uses_tokens_already_appended_by_bookkeeping(output_count, max_tokens, skip):
    calls = []

    def propose(*args, skip_proposal=False, max_model_len=None):
        calls.append(skip_proposal)
        assert max_model_len == 512
        return [[]] if skip_proposal[0] else [[7]]

    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(use_dflash=lambda: True),
        requests={
            "request": SimpleNamespace(
                output_token_ids=[3] * output_count,
                sampling_params=SimpleNamespace(max_tokens=max_tokens),
            )
        },
        input_batch=SimpleNamespace(req_ids=["request"]),
        drafter=SimpleNamespace(propose_spyre=propose),
        _get_positions=lambda count: torch.arange(count),
        effective_drafter_max_model_len=512,
    )
    result = TorchSpyreModelRunner.propose_draft_token_ids(
        runner,
        SimpleNamespace(total_num_scheduled_tokens=4),
        [[3]],
        SimpleNamespace(all_greedy=True),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert calls == [[skip]]
    assert result == ([[]] if skip else [[7]])


def test_skipped_final_proposal_commits_only_accepted_context():
    writes = []
    features = torch.randn(16, 128)
    drafter = SimpleNamespace(
        spyre_device=torch.device("cpu"),
        model=SimpleNamespace(
            combine_hidden_states=lambda states: states,
            precompute_and_store_context_kv=lambda *args: writes.append(args),
        ),
        draft_block_size=16,
        max_model_len=512,
    )
    result = SpyreDFlashProposer.propose_spyre(
        drafter,
        [[7, 9]],
        [features],
        torch.arange(10, 14),
        SimpleNamespace(num_draft_tokens=[3]),
        SimpleNamespace(
            slot_mapping=torch.arange(128, 132), query_start_loc_cpu=torch.tensor([0, 4])
        ),
        skip_proposal=True,
    )
    assert result == [[]]
    assert len(writes) == 1
    context, positions, slots = writes[0]
    torch.testing.assert_close(context, features)
    assert positions.tolist() == [10, 11, 12, 13] + [0] * 12
    assert slots.tolist() == [128, 129] + [0] * 14


@pytest.mark.parametrize("rows", [1, 16, 64])
def test_context_projection_packs_live_transposed_weights_without_query_columns(monkeypatch, rows):
    torch.manual_seed(93)
    attentions, references = [], []
    for _ in range(2):
        query = torch.randn(256, 192)
        key, value = torch.randn(2, 128, 192)
        attention = torch.nn.Module()
        attention.qkv_proj = torch.nn.Module()
        attention.qkv_proj.weight = torch.nn.Parameter(
            torch.cat((query, key, value)).t().contiguous()
        )
        attention.q_size, attention.kv_size = 256, 128
        attention.num_kv_heads, attention.head_dim = 2, 64
        attention.k_norm = torch.nn.Identity()
        attention.rotary_emb = lambda positions, query, key: (query, key)
        attentions.append(attention)
        references.append((key, value))
    model = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=attention) for attention in attentions],
        hidden_norm=torch.nn.Identity(),
    )
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    SpyreDFlashQwen3Model.prepare_for_spyre(model)
    context = torch.randn(rows, 192)
    for attention, reference in zip(attentions, references):
        assert attention._context_kv_weight_t.shape == (192, 256)
        assert attention._context_kv_weight_t.is_contiguous()
        assert "_context_kv_weight_t" not in attention.state_dict()
        actual = model._context_project(attention, context, torch.arange(rows))
        for result, weight in zip(actual, reference):
            expected = torch.nn.functional.linear(context, weight).view(rows, 2, 64)
            torch.testing.assert_close(result, expected)


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
        max_batch_size=1,
        dtype=torch.float16,
        hidden_size=128,
        dflash_config={"target_layer_ids": [0]},
        _context_warmed_rows=set(),
        _draft_warmed=False,
        parallel_drafting_token_id=42,
        build_per_group_and_layer_attn_metadata=build,
        vllm_config=None,
        _draft_attn_layer_names={"draft"},
    )


@pytest.mark.parametrize("limit", [143, 144, 512])
def test_accepted_context_and_next_draft_use_the_request_pages(proposer, limit):
    result = SpyreDFlashProposer.propose_spyre(
        proposer,
        [[7, 9]],
        [torch.randn(16, 128)],
        torch.arange(126, 130),
        SimpleNamespace(num_draft_tokens=[3]),
        SimpleNamespace(
            query_start_loc_cpu=torch.tensor([0, 4]),
            slot_mapping=torch.arange(500, 504),
            block_table_tensor=torch.tensor([[3, 1, 4, 2]]),
        ),
        max_model_len=limit,
    )
    assert proposer.model.context[0][2].tolist() == [500, 501] + [0] * 14
    if limit == 143:
        assert result == [[]]
        assert not proposer.model.forwards
    else:
        assert result == [[9] * 15]
        ids, positions = proposer.model.forwards[0]
        assert ids.tolist() == [9] + [42] * 15
        assert positions.tolist() == list(range(128, 144))
        assert proposer.metadata[0].slot_mapping.tolist() == list(range(128, 144))
        assert proposer.metadata[0].causal is False
        assert len(proposer.model.forwards) == 1


def test_partial_prefill_commits_context_without_a_proposal(proposer):
    result = SpyreDFlashProposer.propose_spyre(
        proposer,
        [[]],
        [torch.randn(16, 128)],
        torch.arange(4),
        None,
        SimpleNamespace(
            query_start_loc_cpu=torch.tensor([0, 4]), slot_mapping=torch.arange(256, 260)
        ),
    )
    assert result == [[]]
    assert proposer.model.context[0][2].tolist() == list(range(256, 260)) + [0] * 12
    assert not proposer.model.forwards


def test_context_limit_replaces_previous_host_drafts_with_empty_rows(proposer):
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(use_dflash=lambda: True),
        input_batch=SimpleNamespace(req_ids=["request"]),
        use_async_scheduling=False,
    )
    for position, scheduled, drafts, expected_width in (
        (494, 1, 0, 15),
        (495, 16, 15, 15),
        (496, 16, 15, 0),
        (497, 1, 0, 0),
    ):
        positions = torch.arange(position, position + scheduled)
        table = torch.tensor([[3, 1, 4, 2]])
        common = SimpleNamespace(
            query_start_loc_cpu=torch.tensor([0, scheduled]),
            slot_mapping=table[0, positions // 128] * 128 + positions % 128,
            block_table_tensor=table,
        )
        assert TorchSpyreModelRunner._input_fits_in_drafter(runner, common)
        runner._draft_token_ids = SpyreDFlashProposer.propose_spyre(
            proposer,
            [[7]],
            [torch.randn(scheduled, 128)],
            positions,
            SimpleNamespace(num_draft_tokens=[drafts]) if drafts else None,
            common,
        )
        TorchSpyreModelRunner._copy_draft_token_ids_to_cpu(runner, None)
        tokens, request_ids = TorchSpyreModelRunner._get_draft_token_ids_cpu(runner)
        assert tokens == [[7] * expected_width]
        assert request_ids == ["request"]
        assert runner._draft_token_req_ids == ["request"]
    assert len(proposer.model.forwards) == 2
    assert len(proposer.model.context) == 4


def test_context_warms_once_per_row_bucket_and_draft_warms_once(proposer, monkeypatch):
    monkeypatch.setattr(dflash.attn_layer, "publish_null_slots", lambda _: None)
    for count in (16, 16, 64, 16, 64):
        SpyreDFlashProposer.dummy_run(proposer, count)
    assert len(proposer.model.context) == 2
    assert len(proposer.model.forwards) == 1
    assert proposer.model.anchors == [[0]]


def test_inherited_loader_shares_untied_embeddings_and_lm_head_separately(monkeypatch):
    from vllm.v1.spec_decode import llm_base_proposer

    monkeypatch.setattr(llm_base_proposer, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(73, 128)),
        lm_head=torch.nn.Linear(128, 73, bias=False),
    )
    proposer = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(embed_tokens=torch.nn.Embedding(73, 128)),
            lm_head=torch.nn.Linear(128, 73, bias=False),
            has_own_embed_tokens=False,
            has_own_lm_head=False,
        ),
        vllm_config=SimpleNamespace(speculative_config=None),
        use_local_argmax_reduction=False,
    )
    SpyreDFlashProposer._maybe_share_embeddings(proposer, target)
    SpyreDFlashProposer._maybe_share_lm_head(proposer, target)
    assert proposer.model.model.embed_tokens is target.model.embed_tokens
    assert proposer.model.lm_head is target.lm_head
    assert proposer.model.lm_head.weight is not proposer.model.model.embed_tokens.weight
