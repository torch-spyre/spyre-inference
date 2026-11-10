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

"""spec_decode.py: the accepted-context bookkeeping and the draft step, against a
recording draft model."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from spyre_inference.v1.spec_decode.dflash import accepted_context_length
from spyre_inference.v1.worker.spyre import model_runner, spec_decode
from spyre_inference.v1.worker.spyre.spec_decode import SpyreDFlashSpeculator, plan_drafts

BLOCK = 4  # one anchor and three drafts
PAGE = 64
MASK = 99


@pytest.mark.parametrize(
    "scheduled,drafts,sampled", [(64, 0, 1), (5, 0, 1), (4, 3, 1), (4, 3, 2), (4, 3, 4)]
)
def test_committed_rows_match_the_v1_bookkeeping(scheduled, drafts, sampled):
    """MRV2 reports num_rejected = drafts + 1 - sampled; V1 counted sampled tokens."""
    committed, _ = plan_drafts(
        [0, scheduled], [sampled], [drafts + 1 - sampled], torch.arange(scheduled), BLOCK, 512
    )
    assert len(committed[0]) == accepted_context_length(scheduled, drafts, sampled)


def test_a_prefill_chunk_commits_its_rows_and_drafts_nothing():
    committed, starts = plan_drafts([0, 64], [0], [0], torch.arange(64), BLOCK, 512)
    assert committed == [range(0, 64)] and starts == [None]


def test_the_draft_block_follows_the_last_accepted_position():
    positions = torch.tensor([10, 11, 12, 13, 40, 41, 42, 43])
    committed, starts = plan_drafts([0, 4, 8], [2, 4], [2, 0], positions, BLOCK, 512)
    assert committed == [range(0, 2), range(4, 8)]
    assert starts == [12, 44]


def test_a_block_past_the_context_limit_is_not_drafted():
    positions = torch.tensor([506, 507, 508, 509])
    _, starts = plan_drafts([0, 4], [4], [0], positions, BLOCK, 514)
    assert starts == [510]
    _, starts = plan_drafts([0, 4], [4], [0], positions, BLOCK, 513)
    assert starts == [None]


def test_more_rejected_rows_than_scheduled_is_an_error():
    with pytest.raises(ValueError, match="rejected"):
        plan_drafts([0, 4], [0], [4], torch.arange(4), BLOCK, 512)


class _DraftModel:
    """Records every call; proposals are the anchor followed by its successors."""

    def __init__(self):
        self.stored: list[tuple] = []
        self.forwards: list[tuple] = []
        self.anchors: list[list[int]] = []

    def combine_hidden_states(self, features):
        return features

    def precompute_and_store_context_kv(self, context, positions, slots):
        self.stored.append((context.shape[0], positions.cpu().tolist(), slots.tolist()))

    def __call__(self, *, input_ids, positions):
        self.forwards.append((input_ids.cpu().tolist(), positions.cpu().tolist()))
        return torch.zeros(input_ids.shape[0], 8)

    def propose_block(self, hidden, anchor_ids):
        self.anchors.append(anchor_ids.tolist())
        return [[a + 1 + k for k in range(BLOCK - 1)] for a in anchor_ids.tolist()]


@pytest.fixture
def speculator(monkeypatch):
    """A speculator over a recording model: one KV group of 64-token pages, and the
    step's gathered block tables giving batch row r page r + 1 (page 0 is the null page)."""
    monkeypatch.setattr(spec_decode, "convert", lambda t, device=None: t)
    built = []
    monkeypatch.setattr(spec_decode, "build_attn_metadata", lambda **kw: built.append(kw) or {})
    monkeypatch.setattr(spec_decode, "set_forward_context", lambda *a, **k: _Null())
    monkeypatch.setattr(spec_decode.attn_layer, "publish_null_slots", lambda n: None)

    s = SpyreDFlashSpeculator.__new__(SpyreDFlashSpeculator)
    s.spyre_device = torch.device("cpu")
    s.model = _DraftModel()
    s.num_query_per_req = BLOCK
    s.num_speculative_steps = BLOCK - 1
    s.max_num_reqs = 4
    s.parallel_drafting_token_id = MASK
    s.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=512))
    s.draft_tokens = torch.full((4, BLOCK - 1), -7, dtype=torch.int64)
    s.draft_attn_layer_names = {"draft.0"}
    s.draft_kv_cache_group_ids = [0]
    s.attn_groups = [[]]
    s.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["draft.0"])])
    s.block_tables = SimpleNamespace(
        block_sizes=[PAGE],
        input_block_tables=[torch.tensor([[1, 0], [2, 0], [3, 0], [4, 0]], dtype=torch.int32)],
    )
    s._warmed_context_rows = set()
    s._draft_warmed = False
    s.built = built
    return s


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _batch(query_start_loc, positions, idx_mapping, padded_rows):
    num_reqs = len(query_start_loc) - 1
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_tokens=query_start_loc[-1],
        positions=torch.tensor(positions + [0] * (padded_rows - len(positions))),
        query_start_loc_np=np.array(query_start_loc, dtype=np.int32),
        idx_mapping=torch.tensor(idx_mapping, dtype=torch.int32),
    )


def _propose(s, batch, slots, sampled, rejected, last_sampled, padded_rows, **kw):
    features = torch.zeros(padded_rows, 8)
    return s.propose(
        batch,
        {},
        {"draft.0": torch.tensor(slots + [0] * (padded_rows - len(slots)))},
        features,
        [features[:, :4], features[:, 4:]],
        torch.tensor(sampled),
        torch.tensor(rejected),
        torch.tensor(last_sampled).view(-1, 1),
        None,
        None,
        None,
        **kw,
    )


def test_a_verify_step_stores_accepted_context_and_drafts_from_it(speculator):
    """Request 0 (state slot 2) accepts one draft; request 1 (state slot 0) all three."""
    batch = _batch([0, 4, 8], [20, 21, 22, 23, 30, 31, 32, 33], [2, 0], 16)
    slots = [148, 149, 150, 151, 94, 95, 96, 97]
    drafts = _propose(speculator, batch, slots, [2, 4], [2, 0], [500, 0, 7, 0], 16)

    rows, positions, stored = speculator.model.stored[0]
    assert rows == 16 and positions[:8] == batch.positions[:8].tolist()
    # Rejected rows and the bucket padding write the null page.
    assert stored == [148, 149, 0, 0, 94, 95, 96, 97] + [0] * 8
    # Anchors come from each request's own state slot.
    assert speculator.model.anchors == [[7, 500]]
    ids, positions = speculator.model.forwards[0]
    assert ids == [7, MASK, MASK, MASK, 500, MASK, MASK, MASK]
    assert positions == [22, 23, 24, 25, 34, 35, 36, 37]
    meta = speculator.built[0]
    assert meta["causal"] is False and meta["num_reqs"] == 2
    assert meta["slot_mappings"][0].tolist() == [1 * PAGE + p for p in (22, 23, 24, 25)] + [
        2 * PAGE + p for p in (34, 35, 36, 37)
    ]
    assert drafts.tolist() == [[8, 9, 10], [501, 502, 503]]


def test_three_drafting_requests_pad_to_four(speculator):
    batch = _batch([0, 4, 8, 12], list(range(12)), [0, 1, 2], 16)
    _propose(speculator, batch, list(range(12)), [4, 4, 4], [0, 0, 0], [1, 2, 3, 0], 16)
    ids, _ = speculator.model.forwards[0]
    assert len(ids) == 4 * BLOCK and ids[3 * BLOCK :] == [0, MASK, MASK, MASK]
    assert speculator.built[0]["num_tokens"] == 3 * BLOCK


def test_requests_that_draft_nothing_get_zeros_and_only_store_context(speculator):
    """A prefill chunk, and a step where no request can use its drafts."""
    batch = _batch([0, 16], list(range(16)), [0], 16)
    drafts = _propose(speculator, batch, list(range(64, 80)), [0], [0], [5, 0, 0, 0], 16)
    assert drafts.tolist() == [[0, 0, 0]] and not speculator.model.forwards

    batch = _batch([0, 4], [16, 17, 18, 19], [0], 16)
    drafts = _propose(
        speculator, batch, [80, 81, 82, 83], [4], [0], [5, 0, 0, 0], 16, num_speculative_tokens=0
    )
    assert drafts.tolist() == [[0, 0, 0]] and not speculator.model.forwards
    assert len(speculator.model.stored) == 2


def test_warmup_stores_each_row_count_once_and_drafts_every_batch_once(speculator):
    for rows in (128, 16, 128):
        batch = _batch([0, rows], [0] * rows, [0], rows)
        _propose(speculator, batch, [0] * rows, [1], [0], [0] * 4, rows, dummy_run=True)
    assert [(rows, set(slots)) for rows, _, slots in speculator.model.stored] == [
        (128, {0}),
        (16, {0}),
    ]
    assert [len(ids) for ids, _ in speculator.model.forwards] == [4 * BLOCK, 2 * BLOCK, BLOCK]


def test_the_runner_replaces_upstreams_speculator(monkeypatch):
    def upstream_init(self, vllm_config, device):
        self.device = device
        self.speculator = "upstream"

    monkeypatch.setattr(GPUModelRunner, "__init__", upstream_init)
    monkeypatch.setattr(model_runner, "validate_dflash_config", lambda config: None)
    monkeypatch.setattr(
        model_runner,
        "SpyreDFlashSpeculator",
        lambda config, device, spyre_device: ("spyre", device, spyre_device),
    )
    spyre = torch.device("meta")
    runner = model_runner.SpyreModelRunnerV2(SimpleNamespace(), spyre)
    assert runner.speculator == ("spyre", torch.device("cpu"), spyre)


def test_the_runner_builds_no_speculator_without_one(monkeypatch):
    monkeypatch.setattr(
        GPUModelRunner, "__init__", lambda self, c, d: setattr(self, "speculator", None)
    )
    monkeypatch.setattr(model_runner, "validate_dflash_config", lambda config: None)
    runner = model_runner.SpyreModelRunnerV2(SimpleNamespace(), torch.device("meta"))
    assert runner.speculator is None
