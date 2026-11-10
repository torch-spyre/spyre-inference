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

"""spec_decode.py's overrides of upstream's DFlashSpeculator hooks, against a recording
draft model."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from spyre_inference.v1.worker.spyre import model_runner, spec_decode
from spyre_inference.v1.worker.spyre.spec_decode import SpyreDFlashSpeculator, _DraftBatchPadding

BLOCK = 4  # one anchor and three drafts
MASK = 99


class _DraftModel:
    def __init__(self):
        self.stored: list[tuple] = []
        self.forwards: list[tuple] = []

    def combine_hidden_states(self, features):
        return features * 2

    def precompute_and_store_context_kv(self, context, positions, slots):
        self.stored.append((context.tolist(), positions.tolist(), slots.tolist()))

    def __call__(self, *, input_ids, positions):
        self.forwards.append((input_ids.tolist(), positions.tolist()))
        return torch.zeros(input_ids.shape[0], 8)

    def propose_block(self, hidden, anchor_ids):
        return [[a + 1 + k for k in range(BLOCK - 1)] for a in anchor_ids.tolist()]


class _Holder:
    def __init__(self):
        self.published: list[list[int]] = []

    def publish(self, slots):
        self.published.append(slots.tolist())


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def speculator(monkeypatch):
    monkeypatch.setattr(spec_decode, "convert", lambda t, device=None, dtype=None: t)
    monkeypatch.setattr(spec_decode, "set_forward_context", lambda *a, **k: _Null())
    nulls = []
    monkeypatch.setattr(spec_decode.attn_layer, "publish_null_slots", nulls.append)

    s = SpyreDFlashSpeculator.__new__(SpyreDFlashSpeculator)
    s.spyre_device = torch.device("cpu")
    s.vllm_config = None
    s.model = _DraftModel()
    s.num_query_per_req = BLOCK
    s.max_num_reqs = 4
    s.parallel_drafting_token_id = MASK
    s.draft_tokens = torch.full((4, BLOCK - 1), -7, dtype=torch.int64)
    s.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(64, dtype=torch.int32), positions=torch.zeros(64, dtype=torch.int64)
    )
    s.context_positions = torch.arange(100, 164)
    s._context_slot_mappings = torch.arange(1000, 1064).view(1, -1)
    s._context = None
    s.holder = _Holder()
    group = SimpleNamespace(
        kv_cache_group_id=0, get_metadata_builder=lambda: SimpleNamespace(_slot_mapping=s.holder)
    )
    s.attn_groups = [[group]]
    s.draft_kv_cache_group_id = 0
    s.block_tables = SimpleNamespace(slot_mappings=torch.arange(500, 564).view(1, -1))
    s.nulls = nulls
    return s


@pytest.mark.parametrize("num_reqs,padded", [(1, 1), (2, 2), (3, 4), (4, 4)])
def test_the_draft_step_pads_to_a_power_of_two_request_count(num_reqs, padded):
    desc = _DraftBatchPadding(BLOCK).dispatch(num_reqs, num_reqs * BLOCK, BLOCK, 0)
    assert desc.cg_mode is CUDAGraphMode.NONE
    assert (desc.num_reqs, desc.num_tokens) == (num_reqs, padded * BLOCK)


def test_context_stays_at_the_targets_padded_width_and_padding_writes_the_null_page(speculator):
    """A 3-row step at a 6-row target bucket: rows 3-5 store against page 0."""
    aux = [torch.ones(6, 1), torch.zeros(6, 1)]
    speculator._store_hidden_states(3, None, aux)
    speculator._precompute_context_kv(0, 3)
    context, positions, slots = speculator.model.stored[0]
    assert context == [[2.0, 0.0]] * 6
    assert positions == [100, 101, 102, 0, 0, 0]
    assert slots == [1000, 1001, 1002, 0, 0, 0]


def test_a_padded_draft_step_publishes_null_slots_for_the_padded_rows(speculator):
    """Upstream's eager metadata publishes the real rows only; the traced KV write needs
    one slot per row of the padded batch."""
    speculator.input_buffers.input_ids[:12] = torch.tensor([7, MASK, MASK, MASK] * 3)
    speculator.input_buffers.input_ids[0], speculator.input_buffers.input_ids[4] = 7, 8
    speculator.input_buffers.input_ids[8] = 9
    speculator._generate_draft(3, 4 * BLOCK, None, {}, None)

    assert speculator.holder.published == [list(range(500, 512)) + [0] * BLOCK]
    ids, _ = speculator.model.forwards[0]
    assert len(ids) == 4 * BLOCK
    assert speculator.draft_tokens[:3].tolist() == [[8, 9, 10], [9, 10, 11], [10, 11, 12]]


def test_an_unpadded_draft_step_keeps_upstreams_slots(speculator):
    speculator._generate_draft(2, 2 * BLOCK, None, {}, None)
    assert speculator.holder.published == []


def test_a_warmup_dummy_stores_context_against_the_null_page_and_skips_upstream(
    speculator, monkeypatch
):
    monkeypatch.setattr(
        spec_decode.DFlashSpeculator, "propose", lambda *a, **k: pytest.fail("upstream ran")
    )
    batch = SimpleNamespace(num_reqs=2)
    out = speculator.propose(batch, None, None, None, [torch.ones(5, 2)], dummy_run=True)
    assert out.shape == (2, BLOCK - 1)
    ((_, _, slots),) = speculator.model.stored
    assert slots == [0] * 5


def test_a_real_step_runs_upstreams_propose(speculator, monkeypatch):
    seen = []
    monkeypatch.setattr(
        spec_decode.DFlashSpeculator, "propose", lambda self, *a, **k: seen.append(k) or "up"
    )
    assert speculator.propose("batch", None, None, None, None, num_speculative_tokens=3) == "up"
    assert seen == [{"num_speculative_tokens": 3}]


def test_warmup_drafts_every_padded_batch_once(speculator):
    speculator.warm_draft_batches()
    assert [len(ids) for ids, _ in speculator.model.forwards] == [4 * BLOCK, 2 * BLOCK, BLOCK]
    assert speculator.nulls == [4 * BLOCK, 2 * BLOCK, BLOCK]
    ids, positions = speculator.model.forwards[-1]
    assert ids == [0, MASK, MASK, MASK] and positions == list(range(BLOCK))
    assert speculator.draft_tokens.eq(0).all()


def test_set_attn_splits_the_draft_groups_after_upstream_builds_them(monkeypatch):
    calls = []
    monkeypatch.setattr(
        spec_decode.DFlashSpeculator,
        "set_attn",
        lambda self, *a: (calls.append("upstream"), setattr(self, "attn_groups", [["g"]])),
    )
    monkeypatch.setattr(
        spec_decode, "split_by_window", lambda groups, *a: calls.append(("split", groups, a))
    )
    s = SpyreDFlashSpeculator.__new__(SpyreDFlashSpeculator)
    s.device = torch.device("cpu")
    s.vllm_config = SimpleNamespace(attention_config=SimpleNamespace())
    monkeypatch.setattr(SpyreDFlashSpeculator, "attn_vllm_config", "draft-config")
    s.set_attn(None, None, SimpleNamespace(block_sizes=[128]), None, None)
    assert calls == ["upstream", ("split", [["g"]], ("draft-config", s.device, [128]))]


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
