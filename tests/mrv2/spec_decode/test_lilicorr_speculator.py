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

"""SpyreLiLiCorrSpeculator: anchors, sample slots, padded batches and warmup."""

from types import SimpleNamespace as NS

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import LiLiCorrSpeculator

from spyre_inference.v1.worker.spyre import spec_decode
from spyre_inference.v1.worker.spyre.spec_decode import (
    SpyreLiLiCorrSpeculator,
    make_spyre_speculator,
)

BLOCK = 4  # one anchor and three draft slots
STEPS = BLOCK - 1
TOPK = 2
HIDDEN = 4
MAX_REQS = 4
MASK = 99


class _FakeModel:
    def __init__(self) -> None:
        self.scored: list[tuple] = []
        self.lilicorr = NS(candidate_topk=TOPK, score=self.score)
        self.model = NS(lilicorr=self.lilicorr)

    def compute_candidates(self, hidden):
        # Key the ids by the gathered row value, which _run_model sets to the
        # global row index, so the expected tokens are visible downstream.
        rows = hidden[:, 0].long()[:, None]
        ids = rows * TOPK + torch.arange(TOPK)
        return ids, torch.zeros(hidden.shape[0], TOPK)

    def score(
        self,
        *,
        token_embeddings,
        candidate_log_probs,
        pass_hidden,
        anchor_hidden,
        anchor_valid,
    ):
        self.scored.append(
            (token_embeddings, candidate_log_probs, pass_hidden, anchor_hidden, anchor_valid)
        )
        batch, slots, topk, _ = token_embeddings.shape
        return torch.zeros(batch, slots, topk, topk)


class _FakeCandidateSampler:
    """Records the upstream call and emulates its greedy output."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def sample(
        self,
        candidate_ids,
        scores,
        num_reqs,
        sample_pos,
        idx_mapping,
        temperature,
        seeds,
        draft_tokens,
        draft_logits,
        use_fp64,
    ) -> None:
        self.calls.append(
            (
                candidate_ids,
                scores,
                num_reqs,
                sample_pos,
                idx_mapping,
                temperature,
                seeds,
                draft_logits,
                use_fp64,
            )
        )
        # Zero scores make the real walk pick candidate zero at every slot, and
        # rows whose request mapping is -1 are padding and stay untouched.
        steps = candidate_ids.shape[1]
        mapping = idx_mapping[: num_reqs * steps].view(num_reqs, steps)[:, 0]
        for row in range(num_reqs):
            if int(mapping[row]) >= 0:
                draft_tokens[row] = candidate_ids[row, :, 0]


@pytest.fixture
def speculator(monkeypatch):
    monkeypatch.setattr(spec_decode, "convert", lambda t, device=None, dtype=None: t)
    nulls = []
    monkeypatch.setattr(spec_decode.attn_layer, "publish_null_slots", nulls.append)

    s = SpyreLiLiCorrSpeculator.__new__(SpyreLiLiCorrSpeculator)
    s.spyre_device = torch.device("cpu")
    s.num_query_per_req = BLOCK
    s.num_speculative_steps = STEPS
    s.max_num_reqs = MAX_REQS
    s.parallel_drafting_token_id = MASK
    s.draft_tokens = torch.full((MAX_REQS, STEPS), -7, dtype=torch.int64)
    s.input_buffers = NS(
        input_ids=torch.zeros(64, dtype=torch.int32),
        positions=torch.zeros(64, dtype=torch.int64),
    )
    s.anchor_hidden = torch.zeros(MAX_REQS, HIDDEN)
    s.anchor_valid = torch.ones(MAX_REQS, dtype=torch.float16)
    s._sample_row_cache = {}
    s._sample_row_device_cache = {}
    s._anchor_index_cache = {}
    s._anchor_mask_cache = {}
    s._context = None
    s.attn_groups = []
    s.nulls = nulls
    s.model = _FakeModel()
    s.target_embeddings = lambda ids: torch.zeros(ids.shape[0], HIDDEN)
    s._run_model = lambda num_tokens, *args, **kwargs: torch.arange(
        num_tokens, dtype=torch.float32
    )[:, None].expand(num_tokens, HIDDEN)
    s.candidate_sampler = _FakeCandidateSampler()
    s.sample_pos = torch.zeros(MAX_REQS * STEPS, dtype=torch.int64)
    s.sample_idx_mapping = torch.full((MAX_REQS * STEPS,), -1, dtype=torch.int64)
    s.temperature = torch.zeros(MAX_REQS)
    s.seeds = torch.arange(MAX_REQS)
    s.draft_logits = None
    s.use_fp64_gumbel = False
    return s


def test_anchor_is_the_last_committed_normalized_row(speculator):
    context = torch.arange(6 * HIDDEN, dtype=torch.float32).view(6, HIDDEN)
    speculator._context = context
    speculator.model.model.normalize_context = lambda rows: rows * 2

    speculator.prepare_context_anchor(2, torch.tensor([0, 2, 6]), torch.tensor([0, 2]))
    torch.testing.assert_close(speculator.anchor_hidden[:2], context[[1, 3]] * 2)
    assert speculator.anchor_valid[:3].tolist() == [1.0, 1.0, 0.0]
    assert not speculator.anchor_hidden[2:].any()

    # A smaller reordered batch must not retain the previous request's anchor.
    speculator.prepare_context_anchor(1, torch.tensor([0, 3]), torch.tensor([1]))
    torch.testing.assert_close(speculator.anchor_hidden[0], context[1] * 2)
    assert not speculator.anchor_hidden[1:].any()
    assert speculator.anchor_valid[1:].eq(0).all()

    # Same request count: the cached device buffers are reused, not rebuilt.
    speculator.prepare_context_anchor(1, torch.tensor([0, 5]), torch.tensor([2]))
    torch.testing.assert_close(speculator.anchor_hidden[0], context[2] * 2)


def test_a_padded_batch_scores_the_mask_slots_and_keeps_padding_out(speculator):
    # As prepare_dflash_inputs does: real rows map to requests, padding to -1.
    speculator.sample_idx_mapping.view(MAX_REQS, STEPS)[:3] = torch.arange(3)[:, None]
    speculator._generate_draft(3, 4 * BLOCK, None, {}, None)

    ((embeddings, log_probs, pass_hidden, anchor_hidden, anchor_valid),) = speculator.model.scored
    assert embeddings.shape == (4, STEPS, TOPK, HIDDEN)
    assert log_probs.shape == (4, STEPS, TOPK)
    # Mask slots are rows 1..BLOCK-1 of each padded request block.
    assert pass_hidden[:, :, 0].flatten().tolist() == [
        req * BLOCK + step for req in range(4) for step in range(1, BLOCK)
    ]
    assert anchor_hidden.shape == (4, HIDDEN)
    assert anchor_valid.shape == (4,)
    # The upstream sampler receives the padded batch and the host score lattice.
    ((ids, scores, num_reqs, _, _, _, _, draft_logits, use_fp64),) = (
        speculator.candidate_sampler.calls
    )
    assert ids.shape == (4, STEPS, TOPK)
    assert scores.shape == (4, STEPS, TOPK, TOPK)
    assert (num_reqs, draft_logits, use_fp64) == (4, None, False)
    assert speculator.draft_tokens[:3].tolist() == [
        [(req * BLOCK + 1 + step) * TOPK for step in range(STEPS)] for req in range(3)
    ]
    # Invalid (padded) rows are masked, so their buffer entries stay untouched.
    assert speculator.draft_tokens[3].eq(-7).all()


def test_sample_rows_are_cached_per_padded_width(speculator):
    first = speculator._sample_rows(4)
    assert first.tolist() == [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
    assert speculator._sample_rows(4) is first
    assert speculator._sample_rows(2).tolist() == [1, 2, 3, 5, 6, 7]
    # The device copy is built once per width as well.
    assert speculator._sample_rows_device(4) is speculator._sample_rows_device(4)


def test_warmup_covers_the_dispatched_widths_and_anchor_gathers(speculator):
    warmed: list[int] = []
    shapes: list[tuple[int, ...]] = []

    speculator._generate_draft = lambda reqs, tokens, *a, **k: warmed.append(reqs)
    speculator._context = torch.arange(MAX_REQS * HIDDEN, dtype=torch.float32).view(
        MAX_REQS, HIDDEN
    )
    speculator.model.model.normalize_context = lambda rows: shapes.append(tuple(rows.shape)) or rows

    speculator.warm_draft_batches()

    assert warmed == [4, 2, 1]
    assert shapes == [(1, HIDDEN), (2, HIDDEN), (3, HIDDEN), (4, HIDDEN)]
    assert speculator.nulls == [4 * BLOCK, 2 * BLOCK, BLOCK]
    assert speculator.draft_tokens.eq(0).all()


def test_factory_keeps_the_upstream_lilicorr_speculator(monkeypatch):
    monkeypatch.setattr(spec_decode, "SpyreLiLiCorrSpeculator", lambda *args: ("lilicorr", args))
    monkeypatch.setattr(spec_decode, "SpyreDFlashSpeculator", lambda *args: ("dflash", args))
    lilicorr = object.__new__(LiLiCorrSpeculator)
    assert make_spyre_speculator(None, None, None, lilicorr)[0] == "lilicorr"
    assert make_spyre_speculator(None, None, None, object())[0] == "dflash"
