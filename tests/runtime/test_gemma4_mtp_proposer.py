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

"""The Spyre Gemma-4 MTP proposer drafts one row per request. Host-side only."""

import contextlib
import types

import pytest
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata

from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod
from spyre_inference.v1.spec_decode import gemma4_proposer
from spyre_inference.v1.spec_decode.gemma4_proposer import SpyreGemma4Proposer

HIDDEN = 4


def _verify_step_metadata() -> CommonAttentionMetadata:
    """Two requests verifying 3 drafts each: 4 query tokens apiece."""
    query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32)
    seq_lens = torch.tensor([20, 30], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        num_reqs=2,
        num_actual_tokens=8,
        max_query_len=4,
        max_seq_len=30,
        block_table_tensor=torch.arange(8, dtype=torch.int32).view(2, 4),
        slot_mapping=torch.arange(100, 108),
    )


def _proposer(monkeypatch, calls, *, all_chunked_prefill=False):
    proposer = object.__new__(SpyreGemma4Proposer)
    proposer._runner = types.SimpleNamespace(
        _is_all_reqs_chunked_prefill=lambda: all_chunked_prefill
    )
    proposer.supports_mm_inputs = False
    proposer.vllm_config = None
    proposer.built = []
    proposer.on_device = []
    proposer._drafts_on_card = lambda batch_size, sampling_metadata: False

    def build(cad, draft_index=0):
        proposer.built.append(cad)
        return [], {"layer": "metadata"}

    def model(*, input_ids, positions, hidden_states, inputs_embeds, on_device=False):
        calls.append((input_ids.clone(), positions.clone(), hidden_states.clone()))
        proposer.on_device.append(on_device)
        return hidden_states + 1, hidden_states * 2

    def sample(last_hidden_states, sampling_metadata):
        return last_hidden_states[:, 0].long() + 1000, None

    proposer.build_per_group_and_layer_attn_metadata = build
    proposer.model = model
    proposer._sample_draft_tokens = sample
    monkeypatch.setattr(
        gemma4_proposer, "set_forward_context", lambda *a, **k: contextlib.nullcontext()
    )
    return proposer


def test_drafts_from_the_sampled_rows_only(monkeypatch):
    calls = []
    proposer = _proposer(monkeypatch, calls)
    cad = _verify_step_metadata()
    target_hidden = torch.arange(8 * HIDDEN, dtype=torch.float32).view(8, HIDDEN)
    rows = torch.tensor([1, 7], dtype=torch.int32)  # request 0 rejected 2 of its 3 drafts
    rejected = torch.tensor([2, 0], dtype=torch.int32)

    drafts = proposer.propose(
        num_speculative_tokens=3,
        target_token_ids=torch.zeros(8, dtype=torch.int32),
        target_positions=torch.arange(10, 18),
        target_hidden_states=target_hidden,
        next_token_ids=torch.tensor([5, 6], dtype=torch.int32),
        token_indices_to_sample=rows,
        common_attn_metadata=cad,
        sampling_metadata=None,
        num_rejected_tokens_gpu=rejected,
    )

    # One metadata build, at one query per request, ending before the rejected drafts.
    (draft_cad,) = proposer.built
    assert draft_cad.query_start_loc.tolist() == [0, 1, 2]
    assert draft_cad.seq_lens.tolist() == [18, 30]
    assert draft_cad.max_query_len == 1 and draft_cad.num_actual_tokens == 2
    assert draft_cad.max_seq_len == 30
    assert draft_cad.slot_mapping.tolist() == [101, 107]

    assert len(calls) == 3
    first_ids, first_pos, first_hidden = calls[0]
    assert first_ids.tolist() == [5, 6]
    torch.testing.assert_close(first_hidden, target_hidden[[1, 7]])
    for step, (ids, pos, hidden) in enumerate(calls):
        assert pos.tolist() == [11, 17], "positions stay fixed across draft steps"
        if step:
            prev_ids, _, prev_hidden = calls[step - 1]
            assert ids.tolist() == ((prev_hidden + 1)[:, 0].long() + 1000).tolist()
            torch.testing.assert_close(hidden, prev_hidden * 2)
    assert drafts.shape == (2, 3)
    assert drafts[:, 0].tolist() == ((target_hidden[[1, 7]] + 1)[:, 0].long() + 1000).tolist()


def test_prefill_drafts_from_each_requests_last_token(monkeypatch):
    calls = []
    proposer = _proposer(monkeypatch, calls)
    cad = _verify_step_metadata()

    proposer.propose(
        num_speculative_tokens=1,
        target_token_ids=torch.zeros(8, dtype=torch.int32),
        target_positions=torch.arange(8),
        target_hidden_states=torch.zeros(8, HIDDEN),
        next_token_ids=torch.tensor([5, 6], dtype=torch.int32),
        token_indices_to_sample=None,
        common_attn_metadata=cad,
        sampling_metadata=None,
    )

    (draft_cad,) = proposer.built
    assert draft_cad.seq_lens.tolist() == [20, 30]
    assert calls[0][1].tolist() == [3, 7]


@pytest.mark.parametrize("transposed", [False, True], ids=["vllm", "spyre_transposed"])
def test_projections_are_replicated_so_drafting_runs_no_collective(monkeypatch, transposed):
    """Each rank holds the full weights, gathered along the split dim of the stored layout.

    Spyre's linear methods store a weight as ``[in, out]``, so there the column split's
    shards are columns and the row split's are rows.
    """
    full_pre = torch.arange(4 * 6, dtype=torch.float16).view(4, 6)  # [hidden, 2 * backbone]
    full_post = torch.arange(3 * 4, dtype=torch.float16).view(3, 4)  # [backbone, hidden]
    if transposed:
        full_pre, full_post = full_pre.t().contiguous(), full_post.t().contiguous()
    out_dim, in_dim = (1, 0) if transposed else (0, 1)
    rank = 1
    pre, post = torch.nn.Module(), torch.nn.Module()
    pre.weight = torch.nn.Parameter(full_pre.chunk(2, dim=out_dim)[rank], requires_grad=False)
    post.weight = torch.nn.Parameter(full_post.chunk(2, dim=in_dim)[rank], requires_grad=False)
    pre.bias = post.bias = None
    if transposed:
        pre.quant_method = object.__new__(SpyreUnquantizedLinearMethod)
        post.quant_method = object.__new__(SpyreUnquantizedLinearMethod)
    pre.gather_output = True
    post.input_is_parallel, post.reduce_results = False, True

    def all_gather(parts, tensor, group):
        # Every rank's shard of whichever weight this rank sent.
        pre_shards, post_shards = full_pre.chunk(2, dim=out_dim), full_post.chunk(2, dim=in_dim)
        shards = pre_shards if tensor.shape == pre_shards[rank].shape else post_shards
        for part, shard in zip(parts, shards):
            part.copy_(shard)

    tp_group = types.SimpleNamespace(world_size=2, cpu_group="gloo")
    monkeypatch.setattr(gemma4_proposer, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)

    proposer = object.__new__(SpyreGemma4Proposer)
    proposer.model = types.SimpleNamespace(
        model=types.SimpleNamespace(pre_projection=pre, post_projection=post)
    )
    proposer._replicate_projections()

    assert torch.equal(pre.weight, full_pre) and pre.weight.dtype == torch.float16
    assert torch.equal(post.weight, full_post)
    assert not pre.gather_output
    assert post.input_is_parallel and not post.reduce_results


def test_a_single_rank_keeps_its_projections(monkeypatch):
    monkeypatch.setattr(
        gemma4_proposer, "get_tp_group", lambda: types.SimpleNamespace(world_size=1)
    )
    proposer = object.__new__(SpyreGemma4Proposer)
    proposer.model = None  # never touched
    proposer._replicate_projections()


def _spec(draft_len, positions):
    text_config = types.SimpleNamespace(max_position_embeddings=positions)
    hf_config = types.SimpleNamespace(text_config=text_config)
    return types.SimpleNamespace(
        draft_model_config=types.SimpleNamespace(max_model_len=draft_len, hf_config=hf_config)
    )


def test_the_drafter_drafts_over_the_whole_target_context():
    """Past the drafter's max_model_len upstream zeroes every draft, so all are rejected."""
    from spyre_inference.platform import TorchSpyrePlatform

    spec = _spec(draft_len=2048, positions=2048)
    TorchSpyrePlatform._extend_drafter_context(spec, 8192)

    assert spec.draft_model_config.max_model_len == 8192
    # The rotary cache must cover those positions too.
    assert spec.draft_model_config.hf_config.text_config.max_position_embeddings == 8192


def test_a_drafter_with_enough_context_is_left_alone():
    from spyre_inference.platform import TorchSpyrePlatform

    spec = _spec(draft_len=8192, positions=131072)
    TorchSpyrePlatform._extend_drafter_context(spec, 4096)

    assert spec.draft_model_config.max_model_len == 8192
    assert spec.draft_model_config.hf_config.text_config.max_position_embeddings == 131072


def test_a_step_of_only_prefill_chunks_drafts_nothing(monkeypatch):
    """The scheduler drops a prefill chunk's drafts, so the draft forwards are skipped."""
    calls = []
    proposer = _proposer(monkeypatch, calls, all_chunked_prefill=True)
    cad = _verify_step_metadata()

    drafts = proposer.propose(
        num_speculative_tokens=3,
        target_token_ids=torch.zeros(8, dtype=torch.int64),
        target_positions=torch.arange(8),
        target_hidden_states=torch.zeros(8, HIDDEN),
        next_token_ids=torch.tensor([1, 2]),
        token_indices_to_sample=None,
        common_attn_metadata=cad,
        sampling_metadata=None,
    )

    assert calls == [] and proposer.built == []
    assert drafts.shape == (2, 3)


def test_greedy_drafts_stay_on_the_card_between_passes(monkeypatch):
    calls = []
    proposer = _proposer(monkeypatch, calls)
    proposer._drafts_on_card = lambda batch_size, sampling_metadata: True
    projected = []

    def greedy(last_hidden_states):
        projected.append(last_hidden_states.clone())
        padded = torch.cat([last_hidden_states, torch.full((2, HIDDEN), -1.0)])
        return padded[:, 0].long() + 2000  # the padded rows come back too

    proposer._greedy_sample = greedy
    target_hidden = torch.arange(8 * HIDDEN, dtype=torch.float32).view(8, HIDDEN)

    drafts = proposer.propose(
        num_speculative_tokens=2,
        target_token_ids=torch.zeros(8, dtype=torch.int32),
        target_positions=torch.arange(8),
        target_hidden_states=target_hidden,
        next_token_ids=torch.tensor([5, 6], dtype=torch.int32),
        token_indices_to_sample=None,
        common_attn_metadata=_verify_step_metadata(),
        sampling_metadata=None,
    )

    assert proposer.on_device == [True, True]
    first = target_hidden[[3, 7]]
    torch.testing.assert_close(calls[1][2], first * 2, msg="the draft state feeds the next pass")
    torch.testing.assert_close(projected[1], first * 2 + 1)
    assert drafts.tolist() == [
        [int(first[0, 0]) + 1 + 2000, int(first[0, 0]) * 2 + 1 + 2000],
        [int(first[1, 0]) + 1 + 2000, int(first[1, 0]) * 2 + 1 + 2000],
    ]
    assert calls[1][0].tolist() == drafts[:, 0].tolist()


def _draft_wrapper(bucket_sizes, max_num_reqs):
    from spyre_inference.v1.worker.spyre_model_runner import _SpyreDraftModelWrapper
    from spyre_inference.v1.worker.spyre_shape_bucketer import logits_row_buckets

    def find_bucket(n):
        return next((b for b in bucket_sizes if b >= n), None)

    return _SpyreDraftModelWrapper(
        torch.nn.Identity(),
        torch.device("cpu"),
        logits_row_buckets=logits_row_buckets(bucket_sizes, max_num_reqs),
        shape_bucketer=types.SimpleNamespace(find_bucket=find_bucket),
        model_dtype=torch.float16,
    )


def test_the_draft_state_stays_on_the_card_only_at_a_compiled_lm_head_width():
    wrapper = _draft_wrapper([1, 2, 4, 8, 16], max_num_reqs=6)

    assert wrapper.lm_head_takes_padded_rows(3)  # bucket 4
    assert not wrapper.lm_head_takes_padded_rows(5), "bucket 8 is past the 6-row lm_head"


def test_model_name_matches_the_vllm_class_of_a_spyre_adaptation():
    """Upstream's per-model drafter branches, e.g. the Gemma-4 image-token id, match on it."""

    class SpyreGemma4ForConditionalGeneration(torch.nn.Module):
        pass

    proposer = object.__new__(SpyreGemma4Proposer)
    model = SpyreGemma4ForConditionalGeneration()
    assert proposer.get_model_name(model) == "Gemma4ForConditionalGeneration"
