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

import pytest
import torch
from torch.nn import functional as F
from xpress_reference import merged_refine

from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead


@pytest.fixture
def head_and_weights():
    torch.manual_seed(7)
    head = XPressRefinerHead(73, 128, 16, rank=64, mlp_hidden=128, topc=0).double()
    state = head.state_dict()
    weights = {
        "xpress_head." + source: torch.randn_like(state[dest]) * 0.05
        for source, dest in head._PUBLISHED_KEYS.items()
    }
    head.load_checkpoint_weights(weights)
    return head, weights


def _unfolded_bias(weights, hidden, previous):
    w = {key.removeprefix("xpress_head."): value for key, value in weights.items()}
    summary = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
    features = torch.cat(
        (
            F.linear(hidden, w["down_h.weight"]),
            F.linear(summary, w["down_g.weight"]),
            F.embedding(previous, w["w1.weight"]),
        ),
        dim=-1,
    )
    x = F.linear(features, w["in_proj.weight"])
    mixed = torch.bmm(w["mix.L"].tril(), x.permute(2, 1, 0)).permute(2, 1, 0)
    x = x + mixed
    x = x + F.linear(
        F.silu(F.linear(x, w["mlp.gate_proj.weight"])) * F.linear(x, w["mlp.up_proj.weight"]),
        w["mlp.down_proj.weight"],
    )
    return F.linear(x, w["w2.weight"])


def test_fold_matches_training_sublayer(head_and_weights):
    head, weights = head_and_weights
    hidden = torch.randn(3, 16, 128, dtype=torch.float64)
    previous = torch.randint(73, (3, 16))
    expected = _unfolded_bias(weights, hidden, previous)[:, 1:]
    actual = head.refine_full(
        torch.zeros_like(expected), previous, head.project_hidden_cache(hidden)
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_later_tokens_cannot_change_earlier_bias(head_and_weights):
    head, _ = head_and_weights
    hidden = torch.randn(2, 16, 128, dtype=torch.float64)
    previous = torch.randint(73, (2, 16))
    changed = previous.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 73
    cache = head.project_hidden_cache(hidden)
    base = torch.zeros(2, 15, 73, dtype=torch.float64)
    first = head.refine_full(base, previous, cache)
    second = head.refine_full(base, changed, cache)
    torch.testing.assert_close(first[:, :7], second[:, :7], atol=0, rtol=0)
    assert not torch.equal(first[:, 7:], second[:, 7:])


@pytest.mark.parametrize("passes", [0, 1, 2, 6])
def test_jacobi_matches_unfolded_reference(head_and_weights, passes):
    head, weights = head_and_weights
    hidden = torch.randn(2, 16, 128, dtype=torch.float64)
    base = torch.randn(2, 16, 73, dtype=torch.float64) * 0.02
    anchor = torch.tensor([3, 17])
    block = base.argmax(-1)
    block[:, 0] = anchor
    for _ in range(passes):
        previous = torch.cat((anchor[:, None], block[:, :-1]), dim=1)
        block = (base + _unfolded_bias(weights, hidden, previous)).argmax(-1)
        block[:, 0] = anchor
    actual = merged_refine(head, base, hidden, anchor, passes)
    assert torch.equal(actual, block[:, 1:])


@pytest.mark.parametrize("topc", [0, 1, 7, 73])
@pytest.mark.parametrize("passes", [0, 1, 6])
def test_shortlist_matches_full_bias_restricted_to_fixed_candidates(head_and_weights, topc, passes):
    head, weights = head_and_weights
    head.topc = topc
    hidden = torch.randn(2, 16, 128, dtype=torch.float64)
    base = torch.randn(2, 16, 73, dtype=torch.float64) * 0.02
    anchor = torch.tensor([3, 17])
    expected = base[:, 1:].argmax(-1)
    candidates = base[:, 1:].topk(topc, dim=-1).indices if topc else None
    for _ in range(passes):
        previous = torch.cat((anchor[:, None], anchor[:, None], expected[:, :-1]), dim=1)
        scores = (base + _unfolded_bias(weights, hidden, previous))[:, 1:]
        if candidates is None:
            expected = scores.argmax(-1)
        else:
            indices = scores.gather(-1, candidates).argmax(-1, keepdim=True)
            expected = candidates.gather(-1, indices).squeeze(-1)
    actual = merged_refine(head, base, hidden, anchor, passes)
    assert torch.equal(actual, expected)
    if candidates is not None:
        assert (actual.unsqueeze(-1) == candidates).any(-1).all()


def test_hoisted_projection_and_candidate_readout_match_unfolded_bias(head_and_weights):
    head, weights = head_and_weights
    hidden = torch.randn(2, 16, 128, dtype=torch.float64)
    previous = torch.randint(73, (2, 16))
    candidates = torch.randint(73, (2, 15, 7))
    expected = _unfolded_bias(weights, hidden, previous)[:, 1:].gather(-1, candidates)
    actual = head.refine_candidates(
        torch.zeros(2, 15, 7, dtype=torch.float64),
        previous,
        head.project_hidden_cache(hidden),
        head.gather_readout(candidates.flatten()),
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_published_and_serving_checkpoint_roundtrip(head_and_weights):
    original, published = head_and_weights
    loaded = XPressRefinerHead(73, 128, 16, rank=64, mlp_hidden=128).double()
    loaded.load_checkpoint_weights(published)
    restored = XPressRefinerHead(73, 128, 16, rank=64, mlp_hidden=128).double()
    restored.load_state_dict(loaded.state_dict())
    for name, expected in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], expected, atol=0, rtol=0)
        torch.testing.assert_close(restored.state_dict()[name], expected, atol=0, rtol=0)


def test_missing_mixer_is_rejected(head_and_weights):
    head, weights = head_and_weights
    del weights["xpress_head.mix.L"]
    with pytest.raises(ValueError, match="mix.L"):
        head.load_checkpoint_weights(weights)


def test_wrong_block_size_is_rejected(head_and_weights):
    head, _ = head_and_weights
    with pytest.raises(ValueError, match="block size"):
        head.project_hidden_cache(torch.zeros(1, 8, 128))
