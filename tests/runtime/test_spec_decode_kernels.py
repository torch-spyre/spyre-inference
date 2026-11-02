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

"""Host stand-ins for upstream's spec-decode Triton kernels, against the kernels' contracts."""

from types import SimpleNamespace

import pytest
import torch

from spyre_inference.v1.sample.sampling_kernels import has_sampling_kernels
from spyre_inference.v1.sample.topk_topp_sampler import apply_top_k_top_p_host
from spyre_inference.v1.spec_decode import kernels

PLACEHOLDER = -1


def test_next_token_is_last_valid_sample_else_backup():
    sampled = torch.tensor([[5, -1, -1], [7, 8, 9], [-1, -1, -1], [4, 6, -1]])
    discard = torch.tensor([False, False, False, True])
    backup = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    next_ids = torch.empty(4, dtype=torch.int32)
    valid = torch.empty(4, dtype=torch.int32)

    kernels.eagle_prepare_next_token_padded(sampled, discard, backup, next_ids, valid, 100, 3, 4)

    assert next_ids.tolist() == [5, 9, 12, 13]
    assert valid.tolist() == [1, 3, 0, 0]


def test_sample_index_steps_back_over_rejected_drafts():
    cu_num_draft = torch.tensor([2, 4, 4])  # drafts per request: 2, 2, 0
    valid = torch.tensor([3, 1, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 6, 7], dtype=torch.int32)
    indices = torch.empty(3, dtype=torch.int32)
    rejected = torch.empty(3, dtype=torch.int32)

    kernels.eagle_prepare_inputs_padded(cu_num_draft, valid, query_start_loc, indices, rejected, 3)

    assert rejected.tolist() == [0, 2, 0]
    assert indices.tolist() == [2, 3, 6]


def test_greedy_rejection_stops_at_first_mismatch_and_skips_random_rows():
    output = torch.full((3, 3), PLACEHOLDER, dtype=torch.int32)
    cu_num_draft = torch.tensor([2, 4, 5])
    drafts = torch.tensor([1, 2, 3, 4, 5])
    argmax = torch.tensor([1, 2, 3, 9, 7])
    bonus = torch.tensor([[10], [11], [12]])

    kernels.rejection_greedy_sample(output, cu_num_draft, drafts, argmax, bonus, None, 2)
    assert output.tolist() == [[1, 2, 10], [3, 9, -1], [7, -1, -1]]

    output.fill_(PLACEHOLDER)
    is_greedy = torch.tensor([True, False, True])
    kernels.rejection_greedy_sample(output, cu_num_draft, drafts, argmax, bonus, is_greedy, 2)
    assert output[1].tolist() == [-1, -1, -1]


def test_random_rejection_accepts_on_probability_ratio():
    output = torch.full((2, 3), PLACEHOLDER, dtype=torch.int32)
    cu_num_draft = torch.tensor([2, 3])
    drafts = torch.tensor([0, 1, 2])
    target_probs = torch.zeros(3, 4)
    target_probs[0, 0], target_probs[1, 1], target_probs[2, 2] = 0.5, 0.1, 0.9
    uniform = torch.tensor([0.4, 0.5, 0.2], dtype=torch.float64)
    recovered = torch.tensor([6, 7, 8])
    bonus = torch.tensor([[20], [21]])
    is_greedy = torch.tensor([False, False])

    kernels.rejection_random_sample(
        output, cu_num_draft, drafts, None, target_probs, bonus, recovered, uniform,
        is_greedy, 2, 4, NO_DRAFT_PROBS=True,
    )  # fmt: skip

    assert output.tolist() == [[0, 7, -1], [2, 21, -1]]


def test_padded_draft_is_always_rejected():
    output = torch.full((1, 2), PLACEHOLDER, dtype=torch.int32)
    kernels.rejection_random_sample(
        output, torch.tensor([1]), torch.tensor([-1]), None, torch.ones(1, 4),
        torch.tensor([[9]]), torch.tensor([3]), torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([False]), 1, 4, NO_DRAFT_PROBS=True,
    )  # fmt: skip
    assert output.tolist() == [[3, -1]]


def test_expand_repeats_per_request_and_replaces_greedy_temperature():
    output = torch.empty(3)
    kernels.expand(output, torch.tensor([0.0, 0.7]), torch.tensor([2, 3]), 0, 1)
    torch.testing.assert_close(output, torch.tensor([1.0, 1.0, 0.7]))


def test_recovered_token_excludes_the_draft_or_takes_the_residual():
    target = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    inv_q = torch.ones(2, 3)
    out = torch.empty(2, dtype=torch.int64)

    kernels.sample_recovered_tokens(
        out, torch.tensor([1, 2]), torch.tensor([0, 2]), None, target, inv_q, 3,
        NO_DRAFT_PROBS=True,
    )  # fmt: skip
    assert out.tolist() == [1, 0]

    draft = torch.tensor([[0.1, 0.8, 0.1], [0.0, 0.1, 0.9]])
    kernels.sample_recovered_tokens(
        out, torch.tensor([1, 2]), torch.tensor([1, 2]), draft, target, inv_q, 3
    )
    assert out.tolist() == [0, 0]


def test_install_replaces_every_dispatcher():
    import vllm.v1.sample.rejection_sampler as rejection_sampler
    import vllm.v1.spec_decode.llm_base_proposer as base_proposer

    kernels.install()

    host = [
        (base_proposer._eagle_prepare_inputs_padded, kernels.eagle_prepare_inputs_padded),
        (base_proposer._eagle_prepare_next_token_padded, kernels.eagle_prepare_next_token_padded),
        (rejection_sampler._expand, kernels.expand),
    ]
    for dispatch, fn in host:
        assert dispatch.fn is fn
    for name in (
        "_rejection_greedy_sample",
        "_rejection_random_sample",
        "_sample_recovered_tokens",
    ):
        assert isinstance(getattr(rejection_sampler, name), kernels._HostDispatch), name
    # Upstream registers warmups on the dispatchers when it builds the proposer and sampler.
    base_proposer._eagle_prepare_inputs_padded.register_warmup(max_batch_size=8)
    assert rejection_sampler.rejection_sample is kernels.rejection_sample
    assert rejection_sampler.apply_top_k_top_p is apply_top_k_top_p_host


def test_expand_batch_to_tokens_runs_on_the_host_after_install():
    """The sampler warmup's first dispatcher call: without install() it needs vLLM's _C."""
    import vllm.v1.sample.rejection_sampler as rejection_sampler

    kernels.install()

    cu_num_tokens = torch.tensor([2, 3, 6], dtype=torch.int32)
    out = rejection_sampler.expand_batch_to_tokens(
        torch.tensor([5, 0, 7]), cu_num_tokens, 6, replace_from=0, replace_to=1
    )
    assert out.tolist() == [5, 5, 1, 7, 7, 7]


needs_kernels = pytest.mark.skipif(
    not has_sampling_kernels(), reason="sampling kernels extension not built"
)

NEG_INF = -float("inf")


def _sample(drafts, logits, *, greedy, num_draft=None, generators=None):
    """``kernels.rejection_sample`` over one draft per request unless told otherwise."""
    num_draft = num_draft or [1] * len(drafts)
    batch = len(num_draft)
    metadata = SimpleNamespace(
        all_greedy=greedy,
        all_random=not greedy,
        temperature=None if greedy else torch.ones(batch),
        generators=generators or {},
    )
    return kernels.rejection_sample(
        torch.tensor(drafts), num_draft, max(num_draft), torch.tensor(num_draft).cumsum(0),
        None, torch.as_tensor(logits, dtype=torch.float32),
        torch.arange(100, 100 + batch).unsqueeze(1), metadata,
    )  # fmt: skip


@needs_kernels
def test_greedy_verify_takes_the_target_argmax():
    logits = [[0.0, 5.0, 1.0], [3.0, 0.0, 1.0], [0.0, 0.0, 9.0]]
    out = _sample([1, 2, 0], logits, greedy=True, num_draft=[2, 1])
    # Request 0 accepts 1 then corrects 2 to the argmax 0; request 1 corrects 0 to 2.
    assert out.tolist() == [[1, 0, -1], [2, -1, -1]]


@needs_kernels
def test_random_verify_never_keeps_an_impossible_draft_or_samples_a_masked_token():
    logits = torch.tensor([[0.0, NEG_INF, 0.0, NEG_INF]]).repeat(200, 1)
    out = _sample([1] * 200, logits, greedy=False)
    assert set(out[:, 0].tolist()) <= {0, 2}
    assert set(out[:, 1].tolist()) == {-1}


@needs_kernels
def test_random_verify_always_keeps_a_certain_draft_then_the_bonus():
    logits = torch.tensor([[0.0, NEG_INF, NEG_INF]]).repeat(50, 1)
    out = _sample([0] * 50, logits, greedy=False)
    assert out[:, 0].tolist() == [0] * 50
    assert out[:, 1].tolist() == list(range(100, 150))


@needs_kernels
def test_random_verify_accepts_at_the_target_probability_and_resamples_the_rest():
    n = 6000
    out = _sample([2] * n, torch.zeros(n, 3), greedy=False)
    first = out[:, 0]
    accepted = (first == 2).sum().item()
    assert 0.29 < accepted / n < 0.38, "the draft holds 1/3 of the target mass"
    recovered = first[first != 2]
    assert set(recovered.tolist()) <= {0, 1}
    share = (recovered == 0).float().mean().item()
    assert 0.45 < share < 0.55, "the residual is uniform over the other two tokens"


@needs_kernels
def test_random_verify_is_reproducible_under_seeded_generators():
    logits = torch.zeros(3, 64)

    def draw():
        generator = torch.Generator().manual_seed(7)
        return _sample([0, 1, 2], logits, greedy=False, num_draft=[3], generators={0: generator})

    assert draw().tolist() == draw().tolist()


def test_falls_back_to_upstream_without_the_kernels(monkeypatch):
    calls = []
    monkeypatch.setattr(kernels, "has_sampling_kernels", lambda: False)
    monkeypatch.setattr(kernels, "_upstream_rejection_sample", lambda *a, **k: calls.append(1))
    _sample([0], [[0.0, 1.0]], greedy=True)
    assert calls == [1]


@pytest.mark.parametrize("with_top_p", [False, True])
def test_window_probs_match_the_full_vocabulary_constraints(with_top_p):
    gen = torch.Generator().manual_seed(0)
    rows, vocab = 6, 500
    logits = torch.randn(rows, vocab, generator=gen) * 3
    # Ties at row 0's k-th value, which the window must widen to keep.
    logits[0, :20] = 20.0
    temperature = torch.rand(rows, generator=gen) + 0.5
    top_k = torch.tensor([5, 1, 40, 17, 3, 64], dtype=torch.int32)
    top_p = torch.rand(rows, generator=gen) * 0.5 + 0.5 if with_top_p else None
    if top_p is not None:
        top_p[0] = 1.0  # top-p would otherwise trim row 0's ties

    probs, ids = kernels._window_target_probs(logits, temperature, top_k, top_p)

    scaled = logits / temperature.unsqueeze(1)
    expected = apply_top_k_top_p_host(scaled.clone(), top_k, top_p).softmax(dim=-1)
    torch.testing.assert_close(probs, expected.gather(1, ids))
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(rows))
    assert (probs[0] > 0).sum() == 20, "every token tied at the k-th value is kept"


def _window_sample(drafts, logits, top_k, *, generators=None):
    """``kernels._window_rejection_sample`` over one draft per request."""
    batch = len(drafts)
    spec = SimpleNamespace(
        draft_token_ids=torch.tensor(drafts),
        num_draft_tokens=[1] * batch,
        cu_num_draft_tokens=torch.arange(1, batch + 1),
        max_spec_len=1,
    )
    sampling = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.ones(batch),
        top_k=torch.full((batch,), top_k, dtype=torch.int32),
        top_p=None,
        generators=generators or {},
    )
    logits = torch.as_tensor(logits, dtype=torch.float32)
    bonus = torch.arange(100, 100 + batch).unsqueeze(1)
    return kernels._window_rejection_sample(spec, logits, bonus, sampling)


@needs_kernels
def test_window_verify_keeps_a_certain_draft_then_the_bonus():
    logits = torch.tensor([[9.0, -50.0, -50.0, -50.0]]).repeat(20, 1)
    out = _window_sample([0] * 20, logits, top_k=2)
    assert out[:, 0].tolist() == [0] * 20
    assert out[:, 1].tolist() == list(range(100, 120))


def test_window_verify_without_drafts_returns_the_bonus_tokens():
    """A prefill chunk schedules no drafts; each request's bonus token is its output."""
    spec = SimpleNamespace(
        draft_token_ids=torch.empty(0, dtype=torch.int32),
        num_draft_tokens=[0, 0],
        cu_num_draft_tokens=torch.zeros(2, dtype=torch.int32),
        max_spec_len=0,
    )
    sampling = SimpleNamespace(all_random=True, top_k=torch.tensor([64, 64]))
    bonus = torch.tensor([[7], [9]])
    out = kernels._window_rejection_sample(spec, torch.empty(0, 16), bonus, sampling)
    assert out.tolist() == [[7], [9]]


@needs_kernels
def test_window_verify_rejects_a_draft_outside_the_top_k():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]]).repeat(200, 1)
    out = _window_sample([3] * 200, logits, top_k=2)
    assert set(out[:, 0].tolist()) <= {0, 1}, "recovered tokens stay inside the top-k"
    assert set(out[:, 1].tolist()) == {-1}


@needs_kernels
def test_window_verify_accepts_at_the_top_k_probability_and_resamples_the_rest():
    n = 6000
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]]).repeat(n, 1)
    out = _window_sample([1] * n, logits, top_k=2)
    accepted = (out[:, 0] == 1).float().mean().item()
    expected = torch.tensor([3.0, 2.0]).softmax(0)[1].item()  # ~0.27 after top-k 2
    assert abs(accepted - expected) < 0.03
    assert set(out[out[:, 0] != 1, 0].tolist()) == {0}, "the residual is the other top-k token"


@pytest.mark.parametrize(
    ("change", "eligible"),
    [
        ({}, True),
        ({"top_k": None}, False),
        ({"top_k": torch.tensor([64, 1000], dtype=torch.int32)}, False),  # one row unset
        ({"max_num_logprobs": 5}, False),
        ({"all_greedy": True}, False),
    ],
)
def test_window_path_only_takes_batches_it_reproduces(monkeypatch, change, eligible):
    monkeypatch.setattr(kernels, "has_sampling_kernels", lambda: True)
    sampler = SimpleNamespace(synthetic_mode=False, use_fp64_gumbel=False)
    fields = {
        "top_k": torch.tensor([64, 40], dtype=torch.int32),
        "all_greedy": False,
        "max_num_logprobs": None,
    }
    sampling = SimpleNamespace(**{**fields, **change})
    assert kernels._window_eligible(sampler, None, sampling, 1000) is eligible
    assert not kernels._window_eligible(sampler, torch.zeros(2, 1000), sampling, 1000)


def test_a_vocabulary_wide_top_k_takes_the_full_path(monkeypatch):
    """Gemma's unset top-k arrives as vocab - 1; topk over it costs more than the full path."""
    monkeypatch.setattr(kernels, "has_sampling_kernels", lambda: True)
    sampler = SimpleNamespace(synthetic_mode=False, use_fp64_gumbel=False)
    vocab = 262144

    def sampling(top_k):
        top_k = torch.tensor([top_k], dtype=torch.int32)
        return SimpleNamespace(top_k=top_k, all_greedy=False, max_num_logprobs=None)

    assert kernels._window_eligible(sampler, None, sampling(64), vocab)
    assert not kernels._window_eligible(sampler, None, sampling(vocab - 1), vocab)
