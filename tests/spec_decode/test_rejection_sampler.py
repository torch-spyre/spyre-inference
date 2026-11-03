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
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.platforms.cpu import CpuPlatform
from vllm.triton_utils import dispatcher
from vllm.v1.sample import rejection_sampler as upstream
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from spyre_inference.platform import TorchSpyrePlatform
from spyre_inference.v1.sample.sampler import greedy_sample
from spyre_inference.v1.sample.topk_topp_sampler import SpyreTopKTopPSampler
from spyre_inference.v1.spec_decode import rejection_sampler as native


@pytest.fixture(scope="module", autouse=True)
def registered_kernels():
    names = (
        "rejection_greedy_sample_kernel",
        "rejection_random_sample_kernel",
        "expand_kernel",
        "sample_recovered_tokens_kernel",
    )
    originals = {name: getattr(upstream, name) for name in names}
    dispatchers = (
        upstream._rejection_greedy_sample,
        upstream._rejection_random_sample,
        upstream._expand,
        upstream._sample_recovered_tokens,
    )
    registry = dispatcher._registry.copy()
    try:
        with patch.object(
            CpuPlatform,
            "register_triton_kernel_overrides",
            side_effect=AssertionError("CpuPlatform overrides require vllm._C"),
        ):
            TorchSpyrePlatform.register_triton_kernel_overrides()
        yield dispatchers
    finally:
        dispatcher._rebind_kernels(
            [(getattr(upstream, name), original) for name, original in originals.items()]
        )
        for name, original in originals.items():
            setattr(upstream, name, original)
        dispatcher._registry.clear()
        dispatcher._registry.update(registry)


@pytest.mark.parametrize("synthetic", [False, True])
@pytest.mark.parametrize("use_fp64", [False, True])
def test_platform_registration_preserves_dispatchers_and_warmup(
    registered_kernels, synthetic, use_fp64
):
    assert registered_kernels == (
        upstream._rejection_greedy_sample,
        upstream._rejection_random_sample,
        upstream._expand,
        upstream._sample_recovered_tokens,
    )
    for owner, implementation in zip(
        registered_kernels,
        (native.greedy_sample, native.random_sample, native.expand, native.recover),
        strict=True,
    ):
        assert owner.kernel.func is implementation
    sampler = SimpleNamespace(logprobs_mode="raw_logprobs", use_fp64_gumbel=use_fp64)
    spec_config = SimpleNamespace(
        rejection_sample_method="synthetic" if synthetic else "standard",
        synthetic_acceptance_rates=[0.9, 0.8, 0.7],
    )
    registry = JitWarmupRegistry(None)
    with registry.activate():
        rejection_sampler = upstream.RejectionSampler(sampler, spec_config, torch.device("cpu"))
    assert len(registry) == 4
    assert rejection_sampler.sampler is sampler
    assert rejection_sampler.synthetic_mode == synthetic


def test_greedy_prefix_correction_bonus_and_empty_request():
    draft = torch.tensor([2, 3, 4, 1, 5, 0, 6])
    target = torch.tensor([2, 7, 4, 1, 5, 2, 6])
    result = upstream.rejection_sample(
        draft,
        [0, 3, 2, 2],
        3,
        torch.tensor([0, 3, 5, 7]),
        None,
        torch.nn.functional.one_hot(target, 8).float(),
        torch.tensor([[4], [0], [7], [1]]),
        SimpleNamespace(all_greedy=True, all_random=False),
    )
    assert result.dtype == torch.int32
    assert result.tolist() == [[4, -1, -1, -1], [2, 7, -1, -1], [1, 5, 7, -1], [2, -1, -1, -1]]


def test_random_acceptance_and_greedy_rows_are_untouched():
    output = torch.full((4, 4), -1, dtype=torch.int32)
    output[1, 0] = 8
    upstream._rejection_random_sample(
        output,
        torch.tensor([3, 4, 5, 6]),
        torch.tensor([1, 0, 2, 1, -1, 1]),
        torch.tensor([[0.2, 0.6, 0.2]] * 6),
        torch.tensor([[0.3, 0.3, 0.4]] * 6),
        torch.tensor([2, 2, 2, 2]),
        torch.tensor([0, 1, 0, 0, 2, 0]),
        torch.tensor([0.4, 0.9, 0.9, 0.9, 0.1, 0.6]),
        torch.tensor([False, True, False, False]),
        3,
        None,
    )
    assert output.tolist() == [[1, 0, 2, 2], [8, -1, -1, -1], [2, -1, -1, -1], [0, -1, -1, -1]]


@pytest.mark.parametrize("no_draft", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_recovery_residual_and_first_index_ties(no_draft, dtype):
    p = torch.tensor([[0.25, 0.25, 0.5], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]])
    q = torch.tensor([[0.25, 0.25, 0.5], [0.5, 0.1, 0.4], [0.2, 0.3, 0.5]])
    ids = torch.tensor([2, -1, 1])
    noise = torch.tensor(
        [[99.0, 1.0, 1.0], [2.0, 2.0, 1.0], [99.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=dtype,
    )
    originals = [tensor.clone() for tensor in (p, q, noise)]
    result = torch.empty(3, dtype=torch.long)
    upstream._sample_recovered_tokens(
        result,
        torch.tensor([0, 2, 2, 3]),
        ids,
        None if no_draft else q,
        p,
        noise,
        max_spec_len=2,
    )
    assert result.tolist() == ([0, 1, 2] if no_draft else [0, 1, 0])
    for actual, original in zip((p, q, noise), originals, strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize("no_draft", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_recovery_keeps_request_noise_separate_with_equal_and_empty_drafts(no_draft, dtype):
    result = torch.empty(6, dtype=torch.int32)
    upstream._sample_recovered_tokens(
        result,
        torch.tensor([0, 2, 4, 4, 6]),
        torch.full((6,), -1),
        None if no_draft else torch.tensor([[0.2, 0.2, 0.6]] * 6),
        torch.tensor([[0.3, 0.3, 0.4]] * 6),
        torch.tensor([[99, 1, 1], [4, 1, 1], [1, 4, 1], [99, 1, 1], [1, 1, 8]], dtype=dtype),
        max_spec_len=2,
    )
    assert result.tolist() == ([0, 0, 1, 1, 2, 2] if no_draft else [0, 0, 1, 1, 0, 0])


@pytest.mark.parametrize("no_draft", [True, False])
def test_recovery_with_only_empty_requests(no_draft):
    result = torch.empty(0, dtype=torch.int32)
    upstream._sample_recovered_tokens(
        result,
        torch.zeros(4, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        None if no_draft else torch.empty(0, 3),
        torch.empty(0, 3),
        torch.ones(4, 3),
        max_spec_len=3,
    )
    assert result.numel() == 0


@pytest.mark.parametrize("dtype,expected", [(torch.float32, 0), (torch.float64, 1)])
def test_recovery_preserves_noise_precision(dtype, expected):
    result = torch.empty(1, dtype=torch.int32)
    upstream._sample_recovered_tokens(
        result,
        torch.tensor([1]),
        torch.tensor([-1]),
        None,
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([[1.0, 1.0 + 1e-10]], dtype=dtype),
        max_spec_len=1,
    )
    assert result.item() == expected


def test_expand_sampling_parameters_with_zero_draft_requests():
    actual = upstream.expand_batch_to_tokens(
        torch.tensor([0.0, 0.5, 2.0, 1.0]),
        torch.tensor([0, 2, 2, 5]),
        5,
        replace_from=1,
        replace_to=9,
    )
    assert actual.tolist() == [0.5, 0.5, 9, 9, 9]


def test_synthetic_acceptance_never_accepts_placeholder():
    output = torch.full((1, 4), -1, dtype=torch.int32)
    upstream._rejection_greedy_sample(
        output,
        torch.tensor([3]),
        torch.tensor([1, -1, 2]),
        torch.tensor([0, 0, 0]),
        torch.tensor([3]),
        None,
        3,
        torch.tensor([0.1, 0.1, 0.1]),
        torch.ones(3),
    )
    assert output.tolist() == [[1, 0, -1, -1]]


@pytest.mark.parametrize("no_draft", [False, True])
def test_random_synthetic_acceptance_rejects_placeholder_and_keeps_empty_bonus(no_draft):
    output = torch.full((2, 3), -1, dtype=torch.int32)
    upstream._rejection_random_sample(
        output,
        torch.tensor([2, 2]),
        torch.tensor([1, -1]),
        None if no_draft else torch.zeros(2, 3),
        torch.zeros(2, 3),
        torch.tensor([2, 2]),
        torch.tensor([0, 0]),
        torch.tensor([0.1, 0.1]),
        torch.tensor([False, False]),
        2,
        torch.ones(2),
    )
    assert output.tolist() == [[1, 0, -1], [2, -1, -1]]


def test_random_rejection_with_zero_draft_probability():
    output = torch.full((1, 2), -1, dtype=torch.int32)
    upstream._rejection_random_sample(
        output,
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.4, 0.6]]),
        torch.tensor([1]),
        torch.tensor([0]),
        torch.tensor([0.0]),
        torch.tensor([False]),
        1,
        None,
    )
    assert output.tolist() == [[0, -1]]


@pytest.mark.parametrize("spyre", [False, True])
@pytest.mark.parametrize("with_draft_probs", [False, True])
def test_random_rejection_preserves_target_distribution(with_draft_probs, spyre, monkeypatch):
    if spyre:
        monkeypatch.setattr(upstream, "rejection_sample", native.rejection_sample)
    torch.manual_seed(129)
    n = 12000
    p = torch.tensor([0.1, 0.3, 0.6])
    q = torch.tensor([0.6, 0.2, 0.2]) if with_draft_probs else torch.tensor([0.0, 1.0, 0.0])
    draft = torch.multinomial(q, n, replacement=True)
    result = upstream.rejection_sample(
        draft,
        [1] * n,
        1,
        torch.arange(1, n + 1),
        q.expand(n, -1).contiguous() if with_draft_probs else None,
        p.log().expand(n, -1).contiguous(),
        torch.zeros(n, 1, dtype=torch.int32),
        SimpleNamespace(
            all_greedy=False,
            all_random=True,
            temperature=torch.ones(n),
            generators={},
        ),
    )
    frequency = result[:, 0].bincount(minlength=3).float() / n
    torch.testing.assert_close(frequency, p, atol=0.015, rtol=0)


def _random_metadata(rows, temperature, top_k=None, top_p=None):
    return SamplingMetadata(
        temperature=torch.full((rows,), temperature),
        all_greedy=False,
        all_random=True,
        top_k=None if top_k is None else torch.full((rows,), top_k, dtype=torch.int32),
        top_p=None if top_p is None else torch.full((rows,), top_p),
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(rows),
        presence_penalties=torch.zeros(rows),
        repetition_penalties=torch.ones(rows),
        output_token_ids=[[] for _ in range(rows)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _packed_metadata(drafts):
    metadata = SpecDecodeMetadata.make_dummy(drafts, torch.device("cpu"))
    widths = torch.tensor([len(draft) + 1 for draft in drafts])
    metadata.bonus_logits_indices = widths.cumsum(0) - 1
    metadata.logits_indices = torch.arange(int(widths.sum()))
    target = torch.ones(len(metadata.logits_indices), dtype=torch.bool)
    target[metadata.bonus_logits_indices] = False
    metadata.target_logits_indices = metadata.logits_indices[target]
    return metadata


@pytest.fixture()
def host_sampler():
    sampler = Sampler()
    sampler.topk_topp_sampler = SpyreTopKTopPSampler(sampler.logprobs_mode, sampler.use_fp64_gumbel)
    sampler.greedy_sample = greedy_sample
    return sampler


@pytest.mark.parametrize(
    "temperature,top_k,top_p",
    [(0.7, None, None), (1.3, 3, None), (0.8, None, 0.75), (1.1, 3, 0.8)],
    ids=["temperature", "topk", "topp", "topk_topp"],
)
def test_full_sampler_preserves_two_token_target_distribution(
    host_sampler, temperature, top_k, top_p
):
    torch.random.default_generator.manual_seed(129)
    draws, vocab = 12000, 4
    first = torch.tensor([1.2, 0.8, 0.2, -0.9])
    transitions = torch.tensor(
        [[0.3, 1.5, 0.6, -0.4], [1.2, -0.7, 0.4, 1.1], [-0.2, 0.2, 1.6, 0.8], [1.5, 0.3, -0.8, 0.7]]
    )
    # The target's second-token distribution depends on the first token. A
    # reject-first draw must resume from the recovered token, not the draft.
    logits = torch.stack((first, transitions[0], transitions[1])).repeat(draws, 1)
    metadata = _packed_metadata([[0, 1]] * draws)
    sampling = _random_metadata(draws, temperature, top_k, top_p)
    result = upstream.RejectionSampler(host_sampler)(metadata, None, logits, sampling)
    tokens = result.sampled_token_ids
    assert torch.all(tokens[:, 0] >= 0)
    second_present = tokens[:, 1] >= 0
    assert second_present.any() and (~second_present).any()
    assert torch.all(tokens[second_present, 0] == 0)
    bonus_present = tokens[:, 2] >= 0
    assert bonus_present.any()
    assert torch.all(tokens[bonus_present, :2] == torch.tensor([0, 1]))

    resumed = host_sampler(
        logits=transitions[tokens[~second_present, 0]],
        sampling_metadata=_random_metadata(int((~second_present).sum()), temperature, top_k, top_p),
    )
    tokens[~second_present, 1] = resumed.sampled_token_ids.flatten()

    scores = torch.cat((first[None], transitions)).double() / temperature
    if top_k is not None:
        scores[scores < scores.topk(top_k).values[:, -1:]] = -torch.inf
    probabilities = scores.softmax(-1)
    if top_p is not None:
        sorted_probs, indices = probabilities.sort(descending=True)
        keep = sorted_probs.cumsum(-1) - sorted_probs < top_p
        probabilities = torch.zeros_like(probabilities).scatter_(1, indices, sorted_probs * keep)
        probabilities /= probabilities.sum(-1, keepdim=True)
    expected = probabilities[0, :, None] * probabilities[1:]
    observed = (tokens[:, 0] * vocab + tokens[:, 1]).bincount(minlength=vocab**2)
    torch.testing.assert_close(
        observed.reshape(vocab, vocab).double() / draws, expected, atol=0.015, rtol=0
    )


def test_stochastic_topk_one_covers_rejection_bonus_and_empty_drafts(host_sampler):
    drafts = [[0, 1], [0, 2], [2, 1], []]
    logits = torch.eye(3).repeat(3, 1)
    logits = torch.cat((logits, torch.tensor([[0.0, 0.0, 1.0]])))
    result = upstream.RejectionSampler(host_sampler)(
        _packed_metadata(drafts), None, logits, _random_metadata(4, 0.9, top_k=1)
    )
    assert result.sampled_token_ids.tolist() == [[0, 1, 2], [0, 1, -1], [0, -1, -1], [2, -1, -1]]


def test_rejection_sampling_requires_native_extension(monkeypatch):
    monkeypatch.setattr(native, "_HAS_REJECTION_KERNELS", False)
    with pytest.raises(RuntimeError, match="Rejection sampling requires the C\\+\\+ kernels"):
        upstream.expand_batch_to_tokens(torch.ones(1), torch.tensor([1]), 1)


@pytest.mark.parametrize("output_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("draft_dtype", [torch.int32, torch.int64])
def test_greedy_accepts_mixed_id_dtypes_and_preserves_unwritten_tail(output_dtype, draft_dtype):
    output = torch.full((2, 3), 42, dtype=output_dtype)
    upstream._rejection_greedy_sample(
        output,
        torch.tensor([2, 3], dtype=torch.int32),
        torch.tensor([70001, 90003, 100007], dtype=draft_dtype),
        torch.tensor([70001, 90005, 100007], dtype=torch.int64),
        torch.tensor([[150000], [120000]], dtype=torch.int32),
        None,
        2,
        None,
        None,
    )
    assert output.tolist() == [[70001, 90005, 42], [100007, 120000, 42]]


@pytest.mark.parametrize("synthetic", [False, True])
def test_random_acceptance_preserves_fp64_thresholds(synthetic):
    # The uniforms become indistinguishable if a wrapper narrows them to FP32.
    threshold = torch.tensor(0.6).double()
    uniform = torch.stack(
        [
            torch.nextafter(threshold, threshold.new_tensor(float("-inf"))),
            threshold,
            torch.nextafter(threshold, threshold.new_tensor(float("inf"))),
        ]
    )
    output = torch.full((3, 2), -1, dtype=torch.int32)
    upstream._rejection_random_sample(
        output,
        torch.tensor([1, 2, 3], dtype=torch.int32),
        torch.ones(3, dtype=torch.int32),
        None,
        torch.tensor([[0.4, 0.6]] * 3),
        torch.full((3, 1), 1, dtype=torch.int64),
        torch.zeros(3, dtype=torch.int64),
        uniform,
        torch.zeros(3, dtype=torch.bool),
        1,
        torch.tensor([0.6]) if synthetic else None,
    )
    expected = [[1, 1], [0, -1], [0, -1]] if synthetic else [[1, 1], [1, 1], [0, -1]]
    assert output.tolist() == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int32, torch.int64])
def test_expand_preserves_parameter_dtype(dtype):
    floating = dtype.is_floating_point
    values = torch.tensor([0, 0.5 if floating else 70001, 1], dtype=dtype)
    actual = upstream.expand_batch_to_tokens(
        values, torch.tensor([0, 2, 3], dtype=torch.int32), 3, replace_from=1, replace_to=9
    )
    assert actual.dtype == dtype
    assert actual.tolist() == ([0.5, 0.5, 9] if floating else [70001, 70001, 9])


@pytest.mark.parametrize("cumulative", [[-1, 3], [2, 1], [1, 4], [1, 2]])
def test_native_rejects_invalid_cumulative_before_writing(cumulative):
    output = torch.full((2, 4), 42, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="cumulative must"):
        upstream._rejection_greedy_sample(
            output,
            torch.tensor(cumulative),
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2, 3]),
            torch.tensor([0, 0]),
            None,
            3,
            None,
            None,
        )
    assert torch.all(output == 42)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("no_draft", [False, True])
def test_recovery_preserves_zero_times_infinite_noise(dtype, no_draft):
    # Masking a drafted token or a zero residual still multiplies 0 * inf.
    # torch.argmax returns the first resulting NaN, even after a finite score.
    output = torch.empty(2, dtype=torch.int32)
    upstream._sample_recovered_tokens(
        output,
        torch.tensor([0, 2, 2], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        None if no_draft else torch.tensor([[0.1, 0.3, 0.4]] * 2),
        torch.tensor([[0.2, 0.3, 0.5]] * 2),
        torch.tensor([[1, 1, 1], [1, float("inf"), 1], [1, 1, 1]], dtype=dtype),
        max_spec_len=2,
    )
    assert output.tolist() == [1, 1]


# Drafts without probabilities (ngram, suffix): recovery is a fused_gumbel_argmax draw over
# the masked target logits, and acceptance reads only each draft's target probability.


def _seeded_metadata(batch_size, generators=None):
    return SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.ones(batch_size),
        generators=generators or {},
    )


def test_recovery_matches_fused_gumbel_argmax_over_masked_logits():
    torch.manual_seed(7)
    logits = torch.randn(6, 1000).mul(3)
    draft_ids = torch.tensor([3, -1, 17, 999, 0, 42])
    generators = {i: torch.Generator().manual_seed(10 + i) for i in range(4)}
    counts = [2, 0, 3, 1]
    seeds = [
        torch.randint(0, 2**62, (1,), generator=torch.Generator().manual_seed(10 + i)).item()
        for i in (0, 2, 3)
    ]
    masked = logits.clone()
    for row, draft_id in enumerate(draft_ids.tolist()):
        if draft_id >= 0:
            masked[row, draft_id] = float("-inf")
    expected = torch.ops._spyre_C.fused_gumbel_argmax(
        masked, torch.tensor(seeds).repeat_interleave(torch.tensor([2, 3, 1]))
    )
    result = native.recover_without_draft_probs(logits, draft_ids, counts, generators)
    assert result.tolist() == expected.tolist()


def test_recovery_draws_only_from_the_residual():
    logits = torch.full((3, 8), float("-inf"))
    logits[0, 7] = 0.0  # a pad (-1) excludes nothing, not the last id
    logits[1, [2, 6]] = torch.tensor([5.0, 0.0])  # the real draft 2 is excluded
    logits[2, 4] = 0.0  # the draft takes all the mass: nothing to draw from
    result = native.recover_without_draft_probs(logits, torch.tensor([-1, 2, 4]), [3], {})
    assert result.tolist() == [7, 6, 0]


def test_recovery_reproduces_seeded_requests_and_skips_empty_ones():
    logits = torch.randn(2, 64)
    idle = torch.Generator().manual_seed(1)
    idle_state = idle.get_state()

    def draw():
        generators = {0: idle, 1: torch.Generator().manual_seed(99)}
        return native.recover_without_draft_probs(
            logits, torch.tensor([-1, -1]), [0, 2], generators
        ).tolist()

    assert draw() == draw()
    torch.testing.assert_close(idle.get_state(), idle_state, rtol=0, atol=0)


def test_acceptance_matches_the_random_kernel():
    """Same decisions as _rejection_random_sample given the same uniforms and recovery."""
    torch.manual_seed(11)
    vocab, counts = 16, [3, 0, 2, 3]
    num_tokens = sum(counts)
    # Peaked rows so both accepts and rejects occur.
    target_probs = torch.randn(num_tokens, vocab).mul(4).softmax(dim=-1)
    draft_ids = torch.cat([target_probs[:4].argmax(-1), torch.tensor([5, -1, 2, 9])])
    recovered = torch.randint(0, vocab, (num_tokens,))
    uniform = torch.rand(num_tokens, dtype=torch.float64)
    bonus = torch.tensor([[100], [101], [102], [103]])
    is_greedy = torch.tensor([False, False, True, False])
    cumulative = torch.tensor(counts).cumsum(0)

    expected = torch.full((4, 4), -1, dtype=torch.int32)
    native.random_sample(
        expected, cumulative, draft_ids, None, target_probs, bonus, recovered, uniform,
        is_greedy, 3, vocab, None, True, False,
    )  # fmt: skip
    result = torch.full((4, 4), -1, dtype=torch.int32)
    target_draft = target_probs[torch.arange(num_tokens), draft_ids.clamp(min=0)]
    native.accept_without_draft_probs(
        result, counts, draft_ids, target_draft, bonus, recovered, uniform, is_greedy
    )
    assert result.tolist() == expected.tolist()


def test_acceptance_rejects_padded_draft_even_at_zero_uniform():
    output = torch.full((1, 3), -1, dtype=torch.int32)
    native.accept_without_draft_probs(
        output,
        [2],
        torch.tensor([3, PAD]),
        torch.ones(2),
        torch.tensor([[7]]),
        torch.tensor([0, 5]),
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([False]),
    )
    assert output.tolist() == [[3, 5, -1]]


def test_rejection_sample_without_draft_probs_skips_softmax_and_exponential_noise():
    logits = torch.randn(3, 64)
    with (
        patch.object(torch.Tensor, "softmax", wraps=torch.Tensor.softmax) as softmax,
        patch.object(torch.Tensor, "exponential_", wraps=torch.Tensor.exponential_) as exp,
    ):
        native.rejection_sample(
            torch.tensor([0, 1, -1]), [1, 2], 2, torch.tensor([1, 3]), None, logits,
            torch.tensor([[7], [8]]), _seeded_metadata(2),
        )  # fmt: skip
    softmax.assert_not_called()
    exp.assert_not_called()


@pytest.mark.parametrize("case", ["draft_probs", "fp64_gumbel"])
def test_upstream_recovery_for_draft_probs_and_fp64_gumbel(case, monkeypatch):
    calls = []
    original = upstream.sample_recovered_tokens
    monkeypatch.setattr(
        upstream, "sample_recovered_tokens", lambda *a: calls.append(a) or original(*a)
    )
    native.rejection_sample(
        torch.tensor([0]), [1], 1, torch.tensor([1]),
        torch.ones(1, 4) / 4 if case == "draft_probs" else None, torch.zeros(1, 4),
        torch.tensor([[7]]), _seeded_metadata(1), use_fp64_gumbel=case == "fp64_gumbel",
    )  # fmt: skip
    assert len(calls) == 1


def test_install_routes_upstream_rejection_sample(monkeypatch):
    monkeypatch.setattr(upstream, "rejection_sample", upstream.rejection_sample)
    native.install()
    assert upstream.rejection_sample is native.rejection_sample


def _greedy_and_random_batch():
    """Request 0 greedy, request 1 random; three draft positions each."""
    torch.manual_seed(5)
    return dict(
        draft_token_ids=torch.tensor([3, 1, 4, 1, 5, 9]),
        num_draft_tokens=[3, 3],
        max_spec_len=3,
        cu_num_draft_tokens=torch.tensor([3, 6]),
        draft_probs=None,
        target_logits=torch.randn(6, 16),
        bonus_token_ids=torch.tensor([[7], [8]]),
    )


@pytest.mark.parametrize("all_greedy", [True, False])
def test_rejection_sample_takes_the_target_argmax_with_greedy_argmax(all_greedy, monkeypatch):
    from spyre_inference.v1.sample import sampler

    calls = []
    greedy_argmax = sampler.greedy_sample
    monkeypatch.setattr(
        sampler, "greedy_sample", lambda logits: calls.append(logits.shape) or greedy_argmax(logits)
    )
    batch = _greedy_and_random_batch()
    metadata = SimpleNamespace(
        all_greedy=all_greedy,
        all_random=False,
        temperature=torch.tensor([0.0, 0.0 if all_greedy else 1.0]),
        generators={},
    )
    result = native.rejection_sample(**batch, sampling_metadata=metadata)
    expected = upstream.rejection_sample(**batch, sampling_metadata=metadata)
    assert calls == [(6, 16)]
    # Greedy rows are deterministic, so they match upstream exactly.
    assert result[0].tolist() == expected[0].tolist()
    if all_greedy:
        assert result.tolist() == expected.tolist()


def test_rejection_sample_with_no_draft_tokens():
    metadata = SimpleNamespace(all_greedy=True, all_random=False)
    result = native.rejection_sample(
        torch.empty(0, dtype=torch.int64),
        [0, 0],
        2,
        torch.tensor([0, 0]),
        None,
        torch.empty(0, 16),
        torch.tensor([[7], [8]]),
        metadata,
    )
    assert result.tolist() == [[7, -1, -1], [8, -1, -1]]


# Padded drafts: TorchSpyreScheduler pads every draft list to k with -1, which the
# rejection kernels must always reject without disturbing the real drafts before it.

PAD = -1


def test_greedy_kernel_rejects_padded_draft():
    output = torch.full((1, 4), upstream.PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    native.greedy_sample(
        output,
        torch.tensor([3]),
        torch.tensor([10, PAD, PAD]),
        torch.tensor([10, 20, 30]),
        torch.tensor([40]),
        None,
        3,
        None,
        None,
        False,
    )
    # The real draft is accepted; the first pad is replaced by the target's own token,
    # which is what the bonus slot would have held without padding.
    assert output.tolist() == [[10, 20, -1, -1]]


def test_random_kernel_rejects_padded_draft_even_at_zero_uniform():
    vocab = 8
    output = torch.full((1, 3), upstream.PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    native.random_sample(
        output,
        torch.tensor([2]),
        torch.tensor([3, PAD]),
        None,
        torch.full((2, vocab), 1.0 / vocab),
        torch.tensor([7]),
        torch.tensor([0, 5]),
        # 0.0 accepts any real draft: only the pad check can reject position 1.
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([False]),
        2,
        vocab,
        None,
        True,
        False,
    )
    assert output.tolist() == [[3, 5, -1]]


def test_greedy_rejection_sample_padded_verification():
    """Two requests at k=3: one fully drafted, one with a single real draft."""
    target_logits = torch.full((6, 100), -10.0)
    for row, token in enumerate([10, 20, 30, 11, 21, 31]):
        target_logits[row, token] = 10.0
    result = native.rejection_sample(
        torch.tensor([10, 20, 30, 11, PAD, PAD]),
        [3, 3],
        3,
        torch.tensor([3, 6]),
        None,
        target_logits,
        torch.tensor([[40], [41]]),
        SimpleNamespace(all_greedy=True, all_random=False),
    )
    assert result.tolist() == [[10, 20, 30, 40], [11, 21, -1, -1]]
