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

from spyre_inference.platform import TorchSpyrePlatform
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


@pytest.mark.parametrize("with_draft_probs", [False, True])
def test_random_rejection_preserves_target_distribution(with_draft_probs):
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
