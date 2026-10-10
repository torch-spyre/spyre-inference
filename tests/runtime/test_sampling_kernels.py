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

"""Fused host sampling kernels ported from vLLM's CPU backend."""

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from spyre_inference.v1.sample.sampler import greedy_sample
from spyre_inference.v1.sample.sampling_kernels import has_sampling_kernels
from spyre_inference.v1.sample.topk_topp_sampler import (
    SpyreTopKTopPSampler,
    apply_top_k_top_p_sort_free,
)

# Skip, like vLLM's optional extensions, where the kernels are absent (a prebaked image
# without them, a dev checkout that never built them). CI's non-prebaked runs still fail
# on a missing extension: the build_kernels job and the post-build check in
# run-matrix-config enforce it, independent of these tests.
pytestmark = pytest.mark.skipif(
    not has_sampling_kernels(), reason="csrc/ sampling kernels are not built"
)


# Vocabs off a SIMD-width multiple exercise the kernels' remainder handling.
@pytest.mark.parametrize("rows", [1, 3, 16])
@pytest.mark.parametrize("vocab", [7, 32000, 128257])
def test_greedy_matches_argmax(rows: int, vocab: int) -> None:
    x = torch.randn(rows, vocab)
    top = x.max() + 1
    x[:, vocab // 2] = top
    x[:, vocab // 3] = top  # tie: argmax keeps the lowest index
    assert torch.equal(torch.ops._spyre_C.greedy_argmax(x), x.argmax(dim=-1))


def _i64(x: int) -> int:
    """A uint64 constant as the int64 bit pattern torch stores."""
    return x - (1 << 64) if x >= 1 << 63 else x


def _shr(x: torch.Tensor, n: int) -> torch.Tensor:
    """Logical right shift of int64 tensors holding uint64 bits."""
    return (x >> n) & ((1 << (64 - n)) - 1)


def _splitmix64(z: torch.Tensor) -> torch.Tensor:
    z = (z ^ _shr(z, 30)) * _i64(0xBF58476D1CE4E5B9)
    z = (z ^ _shr(z, 27)) * _i64(0x94D049BB133111EB)
    return z ^ _shr(z, 31)


def _reference_gumbel_argmax(logits: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    """PyTorch baseline for fused_gumbel_argmax: the same hashed per-element noise, in
    float64, with none of the kernel's pruning or rejection shortcuts."""
    i = torch.arange(1, logits.shape[1] + 1, dtype=torch.int64)
    key = _splitmix64(seeds.to(torch.int64)).unsqueeze(1)
    bits = _shr(_splitmix64(key + i * _i64(0x9E3779B97F4A7C15)), 11)
    u = (bits.double() + 0.5) * 2.0**-53
    return (logits.double() - torch.log(-torch.log(u))).argmax(dim=-1)


# Identical, not just distributionally close: this also pins the kernel's shortcuts as exact.
@pytest.mark.parametrize("vocab,keep", [(7, None), (32000, None), (151936, None), (262144, 64)])
def test_gumbel_matches_pytorch_reference(vocab: int, keep: int | None) -> None:
    torch.manual_seed(vocab)
    logits = torch.randn(32, vocab) * 3
    if keep is not None:  # a top-k style mask: only `keep` tokens per row stay finite
        kept = torch.rand(32, vocab).argsort(dim=1)[:, :keep]
        masked = torch.full_like(logits, float("-inf"))
        logits = masked.scatter_(1, kept, logits.gather(1, kept))
    logits[0] = float("-inf")  # an all-masked row picks token 0, like argmax
    seeds = torch.randint(-(2**63), 2**63 - 1, (32,), dtype=torch.long)
    assert torch.equal(
        torch.ops._spyre_C.fused_gumbel_argmax(logits, seeds),
        _reference_gumbel_argmax(logits, seeds),
    )


@pytest.mark.parametrize(
    "k,p", [(None, None), (64, None), (64, 0.95)], ids=["none", "topk", "topk_topp"]
)
def test_topk_sampler_matches_pytorch_reference(k: int | None, p: float | None) -> None:
    rows = 16
    torch.manual_seed(0)
    logits = torch.randn(rows, 262144) * 3
    k_t = None if k is None else torch.full((rows,), k)
    p_t = None if p is None else torch.full((rows,), p)

    sampler = SpyreTopKTopPSampler("raw_logprobs", False)
    torch.manual_seed(1)
    out = sampler.forward_native(logits.clone(), {}, k_t, p_t)[0]

    # The sampler's own filter, then its per-row seed draw.
    if k_t is not None and p_t is not None:
        filtered = apply_top_k_top_p_sort_free(logits.clone(), k_t, p_t)
    else:
        filtered = apply_top_k_top_p_pytorch(logits.clone(), k_t, p_t, allow_cpu_sync=True)
    torch.manual_seed(1)
    seeds = torch.randint(0, 2**62, (rows,), dtype=torch.long)
    assert torch.equal(out, _reference_gumbel_argmax(filtered, seeds))


def test_gumbel_matches_softmax_distribution() -> None:
    vocab, draws = 8, 200_000
    logp = torch.log_softmax(torch.randn(vocab) * 1.5, dim=-1)
    seeds = torch.randint(0, 2**31, (draws,), dtype=torch.long)
    out = torch.ops._spyre_C.fused_gumbel_argmax(logp.expand(draws, vocab), seeds)
    freq = torch.bincount(out, minlength=vocab).float() / draws
    assert (freq - logp.exp()).abs().max() < 0.01


def test_gumbel_reaches_large_vocab_tail() -> None:
    # A shared noise table limits each row to 2^20 noise windows, which makes
    # much of a large vocab unreachable and under-samples the tail.
    vocab, draws, chunk = 151_936, 8192, 256
    row = -1.1 * torch.log(torch.arange(1, vocab + 1, dtype=torch.float64))  # Zipf(1.1)
    p = torch.softmax(row, dim=-1)
    tail = p < 2.0**-20
    gen = torch.Generator().manual_seed(0)
    hits = 0
    for _ in range(draws // chunk):
        seeds = torch.randint(0, 2**62, (chunk,), dtype=torch.long, generator=gen)
        logits = row.float().expand(chunk, vocab).contiguous()
        hits += int(tail[torch.ops._spyre_C.fused_gumbel_argmax(logits, seeds)].sum())
    expected = float(p[tail].sum())
    sigma = (expected * (1 - expected) / draws) ** 0.5
    assert abs(hits / draws - expected) < 4 * sigma


def test_gumbel_never_picks_masked_tokens() -> None:
    logits = torch.randn(64, 32000)
    logits[:, 100:] = float("-inf")
    seeds = torch.randint(0, 2**31, (64,), dtype=torch.long)
    assert (torch.ops._spyre_C.fused_gumbel_argmax(logits, seeds) < 100).all()


def test_seeded_requests_are_reproducible() -> None:
    sampler = SpyreTopKTopPSampler("raw_logprobs", False)
    logits = torch.randn(4, 32000)

    def draw() -> torch.Tensor:
        gens = {i: torch.Generator().manual_seed(i) for i in (0, 2)}
        return sampler.forward_native(logits.clone(), gens, None, None)[0]

    a, b = draw(), draw()
    assert torch.equal(a[[0, 2]], b[[0, 2]])


def test_spyre_sampler_greedy_matches_stock() -> None:
    logits = torch.randn(8, 32000)
    assert torch.equal(greedy_sample(logits), logits.argmax(dim=-1))


def test_gumbel_strided_seeds_match_contiguous() -> None:
    logits = torch.randn(8, 32000)
    seeds = torch.randint(0, 2**31, (16,), dtype=torch.long)[::2]
    assert not seeds.is_contiguous()
    assert torch.equal(
        torch.ops._spyre_C.fused_gumbel_argmax(logits, seeds),
        torch.ops._spyre_C.fused_gumbel_argmax(logits, seeds.contiguous()),
    )
    with pytest.raises(RuntimeError, match="seeds must be int64"):
        torch.ops._spyre_C.fused_gumbel_argmax(logits, seeds.int())


def test_greedy_nan_matches_argmax() -> None:
    # torch.argmax treats NaN as the maximum and returns the first one.
    vocab = 32003  # off a SIMD-width multiple, so the last NaN lands in the scalar tail
    x = torch.randn(6, vocab)
    x[0, 100] = float("nan")  # before the real max
    x[0, 200] = 50.0
    x[1, 200] = 50.0
    x[1, 300] = float("nan")  # after the real max
    x[2, [7, 9000]] = float("nan")
    x[3, vocab - 1] = float("nan")
    x[4, 10] = float("inf")
    x[4, 20] = float("-inf")  # inf + -inf: NaN sum without a NaN logit
    assert torch.equal(torch.ops._spyre_C.greedy_argmax(x), x.argmax(dim=-1))
