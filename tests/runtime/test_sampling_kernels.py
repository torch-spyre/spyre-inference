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

from spyre_inference.v1.sample.sampler import SpyreSampler
from spyre_inference.v1.sample.topk_topp_sampler import SpyreTopKTopPSampler


# Vocabs off a SIMD-width multiple exercise the kernels' remainder handling.
@pytest.mark.parametrize("rows", [1, 3, 16])
@pytest.mark.parametrize("vocab", [7, 32000, 128257])
def test_greedy_matches_argmax(rows: int, vocab: int) -> None:
    x = torch.randn(rows, vocab)
    top = x.max() + 1
    x[:, vocab // 2] = top
    x[:, vocab // 3] = top  # tie: argmax keeps the lowest index
    assert torch.equal(torch.ops._spyre_C.greedy_argmax(x), x.argmax(dim=-1))


def test_gumbel_matches_softmax_distribution() -> None:
    vocab, draws = 8, 200_000
    logp = torch.log_softmax(torch.randn(vocab) * 1.5, dim=-1)
    seeds = torch.randint(0, 2**31, (draws,), dtype=torch.long)
    out = torch.ops._spyre_C.fused_gumbel_argmax(logp.expand(draws, vocab), seeds)
    freq = torch.bincount(out, minlength=vocab).float() / draws
    assert (freq - logp.exp()).abs().max() < 0.01


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
    assert torch.equal(SpyreSampler.greedy_sample(logits), logits.argmax(dim=-1))
