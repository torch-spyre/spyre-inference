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

"""SpyreTopKTopPSampler takes the sort-free top-k path without changing tokens."""

import pytest
import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.sampler import Sampler

from spyre_inference.v1.sample.topk_topp_sampler import (
    SpyreTopKTopPSampler,
    apply_top_k_top_p_sort_free,
)


def _meta(rows: int, k: int | None = None, p: float | None = None) -> SamplingMetadata:
    z = torch.zeros(rows)
    return SamplingMetadata(
        temperature=torch.full((rows,), 0.8),
        all_greedy=False,
        all_random=True,
        top_p=None if p is None else torch.full((rows,), p),
        top_k=None if k is None else torch.full((rows,), k, dtype=torch.long),
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=z.clone(),
        presence_penalties=z.clone(),
        repetition_penalties=torch.ones(rows),
        output_token_ids=[[] for _ in range(rows)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


# Compares two upstream functions (full sort vs the sort-free top-k), never
# touching SpyreTopKTopPSampler: it pins the upstream invariant this PR relies on.
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("vocab", [4096, 32000])
def test_topk_filter_matches_full_sort(rows: int, vocab: int) -> None:
    x = torch.randn(rows, vocab, dtype=torch.float32)
    k = torch.full((rows,), 50, dtype=torch.long)
    sort_path = apply_top_k_top_p_pytorch(x.clone(), k, None, allow_cpu_sync=False)
    topk_path = apply_top_k_top_p_pytorch(x.clone(), k, None, allow_cpu_sync=True)
    kept_sort = ~torch.isinf(sort_path)
    kept_topk = ~torch.isinf(topk_path)
    assert torch.equal(kept_sort, kept_topk)
    assert torch.equal(sort_path[kept_sort], topk_path[kept_topk])


# Mixed per-row k, plus a no-top-k row (k == vocab) that takes the full sort.
# fp16-rounded logits tie; "wide" ties 300 tokens at the top, past the 2*k window.
# When a tie straddles the top-p cutoff, which of the tied tokens survives is
# arbitrary in upstream's unstable sort too, so there only the kept values (not
# their positions) must match.
@pytest.mark.parametrize("ties", ["none", "fp16", "wide"])
@pytest.mark.parametrize("rows", [1, 8, 32])
@pytest.mark.parametrize("vocab", [4096, 262144])
def test_topk_topp_sort_free_matches_full_sort(rows: int, vocab: int, ties: str) -> None:
    torch.manual_seed(rows)
    x = torch.randn(rows, vocab) * 3
    if ties == "fp16":
        x = x.half().float()
    elif ties == "wide":
        x[:, :300] = 20.0
    p = torch.rand(rows) * 0.5 + 0.5
    for k in (torch.randint(1, 65, (rows,)), torch.tensor([vocab] + [64] * (rows - 1))):
        expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
        actual = apply_top_k_top_p_sort_free(x.clone(), k, p)
        if ties == "none":
            assert torch.equal(actual, expected)
        assert torch.equal(actual.sort(dim=-1).values, expected.sort(dim=-1).values)


# The override applies top-k up front and delegates the rest upstream, so it must
# stay token-for-token identical to the stock joint sort. Both cases exercise the
# override's top-k pre-filter (top-p-only would delegate to super unchanged).
# fp64 Gumbel keeps fresh noise on both sides; the fused kernel's table draw is
# covered in test_sampling_kernels.
@pytest.mark.parametrize("k,p", [(50, None), (50, 0.8)], ids=["topk", "topk_topp"])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("vocab", [4096, 32000])
def test_swapped_sampler_matches_stock_tokens(
    rows: int, vocab: int, k: int | None, p: float | None
) -> None:
    meta = _meta(rows, k=k, p=p)
    logits = torch.randn(rows, vocab, dtype=torch.float16)

    stock = Sampler(use_fp64_gumbel=True)
    torch.manual_seed(1234)
    out_stock = stock(logits=logits.clone(), sampling_metadata=meta).sampled_token_ids

    swapped = Sampler(use_fp64_gumbel=True)
    swapped.topk_topp_sampler = SpyreTopKTopPSampler(swapped.logprobs_mode, swapped.use_fp64_gumbel)
    torch.manual_seed(1234)
    out_swap = swapped(logits=logits.clone(), sampling_metadata=meta).sampled_token_ids

    assert torch.equal(out_stock, out_swap)


@pytest.mark.parametrize("rows", [1, 8])
def test_log_space_gumbel_matches_softmax_draw(rows: int) -> None:
    """The log-space draw (logits - log q) matches the stock softmax(logits)/q
    draw for the same exponential noise q."""
    sampler = SpyreTopKTopPSampler("raw_logprobs", True)
    logits = torch.randn(rows, 32000, dtype=torch.float32) * 5.0

    torch.manual_seed(7)
    sampled, _ = sampler.forward_native(logits.clone(), {}, None, None)

    torch.manual_seed(7)
    q = torch.empty_like(logits, dtype=torch.float64).exponential_()
    ref = (logits.softmax(dim=-1) / q).argmax(dim=-1).view(-1)

    assert torch.equal(sampled, ref)


def test_runner_installs_spyre_topk_sampler() -> None:
    """The runner's __init__ installs SpyreSampler -- guards the swap itself."""
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.config.compilation import CompilationConfig

    from spyre_inference.v1.sample.sampler import SpyreSampler
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B",
            max_model_len=1,
            dtype=torch.float16,
            trust_remote_code=True,
        ),
        cache_config=CacheConfig(block_size=128),
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )
    runner = TorchSpyreModelRunner(vllm_config, torch.device("cpu"))
    assert type(runner.sampler) is SpyreSampler
    assert type(runner.sampler.topk_topp_sampler) is SpyreTopKTopPSampler
