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

from spyre_inference.v1.sample.topk_topp_sampler import SpyreTopKTopPSampler


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


# The override applies top-k up front and delegates the rest upstream, so it must
# stay token-for-token identical to the stock joint sort. Both cases exercise the
# override's top-k pre-filter (top-p-only would delegate to super unchanged).
@pytest.mark.parametrize("k,p", [(50, None), (50, 0.8)], ids=["topk", "topk_topp"])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("vocab", [4096, 32000])
def test_swapped_sampler_matches_stock_tokens(
    rows: int, vocab: int, k: int | None, p: float | None
) -> None:
    meta = _meta(rows, k=k, p=p)
    logits = torch.randn(rows, vocab, dtype=torch.float16)

    stock = Sampler()
    torch.manual_seed(1234)
    out_stock = stock(logits=logits.clone(), sampling_metadata=meta).sampled_token_ids

    swapped = Sampler()
    swapped.topk_topp_sampler = SpyreTopKTopPSampler(swapped.logprobs_mode, swapped.use_fp64_gumbel)
    torch.manual_seed(1234)
    out_swap = swapped(logits=logits.clone(), sampling_metadata=meta).sampled_token_ids

    assert torch.equal(out_stock, out_swap)


@pytest.mark.parametrize("rows", [1, 8])
def test_log_space_gumbel_matches_softmax_draw(rows: int) -> None:
    """The log-space draw (logits - log q) matches the stock softmax(logits)/q
    draw for the same exponential noise q."""
    sampler = SpyreTopKTopPSampler("raw_logprobs", False)
    logits = torch.randn(rows, 32000, dtype=torch.float32) * 5.0

    torch.manual_seed(7)
    sampled, _ = sampler.forward_native(logits.clone(), {}, None, None)

    torch.manual_seed(7)
    q = torch.empty_like(logits).exponential_()
    ref = (logits.softmax(dim=-1) / q).argmax(dim=-1).view(-1)

    assert torch.equal(sampled, ref)


def test_runner_installs_spyre_topk_sampler() -> None:
    """The runner's __init__ installs SpyreTopKTopPSampler -- guards the swap itself."""
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.config.compilation import CompilationConfig

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
    assert type(runner.sampler.topk_topp_sampler) is SpyreTopKTopPSampler
