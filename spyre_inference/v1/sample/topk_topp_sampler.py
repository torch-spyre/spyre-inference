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

import torch
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p_pytorch,
)

import spyre_inference.v1.sample.sampling_kernels  # noqa: F401


def apply_top_k_top_p_sort_free(
    logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor
) -> torch.Tensor:
    """``apply_top_k_top_p_pytorch`` without the full-vocab sort.

    Top-p only ever keeps top-k survivors, so it runs on a ``topk`` window.
    vllm keeps every token tied with a row's k-th value, and fp16 logits tie
    there often, so the window carries slack and widens to the exact tie count
    when ties run past it.

    The logits tensor is updated in-place."""
    max_k = int(k.max())
    vocab = logits.shape[1]
    if max_k >= vocab:
        return apply_top_k_top_p_pytorch(logits, k, p)
    vals, idx = logits.topk(min(2 * max_k, vocab), dim=-1)
    kth = vals.gather(1, k.long().unsqueeze(1) - 1)
    if (vals[:, -1:] == kth).any():
        vals, idx = logits.topk(int((logits >= kth).sum(dim=-1).max()), dim=-1)
    vals.masked_fill_(vals < kth, -float("inf"))
    # Upstream's top-p, on the ascending order it sorts into.
    vals, idx = vals.flip(-1), idx.flip(-1)
    probs_sum = vals.softmax(dim=-1).cumsum_(dim=-1)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False
    vals.masked_fill_(top_p_mask, -float("inf"))
    return logits.fill_(-float("inf")).scatter_(1, idx, vals)


class SpyreTopKTopPSampler(TopKTopPSampler):
    """Sort-free top-k plus a fused Gumbel-max draw for random sampling.

    Upstream only takes the sort-free top-k path under ``allow_cpu_sync`` (CPU
    platform only); Spyre D2Hs logits before sampling, so that host-device sync
    is free and the full-vocab sort it otherwise runs is pure waste. Top-p
    combined with top-k skips the sort too (``apply_top_k_top_p_sort_free``);
    top-p alone sorts.

    The draw runs in vLLM's fused Gumbel-max kernel, reading its noise from a
    fixed 2^20-entry table at a per-row random offset rather than generating a
    fresh ``[B, V]`` noise tensor. That draw is why ``forward_native``
    reimplements upstream's tail rather than delegating to ``super()`` (which
    would softmax + ``random_sample``). ``use_fp64_gumbel`` keeps fresh fp64
    noise, drawn in log space: ``argmax(softmax(x)/q) == argmax(x - log q)`` for
    ``q ~ Exp(1)``, which skips the softmax."""

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Mirrors upstream TopKTopPSampler.forward_native (vLLM 0.28.0) with two
        # Spyre changes: sort-free top-k (and top-k + top-p) and the fused
        # Gumbel-max kernel. Re-sync with upstream on a vLLM bump.
        if k is not None and p is not None:
            logits = apply_top_k_top_p_sort_free(logits, k, p)
        else:
            logits = apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        if not self.use_fp64_gumbel:
            # Per-row seeds offset into a precomputed Gumbel table, so the draw
            # is one pass with no noise tensor; seeded requests stay reproducible.
            seeds = torch.randint(0, 2**31, (logits.shape[0],), dtype=torch.long)
            for i, generator in generators.items():
                seeds[i] = torch.randint(0, 2**31, (1,), generator=generator)
            return (
                torch.ops._spyre_C.fused_gumbel_argmax(
                    logits.float(),  # ty: ignore[invalid-argument-type]
                    seeds,  # ty: ignore[invalid-argument-type]
                ),
                logits_to_return,
            )
        q = torch.empty(logits.shape, dtype=torch.float64, device=logits.device)
        if len(generators) != logits.shape[0]:
            q.exponential_()
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
        return (logits - q.log_()).argmax(dim=-1).view(-1), logits_to_return
