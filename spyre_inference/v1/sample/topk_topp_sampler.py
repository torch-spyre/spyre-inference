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
    empty_exponential_noise_like,
)


class SpyreTopKTopPSampler(TopKTopPSampler):
    """Force the sort-free top-k path. Upstream only takes it under
    ``allow_cpu_sync`` (CPU platform only); Spyre D2Hs logits before sampling, so
    that host-device sync is free and the full-vocab sort it otherwise runs is
    pure waste. Top-p still sorts (unaffected)."""

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Mirrors upstream TopKTopPSampler.forward_native (vLLM 0.28.0) with two
        # Spyre changes: the sort-free top-k path (allow_cpu_sync=True) and a
        # log-space Gumbel draw. Re-sync with upstream on a vLLM bump.
        logits = apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        # argmax(softmax(x)/q) == argmax(x - log q) for q~Exp(1): the softmax's
        # per-row normalization is a constant the argmax ignores, so skip it.
        # Noise generation mirrors upstream random_sample.
        q = empty_exponential_noise_like(logits, self.use_fp64_gumbel)
        if len(generators) != logits.shape[0]:
            q.exponential_()
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
        sampled = (logits - q.log_()).argmax(dim=-1).view(-1)
        return sampled, logits_to_return
