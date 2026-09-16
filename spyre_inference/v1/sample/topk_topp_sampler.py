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
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, apply_top_k_only


class SpyreTopKTopPSampler(TopKTopPSampler):
    """Force the sort-free top-k path. Upstream only takes it under
    ``allow_cpu_sync`` (CPU platform only); Spyre D2Hs logits before sampling, so
    that host-device sync is free and the full-vocab sort it otherwise runs is
    pure waste. Applying top-k up front and passing ``k=None`` upstream is
    bit-identical to the joint sort (``-inf`` entries add 0 to the top-p cumsum,
    so they stay masked) and leaves top-p, logprobs and sampling delegated."""

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if k is not None:
            logits = apply_top_k_only(logits, k)
            k = None
        return super().forward_native(logits, generators, k, p)
