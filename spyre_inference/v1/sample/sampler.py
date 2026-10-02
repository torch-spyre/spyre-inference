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
from vllm.config.model import LogprobsMode
from vllm.v1.sample.sampler import Sampler

import spyre_inference.v1.sample.sampling_kernels  # noqa: F401
from spyre_inference.v1.sample.topk_topp_sampler import SpyreTopKTopPSampler


class SpyreSampler(Sampler):
    """Upstream Sampler with host-side greedy and random draws.

    Spyre D2Hs logits before sampling, so both draws run on the host: greedy
    rows through the parallel argmax kernel, random rows through
    SpyreTopKTopPSampler."""

    def __init__(self, logprobs_mode: LogprobsMode, use_fp64_gumbel: bool) -> None:
        super().__init__(logprobs_mode, use_fp64_gumbel)
        self.topk_topp_sampler = SpyreTopKTopPSampler(logprobs_mode, use_fp64_gumbel)

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        return torch.ops._spyre_C.greedy_argmax(logits)  # ty: ignore[invalid-argument-type]
