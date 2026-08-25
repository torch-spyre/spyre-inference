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

"""Host sampler path for Spyre (async noise ring buffer + log-space Gumbel).

Port of sendnn-inference#1046 (Holtz): async Exp(1) log-noise, TP rank-0
sample + broadcast, and log-space Gumbel.

Config levers live in ``spyre_inference.envs`` (``SPYRE_USE_SPYRE_SAMPLER``,
``SPYRE_ASYNC_NOISE_SCALE``).
"""

from __future__ import annotations

import warnings

from vllm.config import VllmConfig
from vllm.v1.sample.sampler import Sampler

import spyre_inference.envs as envs
from spyre_inference.v1.sample.async_ring_buffer import (
    AsyncExponential_RingBuffer,
    AsyncRingBuffer,
)
from spyre_inference.v1.sample.spyre_sampler import SpyreSampler
from spyre_inference.v1.sample.spyre_topk_topp_sampler import SpyreTopKTopPSampler


def build_spyre_sampler(vllm_config: VllmConfig) -> Sampler:
    """Build Holtz SpyreSampler, or fall back to upstream Sampler.

    Falls back when ``SPYRE_USE_SPYRE_SAMPLER=0`` or when ``vllm_config`` lacks
    ``max_num_seqs`` / vocab size (same as sendnn ``SpyreCausalLM``).
    """
    logprobs_mode = vllm_config.model_config.logprobs_mode
    if not envs.SPYRE_USE_SPYRE_SAMPLER:
        return Sampler(logprobs_mode=logprobs_mode)
    if not SpyreSampler.is_vllm_config_compatible(vllm_config):
        warnings.warn(
            "The provided vllm_config is not compatible with SpyreSampler. "
            "Falling back to default Sampler with reduced performance on Spyre platform.",
            stacklevel=2,
        )
        return Sampler(logprobs_mode=logprobs_mode)
    return SpyreSampler(
        vllm_config=vllm_config,
        logprobs_mode=logprobs_mode,
    )


__all__ = [
    "AsyncExponential_RingBuffer",
    "AsyncRingBuffer",
    "SpyreSampler",
    "SpyreTopKTopPSampler",
    "build_spyre_sampler",
]
