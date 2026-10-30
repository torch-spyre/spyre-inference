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

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.32.0 dispatcher adapters for the vendored csrc/cpu/spec_decode_utils.cpp.

Adapted from vllm/utils/cpu_triton_utils.py at 5df63a7203. The native kernels
accept the caller's dtypes directly and implement synthetic acceptance.

``install`` additionally swaps upstream's ``rejection_sample`` for a copy that takes the
target argmax with ``greedy_argmax`` and draws recovered tokens' noise in-kernel; vLLM's
RejectionSampler.forward runs unchanged.
"""

import torch
from vllm.v1.sample import rejection_sampler as upstream
from vllm.v1.sample.metadata import SamplingMetadata

from spyre_inference.v1.sample import sampler
from spyre_inference.v1.sample.sampling_kernels import draw_seeds, has_sampling_kernels

_HAS_REJECTION_KERNELS = has_sampling_kernels() and all(
    hasattr(torch.ops._spyre_C, name)
    for name in (
        "rejection_greedy_sample_kernel_impl",
        "rejection_random_sample_kernel_impl",
        "expand_kernel_impl",
        "sample_recovered_tokens_kernel_impl",
        "sample_recovered_tokens_seeded_kernel_impl",
    )
)


def _kernels():
    if not _HAS_REJECTION_KERNELS:
        raise RuntimeError(
            "Rejection sampling requires the C++ kernels. "
            "Rebuild spyre-inference's csrc/ CPU extension before using speculative decoding."
        )
    return torch.ops._spyre_C


def greedy_sample(
    output,
    cumulative,
    draft_ids,
    target_ids,
    bonus_ids,
    is_greedy,
    max_spec_len,
    uniform,
    conditional_rates,
    SYNTHETIC_MODE,
):
    _kernels().rejection_greedy_sample_kernel_impl(
        output,
        cumulative,
        draft_ids,
        target_ids,
        bonus_ids,
        is_greedy,
        max_spec_len,
        uniform,
        conditional_rates,
        SYNTHETIC_MODE,
    )


def random_sample(
    output,
    cumulative,
    draft_ids,
    draft_probs,
    target_probs,
    bonus_ids,
    recovered_ids,
    uniform,
    is_greedy,
    max_spec_len,
    vocab_size,
    conditional_rates,
    NO_DRAFT_PROBS,
    SYNTHETIC_MODE,
):
    _kernels().rejection_random_sample_kernel_impl(
        output,
        cumulative,
        draft_ids,
        draft_probs,
        target_probs,
        bonus_ids,
        recovered_ids,
        uniform,
        is_greedy,
        max_spec_len,
        vocab_size,
        conditional_rates,
        NO_DRAFT_PROBS,
        SYNTHETIC_MODE,
    )


def expand(output, values, cumulative, replace_from, replace_to, MAX_NUM_TOKENS):
    _kernels().expand_kernel_impl(output, values, cumulative, replace_from, replace_to)


def recover(
    output,
    cumulative,
    draft_ids,
    draft_probs,
    target_probs,
    inverse_noise,
    vocab_size,
    BLOCK_SIZE,
    NO_DRAFT_PROBS,
    USE_FP64_GUMBEL,
):
    _kernels().sample_recovered_tokens_kernel_impl(
        output,
        cumulative,
        draft_ids,
        draft_probs,
        target_probs,
        inverse_noise,
        vocab_size,
        NO_DRAFT_PROBS,
    )


_upstream_sample_recovered_tokens = upstream.sample_recovered_tokens


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Upstream's ``sample_recovered_tokens`` without its [batch, vocab] exponential_ draw."""
    if use_fp64_gumbel:
        # The kernel's noise is fixed; keep upstream's fp64 q, as SpyreTopKTopPSampler does.
        return _upstream_sample_recovered_tokens(
            max_spec_len,
            num_draft_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            sampling_metadata,
            device,
            use_fp64_gumbel,
        )
    # Like upstream, a request without drafts leaves its generator untouched.
    generators = {
        i: generator
        for i, generator in sampling_metadata.generators.items()
        if num_draft_tokens[i] > 0
    }
    recovered_token_ids = torch.empty_like(draft_token_ids)
    _kernels().sample_recovered_tokens_seeded_kernel_impl(
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        draw_seeds(len(num_draft_tokens), generators),
        target_probs.shape[-1],
        draft_probs is None,
    )
    return recovered_token_ids


def rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """vLLM 0.32.0's ``rejection_sample`` with two calls swapped: ``greedy_argmax`` for the
    target argmax and ``sample_recovered_tokens`` above for the recovered tokens."""
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_logits.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)

    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        upstream.PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == upstream.GREEDY_TEMPERATURE

    # Synthetic mode needs the uniforms in the greedy kernel too.
    uniform_probs: torch.Tensor | None = None
    if synthetic_mode or not sampling_metadata.all_greedy:
        uniform_probs = upstream.generate_uniform_probs(
            num_tokens,
            num_draft_tokens,
            sampling_metadata.generators,
            device,
        )

    if not sampling_metadata.all_random:
        upstream._rejection_greedy_sample(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            sampler.greedy_sample(target_logits),
            bonus_token_ids,
            is_greedy,
            max_spec_len,
            uniform_probs,
            synthetic_conditional_rates,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    assert target_probs.is_contiguous()

    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
        use_fp64_gumbel,
    )

    assert uniform_probs is not None
    upstream._rejection_random_sample(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        synthetic_conditional_rates,
    )
    return output_token_ids


def install() -> None:
    """Route ``RejectionSampler.forward`` through ``rejection_sample`` above."""
    upstream.rejection_sample = rejection_sample  # ty: ignore[invalid-assignment]
