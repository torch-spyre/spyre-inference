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

"""Host replacements for the Triton kernels upstream speculative decoding launches.

Sampling and spec-decode bookkeeping run on the host here, and the build has neither
Triton nor the CPU backend's ``torch.ops._C`` stand-ins (``VLLM_TARGET_DEVICE=empty``).
Each function keeps its kernel's in-place output contract. Batches are at most
``max_num_seqs`` rows, so per-request Python loops cost less than the vectorized
alternatives would.
"""

from __future__ import annotations

import dataclasses

import torch
from vllm.v1.sample import rejection_sampler as _rejection_sampler

from spyre_inference.v1.sample.sampling_kernels import has_sampling_kernels
from spyre_inference.v1.sample.topk_topp_sampler import apply_top_k_top_p_host

# Bound before install() replaces it: the fallback when Spyre's kernels do not apply.
_upstream_rejection_sample = _rejection_sampler.rejection_sample

# A top-k this wide (a "disabled" top-k is often vocab - 1) is no window: its ``topk``
# would cost more than the full-vocabulary path.
_WINDOW_MAX_TOP_K = 1024


class _HostDispatch:
    """Stands in for upstream's Triton kernel dispatcher; there is nothing to warm up."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def register_warmup(self, *args, **kwargs) -> None:
        pass


def _starts_and_counts(cu_num_tokens: torch.Tensor) -> tuple[list[int], list[int]]:
    """Per-request start offsets and lengths from an inclusive cumulative sum."""
    ends = cu_num_tokens.tolist()
    starts = [0, *ends[:-1]]
    return starts, [end - start for start, end in zip(starts, ends)]


def _store(output: torch.Tensor, rows: list[int], cols: list[int], vals: list[int]) -> None:
    if rows:
        output[rows, cols] = torch.tensor(vals, dtype=output.dtype)


def eagle_prepare_inputs_padded(
    cu_num_draft_tokens: torch.Tensor,
    valid_sampled_tokens_count: torch.Tensor,
    query_start_loc: torch.Tensor,
    token_indices_to_sample: torch.Tensor,
    num_rejected_tokens: torch.Tensor,
    num_reqs: int,
) -> None:
    cu = cu_num_draft_tokens[:num_reqs].long()
    num_draft = torch.diff(cu, prepend=cu.new_zeros(1))
    valid = valid_sampled_tokens_count[:num_reqs].long()
    rejected = torch.where(num_draft > 0, num_draft + 1 - valid, torch.zeros_like(num_draft))
    last_token = query_start_loc[1 : num_reqs + 1].long() - 1
    token_indices_to_sample[:num_reqs] = (last_token - rejected).to(token_indices_to_sample.dtype)
    num_rejected_tokens[:num_reqs] = rejected.to(num_rejected_tokens.dtype)


def eagle_prepare_next_token_padded(
    sampled_token_ids: torch.Tensor,
    discard_request_mask: torch.Tensor,
    backup_next_token_ids: torch.Tensor,
    next_token_ids: torch.Tensor,
    valid_sampled_tokens_count: torch.Tensor,
    vocab_size: int,
    num_sampled_tokens_per_req: int,
    num_reqs: int,
    *args,
    **kwargs,
) -> None:
    tokens = sampled_token_ids[:num_reqs, :num_sampled_tokens_per_req].long()
    valid = (tokens != -1) & (tokens < vocab_size)
    count = valid.sum(dim=1)
    cols = torch.arange(tokens.shape[1]).expand_as(tokens)
    last_index = torch.where(valid, cols, torch.full_like(cols, -1)).amax(dim=1)
    last_token = tokens.gather(1, last_index.clamp(min=0).unsqueeze(1)).squeeze(1)
    discard = discard_request_mask[:num_reqs].bool()
    backup = backup_next_token_ids[:num_reqs].long()
    next_token_ids[:num_reqs] = torch.where(discard | (count == 0), backup, last_token).to(
        next_token_ids.dtype
    )
    valid_sampled_tokens_count[:num_reqs] = torch.where(discard, torch.zeros_like(count), count).to(
        valid_sampled_tokens_count.dtype
    )


def rejection_greedy_sample(
    output_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_argmax: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    is_greedy: torch.Tensor | None,
    max_spec_len: int,
    uniform_probs: torch.Tensor | None = None,
    synthetic_conditional_rates: torch.Tensor | None = None,
    SYNTHETIC_MODE: bool = False,
) -> None:
    starts, counts = _starts_and_counts(cu_num_draft_tokens)
    drafts = draft_token_ids.tolist()
    argmax = target_argmax.tolist()
    bonus = bonus_token_ids.reshape(-1).tolist()
    greedy = None if is_greedy is None else is_greedy.tolist()
    uniform = uniform_probs.tolist() if SYNTHETIC_MODE and uniform_probs is not None else None
    rates = (
        synthetic_conditional_rates.tolist()
        if SYNTHETIC_MODE and synthetic_conditional_rates is not None
        else None
    )

    rows: list[int] = []
    cols: list[int] = []
    vals: list[int] = []
    for req, (start, n) in enumerate(zip(starts, counts)):
        if greedy is not None and not greedy[req]:
            continue
        rejected = False
        for pos in range(n):
            draft, target = drafts[start + pos], argmax[start + pos]
            if uniform is not None and rates is not None:
                accepted = uniform[start + pos] < rates[pos] and draft >= 0
                token, rejected = (draft if accepted else target), not accepted
            else:
                token, rejected = target, draft != target
            rows.append(req)
            cols.append(pos)
            vals.append(token)
            if rejected:
                break
        if not rejected:
            rows.append(req)
            cols.append(n)
            vals.append(bonus[req])
    _store(output_token_ids, rows, cols, vals)


def rejection_random_sample(
    output_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    recovered_token_ids: torch.Tensor,
    uniform_probs: torch.Tensor,
    is_greedy: torch.Tensor | None,
    max_spec_len: int,
    vocab_size: int,
    synthetic_conditional_rates: torch.Tensor | None = None,
    NO_DRAFT_PROBS: bool = False,
    SYNTHETIC_MODE: bool = False,
) -> None:
    rows, safe_drafts = _draft_cells(draft_token_ids)
    _accept_random(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_probs[rows, safe_drafts],
        None if NO_DRAFT_PROBS or draft_probs is None else draft_probs[rows, safe_drafts],
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        synthetic_conditional_rates if SYNTHETIC_MODE else None,
    )


def _draft_cells(draft_token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(row, draft token)`` index pairs, with padded drafts (-1) pointed at token 0."""
    return torch.arange(draft_token_ids.shape[0]), draft_token_ids.long().clamp(min=0)


def _accept_random(
    output_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_p_at_draft: torch.Tensor,
    draft_p_at_draft: torch.Tensor | None,
    bonus_token_ids: torch.Tensor,
    recovered_token_ids: torch.Tensor,
    uniform_probs: torch.Tensor,
    is_greedy: torch.Tensor | None,
    synthetic_conditional_rates: torch.Tensor | None,
) -> None:
    """The random-request half of rejection sampling, from per-draft probabilities."""
    starts, counts = _starts_and_counts(cu_num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    target_p = target_p_at_draft.tolist()
    draft_p = [1.0] * num_tokens if draft_p_at_draft is None else draft_p_at_draft.tolist()
    drafts = draft_token_ids.tolist()
    recovered = recovered_token_ids.tolist()
    uniform = uniform_probs.tolist()
    bonus = bonus_token_ids.reshape(-1).tolist()
    greedy = None if is_greedy is None else is_greedy.tolist()
    rates = None if synthetic_conditional_rates is None else synthetic_conditional_rates.tolist()

    rows: list[int] = []
    cols: list[int] = []
    vals: list[int] = []
    for req, (start, n) in enumerate(zip(starts, counts)):
        if greedy is not None and greedy[req]:
            continue
        rejected = False
        for pos in range(n):
            i = start + pos
            if drafts[i] < 0:
                accepted = False
            elif rates is not None:
                accepted = uniform[i] < rates[pos]
            else:
                # A zero draft probability is rejected rather than divided by.
                accepted = draft_p[i] > 0 and target_p[i] / draft_p[i] >= uniform[i]
            rows.append(req)
            cols.append(pos)
            vals.append(drafts[i] if accepted else recovered[i])
            if not accepted:
                rejected = True
                break
        if not rejected:
            rows.append(req)
            cols.append(n)
            vals.append(bonus[req])
    _store(output_token_ids, rows, cols, vals)


def expand(
    output: torch.Tensor,
    input_val: torch.Tensor,
    cu_num_tokens: torch.Tensor,
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: int | None = None,
) -> None:
    _, counts = _starts_and_counts(cu_num_tokens)
    values = torch.where(
        input_val == replace_from, torch.full_like(input_val, replace_to), input_val
    )
    expanded = values.repeat_interleave(torch.tensor(counts, dtype=torch.long))
    output[: expanded.shape[0]] = expanded.to(output.dtype)


def sample_recovered_tokens(
    output_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    inv_q: torch.Tensor,
    vocab_size: int,
    BLOCK_SIZE: int | None = None,
    NO_DRAFT_PROBS: bool = False,
    USE_FP64_GUMBEL: bool = False,
) -> None:
    _, counts = _starts_and_counts(cu_num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    if num_tokens == 0:
        return
    req_of_token = torch.repeat_interleave(
        torch.arange(len(counts)), torch.tensor(counts, dtype=torch.long)
    )
    prob = _residual_probs(draft_token_ids, None if NO_DRAFT_PROBS else draft_probs, target_probs)
    score = prob.to(inv_q.dtype) * inv_q[req_of_token, :vocab_size]
    recovered = score.argmax(dim=-1).clamp(max=vocab_size - 1)
    output_token_ids[:num_tokens] = recovered.to(output_token_ids.dtype)


def _residual_probs(
    draft_token_ids: torch.Tensor, draft_probs: torch.Tensor | None, target_probs: torch.Tensor
) -> torch.Tensor:
    """The (unnormalized) distribution a rejected position resamples from."""
    num_tokens = draft_token_ids.shape[0]
    if draft_probs is None:
        prob = target_probs[:num_tokens].clone()
        # The draft token itself is excluded; padded drafts (-1) exclude nothing.
        drafts = draft_token_ids.long()
        real = (drafts >= 0).nonzero().squeeze(1)
        prob[real, drafts[real]] = 0
        return prob
    return (target_probs[:num_tokens] - draft_probs[:num_tokens]).clamp(min=0)


def _fused_draw(
    logits: torch.Tensor, num_draft_tokens: list[int], generators: dict[int, torch.Generator]
) -> torch.Tensor:
    """One Gumbel-max draw per row of ``logits`` (fp32, -inf masked) through the C++ kernel.

    Rows get distinct seeds: only a request's first rejected position is ever used, so
    upstream's one noise row per request buys nothing. Like upstream, a request without
    drafts leaves its generator untouched.
    """
    request_seeds = torch.randint(0, 2**62, (len(num_draft_tokens),), dtype=torch.long)
    for i, generator in generators.items():
        if num_draft_tokens[i] > 0:
            request_seeds[i] = torch.randint(0, 2**62, (1,), generator=generator)
    counts = torch.tensor(num_draft_tokens, dtype=torch.long)
    starts = torch.cumsum(counts, 0) - counts
    position = torch.arange(logits.shape[0]) - starts.repeat_interleave(counts)
    seeds = request_seeds.repeat_interleave(counts) + position
    return torch.ops._spyre_C.fused_gumbel_argmax(
        logits.contiguous(),  # ty: ignore[invalid-argument-type]
        seeds,  # ty: ignore[invalid-argument-type]
    )


def rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Upstream's ``rejection_sample`` on Spyre's host sampling kernels.

    Greedy requests take the target argmax from ``greedy_argmax``. Random requests with
    greedy drafts (no ``draft_probs``) skip the full-vocabulary softmax: acceptance needs
    only the target probability of each draft token, ``exp(logit - logsumexp)``, and the
    recovered token is a ``fused_gumbel_argmax`` draw over the logits with the draft
    masked out, which samples the renormalized residual exactly.
    """
    if synthetic_mode or use_fp64_gumbel or not has_sampling_kernels():
        return _upstream_rejection_sample(
            draft_token_ids, num_draft_tokens, max_spec_len, cu_num_draft_tokens,
            draft_probs, target_logits, bonus_token_ids, sampling_metadata,
            synthetic_mode=synthetic_mode,
            synthetic_conditional_rates=synthetic_conditional_rates,
            use_fp64_gumbel=use_fp64_gumbel,
        )  # fmt: skip

    output = torch.full((len(num_draft_tokens), max_spec_len + 1), -1, dtype=torch.int32)
    logits = target_logits.float().contiguous()
    is_greedy = None if sampling_metadata.all_greedy else sampling_metadata.temperature == 0

    if not sampling_metadata.all_random:
        target_argmax = torch.ops._spyre_C.greedy_argmax(logits)  # ty: ignore[invalid-argument-type]
        rejection_greedy_sample(
            output, cu_num_draft_tokens, draft_token_ids, target_argmax, bonus_token_ids,
            is_greedy, max_spec_len,
        )  # fmt: skip
        if sampling_metadata.all_greedy:
            return output

    num_tokens = draft_token_ids.shape[0]
    uniform_probs = _rejection_sampler.generate_uniform_probs(
        num_tokens, num_draft_tokens, sampling_metadata.generators, logits.device
    )
    rows, safe_drafts = _draft_cells(draft_token_ids)
    if draft_probs is None:
        target_p = (logits[rows, safe_drafts] - logits.logsumexp(dim=-1)).exp()
        draft_p = None
        masked = logits.clone()
        drafts = draft_token_ids.long()
        real = (drafts >= 0).nonzero().squeeze(1)
        masked[real, drafts[real]] = -float("inf")
    else:
        target_probs = logits.softmax(dim=-1)
        target_p = target_probs[rows, safe_drafts]
        draft_p = draft_probs[rows, safe_drafts]
        masked = _residual_probs(draft_token_ids, draft_probs, target_probs).log_()
    recovered = _fused_draw(masked, num_draft_tokens, sampling_metadata.generators)

    _accept_random(
        output, cu_num_draft_tokens, draft_token_ids, target_p, draft_p, bonus_token_ids,
        recovered, uniform_probs, is_greedy, None,
    )  # fmt: skip
    return output


def _window_target_probs(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's sampling distribution after temperature, top-k and top-p, over its window.

    Returns ``(probs, token_ids)``, ``[rows, window]``: every token outside the window has
    probability zero, so past one ``topk`` over the vocabulary nothing touches it. Matches
    ``apply_top_k_top_p_sort_free`` on the temperature-scaled logits, ties at a row's k-th
    value included; the positive temperature changes neither the order nor the ties.
    """
    k = top_k.long().unsqueeze(1)
    vocab = logits.shape[1]
    vals, ids = logits.topk(min(2 * int(k.max()), vocab), dim=-1)
    kth = vals.gather(1, k - 1)
    if (vals[:, -1:] == kth).any():
        vals, ids = logits.topk(int((logits >= kth).sum(dim=-1).max()), dim=-1)
    scaled = (vals / temperature.unsqueeze(1)).masked_fill(vals < kth, -float("inf"))
    if top_p is not None:
        ascending = scaled.flip(-1)
        cumulative = ascending.softmax(dim=-1).cumsum(dim=-1)
        drop = cumulative <= 1 - top_p.unsqueeze(1)
        drop[:, -1] = False
        scaled = ascending.masked_fill(drop, -float("inf")).flip(-1)
    return scaled.softmax(dim=-1), ids


def _window_rejection_sample(
    metadata,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata,
) -> torch.Tensor:
    """Rejection sampling for random requests with greedy drafts, on top-k windows.

    The draft token's acceptance probability is its window probability (zero outside the
    window), and the recovered token is a Gumbel-max draw over the window with the draft
    excluded: the renormalized residual, exactly. Greedy requests keep the full argmax.
    """
    draft_token_ids = metadata.draft_token_ids
    num_draft_tokens = metadata.num_draft_tokens
    cu_num_draft_tokens = metadata.cu_num_draft_tokens
    output = torch.full((len(num_draft_tokens), metadata.max_spec_len + 1), -1, dtype=torch.int32)
    # A step with no drafts scheduled (a prefill chunk) still verifies: just the bonus rows.
    if draft_token_ids.numel() == 0:
        output[:, 0] = bonus_token_ids.view(-1)
        return output
    is_greedy = None if sampling_metadata.all_random else sampling_metadata.temperature == 0
    if is_greedy is not None:
        target_argmax = torch.ops._spyre_C.greedy_argmax(target_logits)  # ty: ignore[invalid-argument-type]
        rejection_greedy_sample(
            output, cu_num_draft_tokens, draft_token_ids, target_argmax, bonus_token_ids,
            is_greedy, metadata.max_spec_len,
        )  # fmt: skip

    counts = torch.tensor(num_draft_tokens, dtype=torch.long)
    # Greedy rows draw at temperature 1 here; their result is discarded above.
    temperature = sampling_metadata.temperature.float()
    temperature = torch.where(temperature > 0, temperature, torch.ones_like(temperature))
    top_p = sampling_metadata.top_p
    probs, ids = _window_target_probs(
        target_logits,
        temperature.repeat_interleave(counts),
        sampling_metadata.top_k.repeat_interleave(counts),
        None if top_p is None else top_p.float().repeat_interleave(counts),
    )
    is_draft = ids == draft_token_ids.long().unsqueeze(1)
    target_p = (probs * is_draft).sum(dim=-1)
    window_choice = _fused_draw(
        probs.masked_fill(is_draft, 0).log(), num_draft_tokens, sampling_metadata.generators
    )
    recovered = ids.gather(1, window_choice.long().unsqueeze(1)).squeeze(1)
    uniform_probs = _rejection_sampler.generate_uniform_probs(
        draft_token_ids.shape[0], num_draft_tokens, sampling_metadata.generators, probs.device
    )
    _accept_random(
        output, cu_num_draft_tokens, draft_token_ids, target_p, None, bonus_token_ids,
        recovered, uniform_probs, is_greedy, None,
    )  # fmt: skip
    return output


def _window_eligible(sampler, draft_probs, sampling_metadata, vocab_size: int) -> bool:
    """Random requests that all set a narrow top-k, with nothing that needs full-vocab outputs."""
    top_k = sampling_metadata.top_k
    return (
        draft_probs is None
        and top_k is not None
        and not sampling_metadata.all_greedy
        and sampling_metadata.max_num_logprobs is None
        and not sampler.synthetic_mode
        and not sampler.use_fp64_gumbel
        and has_sampling_kernels()
        and bool(((top_k > 0) & (top_k <= min(_WINDOW_MAX_TOP_K, vocab_size - 1))).all())
    )


def _window_forward(original):
    """``RejectionSampler.forward`` that rejection-samples on top-k windows when it can.

    Upstream's path makes several full-vocabulary passes over every draft row (copy, cast,
    clone, temperature, the top-k/top-p fill, the draw); with a top-k, all but one
    ``topk`` work on the window. Anything else falls back to ``original``.
    """

    def forward(self, metadata, draft_probs, logits, sampling_metadata):
        from vllm.v1.outputs import SamplerOutput

        if not _window_eligible(self, draft_probs, sampling_metadata, logits.shape[-1]):
            return original(self, metadata, draft_probs, logits, sampling_metadata)
        bonus_token_ids = self.sampler(
            logits=logits[metadata.bonus_logits_indices],
            sampling_metadata=dataclasses.replace(sampling_metadata, max_num_logprobs=-1),
            predict_bonus_token=True,
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        ).sampled_token_ids
        target_logits = logits[metadata.target_logits_indices].float()
        target_logits = self.apply_logits_processors(target_logits, sampling_metadata, metadata)
        output = _window_rejection_sample(
            metadata, target_logits, bonus_token_ids, sampling_metadata
        )
        return SamplerOutput(sampled_token_ids=output, logprobs_tensors=None)

    return forward


def install() -> None:
    """Point upstream's spec-decode kernel launches at the host versions above.

    Also routes the rejection sampler's top-k/top-p and its accept/recover draws through
    Spyre's host sampling kernels. Upstream calls each Triton kernel through a dispatcher
    bound at import time, so the dispatchers are what get replaced, on the modules that
    call them.
    """
    import vllm.v1.sample.rejection_sampler as rejection_sampler
    import vllm.v1.spec_decode.llm_base_proposer as base_proposer

    base_proposer._eagle_prepare_inputs_padded = _HostDispatch(eagle_prepare_inputs_padded)  # ty: ignore[invalid-assignment]
    base_proposer._eagle_prepare_next_token_padded = _HostDispatch(  # ty: ignore[invalid-assignment]
        eagle_prepare_next_token_padded
    )

    def greedy(*args) -> None:
        rejection_greedy_sample(*args, SYNTHETIC_MODE=args[-1] is not None)

    def random(
        output_token_ids, cu_num_draft_tokens, draft_token_ids, draft_probs, target_probs,
        bonus_token_ids, recovered_token_ids, uniform_probs, is_greedy, max_spec_len,
        synthetic_conditional_rates,
    ) -> None:  # fmt: skip
        rejection_random_sample(
            output_token_ids, cu_num_draft_tokens, draft_token_ids, draft_probs, target_probs,
            bonus_token_ids, recovered_token_ids, uniform_probs, is_greedy, max_spec_len,
            target_probs.shape[1], synthetic_conditional_rates,
            NO_DRAFT_PROBS=draft_probs is None,
            SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        )  # fmt: skip

    def recovered(
        output_token_ids, cu_num_draft_tokens, draft_token_ids, draft_probs, target_probs,
        inv_q, max_spec_len: int,
    ) -> None:  # fmt: skip
        sample_recovered_tokens(
            output_token_ids, cu_num_draft_tokens, draft_token_ids, draft_probs, target_probs,
            inv_q, target_probs.shape[1], NO_DRAFT_PROBS=draft_probs is None,
        )  # fmt: skip

    rejection_sampler._rejection_greedy_sample = _HostDispatch(greedy)  # ty: ignore[invalid-assignment]
    rejection_sampler._rejection_random_sample = _HostDispatch(random)  # ty: ignore[invalid-assignment]
    rejection_sampler._expand = _HostDispatch(expand)  # ty: ignore[invalid-assignment]
    rejection_sampler._sample_recovered_tokens = _HostDispatch(recovered)  # ty: ignore[invalid-assignment]
    rejection_sampler.rejection_sample = rejection_sample  # ty: ignore[invalid-assignment]
    sampler_cls = rejection_sampler.RejectionSampler
    if not getattr(sampler_cls.forward, "_spyre_window", False):
        forward = _window_forward(sampler_cls.forward)
        forward._spyre_window = True  # ty: ignore[unresolved-attribute]
        sampler_cls.forward = forward  # ty: ignore[invalid-assignment]
    rejection_sampler.apply_top_k_top_p = apply_top_k_top_p_host  # ty: ignore[invalid-assignment]
