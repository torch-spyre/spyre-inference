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

"""Random sampling on the Spyre card.

One host uniform per row, then an inverse CDF on the device:

    token = (cumsum(softmax(masked_logits / T)) < u * total).sum(-1)

Only the ``[B]`` token ids are copied back. This is route 1 from torch-spyre
#5851. It needs device ``cumsum`` (#5853) and the fp16 upcast (#5856). Top-k
reads ``topk`` values (#5846, and #5854 when k > 128). Top-p sorts the row
(#5854). The draw matches the softmax distribution, not the host Gumbel
kernel's token for the same seed. The same ``u`` is the same token in any row.

Penalties, masks, logprobs, logits processors and structured outputs stay on
the host path. A batch that is not all-random stays there too.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

_active: ContextVar[bool] = ContextVar("spyre_device_random", default=False)
_compiled: Callable[..., torch.Tensor] | None = None


def sampling_active() -> bool:
    return _active.get()


@contextmanager
def activated() -> Generator[None]:
    token = _active.set(True)
    try:
        yield
    finally:
        _active.reset(token)


def batch_is_device_random(meta: SamplingMetadata, has_structured_output: bool) -> bool:
    """True when every row is a temperature draw and nothing else reads the logits."""
    if (
        not meta.all_random
        or meta.temperature is None
        or has_structured_output
        or not meta.no_penalties
        or meta.allowed_token_ids_mask is not None
        or meta.bad_words_token_ids
        or meta.max_num_logprobs is not None
        or meta.logprob_token_ids
        or meta.spec_token_ids
        or meta.logitsprocs.argmax_invariant
        or meta.logitsprocs.non_argmax_invariant
    ):
        return False
    holder = meta.thinking_budget_state_holder
    return holder is None or not holder.has_tracked_requests()


def row_uniforms(rows: int, generators: dict[int, torch.Generator]) -> torch.Tensor:
    """One ``u ~ Uniform[0, 1)`` per row.

    A seeded row draws from its own generator, so the value does not depend on
    which other rows share the step. Unseeded rows share the process generator,
    as the host sampler does.
    """
    uniform = torch.rand(rows)
    for index, generator in generators.items():
        uniform[index] = torch.rand((), generator=generator)
    return uniform


def padding_mask(padded_vocab: int, org_vocab_size: int, device: torch.device) -> torch.Tensor:
    """True on the lm_head pad columns. Those columns are logit 0 and would take mass."""
    mask = torch.zeros(padded_vocab, dtype=torch.bool)
    if org_vocab_size < padded_vocab:
        mask[org_vocab_size:] = True
    return mask.to(device)


def _apply_top_k(logits: torch.Tensor, top_k: torch.Tensor, k_max: int) -> torch.Tensor:
    """Keep every logit tied with the k-th largest. A row whose k is the vocab is unchanged."""
    vocab = logits.shape[-1]
    no_top_k = top_k == vocab
    index = (top_k - 1).clamp(min=0).unsqueeze(1).long()
    kth = torch.topk(logits, k_max, dim=-1).values.gather(1, index)
    kth = kth.masked_fill(no_top_k.unsqueeze(1), float("-inf"))
    return logits.masked_fill(logits < kth, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Nucleus mask. The token that crosses ``p`` stays, and so does the largest."""
    ascending = torch.argsort(logits, dim=-1, descending=False)
    sorted_logits = logits.gather(-1, ascending)
    cdf = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    drop_rank = cdf <= (1 - top_p.unsqueeze(1))
    drop_rank = drop_rank.clone()
    drop_rank[:, -1] = False
    rank = torch.argsort(ascending, dim=-1)
    return logits.masked_fill(drop_rank.gather(-1, rank), float("-inf"))


def _inverse_cdf(logits: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    cdf = torch.cumsum(torch.softmax(logits, dim=-1), dim=-1)
    total = torch.amax(cdf, dim=-1, keepdim=True)
    return torch.sum(cdf < uniform.unsqueeze(1) * total, dim=-1)


def random_token_ids(
    logits: torch.Tensor,
    pad_mask: torch.Tensor,
    temperature: torch.Tensor,
    uniform: torch.Tensor,
    top_k: torch.Tensor | None,
    k_max: int,
    top_p: torch.Tensor | None,
) -> torch.Tensor:
    """``[B, V]`` logits to ``[B]`` ids. ``pad_mask`` is ``[V]`` and broadcasts over rows."""
    # Mask the pad in the logits' dtype, then upcast. An fp32 mask of a host bool
    # does not compile on current torch-spyre; fp16 does, and #5856 covers the cast.
    values = logits.masked_fill(pad_mask, float("-inf")).float()
    values = values / temperature.unsqueeze(1)
    if top_k is not None:
        values = _apply_top_k(values, top_k, k_max)
    if top_p is not None:
        values = _apply_top_p(values, top_p)
    return _inverse_cdf(values, uniform)


def sample(
    logits: torch.Tensor,
    pad_mask: torch.Tensor,
    temperature: torch.Tensor,
    uniform: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    *,
    compile_kernel: bool,
) -> torch.Tensor:
    """Move the per-row vectors across, then run the draw.

    ``k_max`` is a Python int, so each distinct top-k compiles its own graph.
    Top-p does too. Those graphs are not registered with the compile guard:
    the k is only known when a request arrives. Warmup compiles the
    temperature-only graph at each row width.
    """
    device = logits.device
    temperature = temperature.to(device=device, dtype=torch.float32)
    uniform = uniform.to(device=device, dtype=torch.float32)
    k_max = 0
    if top_k is not None:
        top_k = top_k.to(device=device)
        k_max = int(top_k.max())
    if top_p is not None:
        top_p = top_p.to(device=device, dtype=torch.float32)
    return _kernel(compile_kernel)(logits, pad_mask, temperature, uniform, top_k, k_max, top_p)


def _kernel(compile_kernel: bool) -> Callable[..., torch.Tensor]:
    global _compiled
    if not compile_kernel:
        return random_token_ids
    if _compiled is None:
        from vllm.platforms import current_platform

        # Not @maybe_compile: SpyreLogitsProcessor already stores one compiled
        # kernel for the TP all-gather, and a second method would replace it.
        _compiled = torch.compile(
            random_token_ids,
            backend=current_platform.simple_compile_backend,
            fullgraph=True,
            dynamic=False,
        )
    return _compiled


def sampler_output(token_ids: torch.Tensor) -> SamplerOutput:
    """The shape ``Sampler.forward`` returns for an all-random batch with no logprobs."""
    return SamplerOutput(
        sampled_token_ids=token_ids.to(torch.int32).reshape(-1, 1),
        logprobs_tensors=None,
    )
