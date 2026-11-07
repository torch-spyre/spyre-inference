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

"""Greedy sampling on the Spyre card.

``torch.argmax(logits, dim=-1)`` over the fp16 ``[B, V]`` logits, then only the
``[B]`` token ids are copied to the host. This assumes torch-spyre #5845: the
reduction runs on the device and matches CPU ``torch.argmax`` (lowest index on a
tie, NaN's index when a row contains NaN). The topk issues #5846-#5850 are the
other way to pick a greedy token; this path does not call ``topk``.

Penalties, masks, logprobs and structured outputs stay on the host path. A batch
that is not all-greedy stays there too.
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

_active: ContextVar[bool] = ContextVar("spyre_device_greedy", default=False)


def sampling_active() -> bool:
    return _active.get()


@contextmanager
def activated() -> Generator[None]:
    token = _active.set(True)
    try:
        yield
    finally:
        _active.reset(token)


def batch_is_device_greedy(meta: SamplingMetadata, has_structured_output: bool) -> bool:
    """True when every row can be drawn with ``argmax`` and nothing else reads the logits."""
    if not meta.all_greedy or has_structured_output:
        return False
    if (
        not meta.no_penalties
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


def padding_mask(padded_vocab: int, org_vocab_size: int, device: torch.device) -> torch.Tensor:
    """True on the lm_head pad columns. Those columns are logit 0 and can win an argmax."""
    mask = torch.zeros(padded_vocab, dtype=torch.bool)
    if org_vocab_size < padded_vocab:
        mask[org_vocab_size:] = True
    return mask.to(device)


def greedy_token_ids(logits: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """``[B, V]`` logits to ``[B]`` ids. ``pad_mask`` is ``[V]`` and broadcasts over rows."""
    return torch.argmax(logits.masked_fill(pad_mask, float("-inf")), dim=-1)


def sampler_output(token_ids: torch.Tensor) -> SamplerOutput:
    """The shape ``Sampler.forward`` returns for an all-greedy batch with no logprobs."""
    return SamplerOutput(
        sampled_token_ids=token_ids.to(torch.int32).reshape(-1, 1),
        logprobs_tensors=None,
    )
