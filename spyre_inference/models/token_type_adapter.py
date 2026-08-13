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

"""Side-buffer adapter for vLLM's BERT ``token_type_ids`` bit-pack transport.

vLLM packs segment ids into the high bits of ``input_ids`` (see
``TOKEN_TYPE_SHIFT`` in ``vllm.model_executor.models.bert``) so torch.compile
sees one persistent tensor. That packing is a vLLM transport hack, not a BERT
requirement, and Spyre does not lower the integer bitwise unpack
(torch-spyre#3509).

Instead of enabling those ops or doing CPU pack/unpack:

* ``encode`` — leave ``input_ids`` alone; copy the real ``token_type_ids`` into
  a persistent buffer sized like ``input_ids`` (zeros past
  ``token_type_ids.shape[0]`` — right-pad slots are segment 0).
* ``decode`` — return that buffer; do not mutate ``input_ids``.
"""

from __future__ import annotations

import torch

_buffer: torch.Tensor | None = None


def encode_token_type_ids(input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> None:
    """Store ``token_type_ids`` in the side buffer; do not pack into ``input_ids``."""
    global _buffer
    n = token_type_ids.shape[0]
    dtype = token_type_ids.dtype
    device = input_ids.device
    if (
        _buffer is None
        or _buffer.shape != input_ids.shape
        or _buffer.device != device
        or _buffer.dtype != dtype
    ):
        _buffer = torch.zeros(input_ids.shape, dtype=dtype, device=device)
    else:
        _buffer.zero_()
    src = token_type_ids if token_type_ids.device == device else token_type_ids.to(device)
    _buffer[:n].copy_(src[:n])


def decode_token_type_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """Return the side buffer; leave ``input_ids`` unchanged."""
    if _buffer is None or _buffer.shape != input_ids.shape or _buffer.device != input_ids.device:
        # encode was not called (single-segment path) → all segment 0
        return torch.zeros_like(input_ids)
    return _buffer


def install_on(module) -> None:
    """Rebind ``_encode_token_type_ids`` / ``_decode_token_type_ids`` on ``module``.

    Must be applied to both ``vllm...bert`` and ``vllm...roberta``: RoBERTa
    imports those names into its own module globals, so patching ``bert`` alone
    does not update RoBERTa call sites.
    """
    if not hasattr(module, "_encode_token_type_ids") or not hasattr(
        module, "_decode_token_type_ids"
    ):
        raise RuntimeError(
            f"{module.__name__} token_type helpers not found; Spyre adapter "
            "needs updating for this vLLM version."
        )
    module._encode_token_type_ids = encode_token_type_ids
    module._decode_token_type_ids = decode_token_type_ids
