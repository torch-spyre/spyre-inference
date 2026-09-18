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

"""Stick-aligned SDPA for vLLM's generic ViT attention path.

``vllm.v1.attention.ops.vit_attn_wrappers.apply_sdpa`` (used by
``MMEncoderAttention``, the default ViT attention for models like CLIP that don't
define a bespoke vision Attention) calls ``F.scaled_dot_product_attention``
directly on whatever sequence length the image produces. torch-spyre compiles
that op internally on every dispatch (regardless of ``--enforce-eager``) and its
BMM-padding pass asserts when the sequence length isn't a multiple of the
64-element fp16 stick (e.g. CLIP ViT-B/32's 50 patches). Routes through the same
``padded_sdpa`` helper Pixtral's vision tower uses (``multimodal/utils.py``),
with an "attend everywhere" mask since this path has no real one of its own.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from vllm.logger import init_logger

from spyre_inference.multimodal.utils import padded_sdpa

logger = init_logger(__name__)


@lru_cache(maxsize=8)
def _full_attend_mask(seq: int) -> torch.Tensor:
    """Stable per-length mask object so ``padded_sdpa``'s per-mask cache (keyed on
    this tensor's identity) hits across layers instead of rebuilding every call."""
    return torch.ones(seq, seq, dtype=torch.bool)


def _padded_apply_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Drop-in replacement for ``vit_attn_wrappers.apply_sdpa``.

    Input/output shape: ``(batch, seq, num_heads, head_size)``.
    """
    seq = q.shape[1]
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))  # -> (batch, heads, seq, head_size)
    out = padded_sdpa(q, k, v, _full_attend_mask(seq), scale=scale, enable_gqa=enable_gqa)
    return out.transpose(1, 2)


def register() -> None:
    import vllm.v1.attention.ops.vit_attn_wrappers as vit_attn_wrappers

    if getattr(vit_attn_wrappers.apply_sdpa, "_spyre_patched", False):
        return

    _padded_apply_sdpa._spyre_patched = True
    vit_attn_wrappers.apply_sdpa = _padded_apply_sdpa  # ty: ignore[invalid-assignment]
    logger.debug_once(
        "Patched vllm.v1.attention.ops.vit_attn_wrappers.apply_sdpa to pad to "
        "the 64-element stick before calling SDPA."
    )
