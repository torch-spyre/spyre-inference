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

"""BLIP-2 / Q-Former workarounds for Spyre."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)


def patch_blip2_qformer_attention() -> None:
    """Replace Blip2QFormerMultiHeadAttention.forward to run fully on Spyre.

    The original forward uses transpose_for_scores (permute(0,2,1,3)) without a
    subsequent .contiguous(), producing non-contiguous stick layouts that Spyre's
    restickify and bmm_padding passes cannot reconcile.  Following the pattern
    established in hf-adapters' ``make_encoder_block``, the replacement adds an
    explicit .contiguous() call after each head-shape transpose, forcing Spyre to
    materialise a fresh, canonically-tiled buffer before the SDPA decomposition
    runs.
    """
    try:
        from vllm.model_executor.models.blip2 import Blip2QFormerMultiHeadAttention
    except ImportError:
        return

    if getattr(Blip2QFormerMultiHeadAttention.forward, "_spyre_patched", False):
        return

    def _blip2_attn_forward_spyre(self, hidden_states, encoder_hidden_states=None):
        bsz, q_len, _ = hidden_states.shape

        # .contiguous() after the view+transpose is required on Spyre: the fused
        # lowering of transpose -> SDPA reads non-contiguous (transposed) tensors
        # with the wrong stick layout and returns garbage.  See hf-adapters'
        # make_encoder_block for the same pattern.
        q = (
            self.query(hidden_states)
            .view(bsz, q_len, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
            .contiguous()
        )

        kv_src = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        kv_len = kv_src.shape[1]
        k = (
            self.key(kv_src)
            .view(bsz, kv_len, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.value(kv_src)
            .view(bsz, kv_len, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
            .contiguous()
        )

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )

        # [B, H, Lq, D] -> [B, Lq, H*D]
        return attn_out.transpose(1, 2).reshape(bsz, q_len, self.all_head_size)

    _blip2_attn_forward_spyre._spyre_patched = True  # type: ignore[attr-defined]
    Blip2QFormerMultiHeadAttention.forward = _blip2_attn_forward_spyre  # type: ignore[method-assign]
    logger.info(
        "Spyre: patched Blip2QFormerMultiHeadAttention.forward to use "
        "F.scaled_dot_product_attention with .contiguous() (runs fully on Spyre)."
    )


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Apply BLIP-2 Q-Former workarounds."""
    patch_blip2_qformer_attention()
