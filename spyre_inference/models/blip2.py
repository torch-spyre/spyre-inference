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

"""BLIP-2 Q-Former pieces for Spyre, mirroring ``vllm.model_executor.models.blip2``.

``SpyreBlip2QFormerMultiHeadAttention`` is a Spyre-safe Q-Former attention, and
``use_spyre_qformer_attention`` swaps it onto any model that embeds upstream's Q-Former, so
models other than Granite 4 Vision can reuse it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.models import blip2


class SpyreBlip2QFormerMultiHeadAttention(blip2.Blip2QFormerMultiHeadAttention):
    """Q-Former attention that runs fully on Spyre.

    Upstream's ``transpose_for_scores`` is a ``permute(0, 2, 1, 3)`` with no ``.contiguous()``,
    which leaves non-contiguous stick layouts that Spyre's restickify and bmm_padding passes
    cannot reconcile: the fused transpose -> SDPA lowering reads them with the wrong stick
    layout and returns garbage. As in hf-adapters' ``make_encoder_block``, each head-shape
    transpose is followed by ``.contiguous()``, so Spyre materialises a canonically tiled
    buffer before the SDPA decomposition.

    It adds no state, so ``use_spyre_qformer_attention`` can swap it onto a built module.
    """

    def forward(  # ty: ignore[invalid-method-override]
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape

        def project(proj: nn.Module, x: torch.Tensor) -> torch.Tensor:
            return (
                proj(x)
                .view(bsz, x.shape[1], self.num_attention_heads, self.attention_head_size)
                .transpose(1, 2)
                .contiguous()
            )

        q = project(self.query, hidden_states)
        kv_src = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = project(self.key, kv_src)
        v = project(self.value, kv_src)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False, scale=self.scaling
        )
        # [B, H, Lq, D] -> [B, Lq, H*D]
        return out.transpose(1, 2).reshape(bsz, q_len, self.all_head_size)


def use_spyre_qformer_attention(root: nn.Module) -> int:
    """Swap ``SpyreBlip2QFormerMultiHeadAttention`` onto every upstream Q-Former attention
    under ``root``, in place and before weights load. Only this model's own modules change,
    not the upstream class other models share. Returns how many it swapped."""
    swapped = 0
    for module in root.modules():
        if type(module) is blip2.Blip2QFormerMultiHeadAttention:
            module.__class__ = SpyreBlip2QFormerMultiHeadAttention
            swapped += 1
    return swapped
