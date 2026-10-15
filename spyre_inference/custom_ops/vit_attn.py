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

"""Stick-aligned SDPA for vLLM's ``MMEncoderAttention``.

``MMEncoderAttention`` is the default ViT attention for towers that don't define a
bespoke one (CLIP, BLIP). Its SDPA path calls ``F.scaled_dot_product_attention``
on whatever sequence length the image produces, behind the opaque
``torch.ops.vllm.torch_sdpa_wrapper``. torch-spyre's BMM-padding pass asserts when
that length isn't a multiple of the 64-element fp16 stick (e.g. CLIP ViT-B/32's 50
patches), so this routes through the same ``padded_sdpa`` helper Pixtral's tower
uses (``multimodal/utils.py``). The path has no real mask of its own, so only the
padded keys are masked.

``padded_sdpa`` is called directly rather than behind the opaque wrapper, so a
tower whose blocks are marked compilable traces attention into the block graph.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention

from spyre_inference.multimodal.utils import padded_sdpa


@MMEncoderAttention.register_oot(name="MMEncoderAttention")
class SpyreMMEncoderAttention(MMEncoderAttention):
    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Input: ``[B, S, H*D]`` or ``[B, S, H, D]``; output has the same rank."""
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        # padded_sdpa pads both sides to the query's stick-aligned length.
        if cu_seqlens is not None or q_len != kv_len:
            raise NotImplementedError(
                "Spyre MMEncoderAttention supports only unpacked self-attention "
                f"(got cu_seqlens={cu_seqlens is not None}, q_len={q_len}, kv_len={kv_len})."
            )
        is_reshaped = query.dim() != 4
        q, k, v = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)
        out = padded_sdpa(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            None,
            scale=self.scale,
            enable_gqa=self.num_heads > self.num_kv_heads,
        ).transpose(1, 2)
        return out.reshape(bsz, q_len, -1) if is_reshaped else out
