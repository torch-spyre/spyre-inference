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

"""Spyre OOT replacement for MMEncoderAttention."""

import einops
import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention

from .utils import convert

logger = init_logger(__name__)


def _apply_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Input shape: (batch_size x seq_len x num_heads x head_size)."""
    q, k, v = (einops.rearrange(x, "b s h d -> b h s d") for x in [q, k, v])
    output = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
    )
    output = einops.rearrange(output, "b h s d -> b s h d")
    return output


def _sdpa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    if cu_seqlens is None:
        return _apply_sdpa(q, k, v, scale=scale, enable_gqa=enable_gqa)

    outputs = []
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    q_chunks = torch.split(q, lens, dim=1)
    k_chunks = torch.split(k, lens, dim=1)
    v_chunks = torch.split(v, lens, dim=1)
    for q_i, k_i, v_i in zip(q_chunks, k_chunks, v_chunks):
        output_i = _apply_sdpa(q_i, k_i, v_i, scale=scale, enable_gqa=enable_gqa)
        outputs.append(output_i)
    return torch.cat(outputs, dim=1)


@MMEncoderAttention.register_oot(name="MMEncoderAttention")
class SpyreMMEncoderAttention(MMEncoderAttention):
    """Out-of-tree (OOT) MMEncoderAttention implementation for IBM's Spyre."""

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Run the entire attention on CPU: F.scaled_dot_product_attention called
        # eagerly on Spyre tensors hits torch-spyre's compiled SDPA decomposition,
        # whose permute/reshape produces stick layouts that optimize_restickify
        # cannot reconcile ("no mechanism to resolve stick incompatibility").
        target_device = query.device
        query = convert(query, device="cpu")
        key = convert(key, device="cpu")
        value = convert(value, device="cpu")
        if cu_seqlens is not None:
            cu_seqlens = convert(cu_seqlens, device="cpu")

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)

        output = _sdpa_forward(
            q=query,
            k=key,
            v=value,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
            enable_gqa=self.num_heads > self.num_kv_heads,
        )

        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return convert(output, device=target_device)
