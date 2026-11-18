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

"""SigLIP pieces for Spyre, mirroring ``vllm.model_executor.models.siglip``.

``pad_siglip_heads`` widens the heads of an upstream SigLIP tower to the stick, so models other
than Granite 4 Vision that embed one can reuse it.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from spyre_inference.custom_ops.head_pad import pad_input_end, pad_output_end
from spyre_inference.custom_ops.utils import convert
from spyre_inference.multimodal.utils import STICK, align_up


def pad_siglip_heads(tower: nn.Module) -> None:
    """Pad every SigLIP attention's Q/K/V/O projections from head_dim=72 to 128.

    ``tower`` is upstream's ``SiglipVisionTransformer`` (a ``SiglipVisionModel``'s
    ``vision_model``): anything with ``encoder.layers[*].self_attn`` built from vLLM's linears
    and ``MMEncoderAttention``.

    SigLIP so400m has head_dim=72 (not stick-aligned). Any on-device op that materialises a
    D=72 tensor triggers a restickify whose store index contains a fractional coefficient
    (9*c2/8), which torch-spyre's inductor cannot lower (torch-spyre#1353). Each head block of
    qkv_proj and out_proj is zero-extended to the stick-aligned width (mirroring hf-adapters'
    ``pad_attention_heads_linear``), and ``attn.head_size`` is set to match so
    ``SpyreMMEncoderAttention`` handles the attention natively on-device. It keeps the original
    scale, which ``MMEncoderAttention`` fixed at construction.

    The checkpoint is 72 wide, so this runs after loading. Each weight is pulled to the host
    before the slice-assignments (a strided slice-assign is silently wrong on Spyre) and moved
    back. A no-op for a head_dim that is already stick-aligned, and for a tower padded before.
    """
    for layer in tower.encoder.layers:  # ty: ignore[not-iterable]
        attn = layer.self_attn
        if getattr(attn, "_spyre_head_dim_padded", False):
            continue
        orig = attn.head_dim
        pad = align_up(orig, STICK)
        if pad == orig:
            continue
        num_heads = attn.num_heads_per_partition

        # The weights are stored as Wᵀ (SpyreTransposedWeightMethod): qkv is [hidden, 3*H*D], so
        # its heads are on dim 1 (3*H of them, Q, K and V back to back); out is [H*D, hidden],
        # so its heads are on dim 0. Padded on the host, in the same layout.
        qkv, out = attn.qkv_proj, attn.out_proj
        qkv_w = cast(torch.Tensor, qkv.weight)
        dev = qkv_w.device
        qkv_pad = (3 * num_heads, orig, pad)
        qkv.weight = nn.Parameter(
            convert(pad_input_end(convert(qkv_w.detach(), device="cpu"), *qkv_pad), device=dev),
            requires_grad=False,
        )
        if qkv.bias is not None:
            qkv_b = cast(torch.Tensor, qkv.bias)
            qkv.bias = nn.Parameter(
                convert(
                    pad_output_end(convert(qkv_b.detach(), device="cpu"), *qkv_pad), device=dev
                ),
                requires_grad=False,
            )
        out_w = cast(torch.Tensor, out.weight)
        out.weight = nn.Parameter(
            convert(
                pad_output_end(convert(out_w.detach(), device="cpu"), num_heads, orig, pad),
                device=dev,
            ),
            requires_grad=False,
        )
        # out_proj bias is [hidden], the output dim, unchanged.
        attn.attn.head_size = pad
        attn._spyre_head_dim_padded = True
