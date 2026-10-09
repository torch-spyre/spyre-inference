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

"""CLIP boundary-LayerNorm workaround for Spyre.

Only ``vision_model.pre_layrnorm``/``post_layernorm`` and
``text_model.final_layer_norm`` are swapped to ``SpyreLayerNorm``. Those three
sit at the model boundary, outside any per-block compiled graph, which is
what triggers the crash ``SpyreLayerNorm`` works around (see
``spyre_inference.custom_ops.layer_norm``). ``CLIPEncoderLayer.layer_norm1``/
``layer_norm2`` are traced inside the per-block ``torch.compile`` region
already and never hit that crashing path, so they're left as plain
``nn.LayerNorm`` -- swapping them too would be unnecessary.

Applied to the already-loaded model instance (weights included), so the
replacement ``SpyreLayerNorm`` here copies the original's already-loaded
weight/bias explicitly, rather than relying on a later ``load_weights()``
pass to populate them.

Also patches ``MMEncoderAttention._forward_sdpa`` with stick-aligned SDPA and
compiles the vision encoder blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm

logger = init_logger(__name__)


def patch_mm_encoder_attention() -> None:
    """Replace ``MMEncoderAttention._forward_sdpa`` with a stick-aligned SDPA impl.

    The replacement calls ``torch.ops.vllm.spyre_clip_attn_mask`` (a custom op
    registered in ``custom_ops/vit_attn.py``) to obtain the additive mask opaquely,
    then calls ``F.pad`` + ``F.scaled_dot_product_attention`` inline.

    ``CLIPAttention.forward`` always calls ``self.attn(q, k, v)`` with no
    ``cu_seqlens``; the replacement hardcodes ``cu_seqlens=None``.
    """
    try:
        from vllm.model_executor.layers.attention.mm_encoder_attention import (
            MMEncoderAttention,
        )
    except ImportError:
        return

    if getattr(MMEncoderAttention._forward_sdpa, "_spyre_patched", False):
        return

    from spyre_inference.custom_ops.vit_attn import _ensure_clip_attn_mask_registered

    _ensure_clip_attn_mask_registered()

    from spyre_inference.multimodal.utils import STICK, align_up

    def _spyre_forward_sdpa(
        self: MMEncoderAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        assert q_len == kv_len, (
            f"_spyre_forward_sdpa assumes self-attention (q_len == kv_len) "
            f"but got q_len={q_len}, kv_len={kv_len}"
        )
        is_reshaped = query.dim() != 4

        query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)
        q = query.transpose(1, 2)  # [B, H, S, D]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        d = q.shape[-1]
        seq_pad = align_up(q_len, STICK)
        d_pad = align_up(d, STICK)
        device = q.device

        attn_mask = torch.ops.vllm.spyre_clip_attn_mask(q_len, bsz, seq_pad, q.dtype, device)  # ty: ignore[invalid-argument-type]

        pad_needed = (seq_pad, d_pad) != (q_len, d)
        if pad_needed:
            pad = (0, d_pad - d, 0, seq_pad - q_len)
            q = F.pad(q, pad)
            k = F.pad(k, pad)
            v = F.pad(v, pad)
        else:
            # Offset operands must be materialized (torch-spyre#3770).
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)

        if pad_needed:
            out = out[:, :, :q_len, :d]

        out = out.transpose(1, 2)

        if is_reshaped:
            out = out.reshape(bsz, q_len, -1)
        return out

    _spyre_forward_sdpa._spyre_patched = True
    MMEncoderAttention._forward_sdpa = _spyre_forward_sdpa  # type: ignore[method-assign]
    logger.info("Spyre: patched MMEncoderAttention._forward_sdpa with stick-aligned SDPA.")


def _compile_vision_encoder_blocks(model: torch.nn.Module) -> None:
    """Compile each ``CLIPEncoderLayer`` with ``fullgraph=True, dynamic=False``.

    Vision tower blocks are excluded from the model-runner's block compilation
    pass, so this function handles them explicitly. Aliased blocks (pipeline
    parallelism) are compiled once. No-op when the tower or encoder is absent.
    """
    from spyre_inference.v1.worker import compile_guard

    vision_model = getattr(model, "vision_model", None)
    if vision_model is None:
        return
    encoder = getattr(vision_model, "encoder", None)
    if encoder is None:
        return
    layers: nn.ModuleList | None = getattr(encoder, "layers", None)
    if not layers:
        return

    seen: set[int] = set()
    for block in layers:
        if id(block) in seen:
            continue
        seen.add(id(block))
        block.compile(backend="inductor", fullgraph=True, dynamic=False)
        compile_guard.watch(block, f"{type(block).__name__} (CLIP vision block)")

    logger.info("Spyre: compiled %d CLIPEncoderLayer block(s) for the vision tower.", len(seen))


def _to_spyre_layer_norm(ln: torch.nn.LayerNorm, device: torch.device) -> torch.nn.LayerNorm:
    new_ln = SpyreLayerNorm(
        list(ln.normalized_shape),
        eps=ln.eps,
        elementwise_affine=ln.elementwise_affine,
        bias=ln.bias is not None,
    ).to(device=device, dtype=ln.weight.dtype if ln.elementwise_affine else torch.float16)
    if ln.elementwise_affine:
        with torch.no_grad():
            new_ln.weight.copy_(ln.weight)
            if ln.bias is not None:
                new_ln.bias.copy_(ln.bias)
    return new_ln


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Swap CLIP's three boundary LayerNorms for ``SpyreLayerNorm``, in place,
    and compile the vision encoder blocks.

    The ``isinstance`` checks are a second line of defense on top of the
    ``model_type == "clip"`` dispatch gate in ``multimodal/__init__.py``: they
    keep this a no-op (rather than an ``AttributeError`` on ``normalized_shape``)
    for any boundary norm that isn't a plain ``nn.LayerNorm``.
    """
    text_model = getattr(model, "text_model", None)
    if text_model is not None:
        ln = getattr(text_model, "final_layer_norm", None)
        if isinstance(ln, torch.nn.LayerNorm):
            text_model.final_layer_norm = _to_spyre_layer_norm(ln, device)

    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None:
        pre_ln = getattr(vision_model, "pre_layrnorm", None)
        if isinstance(pre_ln, torch.nn.LayerNorm):
            vision_model.pre_layrnorm = _to_spyre_layer_norm(pre_ln, device)
        post_ln = getattr(vision_model, "post_layernorm", None)
        if isinstance(post_ln, torch.nn.LayerNorm):
            vision_model.post_layernorm = _to_spyre_layer_norm(post_ln, device)

    logger.info_once(
        "Spyre: CLIP's boundary LayerNorms (pre_layrnorm/post_layernorm/"
        "final_layer_norm) use SpyreLayerNorm; layer_norm1/layer_norm2 inside "
        "encoder blocks are unaffected."
    )

    patch_mm_encoder_attention()
    _compile_vision_encoder_blocks(model)
