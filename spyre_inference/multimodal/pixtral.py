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

"""Pixtral/Ministral vision-tower workarounds for Spyre.

The tower is plain `nn.Module` code outside vLLM's layer registries, so nothing here
can go through `CustomOp.register_oot`; every fix is a guarded, idempotent
monkeypatch and `apply()` is the only entry point.
"""

from __future__ import annotations

from functools import cache

import torch
import torch.nn as nn
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert
from spyre_inference.multimodal.utils import _padded_attn_mask, align_up, padded_sdpa

logger = init_logger(__name__)


@cache
def rope_perm_matrix(kind: str, head_dim: int, device: torch.device) -> torch.Tensor:
    """Constant `[head_dim, head_dim]` permutation `M` so `x @ M` is a rope shuffle.

    Rotating by a full-width matmul avoids slicing the head into `d/2`-wide halves:
    at head_dim=64 that half is 32, which torch-spyre cannot lay out ("Unexpected
    stick expression ... Mod(var, 32)"). kind="pair" swaps each `(2k, 2k+1)` pair.
    """
    if kind != "pair":
        raise ValueError(f"unknown rope permutation kind {kind!r}")
    m = torch.zeros(head_dim, head_dim, dtype=torch.float16)
    even = torch.arange(0, head_dim, 2)
    m[even, even + 1] = 1.0
    m[even + 1, even] = 1.0
    return convert(m, device=device, dtype=torch.float16)


def rope_rotate_matmul(x, cos, sin, m: torch.Tensor):
    """`x*cos + (x @ m)*sin` — the rope rotation as a stick-aligned matmul."""
    return x * cos + torch.matmul(x, m) * sin


_ROPE_PERM_BUFFER = "spyre_rope_perm"


def install_rope_perm(model: nn.Module, device: torch.device) -> None:
    """Attach the rope permutation to each vision `Attention` as a device buffer.

    Built inside a compiled block, its CPU index writes land in the graph and fail with
    "does not have FixedTiledLayout". Non-persistent, so weight loading never sees it.
    """
    try:
        from vllm.model_executor.models import pixtral
    except ImportError:
        return

    for module in model.modules():
        if isinstance(module, pixtral.Attention) and not hasattr(module, _ROPE_PERM_BUFFER):
            perm = rope_perm_matrix("pair", module.head_dim, device)
            module.register_buffer(_ROPE_PERM_BUFFER, perm, persistent=False)


def patch_vision_attention() -> None:
    """Replace Pixtral's vision `Attention.forward` with the padded on-card SDPA.

    At a patch count coprime with the 64 stick, stock SDPA either fails to restickify
    a batch-matmul operand or returns silently wrong values, so the padding is a
    correctness requirement. The body is upstream's non-xformers branch with only the
    SDPA call swapped; `patch_vision_rope_vit` must run first because
    `apply_rotary_emb_vit` is resolved by name at call time.
    """
    try:
        from vllm.model_executor.models import pixtral
    except ImportError:
        return

    attn_cls = getattr(pixtral, "Attention", None)
    if attn_cls is None or getattr(attn_cls.forward, "_spyre_patched", False):
        return

    def _forward(self, x, mask, freqs_cis):
        batch, patches, _ = x.shape
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, patches, self.n_heads, self.head_dim)
        k = k.reshape(batch, patches, self.n_heads, self.head_dim)
        v = v.reshape(batch, patches, self.n_heads, self.head_dim)
        # Resolves to the `patch_vision_rope_vit` replacement, which takes `perm`.
        perm = getattr(self, _ROPE_PERM_BUFFER, None)
        q, k = pixtral.apply_rotary_emb_vit(q, k, freqs_cis=freqs_cis, perm=perm)  # ty: ignore[unknown-argument]
        # [B, H, L, D] for SDPA.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # A Spyre mask was already padded by `patch_transformer_mask`.
        out = padded_sdpa(q, k, v, mask, mask_is_padded=mask.device.type == "spyre")
        out = out.transpose(1, 2).reshape(batch, patches, self.n_heads * self.head_dim)
        out, _ = self.o_proj(out)
        return out

    _forward._spyre_patched = True
    attn_cls.forward = _forward
    logger.info(
        "Spyre: patched Pixtral vision Attention to stick-aligned padded "
        "on-card SDPA (pad L/D to 64, mask, crop)."
    )


def patch_vision_rope_vit() -> None:
    """Run the Pixtral `VisionTransformer` 2D-RoPE on-card.

    Upstream's rope is complex and gathers per-token freqs by advanced indexing;
    Spyre has neither `complex64` nor `aten::index.Tensor_out`. So `freqs_cis`
    becomes a real packed cos/sin table gathered with `index_select`, and
    `apply_rotary_emb_vit` becomes `x·cos + (x @ P)·sin` over the full stick width.
    """
    try:
        from vllm.model_executor.models import pixtral
    except ImportError:
        return

    orig = getattr(pixtral, "apply_rotary_emb_vit", None)
    vt = getattr(pixtral, "VisionTransformer", None)
    if orig is None or vt is None or getattr(orig, "_spyre_patched", False):
        return

    class _OnCardFreqsTable:
        """Real freqs table on Spyre, gathered per-token by flat `index_select`."""

        def __init__(self, table: torch.Tensor, width: int):
            self._table = table  # (H*W, 2, head_dim) on Spyre
            self._width = width

        def __getitem__(self, idx):
            # `positions[:, 1]` has storage_offset=1, which the device needs
            # stick-aligned, so fold both columns into a flat index on CPU first.
            row, col = idx
            flat = (convert(row, "cpu") * self._width + convert(col, "cpu")).to(torch.int64)
            flat = convert(flat, device=self._table.device, dtype=torch.int64)
            return self._table.index_select(0, flat)  # (seq, 2, head_dim)

    def _freqs_cis_ondev(self):
        # Packed real table (H*W, 2, head_dim): [..., 0, :]=cos, [..., 1, :]=sin.
        if self._freqs_cis is None:
            fc = pixtral.precompute_freqs_cis_2d(
                dim=self.args.hidden_size // self.args.num_attention_heads,
                height=self.max_patches_per_side,
                width=self.max_patches_per_side,
                theta=self.args.rope_theta,
            )  # (H, W, head_dim//2) complex64 on CPU
            cos = fc.real
            sin = fc.imag
            cos_full = cos.repeat_interleave(2, dim=-1)
            sin_signed = torch.stack([-sin, sin], dim=-1).reshape(*sin.shape[:-1], -1)
            packed = torch.stack([cos_full, sin_signed], dim=-2)  # (H, W, 2, head_dim)
            self._freqs_cis = convert(
                packed.reshape(-1, packed.shape[-2], packed.shape[-1]), dtype=torch.float16
            )  # (H*W, 2, head_dim) on CPU
        if self._freqs_cis.device != self.device:
            self._freqs_cis = convert(self._freqs_cis, device=self.device, dtype=torch.float16)
        return _OnCardFreqsTable(self._freqs_cis, self.max_patches_per_side)

    def _apply_rotary_emb_vit(xq, xk, freqs_cis, perm=None):
        # xq, xk: [batch, patches, n_heads, head_dim]; freqs_cis: [patches, 2, head_dim].
        p = perm if perm is not None else rope_perm_matrix("pair", xq.shape[-1], xq.device)
        cos = freqs_cis[:, 0, :][None, :, None, :]  # [1, patches, 1, head_dim]
        sin = freqs_cis[:, 1, :][None, :, None, :]

        return (
            rope_rotate_matmul(xq, cos, sin, p).type_as(xq),
            rope_rotate_matmul(xk, cos, sin, p).type_as(xk),
        )

    _apply_rotary_emb_vit._spyre_patched = True
    pixtral.apply_rotary_emb_vit = _apply_rotary_emb_vit  # ty: ignore[invalid-assignment]
    vt.freqs_cis = property(_freqs_cis_ondev)
    logger.info(
        "Spyre: patched Pixtral VisionTransformer 2D-RoPE to on-card real "
        "rotation (index_select freqs gather + pair-swap matmul)."
    )


def patch_block_attention_mask() -> None:
    """Build Pixtral's block-diagonal vision mask on CPU.

    Upstream zeroes one `[start:end, start:end]` sub-block per image on
    `patch_embeds.device`; with N images those are strided sub-block writes, which are
    not stick-safe. `_padded_attn_mask` pulls the mask to CPU anyway.
    """
    try:
        from transformers.models.pixtral import modeling_pixtral
    except ImportError:
        return

    orig = getattr(modeling_pixtral, "generate_block_attention_mask", None)
    if orig is None or getattr(orig, "_spyre_patched", False):
        return

    def _cpu_mask(patch_embeds_list, tensor):
        if tensor.device.type != "spyre":
            return orig(patch_embeds_list, tensor)
        # Only `dtype` and the two leading dims are read off `tensor`, so a CPU stand-in
        # gives an identical mask without a D2H of patch_embeds.
        stand_in = torch.empty((tensor.shape[0], tensor.shape[1]), dtype=tensor.dtype)
        return orig(patch_embeds_list, stand_in)

    _cpu_mask._spyre_patched = True
    # vLLM imports this symbol inside the function body, so patching the module
    # attribute is picked up at call time.
    modeling_pixtral.generate_block_attention_mask = _cpu_mask  # ty: ignore[invalid-assignment]
    logger.info("Spyre: Pixtral block attention mask built on CPU (N-image sub-block writes).")


def patch_transformer_mask() -> None:
    """Pad and upload the vision attention mask once per image, ahead of the blocks.

    `_padded_attn_mask`'s attribute cache does not survive a compiled block, so inside
    one every layer would re-upload the O(L²) mask.
    """
    try:
        from vllm.model_executor.models import pixtral
    except ImportError:
        return

    tr_cls = getattr(pixtral, "Transformer", None)
    if tr_cls is None or getattr(tr_cls.forward, "_spyre_patched", False):
        return

    def _forward(self, x, mask, freqs_cis):
        if x.device.type == "spyre":
            batch, seq, _ = x.shape
            mask = _padded_attn_mask(mask, batch, seq, align_up(seq), x.dtype, x.device)
        for layer in self.layers:
            x = layer(x, mask=mask, freqs_cis=freqs_cis)
        return x

    _forward._spyre_patched = True
    tr_cls.forward = _forward
    logger.info("Spyre: Pixtral vision mask padded and uploaded once per image.")


def patch_patch_merger() -> None:
    """Run Pixtral `PatchMerger.permute` (spatial s×s regroup) on CPU.

    It uses `F.unfold` (`aten::im2col`), unsupported on Spyre, and a reshape/permute
    rewrite does not lower either — the regroup is a geometry-dependent multi-counter
    stick scatter. The `merging_layer` GEMM stays on-card.
    """
    try:
        from vllm.model_executor.models import pixtral
    except ImportError:
        return

    pm_cls = getattr(pixtral, "PatchMerger", None)
    if pm_cls is None or getattr(pm_cls.forward, "_spyre_patched", False):
        return

    def _forward(self, x, image_sizes):
        dev = x.device
        x_perm = self.permute(convert(x, "cpu"), image_sizes)  # unfold on CPU
        return self.merging_layer(convert(x_perm, device=dev))  # GEMM on-card

    _forward._spyre_patched = True
    pm_cls.forward = _forward
    logger.info(
        "Spyre: patched Pixtral PatchMerger permute to CPU (merging_layer GEMM stays on-card)."
    )


class _DefaultLayoutNorm(nn.Module):
    """Materialize a default-layout input before Pixtral's pre-transformer norm."""

    def __init__(self, norm: nn.Module) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "spyre":
            device = x.device
            x = convert(convert(x, device="cpu").contiguous(), device=device)
        return self.norm(x)


def patch_pre_transformer_norm(model: nn.Module) -> None:
    """Reset the patch-conv layout before RMSNorm's fp32 accumulation.

    Flattening the channel-tiled convolution output preserves a device layout whose
    patch-grid dimension can contain an odd number of fp32 sticks. That layout cannot
    be rescaled for RMSNorm's fp32-to-fp16 conversion. A CPU round trip after flattening
    materializes the logical ``[batch, patches, hidden]`` tensor in its default layout.
    """
    tower = getattr(model, "vision_encoder", None) or getattr(model, "vision_tower", None)
    if tower is None or isinstance(tower.ln_pre, _DefaultLayoutNorm):
        return
    tower.ln_pre = _DefaultLayoutNorm(tower.ln_pre)
    logger.info("Spyre: Pixtral pre-transformer norm input uses the default device layout.")


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Install every Pixtral vision-tower workaround, in dependency order.

    The patch-embedding conv is absent on purpose: `SpyreConv2d` in
    `custom_ops/conv.py` handles it through OOT dispatch.

    Must run after the weights reach `device` and before blocks compile
    (`install_rope_perm`).
    """
    try:
        from vllm.model_executor.models import pixtral
    except ImportError:
        return

    # True whenever xformers merely imports: upstream only disables it on CUDA B200.
    if getattr(pixtral, "USE_XFORMERS_OPS", False):
        raise RuntimeError(
            "xformers is installed; Pixtral on Spyre needs the non-xformers mask path. "
            "Uninstall xformers in this environment."
        )

    # Must precede the attention patch, which resolves apply_rotary_emb_vit by name.
    patch_vision_rope_vit()
    patch_vision_attention()
    install_rope_perm(model, device)
    patch_transformer_mask()
    patch_block_attention_mask()
    patch_patch_merger()
    patch_pre_transformer_norm(model)
