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
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert
from spyre_inference.multimodal.utils import padded_sdpa

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
        q, k = pixtral.apply_rotary_emb_vit(q, k, freqs_cis=freqs_cis)
        # [B, H, L, D] for SDPA.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = padded_sdpa(q, k, v, mask)
        out = out.transpose(1, 2).reshape(batch, patches, self.n_heads * self.head_dim)
        out, _ = self.o_proj(out)
        return out

    _forward._spyre_patched = True
    attn_cls.forward = _forward  # ty: ignore[invalid-assignment]
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
            flat = (row.to("cpu") * self._width + col.to("cpu")).to(torch.int64)
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
            self._freqs_cis = packed.reshape(-1, packed.shape[-2], packed.shape[-1]).to(
                torch.float16
            )  # (H*W, 2, head_dim) on CPU
        if self._freqs_cis.device != self.device:
            self._freqs_cis = convert(self._freqs_cis, device=self.device, dtype=torch.float16)
        return _OnCardFreqsTable(self._freqs_cis, self.max_patches_per_side)

    def _apply_rotary_emb_vit(xq, xk, freqs_cis):
        # xq, xk: [batch, patches, n_heads, head_dim]; freqs_cis: [patches, 2, head_dim].
        p = rope_perm_matrix("pair", xq.shape[-1], xq.device)
        cos = freqs_cis[:, 0, :][None, :, None, :]  # [1, patches, 1, head_dim]
        sin = freqs_cis[:, 1, :][None, :, None, :]

        return (
            rope_rotate_matmul(xq, cos, sin, p).type_as(xq),
            rope_rotate_matmul(xk, cos, sin, p).type_as(xk),
        )

    _apply_rotary_emb_vit._spyre_patched = True
    pixtral.apply_rotary_emb_vit = _apply_rotary_emb_vit  # ty: ignore[invalid-assignment]
    vt.freqs_cis = property(_freqs_cis_ondev)  # ty: ignore[invalid-assignment]
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
        x_perm = self.permute(x.to("cpu"), image_sizes)  # unfold on CPU
        return self.merging_layer(convert(x_perm, device=dev))  # GEMM on-card

    _forward._spyre_patched = True
    pm_cls.forward = _forward  # ty: ignore[invalid-assignment]
    logger.info(
        "Spyre: patched Pixtral PatchMerger permute to CPU (merging_layer GEMM stays on-card)."
    )


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Install every Pixtral vision-tower workaround, in dependency order.

    The patch-embedding conv is absent on purpose: `SpyreConv2d` in
    `custom_ops/conv.py` handles it through OOT dispatch.

    `model` and `device` are unused: every remaining patch rewrites upstream module
    attributes rather than a loaded instance. They stay for the `apply(model, device)`
    contract the sibling architecture modules share.
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
    patch_block_attention_mask()
    patch_patch_merger()
