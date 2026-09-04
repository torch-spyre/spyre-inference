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

import os
from functools import cache

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert

logger = init_logger(__name__)

# Matmul reduction dims must land on the Spyre stick: 64 fp16 elements.
SEQ_ALIGNMENT = 64


def _align_up(n: int, align: int = SEQ_ALIGNMENT) -> int:
    return (n + align - 1) // align * align


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


# Attribute under which a source mask carries its padded counterpart `(key, padded)`.
_MASK_ATTR = "_spyre_padded_mask"


def _padded_attn_mask(
    mask: torch.Tensor,
    b: int,
    seq: int,
    seq_pad: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive `[b, 1, seq_pad, seq_pad]` mask on `device`.

    The tensor is O(L²) and the tower hands the same mask to every layer, so it is
    cached on the mask itself: one upload per image, released with its source.
    """
    key = (b, seq, seq_pad, dtype, str(device))
    cached = getattr(mask, _MASK_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]

    # Assembled on CPU: strided slice-assign is not stick-safe on Spyre.
    neg_inf = torch.finfo(dtype).min
    m = torch.zeros(b, 1, seq_pad, seq_pad, dtype=dtype)
    m[:, :, :, seq:] = neg_inf  # padded keys never attended
    mc = convert(mask, "cpu")
    if mc.dtype == torch.bool:
        m[:, :, :seq, :seq] = torch.zeros(seq, seq, dtype=dtype).masked_fill(
            ~mc.reshape(seq, seq), neg_inf
        )
    else:
        m[:, :, :seq, :seq] = mc.to(dtype).reshape(seq, seq)

    m = convert(m, device)
    setattr(mask, _MASK_ATTR, (key, m))
    return m


def padded_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """SDPA over `[B, H, L, D]` with L and D padded to the 64 stick, then cropped.

    Padded keys are masked to `-inf` and padded queries cropped off. `scale` comes
    from the unpadded head dim, so the padding cannot change it.
    """
    b, _, seq, d = q.shape
    scale = d**-0.5
    seq_pad = _align_up(seq)
    d_pad = _align_up(d)
    device = q.device
    padded = (seq_pad, d_pad) != (seq, d)

    if padded:
        # F.pad's tuple runs from the last dim backwards: (D left, D right, L left, L right).
        pad = (0, d_pad - d, 0, seq_pad - seq)
        q = F.pad(q, pad)
        k = F.pad(k, pad)
        v = F.pad(v, pad)

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=_padded_attn_mask(mask, b, seq, seq_pad, q.dtype, device),
        scale=scale,
    )

    if padded:
        # Offset-0 prefix slice, so torch-spyre#3770 cannot bite. Left as a view: the
        # caller's transpose+reshape materializes it anyway.
        out = out[:, :, :seq, :d]
    return out


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


def offload_projector_norm(model: torch.nn.Module, device: torch.device) -> None:
    """Run Pixtral's `pre_mm_projector_norm` on CPU:
    ("Unsupported coordinate expression 195*c0/64 + c1/1024")."""
    for module_name, module in model.named_modules():
        if module_name.rsplit(".", 1)[-1] != "pre_mm_projector_norm":
            continue
        module.to("cpu")

        def _to_cpu(mod, args):
            return tuple(a.to("cpu") if isinstance(a, torch.Tensor) else a for a in args)

        def _to_dev(mod, args, output, _dev=device):
            if isinstance(output, tuple):
                return tuple(
                    convert(o, device=_dev) if isinstance(o, torch.Tensor) else o for o in output
                )
            return convert(output, device=_dev)

        module.register_forward_pre_hook(_to_cpu)
        module.register_forward_hook(_to_dev)
        logger.info("Spyre: %s offloaded to CPU (D2H in / H2D out)", module_name)


def disable_norm_compile(model: torch.nn.Module) -> None:
    """Keep the vision tower's norms eager; the patch conv stays compiled.

    vLLM never compiles the Pixtral encoder, so `compile_when_outermost` compiles each
    norm itself, and `ln_pre` normalizes a view whose index math carries the grid size
    — a coordinate expression torch-spyre rejects (#1353). `SPYRE_COMPILE_MM_ENCODER=1`
    compiles them anyway.
    """
    if os.environ.get("SPYRE_COMPILE_MM_ENCODER", "0") == "1":
        return

    from vllm.model_executor.layers.layernorm import RMSNorm

    from spyre_inference.custom_ops.lazy_compile import CompileOutermost

    tower = getattr(model, "vision_encoder", None)
    if tower is None:
        return

    n = 0
    for module in tower.modules():
        if (
            isinstance(module, CompileOutermost)
            and isinstance(module, RMSNorm)
            and module.spyre_compile_enabled
        ):
            module.spyre_compile_enabled = False
            n += 1
    if n:
        logger.info("Spyre: %d vision-tower norm(s) run eager (not compiled).", n)


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
    """
    # Must precede the attention patch, which resolves apply_rotary_emb_vit by name.
    patch_vision_rope_vit()
    patch_vision_attention()
    patch_block_attention_mask()
    offload_projector_norm(model, device)
    patch_patch_merger()
    disable_norm_compile(model)
