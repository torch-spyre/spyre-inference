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

"""Helpers shared by the vision-tower workarounds in this package."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from spyre_inference.custom_ops.utils import convert

# Spyre stick width in 2-byte elements (128-byte stick). Matmul reduction dims and
# the sequence axis must land on it.
STICK = 64

# Set by a multimodal patch on the blocks whose body it made traceable; the runner
# per-block compiles a tower's block list only when every block carries it.
_BLOCK_COMPILABLE_ATTR = "_spyre_block_compilable"


def align_up(n: int, align: int = STICK) -> int:
    return (n + align - 1) // align * align


def linear_weight(layer: nn.Module) -> torch.Tensor:
    """``[out, in]``, undoing the Spyre linear's post-load transpose and row padding."""
    weight = cast(torch.Tensor, layer.weight).detach()
    padding = getattr(layer, "spyre_row_padding", None)
    if padding is None:
        return weight
    weight = weight.t()
    return weight[: weight.shape[0] - padding]


@cache
def rope_perm_matrix(
    kind: str, head_dim: int, device: torch.device, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Constant `[head_dim, head_dim]` permutation `M` so `x @ M` is a rope shuffle.

    Rotating by a full-width matmul avoids slicing the head: at head_dim=64 a `d/2` slice
    is 32 wide, which torch-spyre cannot lay out ("Unexpected stick expression ...
    Mod(var, 32)"), and `cat`ing two slices of a matmul result is silently wrong on
    device (uncorrelated data, no warning). kind="pair" swaps each `(2k, 2k+1)` pair;
    kind="half" swaps the two halves, as `rotate_half` does.
    """
    m = torch.zeros(head_dim, head_dim, dtype=dtype)
    if kind == "pair":
        even = torch.arange(0, head_dim, 2)
        m[even, even + 1] = 1.0
        m[even + 1, even] = 1.0
    elif kind == "half":
        half = head_dim // 2
        rows = torch.cat([torch.arange(half, head_dim), torch.arange(0, half)])
        m[rows, torch.arange(head_dim)] = 1.0
    else:
        raise ValueError(f"unknown rope permutation kind {kind!r}")
    return convert(m, device=device, dtype=dtype)


def rope_rotate_matmul(x, cos, sin, m: torch.Tensor):
    """`x*cos + (x @ m)*sin` — the rope rotation as a stick-aligned matmul."""
    return x * cos + torch.matmul(x, m) * sin


def mark_blocks_compilable(blocks: Iterable[nn.Module]) -> None:
    """Opt a tower's blocks into the runner's per-block compile.

    Only for a block body that builds no constant on the host: a CPU-resident buffer
    has no device layout, so the graph would not lower. Mark every block of the list,
    including any past the slice the forward runs: discovery needs all of them.
    """
    for block in blocks:
        setattr(block, _BLOCK_COMPILABLE_ATTR, True)


def blocks_marked_compilable(blocks: Iterable[nn.Module]) -> bool:
    return all(getattr(b, _BLOCK_COMPILABLE_ATTR, False) for b in blocks)


# Attribute under which a `cu_seqlens` tensor caches its block-diagonal mask.
_CU_MASK_ATTR = "_spyre_block_diag_mask"


def block_diagonal_mask(cu_seqlens: torch.Tensor, seq: int) -> torch.Tensor:
    """Bool `[seq, seq]` keep-mask from packed boundaries `[0, len0, len0+len1, ...]`.

    Built on CPU and cached on `cu_seqlens`, which a tower builds once per forward and
    passes to every layer, so only the first layer reads the (device) boundaries back.
    Keyed on its version, so an in-place repartition rebuilds; an inference tensor has
    none, and vLLM never repartitions one in place.
    """
    key = (seq, None if cu_seqlens.is_inference() else cu_seqlens._version)
    cached = getattr(cu_seqlens, _CU_MASK_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    bounds = convert(cu_seqlens, "cpu").to(torch.int64).tolist()
    m = torch.zeros(seq, seq, dtype=torch.bool)
    for start, end in zip(bounds[:-1], bounds[1:]):
        m[start:end, start:end] = True
    setattr(cu_seqlens, _CU_MASK_ATTR, (key, m))
    return m


# Attribute under which a source mask caches its padded counterpart `(key, padded)`.
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

    O(L²) and shared by every layer, so it is cached on the source mask: one upload
    per image, released with its source.
    """
    key = (b, seq, seq_pad, dtype, str(device))
    cached = getattr(mask, _MASK_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]

    # Assembled on CPU: strided slice-assign is not stick-safe on Spyre.
    neg_inf = torch.finfo(dtype).min
    m = torch.zeros(b, 1, seq_pad, seq_pad, dtype=dtype)
    m[:, :, :, seq:] = neg_inf  # padded keys never attended
    if mask.dtype != torch.bool:
        raise TypeError(f"an unpadded mask has to be a bool keep-mask, got {mask.dtype}")
    mc = convert(mask, "cpu")
    m[:, :, :seq, :seq] = torch.zeros(seq, seq, dtype=dtype).masked_fill(
        ~mc.reshape(seq, seq), neg_inf
    )

    m = convert(m, device)
    setattr(mask, _MASK_ATTR, (key, m))
    return m


# `(seq, seq_pad, dtype, device)` -> additive `[1, 1, 1, seq_pad]` row on `device`.
# Batch is not in the key: every image in the step shares this row, and SDPA broadcasts it.
_KEY_PAD_MASKS: dict[tuple, torch.Tensor] = {}


@torch.compiler.assume_constant_result
def _key_pad_mask_key(seq: int, seq_pad: int, dtype: torch.dtype, device: torch.device) -> tuple:
    """Build the row for this key if it is missing, then return the key.

    The row is `[1, 1, 1, seq_pad]` and broadcasts over batch, heads, and queries.
    `finfo.min / 2` matches `encoder_key_pad_mask`: a full `finfo.min` saturates to
    `-inf` in fp16 and NaNs SDPA.

    Runs once at trace time: Dynamo traces through `lru_cache` (pytorch/pytorch#152994)
    and the tail write cannot lower on Spyre. The key, not the row, is the constant: a
    tensor constant becomes a graph attribute torch-spyre cannot lay out, while the dict
    read in `padded_sdpa` makes the row a guarded graph input. Needs `dynamic=False`,
    which `_compile_blocks` passes because Spyre rejects SymInt shapes.
    """
    key = (seq, seq_pad, dtype, device)
    if key not in _KEY_PAD_MASKS:
        m = torch.zeros(1, 1, 1, seq_pad, dtype=dtype)
        m[..., seq:] = torch.finfo(dtype).min / 2
        _KEY_PAD_MASKS[key] = convert(m, device)
    return key


def padded_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """SDPA over `[B, H, L, D]` with L and D padded to the stick, then cropped.

    At a sequence length coprime with the stick, stock SDPA either fails to
    restickify a batch-matmul operand or returns silently wrong values, so the
    padding is a correctness requirement rather than a tuning choice. Padded keys are
    masked to `-inf` and padded queries cropped off.

    `scale` defaults to the head dim seen here, which assumes `q`/`k`/`v` arrive unpadded
    so the padding cannot change it. Pass it explicitly when the head dim is already
    padded, or when the model carries its own scale.

    `mask=None` attends everywhere: only the padded keys are masked, and that path
    is safe inside a compiled block. An additive `mask` already padded to `seq_pad`
    keys is used as is and has to be at `q.dtype`.
    """
    b, _, seq, d = q.shape
    if scale is None:
        scale = d**-0.5
    seq_pad = align_up(seq)
    d_pad = align_up(d)
    device = q.device
    padded = (seq_pad, d_pad) != (seq, d)

    if padded:
        # F.pad's tuple runs from the last dim backwards: (D left, D right, L left, L right).
        pad = (0, d_pad - d, 0, seq_pad - seq)
        q = F.pad(q, pad)
        k = F.pad(k, pad)
        v = F.pad(v, pad)
    else:
        # Offset operands read as offset 0 (torch-spyre#3770), so SDPA is silently
        # wrong here; the padded branch escapes it only because F.pad materializes.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

    if mask is None:
        # An aligned length has no padded keys, so SDPA runs unmasked.
        attn_mask = (
            None
            if seq == seq_pad
            else _KEY_PAD_MASKS[_key_pad_mask_key(seq, seq_pad, q.dtype, device)]
        )
    elif mask.shape[-1] == seq_pad and mask.dtype == q.dtype:
        attn_mask = mask
    elif mask.shape[-1] == seq_pad and seq != seq_pad:
        # Padded to seq_pad keys at another dtype: `_padded_attn_mask` would reshape it
        # to [seq, seq] and fail on the size, not the dtype.
        raise TypeError(
            f"padded_sdpa: a mask padded to {seq_pad} keys has to be {q.dtype}, got {mask.dtype}"
        )
    else:
        attn_mask = _padded_attn_mask(mask, b, seq, seq_pad, q.dtype, device)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        scale=scale,
        enable_gqa=enable_gqa,
    )

    if padded:
        # Offset-0 prefix slice, so torch-spyre#3770 cannot bite. Left as a view: the
        # caller's transpose+reshape materializes it anyway.
        out = out[:, :, :seq, :d]
    return out
