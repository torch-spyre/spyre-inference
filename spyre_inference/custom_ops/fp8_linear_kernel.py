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

"""Spyre FP8 linear: cached ``qfp8wt`` weights, compiled ``aten._scaled_mm``.

Load dequants checkpoint FP8 to CPU fp16 so ``model.to("spyre")`` is a legal
H2D (CPU ``float8.to("spyre")`` is the wrong layout; torch-spyre#4467). The
first Spyre forward eager-quantizes each SuperDSC N-tile
(``quantize_weight_fp8_with_scale``) and caches ``layer._qfp8wt_for_mm``.
Later forwards are one compiled graph:

    scale_a = quantscalepertokenfp8(x)              # in-graph, per-token
    y = _scaled_mm(qfp8ch(x), cached_qfp8wt, scale_a, scale_b)

Per-tensor activations still compute ``scale_a = amax(x) / FP8_E4M3FN_MAX``
eagerly because ``quantscalepertokenfp8`` always reduces over the hidden dim.

Eager qfp8wt must keep ``QFP8WT`` at the compiled GEMM boundary (torch-spyre
#4490). There is no in-graph qfp8wt fallback and no env toggle.

Granite 4096-wide SuperDSC only accepts M∈{1,4} and N∈{4096,1024,128}, so we
tile rows and split fused QKV/gate_up columns. Tile slices are ``clone()``'d
(``.contiguous()`` is a no-op on a row-view and SuperDSC ignores storage_offset).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch.nn.parameter import Parameter
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import register_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel,
)
from vllm.platforms import PlatformEnum

from spyre_inference.v1.worker import compile_guard

logger = init_logger(__name__)

try:
    from torch_spyre._inductor.constants import FP8_E4M3FN_MAX
except ImportError:
    FP8_E4M3FN_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

_REGISTERED = False

_WIDE = 4096
_WIDE_N = (4096, 1024, 128)
_SMALL_M = frozenset({1, 2, 3, 4})
_M_ALIGN = 128


def _m_tiles(m: int, k: int, n: int) -> list[int]:
    """Wide GEMMs: decode M=1, otherwise tiles of 4 (pad up). Narrow: one tile."""
    if max(k, n) < _WIDE:
        return [m]
    if m == 1:
        return [1]
    pad_m = m if m % 4 == 0 else ((m + 3) // 4) * 4
    return [4] * (pad_m // 4)


def _n_tiles(n: int) -> list[int]:
    """Split fused N (QKV 6144, gate_up 25600) into SuperDSC-legal widths."""
    if n in _WIDE_N:
        return [n]
    tiles: list[int] = []
    left = n
    for size in _WIDE_N:
        count, left = divmod(left, size)
        tiles.extend([size] * count)
    if left:
        raise RuntimeError(f"SpyreFp8LinearKernel: N={n} remainder {left} is not in {_WIDE_N}")
    return tiles


def _join(parts: list[torch.Tensor], dim: int) -> torch.Tensor:
    """Concatenate tiles; both ``_fp8_mm`` and ``cat`` already return fresh buffers."""
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=dim)


def _per_tensor_activation_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax().clamp(min=1e-12)
    return (amax / FP8_E4M3FN_MAX).to(dtype=torch.float16).reshape(1)


def _require_qfp8wt(weight: torch.Tensor) -> None:
    """Fail unless the eager-quantized weight is QFP8WT at the GEMM boundary.

    Missing layout APIs used to return here, so a build that canonicalized the
    layout would reach ``aten._scaled_mm`` with no error. Fail closed instead.
    """
    getter = getattr(weight, "device_tensor_layout", None)
    if getter is None:
        raise RuntimeError(
            "eager quantize_weight_fp8_with_scale produced a weight with no "
            "device_tensor_layout; QFP8WT must be readable at the compiled "
            "_scaled_mm boundary."
        )
    layout = getter()
    if layout is None:
        raise RuntimeError(
            "eager quantize_weight_fp8_with_scale produced a weight whose "
            "device_tensor_layout is None; QFP8WT must be present at the "
            "compiled _scaled_mm boundary."
        )
    arr = getattr(layout, "element_arrangement", None)
    if arr is None:
        raise RuntimeError(
            "device_tensor_layout has no element_arrangement; QFP8WT must be "
            "present at the compiled _scaled_mm boundary."
        )
    try:
        from torch_spyre._C import ElementArrangement
    except ImportError as e:
        raise RuntimeError(
            "torch_spyre._C.ElementArrangement is required to verify QFP8WT on the cached weight."
        ) from e
    if arr != ElementArrangement.QFP8WT:
        raise RuntimeError(
            "eager quantize_weight_fp8_with_scale did not produce QFP8WT "
            f"(got {arr}); the cached weight must keep that layout at the "
            "compiled _scaled_mm boundary."
        )


@torch.compile(backend="inductor", dynamic=False)
def _compiled_fp8_mm(
    x: torch.Tensor,
    weight_qfp8wt: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # Weight is already qfp8wt. Fuse per-token scale + qfp8ch + mm in one graph.
    scale_a = torch.ops.spyre.quantscalepertokenfp8(
        x,  # ty: ignore[invalid-argument-type]
        FP8_E4M3FN_MAX,  # ty: ignore[invalid-argument-type]
    )
    x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(
        x,  # ty: ignore[invalid-argument-type]
        scale_a,  # ty: ignore[invalid-argument-type]
    )
    return torch.ops.aten._scaled_mm(
        x_fp8,  # ty: ignore[invalid-argument-type]
        weight_qfp8wt,  # ty: ignore[invalid-argument-type]
        scale_a=scale_a,  # ty: ignore[invalid-argument-type]
        scale_b=weight_scale,  # ty: ignore[invalid-argument-type]
        bias=bias,  # ty: ignore[invalid-argument-type]
        out_dtype=torch.float16,  # ty: ignore[invalid-argument-type]
    )


@torch.compile(backend="inductor", dynamic=False)
def _compiled_fp8_mm_static_scale(
    x: torch.Tensor,
    scale_a: torch.Tensor,
    weight_qfp8wt: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(
        x,  # ty: ignore[invalid-argument-type]
        scale_a,  # ty: ignore[invalid-argument-type]
    )
    return torch.ops.aten._scaled_mm(
        x_fp8,  # ty: ignore[invalid-argument-type]
        weight_qfp8wt,  # ty: ignore[invalid-argument-type]
        scale_a=scale_a,  # ty: ignore[invalid-argument-type]
        scale_b=weight_scale,  # ty: ignore[invalid-argument-type]
        bias=bias,  # ty: ignore[invalid-argument-type]
        # A bfloat16 model cannot use this kernel; `check_and_update_config` rejects
        # that pairing rather than let float16 output reach a bfloat16 graph.
        out_dtype=torch.float16,  # ty: ignore[invalid-argument-type]
    )


# Both compile per distinct tile shape, and `apply_weights` tiles M onto {1, 4} (or a
# 128-multiple) and N onto `_WIDE_N`, so warmup's shapes cover every tile a request can
# produce. A compile here mid-serving is therefore a warmup-coverage gap like any other,
# and unwatched it would land in the guard's never-fatal UNKNOWN class.
compile_guard.watch(_compiled_fp8_mm, "fp8 scaled_mm (per-token scale)")
compile_guard.watch(_compiled_fp8_mm_static_scale, "fp8 scaled_mm (static scale)")


def _fp8_mm(
    x: torch.Tensor,
    weight_qfp8wt: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    per_token: bool,
) -> torch.Tensor:
    if per_token:
        return _compiled_fp8_mm(x, weight_qfp8wt, weight_scale, bias)
    return _compiled_fp8_mm_static_scale(
        x, _per_tensor_activation_scale(x), weight_qfp8wt, weight_scale, bias
    )


def _normalize_weight_scale(weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    scale = weight_scale.detach().to(torch.float16)
    n_out = weight.shape[-1]
    if scale.numel() == 1:
        return scale.reshape(1)
    if scale.numel() != n_out:
        raise NotImplementedError(
            "SpyreFp8LinearKernel expects per-tensor [1] or per-channel "
            f"[N]={n_out} weight_scale, got shape {tuple(weight_scale.shape)}"
        )
    return scale.reshape(1, n_out)


def _n_weight_splits(
    w_fp16: torch.Tensor, weight_scale: torch.Tensor, n_parts: list[int]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if len(n_parts) == 1:
        return [(w_fp16, weight_scale)]
    parts: list[tuple[torch.Tensor, torch.Tensor]] = []
    j = 0
    for ns in n_parts:
        wj = w_fp16[:, j : j + ns].clone()
        sj = weight_scale if weight_scale.numel() == 1 else weight_scale[:, j : j + ns].clone()
        parts.append((wj, sj))
        j += ns
    return parts


def _qfp8wt_splits(
    w_fp16: torch.Tensor, weight_scale: torch.Tensor, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Eager-quantize SuperDSC-legal N-tiles. Cached for the rest of the run."""
    scale = weight_scale.to(device=device, dtype=torch.float16)
    tiles = _n_weight_splits(w_fp16, scale, _n_tiles(int(w_fp16.shape[1])))
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for wj, sj in tiles:
        wq = torch.ops.spyre.quantize_weight_fp8_with_scale(
            wj,  # ty: ignore[invalid-argument-type]
            sj,  # ty: ignore[invalid-argument-type]
        )
        if wq is None:
            raise RuntimeError(
                "quantize_weight_fp8_with_scale returned None; eager qfp8wt "
                "is required for the cached-weight path."
            )
        _require_qfp8wt(wq)
        out.append((wq, sj))
    return out


def _pad_m(x: torch.Tensor, need_m: int) -> torch.Tensor:
    extra = need_m - x.shape[0]
    if extra <= 0:
        return x
    pad = torch.zeros(extra, x.shape[-1], dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=0)


class SpyreFp8LinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        gs = c.weight_quant_key.scale.group_shape
        if gs.is_per_tensor() or gs.is_per_channel():
            return True, None
        return False, "requires per-tensor or per-channel weight scales"

    def __init__(self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]) -> None:
        # Skip CUDA QuantFP8 in FP8ScaledMMLinearKernel.__init__.
        self._per_token_act = c.activation_quant_key.scale.group_shape.is_per_token()
        ScaledMMLinearKernel.__init__(self, c, layer_param_names)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = cast(torch.Tensor, layer.weight)
        weight_scale = cast(torch.Tensor, layer.weight_scale)
        scale = _normalize_weight_scale(weight, weight_scale)
        if weight.dtype == torch.float8_e4m3fn:
            # CPU float8 cannot DMA into qfp8wt. Dequant here so
            # model.to("spyre") is a legal fp16 H2D; first apply caches qfp8wt.
            w = weight.detach().cpu().to(torch.float16)
            s = scale.detach().cpu().to(torch.float16)
            weight = (w * s).contiguous()
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(scale, requires_grad=False)

    @torch._dynamo.disable()
    def _cached_qfp8wt(
        self, layer: torch.nn.Module, device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        splits = getattr(layer, "_qfp8wt_for_mm", None)
        if splits is not None and splits[0][0].device == device:
            return splits
        # Checkpoint FP8 is already dequanted to fp16 in process_weights_after_loading.
        w_fp16 = cast(torch.Tensor, layer.weight).to(device)
        splits = _qfp8wt_splits(w_fp16, cast(torch.Tensor, layer.weight_scale), device)
        layer._qfp8wt_for_mm = splits
        return splits

    # Not an untraceable op. The GEMM is already Dynamo/Inductor:
    # ``_compiled_fp8_mm`` (quantscalepertokenfp8 + qfp8ch + aten._scaled_mm).
    # ``recursive=False`` keeps that nested compile. This wrapper stays
    # eager because SuperDSC only accepts M∈{1,4} and N∈{4096,1024,128}, so
    # Granite QKV/gate_up is a Python tile/split loop with clone()'d views
    # (storage_offset is ignored), and first-forward eager qfp8wt is not
    # Spyre-graphable.
    @torch._dynamo.disable(recursive=False)
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        orig_m = x2d.shape[0]

        splits = self._cached_qfp8wt(layer, x2d.device)
        k = int(splits[0][0].shape[0])
        n = sum(int(wj.shape[1]) for wj, _ in splits)
        wide = max(k, n) >= _WIDE or len(splits) > 1
        if wide:
            m_parts = _m_tiles(orig_m, k, n)
            x2d = _pad_m(x2d, sum(m_parts))
        else:
            # 128-wide: pad M to a torch-spyre-tested size (1–4 or 128).
            if orig_m not in _SMALL_M and orig_m % _M_ALIGN:
                x2d = _pad_m(x2d, ((orig_m + _M_ALIGN - 1) // _M_ALIGN) * _M_ALIGN)
            m_parts = [x2d.shape[0]]

        row_outs = []
        i = 0
        for mt in m_parts:
            xi = x2d[i : i + mt].clone()
            i += mt
            col_outs = []
            col = 0
            for wj, sj in splits:
                ns = wj.shape[1]
                bj = None if bias is None else bias[col : col + ns].clone()
                col_outs.append(_fp8_mm(xi, wj, sj, bj, self._per_token_act))
                col += ns
            row_outs.append(_join(col_outs, dim=-1))
        out = _join(row_outs, dim=0)
        if out.shape[0] > orig_m:
            # The slice is already contiguous at offset 0; clone() compacts the
            # storage so the subsequent reshape (3-D inputs) sees the correct
            # element count. Without it, the padding rows corrupt the trailing
            # dimensions.
            out = out[:orig_m].clone()
        if x.dim() > 2:
            out = out.reshape(*orig_shape[:-1], out.shape[-1])
        return out

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # Required: FP8ScaledMMLinearKernel marks this abstract. Unused on Spyre.
        # Upstream Torch only overrides this hook because parent apply_weights
        # quantizes then calls it with already-FP8 A/B. We replace apply_weights
        # (tiling + cached qfp8wt + in-graph qfp8ch), so this is never entered.
        raise RuntimeError(
            "SpyreFp8LinearKernel runs only through apply_weights "
            "(tiled qfp8ch + cached qfp8wt). apply_scaled_mm is unused."
        )


SpyreFp8DequantLinearKernel = SpyreFp8LinearKernel


def register_spyre_fp8_linear_kernel() -> bool:
    global _REGISTERED
    if _REGISTERED:
        return True
    register_linear_kernel(SpyreFp8LinearKernel, PlatformEnum.OOT, kernel_type="fp8")
    _REGISTERED = True
    logger.info("Registered SpyreFp8LinearKernel for PlatformEnum.OOT (aten._scaled_mm)")
    return True
