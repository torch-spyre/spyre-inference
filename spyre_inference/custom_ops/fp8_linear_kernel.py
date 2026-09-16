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

"""Spyre FP8 linear: keep checkpoint FP8 weights, run compiled ``aten._scaled_mm``.

Forward (same graph as torch-spyre ``test_fp8_scaled_mm_cpu``):

    scale_a = amax(x) / 448                         # eager, outside compile
    y = _scaled_mm(qfp8ch(x), qfp8wt(W), scale_a, scale_b)   # FP16 out

**Decode (M=1)** — the FP8 ops are inlined directly into the model's compiled
graph via ``_apply_decode``, eliminating the graph break and HostCallback
overhead between the model graph and a separate FP8 compiled graph.  This is
possible because torch-spyre's GEMV fix (ef8b274b) allows M=1 ``_scaled_mm``
to compile through SuperDSC.

**Prefill (M>1)** — ``_apply_tiled`` tiles rows (M) and splits fused
QKV/gate_up columns (N) into SuperDSC-legal widths ``{4096, 1024, 128}``.
This path retains ``@torch._dynamo.disable(recursive=False)`` because
inlining the full tiling loop would fuse Granite-sized shapes that SuperDSC
cannot yet handle as a single graph.
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
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.platforms import PlatformEnum

logger = init_logger(__name__)

_REGISTERED = False
FP8_E4M3FN_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

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
    """Cat tiles into a new buffer so RMSNorm/SiLU/attention see offset 0."""
    return (parts[0] if len(parts) == 1 else torch.cat(parts, dim=dim)).clone()


def _activation_scale(x: torch.Tensor, per_token: bool) -> torch.Tensor:
    if per_token:
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        return (amax / FP8_E4M3FN_MAX).to(dtype=torch.float16)
    amax = x.abs().amax().clamp(min=1e-12)
    return (amax / FP8_E4M3FN_MAX).to(dtype=torch.float16).reshape(1)


@torch.compile(backend="inductor", dynamic=False)
def _compiled_fp8_scaled_mm(
    x: torch.Tensor,
    scale_a: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # qfp8wt layout is assigned in this graph; do not pre-quantize weights.
    x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(
        x,  # ty: ignore[invalid-argument-type]
        scale_a,  # ty: ignore[invalid-argument-type]
    )
    w_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(
        weight,  # ty: ignore[invalid-argument-type]
        weight_scale,  # ty: ignore[invalid-argument-type]
    )
    return torch.ops.aten._scaled_mm(
        x_fp8,
        w_fp8,
        scale_a=scale_a,  # ty: ignore[invalid-argument-type]
        scale_b=weight_scale,  # ty: ignore[invalid-argument-type]
        bias=bias,  # ty: ignore[invalid-argument-type]
        out_dtype=torch.float16,  # ty: ignore[invalid-argument-type]
    )


def _fp8_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    per_token: bool,
) -> torch.Tensor:
    return _compiled_fp8_scaled_mm(x, _activation_scale(x, per_token), weight, weight_scale, bias)


def _inline_fp8_scaled_mm(
    x_fp8: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_a: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Traceable FP8 matmul for the model's outer compiled graph.

    Unlike ``_compiled_fp8_scaled_mm`` this has no ``@torch.compile``:
    the enclosing model compile captures these ops directly so the
    ``qfp8wt`` layout is assigned by Inductor on the model graph.
    Activation ``x_fp8`` is pre-quantized by the caller so it is
    computed once and reused across N-tiles.
    """
    w_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(
        weight,
        weight_scale,
    )
    return torch.ops.aten._scaled_mm(
        x_fp8,
        w_fp8,
        scale_a=scale_a,
        scale_b=weight_scale,
        bias=bias,
        out_dtype=torch.float16,
    )


def _fp16_weight_for_qfp8wt(
    weight: torch.Tensor, weight_scale: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """CPU float8 is not qfp8wt. Dequant once; the compiled graph re-quantizes."""
    if weight.dtype != torch.float8_e4m3fn:
        return weight
    w = weight.detach().cpu().to(torch.float16)
    s = weight_scale.detach().cpu()
    return (w * s).contiguous().to(device)


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
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(scale, requires_grad=False)

    def _weight_splits(
        self, layer: torch.nn.Module, w: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        n_parts = _n_tiles(int(w.shape[1]))
        splits = getattr(layer, "_fp8_n_weight_splits", None)
        if splits is None or len(splits) != len(n_parts):
            splits = _n_weight_splits(w, cast(torch.Tensor, layer.weight_scale), n_parts)
            layer._fp8_n_weight_splits = splits
        return splits

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch to inline decode or tiled prefill path.

        Decode (M=1): ``_apply_decode`` — FP8 ops traced by the outer model
        compile; no graph break, no HostCallback overhead.  Requires cached
        weight splits from a prior forward (set up during warmup).
        """
        if not isinstance(x, torch.Tensor):
            raise NotImplementedError(
                "SpyreFp8LinearKernel expects unquantized activations; "
                "in-graph qfp8ch handles quantization."
            )
        orig_shape = x.shape
        x2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x

        if x2d.shape[0] == 1 and hasattr(layer, "_fp8_n_weight_splits"):
            out = self._apply_decode(layer, x2d, bias)
            if x.dim() > 2:
                out = out.reshape(*orig_shape[:-1], out.shape[-1])
            return out

        return self._apply_tiled(layer, x, bias)

    def _apply_decode(
        self,
        layer: torch.nn.Module,
        x2d: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """M=1 path: no row clone / pad; still the isolated ``_compiled_fp8_scaled_mm``."""
        splits = cast(list[tuple[torch.Tensor, torch.Tensor]], layer._fp8_n_weight_splits)
        if len(splits) == 1:
            wj, sj = splits[0]
            return _fp8_mm(x2d, wj, sj, bias, self._per_token_act)

        col_outs: list[torch.Tensor] = []
        col = 0
        for wj, sj in splits:
            ns = int(wj.shape[1])
            bj = None if bias is None else bias[col : col + ns]
            col_outs.append(_fp8_mm(x2d, wj, sj, bj, self._per_token_act))
            col += ns
        return torch.cat(col_outs, dim=-1)

    def _apply_tiled(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        orig_m = x2d.shape[0]

        w = getattr(layer, "_fp16_for_qfp8wt", None)
        if w is None or w.device != x2d.device:
            w = _fp16_weight_for_qfp8wt(
                cast(torch.Tensor, layer.weight),
                cast(torch.Tensor, layer.weight_scale),
                x2d.device,
            )
            layer._fp16_for_qfp8wt = w

        k, n = int(w.shape[0]), int(w.shape[1])
        splits = self._weight_splits(layer, w)
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
        out = _join(row_outs, dim=0)[:orig_m]
        if x.dim() > 2:
            out = out.reshape(*orig_shape[:-1], out.shape[-1])
        return out.clone()

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
        # (tiling + in-graph qfp8ch/qfp8wt), so this is never entered. Do not
        # wrap _fp8_mm here: that helper expects FP16 x/W, not pre-quantized A/B.
        raise RuntimeError(
            "SpyreFp8LinearKernel runs only through apply_weights "
            "(tiled qfp8ch/qfp8wt graph). apply_scaled_mm is unused."
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
