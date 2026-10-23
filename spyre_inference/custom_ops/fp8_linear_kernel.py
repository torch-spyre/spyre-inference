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

"""Spyre FP8 linear: DMA'd ``qfp8wt`` weights, ``spyre.scaled_mm``.

Load DMAs checkpoint FP8 straight into ``qfp8wt`` (torch-spyre#4663), so
``model.to("spyre")`` skips it (CPU ``float8.to("spyre")`` is the wrong layout;
torch-spyre#4467). ``apply_weights`` traces into the block graph, or compiles as
its own graph outside one, since ``spyre.scaled_mm`` has no eager kernel:

    scale_a = quantscalepertokenfp8(x)              # in-graph, per-token
    y = scaled_mm(qfp8ch(x), qfp8wt) * (scale_a * scale_b)

Per-tensor activations use the checkpoint's static ``input_scale``, or else compute
``scale_a = amax(x) / FP8_E4M3FN_MAX``, because ``quantscalepertokenfp8`` always
reduces over the hidden dim.

The DMA'd qfp8wt must keep ``QFP8WT`` at the compiled GEMM boundary (torch-spyre
#4490). There is no in-graph qfp8wt fallback and no env toggle.
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

from .lazy_compile import CompileOutermost, maybe_compile

logger = init_logger(__name__)

try:
    from torch_spyre._inductor.constants import FP8_E4M3FN_MAX
except ImportError:
    FP8_E4M3FN_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

_REGISTERED = False


def _per_tensor_activation_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax().clamp(min=1e-12)
    return (amax / FP8_E4M3FN_MAX).to(dtype=torch.float16).reshape(1)


def _fp8_gemm_epilogue(
    y: torch.Tensor,
    scale_a: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """``y * (scale_a * weight_scale) + bias``: one epilogue, where ``aten._scaled_mm``
    decomposes into two full ``[M, N]`` multiplies."""
    scaled = y * (scale_a * weight_scale)
    return scaled if bias is None else scaled + bias


def _require_qfp8wt(weight: torch.Tensor) -> None:
    """Fail unless the DMA'd weight is QFP8WT at the GEMM boundary.

    Missing layout APIs used to return here, so a build that canonicalized the
    layout would reach ``spyre.scaled_mm`` with no error. Fail closed instead.
    """
    getter = getattr(weight, "device_tensor_layout", None)
    if getter is None:
        raise RuntimeError(
            "the qfp8wt DMA produced a weight with no "
            "device_tensor_layout; QFP8WT must be readable at the compiled "
            "scaled_mm boundary."
        )
    layout = getter()
    if layout is None:
        raise RuntimeError(
            "the qfp8wt DMA produced a weight whose "
            "device_tensor_layout is None; QFP8WT must be present at the "
            "compiled scaled_mm boundary."
        )
    arr = getattr(layout, "element_arrangement", None)
    if arr is None:
        raise RuntimeError(
            "device_tensor_layout has no element_arrangement; QFP8WT must be "
            "present at the compiled scaled_mm boundary."
        )
    try:
        from torch_spyre._C import ElementArrangement
    except ImportError as e:
        raise RuntimeError(
            "torch_spyre._C.ElementArrangement is required to verify QFP8WT on the DMA'd weight."
        ) from e
    if arr != ElementArrangement.QFP8WT:
        raise RuntimeError(
            "the qfp8wt DMA did not produce QFP8WT "
            f"(got {arr}); the weight must keep that layout at the "
            "compiled scaled_mm boundary."
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


def _dma_checkpoint_qfp8wt(weight: torch.Tensor) -> torch.Tensor:
    """DMA a CPU ``float8_e4m3fn`` ``[K, N]`` weight into QFP8WT.

    vLLM hands over ``weight.t()``, but the QFP8WT DMA assumes row-major host strides
    whatever the tensor's are, so it must be compacted first.
    """
    from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

    host = weight.detach().cpu().contiguous()
    qweight = _dma_to_spyre_fp8_kernel(host)
    _require_qfp8wt(qweight)
    return qweight


class SpyreFp8LinearKernel(CompileOutermost, FP8ScaledMMLinearKernel):
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
        # Skip CUDA QuantFP8 in FP8ScaledMMLinearKernel.__init__, and with it
        # CompileOutermost.__init__.
        self._per_token_act = c.activation_quant_key.scale.group_shape.is_per_token()
        ScaledMMLinearKernel.__init__(self, c, layer_param_names)
        self.arm_outer_compile()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = cast(torch.Tensor, layer.weight)
        weight_scale = cast(torch.Tensor, layer.weight_scale)
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"SpyreFp8LinearKernel expects a float8_e4m3fn weight, got {weight.dtype}."
            )
        scale = _normalize_weight_scale(weight, weight_scale)
        # CPU float8 cannot H2D into qfp8wt. DMA it here, so model.to("spyre") skips it.
        if weight.device.type != "spyre":
            weight = _dma_checkpoint_qfp8wt(weight)
        else:
            _require_qfp8wt(weight)
        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_scale = Parameter(scale, requires_grad=False)
        # Static activation checkpoints; upstream leaves None for dynamic ones.
        input_scale = getattr(layer, "input_scale", None)
        if input_scale is not None:
            layer.input_scale = Parameter(
                input_scale.detach().max().to(torch.float16).reshape(1), requires_grad=False
            )

    @maybe_compile(force=True)
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # input_quant_key() is None, so fusion never hands over a QuantizedActivation.
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"SpyreFp8LinearKernel quantizes in-graph and cannot consume {type(x).__name__}."
            )
        orig_shape = x.shape
        x2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        weight = cast(torch.Tensor, layer.weight)
        weight_scale = cast(torch.Tensor, layer.weight_scale)
        if self._per_token_act:
            scale_a = torch.ops.spyre.quantscalepertokenfp8(
                x2d,  # ty: ignore[invalid-argument-type]
                FP8_E4M3FN_MAX,  # ty: ignore[invalid-argument-type]
            )
        elif getattr(layer, "input_scale", None) is not None:
            scale_a = cast(torch.Tensor, layer.input_scale)
        else:
            scale_a = _per_tensor_activation_scale(x2d)
        x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(
            x2d,  # ty: ignore[invalid-argument-type]
            scale_a,  # ty: ignore[invalid-argument-type]
        )
        y = torch.ops.spyre.scaled_mm(
            x_fp8,
            weight,  # ty: ignore[invalid-argument-type]
            # A bfloat16 model cannot use this kernel; `check_and_update_config` rejects
            # that pairing rather than let float16 output reach a bfloat16 graph.
            out_dtype=torch.float16,  # ty: ignore[invalid-argument-type]
        )
        out = _fp8_gemm_epilogue(y, scale_a, weight_scale, bias)
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
        # (DMA'd qfp8wt + in-graph qfp8ch), so this is never entered.
        raise RuntimeError(
            "SpyreFp8LinearKernel runs only through apply_weights "
            "(qfp8ch + DMA'd qfp8wt). apply_scaled_mm is unused."
        )


def register_spyre_fp8_linear_kernel() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    register_linear_kernel(SpyreFp8LinearKernel, PlatformEnum.OOT, kernel_type="fp8")
    _REGISTERED = True
    logger.info("Registered SpyreFp8LinearKernel for PlatformEnum.OOT (spyre.scaled_mm)")
