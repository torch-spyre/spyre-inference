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

"""Spyre FP8 linear: QFP8WT weights, ops traced into the block graph.

Checkpoint ``float8_e4m3fn`` weights are DMA'd straight into QFP8WT
(``_dma_to_spyre_fp8_kernel``, torch-spyre#4663). ``model.to("spyre")`` then
skips that parameter. Every vLLM FP8 method, checkpoint or online, hands this
kernel a float8 weight, so there is no fp16 path. ``apply_weights`` only reads
``layer.weight`` and ``layer.weight_scale``:

    scale_a = quantscalepertokenfp8(x)              # per-token
    y = scaled_mm(qfp8ch(x), qfp8wt) * (scale_a * scale_b)

Those ops sit in the block graph (``fullgraph=True``). There is no nested
``torch.compile`` and no Dynamo disable.

``aten._scaled_mm`` would decompose that product into two full ``[M, N]``
multiplies plus a bias add. Folding the scales first is one epilogue.

Per-tensor activations use the checkpoint's static ``input_scale`` when it has
one, and otherwise compute ``scale_a = amax(x) / FP8_E4M3FN_MAX`` in the same
graph, because ``quantscalepertokenfp8`` always reduces over the hidden dim.

``spyre.scaled_mm`` and ``quantscalepertokenfp8`` lower only inside a compiled
graph, so ``check_and_update_config`` rejects a quantized model under
``--enforce-eager``.

The weight must be ``QFP8WT`` (torch-spyre#4490).

SuperDSC M/N tiling is the inductor backend's job. This kernel does not split
rows or fused QKV/gate_up columns.
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
    """``y * (scale_a * weight_scale)``, plus bias when present.

    One pointwise formula, so inductor emits a single epilogue instead of the
    two scale multiplies and the bias add that ``aten._scaled_mm`` decomposes
    into.
    """
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
            "the QFP8WT DMA produced a weight with no "
            "device_tensor_layout; QFP8WT must be readable at the compiled "
            "scaled_mm boundary."
        )
    layout = getter()
    if layout is None:
        raise RuntimeError(
            "the QFP8WT DMA produced a weight whose "
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
            "torch_spyre._C.ElementArrangement is required to verify QFP8WT on the cached weight."
        ) from e
    if arr != ElementArrangement.QFP8WT:
        raise RuntimeError(
            "the QFP8WT DMA did not produce QFP8WT "
            f"(got {arr}); the cached weight must keep that layout at the "
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
    """DMA a CPU ``float8_e4m3fn`` weight into QFP8WT.

    vLLM's ``process_fp8_weight_*`` returns ``weight.t()``, a non-contiguous
    ``[K, N]``. The QFP8WT DCI addresses the host as contiguous row-major
    ``k * N + n`` and ignores the tensor's real strides, so compact first or
    the GEMM reads a scrambled matrix. K must be divisible by 2 and N by 64.
    """
    from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

    host = weight.detach().cpu().contiguous()
    qweight = _dma_to_spyre_fp8_kernel(host)
    _require_qfp8wt(qweight)
    return qweight


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
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"SpyreFp8LinearKernel expects a float8_e4m3fn weight, got {weight.dtype}."
            )
        scale = _normalize_weight_scale(weight, weight_scale)
        # DMA'd now, before model.to, so the generic H2D never sees float8
        # (torch-spyre#4467, #4663); .to then skips the QFP8WT weight.
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

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Traced into the block graph. Weight and scale are already QFP8WT / fp16
        # parameters.
        # input_quant_key() is None, so fusion does not hand us a pre-quantized
        # activation. The union is only here to match FP8ScaledMMLinearKernel.
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "SpyreFp8LinearKernel quantizes activations in the block graph "
                f"and cannot consume {type(x).__name__}."
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
            # A bfloat16 model cannot use this kernel; `check_and_update_config`
            # rejects that pairing rather than let float16 output reach a
            # bfloat16 graph.
            scale_a = _per_tensor_activation_scale(x2d)
        x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(
            x2d,  # ty: ignore[invalid-argument-type]
            scale_a,  # ty: ignore[invalid-argument-type]
        )
        y = torch.ops.spyre.scaled_mm(
            x_fp8,  # ty: ignore[invalid-argument-type]
            weight,  # ty: ignore[invalid-argument-type]
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
        # (QFP8WT weight + in-graph qfp8ch), so this is never entered.
        raise RuntimeError(
            "SpyreFp8LinearKernel runs only through apply_weights "
            "(qfp8ch + qfp8wt). apply_scaled_mm is unused."
        )


def _spyre_fp8_kernel(module: torch.nn.Module) -> SpyreFp8LinearKernel | None:
    quant_method = getattr(module, "quant_method", None)
    owners = (
        quant_method,
        getattr(module, "scheme", None),
        getattr(quant_method, "scheme", None),
    )
    for owner in owners:
        if isinstance(owner, SpyreFp8LinearKernel):
            return owner
        kernel = getattr(owner, "fp8_linear", None)
        if isinstance(kernel, SpyreFp8LinearKernel):
            return kernel
    return None


def register_spyre_fp8_linear_kernel() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    register_linear_kernel(SpyreFp8LinearKernel, PlatformEnum.OOT, kernel_type="fp8")
    _REGISTERED = True
    logger.info("Registered SpyreFp8LinearKernel for PlatformEnum.OOT (spyre.scaled_mm)")
