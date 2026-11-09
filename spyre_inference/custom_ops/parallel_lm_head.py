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

"""Spyre OOT replacement for ParallelLMHead.

Spyre Device Constraints:
    - Tensor Parallelism: TP>=1 supported with vocabulary sharding (each rank
      computes logits for its vocab partition)
    - Quantization: an FP8 head a compressed-tensors checkpoint ships (FP8 W8A8,
      channel or tensor weight scales) runs ``SpyreFp8LMHeadMethod``, one
      ``spyre.scaled_mm`` against the checkpoint weight DMA'd into ``qfp8wt``,
      compiled even under ``enforce_eager``. An FP16 head stays FP16, as upstream.
      Other quantized heads raise NotImplementedError.
"""

from typing import cast

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

from .fp8_linear_kernel import (
    FP8_E4M3FN_MAX,
    _dma_checkpoint_qfp8wt,
    _fp8_gemm_epilogue,
)
from .lazy_compile import CompileOutermost, maybe_compile
from .linear import SpyreTransposedWeightMethod
from .utils import convert

logger = init_logger(__name__)


class SpyreUnquantizedLMHeadMethod(
    CompileOutermost, SpyreTransposedWeightMethod, UnquantizedEmbeddingMethod
):
    """LM-head projection via the shared transposed-weight fast path.

    No graph encloses it: per-block wraps only the block ModuleList, and a whole-model
    compile intercepts ``__call__``, not ``compute_logits``. Compiled ``dynamic=False``,
    so rows must arrive padded onto one of the ``logits_row_buckets``.
    """

    WEIGHT_T_ATTR = "padded_weight_t"
    ROW_ALIGN = 64 * 32

    @maybe_compile
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return SpyreTransposedWeightMethod.apply(self, layer, x, bias)


def checkpoint_fp8_scales(layer: torch.nn.Module) -> torch.Tensor:
    """The checkpoint's weight scale, one per row of ``layer.weight``, as loaded.

    A per-tensor scale is broadcast. The rows past the vocab shard are zero-filled by
    the loader, weight and scale alike, so their logits are 0.
    """
    weight = cast(torch.Tensor, layer.weight)
    scale = cast(torch.Tensor, layer.weight_scale).detach().float().reshape(-1)
    return scale.expand(weight.shape[0]) if scale.numel() == 1 else scale


class SpyreFp8LMHeadMethod(SpyreUnquantizedLMHeadMethod):
    """``SpyreFp8LinearKernel``'s GEMM on the checkpoint's FP8 head.

    The FP8 weight is padded and transposed on the host, as the FP16 head is, and DMA'd
    into ``qfp8wt``; nothing is requantized. The activation is quantized per token in
    the graph, or with the checkpoint's static ``input_scale``. The runtime scales live
    beside ``padded_weight_t``, so the checkpoint parameters stay loadable on reload.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = cast(torch.Tensor, layer.weight).data
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"SpyreFp8LMHeadMethod expects a float8_e4m3fn head, got {weight.dtype}."
            )
        padding = (-weight.shape[0]) % cast(int, self.ROW_ALIGN)
        layer.spyre_row_padding = padding
        # CPU float8 cannot H2D into qfp8wt. DMA it here, so model.to("spyre") skips it.
        weight_t = _dma_checkpoint_qfp8wt(F.pad(weight, (0, 0, 0, padding)).t())
        setattr(layer, self.WEIGHT_T_ATTR, Parameter(weight_t, requires_grad=False))

        scale = F.pad(checkpoint_fp8_scales(layer), (0, padding)).to(torch.float16)
        scale = scale.reshape(1, -1)
        self._set_runtime_scale(layer, "padded_weight_scale", scale)
        input_scale = getattr(layer, "input_scale", None)
        if input_scale is not None:
            static = input_scale.detach().max().to(torch.float16).reshape(1)
            self._set_runtime_scale(layer, "padded_input_scale", static)

    @staticmethod
    def _set_runtime_scale(layer: torch.nn.Module, name: str, scale: torch.Tensor) -> None:
        # A reload after the device move must leave the scale where the GEMM reads it.
        old = getattr(layer, name, None)
        if old is not None:
            scale = convert(scale, device=old.device)
        setattr(layer, name, Parameter(scale, requires_grad=False))

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise NotImplementedError("SpyreFp8LMHeadMethod does not yet support embedding_bias.")
        if x.dim() > 2:
            # A column slice of a `scaled_mm` output followed by a reshape reads garbage
            # on Spyre, and serving only ever sends 2-D hidden states.
            raise NotImplementedError(
                f"SpyreFp8LMHeadMethod requires 2-D input, got x.shape={tuple(x.shape)}."
            )
        return self._project(layer, x)

    # `spyre.scaled_mm` has no eager kernel, so this compiles under enforce_eager too.
    @maybe_compile(force=True)
    def _project(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        scale_a = getattr(layer, "padded_input_scale", None)
        if scale_a is None:
            scale_a = torch.ops.spyre.quantscalepertokenfp8(
                x,  # ty: ignore[invalid-argument-type]
                FP8_E4M3FN_MAX,  # ty: ignore[invalid-argument-type]
            )
        x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(
            x,  # ty: ignore[invalid-argument-type]
            scale_a,  # ty: ignore[invalid-argument-type]
        )
        y = torch.ops.spyre.scaled_mm(
            x_fp8,
            getattr(layer, self.WEIGHT_T_ATTR),  # ty: ignore[invalid-argument-type]
            out_dtype=torch.float16,  # ty: ignore[invalid-argument-type]
        )
        out = _fp8_gemm_epilogue(y, scale_a, cast(torch.Tensor, layer.padded_weight_scale), None)
        padding = cast(int, layer.spyre_row_padding)
        if padding:
            out = out[:, :-padding]
        return out


def ships_fp8_head(head: torch.nn.Module) -> bool:
    """True when the checkpoint ships ``head`` in FP8 W8A8 with channel or tensor scales:
    compressed-tensors gives a head a linear method only when a config group targets it."""
    from compressed_tensors.quantization import QuantizationStrategy
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsLinearMethod,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8Fp8,
    )

    scheme = getattr(head, "scheme", None)
    return (
        isinstance(getattr(head, "quant_method", None), CompressedTensorsLinearMethod)
        and isinstance(scheme, CompressedTensorsW8A8Fp8)
        and scheme.strategy in (QuantizationStrategy.CHANNEL, QuantizationStrategy.TENSOR)
    )


@ParallelLMHead.register_oot(name="ParallelLMHead")
class SpyreParallelLMHead(ParallelLMHead):
    """Out-of-tree (OOT) ParallelLMHead implementation for IBM's Spyre device.

    The projection lives in ``SpyreUnquantizedLMHeadMethod.apply`` (FP16) or
    ``SpyreFp8LMHeadMethod.apply`` (FP8), reached via
    ``LogitsProcessor._apply_head`` → ``lm_head.quant_method.apply``. The base
    ``ParallelLMHead.forward`` raises and is unused.

    The quant config's method has created and will load the checkpoint's parameters;
    only the projection and the post-load layout are replaced here.
    """

    def _apply(self, fn, recurse=True):
        # The GEMM reads `padded_weight_t` and its runtime scales; once they exist the
        # checkpoint parameters are runtime-dead, so skip moving them to device. Until
        # then they are still live: don't skip them.
        if not hasattr(self, "padded_weight_t"):
            return super()._apply(fn, recurse=recurse)
        host_only = {
            name: self._parameters.pop(name)
            for name in ("weight", "weight_scale", "input_scale")
            if name in self._parameters
        }
        try:
            return super()._apply(fn, recurse=recurse)
        finally:
            self._parameters.update(host_only)

    def __init__(self, *args, **kwargs):
        # Pad the vocab so every TP shard is a whole number of 64-element sticks, which
        # lets the logits gather stay on device (see SpyreCommunicator.all_gather).
        kwargs["padding_size"] = 64 * get_tensor_model_parallel_world_size()
        super().__init__(*args, **kwargs)

        logger.debug("Building SpyreParallelLMHead with TP size %d", self.tp_size)
        if isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            self.quant_method = SpyreUnquantizedLMHeadMethod()
            return
        if not ships_fp8_head(self):
            raise NotImplementedError(
                f"SpyreParallelLMHead does not support {type(self.quant_method).__name__}"
                f" with scheme {type(getattr(self, 'scheme', None)).__name__}."
            )
        self.quant_method = SpyreFp8LMHeadMethod()
        logger.info("LM head %r projects in FP8, as the checkpoint ships it.", self.prefix)
