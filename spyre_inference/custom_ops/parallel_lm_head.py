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
    - Quantization: FP8 W8A8 models use ``SpyreFp8LMHeadMethod`` (one
      ``spyre.scaled_mm`` against a DMA'd ``qfp8wt`` weight) unless the checkpoint
      keeps the head unquantized or the model runs eagerly; all others use the
      unquantized transposed-weight fast path. Chosen at construction, as vLLM
      chooses a linear's quant method.
"""

from typing import cast

import torch
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


def quantize_weight_t_per_channel(w_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a host ``[K, N]`` weight to ``float8_e4m3fn`` with one scale per column.

    A column is one vocab entry, so an outlier entry cannot push the rest of the vocab
    toward e4m3's subnormal range, as one per-tensor scale would. The scale is rounded
    to float16 before dividing, so the weight quantizes against the exact scale the
    epilogue multiplies back. Padded vocab columns are all zero; the clamp keeps their
    scale a normal float16 instead of 0.
    """
    amax = w_t.float().abs().amax(dim=0)
    scale = (amax / FP8_E4M3FN_MAX).clamp(min=torch.finfo(torch.float16).tiny)
    scale = scale.to(torch.float16)
    q = (w_t.float() / scale.float()).clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX)
    return q.to(torch.float8_e4m3fn), scale.reshape(1, -1)


class SpyreFp8LMHeadMethod(SpyreUnquantizedLMHeadMethod):
    """``SpyreFp8LinearKernel``'s GEMM on the padded head, quantized per vocab column.

    The checkpoint head is FP16, so the padded ``Wᵀ`` is quantized on the host and
    DMA'd into ``qfp8wt``; the activation is quantized per token in the graph.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Reached on weight reload: rebuild the FP16 `Wᵀ` from `weight`, then requantize.
        super().process_weights_after_loading(layer)
        self.quantize_weight_t(layer)

    def quantize_weight_t(self, layer: torch.nn.Module) -> None:
        w_t = convert(cast(torch.Tensor, getattr(layer, self.WEIGHT_T_ATTR)).data, device="cpu")
        q, scale = quantize_weight_t_per_channel(w_t)
        # CPU float8 cannot H2D into qfp8wt. DMA it here, so model.to("spyre") skips it.
        setattr(
            layer, self.WEIGHT_T_ATTR, Parameter(_dma_checkpoint_qfp8wt(q), requires_grad=False)
        )
        # A reload after the device move must leave the scale where the GEMM reads it.
        old_scale = getattr(layer, "weight_scale", None)
        if old_scale is not None:
            scale = convert(scale, device=old_scale.device)
        layer.weight_scale = Parameter(scale, requires_grad=False)

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

    @maybe_compile
    def _project(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
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
        out = _fp8_gemm_epilogue(y, scale_a, cast(torch.Tensor, layer.weight_scale), None)
        padding = cast(int, layer.spyre_row_padding)
        if padding:
            out = out[:, :-padding]
        return out


def quantizes_body_in_fp8(quant_config: object) -> bool:
    """True when ``quant_config`` puts the model's linears on ``SpyreFp8LinearKernel``.

    That kernel serves FP8 W8A8: every ``Fp8Config``, and a compressed-tensors
    checkpoint whose schemes are FP8 weights with FP8 activations.
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if isinstance(quant_config, Fp8Config):
        return True
    if isinstance(quant_config, CompressedTensorsConfig):
        return any(
            CompressedTensorsConfig._is_fp8_w8a8(
                scheme.get("weights"), scheme.get("input_activations")
            )
            for scheme in quant_config.target_scheme_map.values()
        )
    return False


def checkpoint_quantizes_head(head: torch.nn.Module, quant_config: object) -> bool:
    """False when the checkpoint keeps ``head`` unquantized, as upstream vLLM honours.

    Matches each config's own rule for its linear layers. Any other config is treated
    as keeping the head FP16, since nothing here can read its opt-out.
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

    prefix = cast(str, getattr(head, "prefix", ""))
    if isinstance(quant_config, Fp8Config):
        return not is_layer_skipped(
            prefix,
            quant_config.ignored_layers,
            quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        )
    if isinstance(quant_config, CompressedTensorsConfig):
        return not should_ignore_layer(
            prefix, ignore=quant_config.ignore, fused_mapping=quant_config.packed_modules_mapping
        )
    return False


@ParallelLMHead.register_oot(name="ParallelLMHead")
class SpyreParallelLMHead(ParallelLMHead):
    """Out-of-tree (OOT) ParallelLMHead implementation for IBM's Spyre device.

    The projection lives in ``SpyreUnquantizedLMHeadMethod.apply`` (FP16) or
    ``SpyreFp8LMHeadMethod.apply`` (FP8), reached via
    ``LogitsProcessor._apply_head`` → ``lm_head.quant_method.apply``. The base
    ``ParallelLMHead.forward`` raises and is unused.

    vLLM's FP8 configs give an LM head no quant method, so the FP8 one is chosen here
    from the same config, and the loader's ``process_weights_after_loading`` quantizes it.
    """

    def _apply(self, fn, recurse=True):
        # The GEMM reads `padded_weight_t`; once it exists `weight` is runtime-dead, so
        # skip moving it to device. Until then `weight` is still live: don't skip it.
        if not hasattr(self, "padded_weight_t"):
            return super()._apply(fn, recurse=recurse)
        weight = self._parameters.pop("weight", None)
        try:
            return super()._apply(fn, recurse=recurse)
        finally:
            if weight is not None:
                self._parameters["weight"] = weight

    def __init__(self, *args, **kwargs):
        # Pad the vocab so every TP shard is a whole number of 64-element sticks, which
        # lets the logits gather stay on device (see SpyreCommunicator.all_gather).
        kwargs["padding_size"] = 64 * get_tensor_model_parallel_world_size()
        super().__init__(*args, **kwargs)

        # A checkpoint-quantized head (AWQ/GPTQ `lm_head_quantized`) is not supported.
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                f"SpyreParallelLMHead does not support {type(self.quant_method).__name__}."
            )

        logger.debug("Building SpyreParallelLMHead with TP size %d", self.tp_size)
        self.quant_method = SpyreUnquantizedLMHeadMethod()
        if not quantizes_body_in_fp8(self.quant_config):
            return
        fp8_method = SpyreFp8LMHeadMethod()
        if not checkpoint_quantizes_head(self, self.quant_config):
            logger.info("LM head %r stays FP16, as the checkpoint asks.", self.prefix)
        elif not fp8_method.spyre_compile_enabled:
            # `spyre.scaled_mm` only runs compiled, and eager mode pads no logits rows.
            logger.info("LM head %r stays FP16 under enforce_eager.", self.prefix)
        else:
            self.quant_method = fp8_method
            logger.info("LM head %r projects in FP8.", self.prefix)
