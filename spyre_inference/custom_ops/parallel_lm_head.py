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
    - Quantization: FP8 models use ``SpyreFp8LMHeadMethod`` (tiled
      ``aten._scaled_mm``); all others use the unquantized transposed-weight
      fast path. The FP8 method is installed post-load by
      ``TorchSpyreModelRunner._initialize_fp8_lm_head``, which detects
      ``SpyreFp8LinearKernel`` in the body layers — covering both
      ``Fp8Config`` and ``compressed-tensors`` FP8 models.
      Unsupported quantization methods raise NotImplementedError.
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
    _fp8_mm,
    _join,
    _m_tiles,
    _pad_m,
    _qfp8wt_splits,
)
from .lazy_compile import CompileOutermost, maybe_compile
from .linear import SpyreTransposedWeightMethod

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


class SpyreFp8LMHeadMethod(SpyreTransposedWeightMethod, UnquantizedEmbeddingMethod):
    """FP8 LM-head projection via tiled ``aten._scaled_mm``.

    The padded transposed weight ``[K, N]`` = ``[hidden_dim, padded_vocab]``
    is stored FP16 and eager-quantized to QFP8WT tiles on first forward (via
    ``_qfp8wt_splits``), matching the ``SpyreFp8LinearKernel`` body pattern.
    The compiled ``_scaled_mm`` graph quantizes only the activation per call.

    N-tiles split the padded vocab into SuperDSC-legal widths
    ``{4096, 1024, 128}``; M-tiles handle the batch dimension.
    """

    WEIGHT_T_ATTR = "padded_weight_t"
    ROW_ALIGN = 64 * 32

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w = getattr(layer, self.WEIGHT_T_ATTR)  # [K, N]

        if bias is not None:
            raise NotImplementedError("SpyreFp8LMHeadMethod does not yet support embedding_bias.")

        if x.dim() > 2:
            raise NotImplementedError(
                f"SpyreFp8LMHeadMethod requires 2-D input, got x.shape={tuple(x.shape)}."
            )

        orig_m = x.shape[0]
        k, n = int(w.shape[0]), int(w.shape[1])

        # Eager-quantize weight tiles to QFP8WT on first call or after device
        # move. _n_weight_splits returns FP16 slices; _scaled_mm requires FP8
        # for mat2, so we must use _qfp8wt_splits (as SpyreFp8LinearKernel does).
        splits = getattr(layer, "_fp8_qfp8wt_splits", None)
        if splits is None or splits[0][0].device != w.device:
            splits = _qfp8wt_splits(w, cast(torch.Tensor, layer.weight_scale), w.device)
            layer._fp8_qfp8wt_splits = splits

        # Cache m_parts: k and n are fixed; only orig_m varies between calls.
        # Decode overwhelmingly sends the same orig_m (typically 1) every step.
        cached = getattr(layer, "_fp8_cached_m_parts", None)
        if cached is None or cached[0] != orig_m:
            m_parts = _m_tiles(orig_m, k, n)
            layer._fp8_cached_m_parts = (orig_m, m_parts)
        else:
            m_parts = cached[1]

        x2d = _pad_m(x, sum(m_parts))

        row_outs: list[torch.Tensor] = []
        i = 0
        for mt in m_parts:
            xi = x2d[i : i + mt].clone()
            i += mt
            row_outs.append(
                _join(
                    [_fp8_mm(xi, wj, sj, None, per_token=False) for wj, sj in splits],
                    dim=-1,
                )
            )
        out = _join(row_outs, dim=0)
        if out.shape[0] > orig_m:
            # Row-slice: clone() compacts M-pad rows into fresh storage.
            out = out[:orig_m].clone()

        padding = cast(int, layer.spyre_row_padding)
        if padding:
            out = out[:, :-padding]
        return out


@ParallelLMHead.register_oot(name="ParallelLMHead")
class SpyreParallelLMHead(ParallelLMHead):
    """Out-of-tree (OOT) ParallelLMHead implementation for IBM's Spyre device.

    The projection lives in ``SpyreUnquantizedLMHeadMethod.apply`` (FP16) or
    ``SpyreFp8LMHeadMethod.apply`` (FP8), reached via
    ``LogitsProcessor._apply_head`` → ``lm_head.quant_method.apply``. The base
    ``ParallelLMHead.forward`` raises and is unused.

    ``quant_method`` starts as ``SpyreUnquantizedLMHeadMethod`` for all models.
    ``TorchSpyreModelRunner._initialize_fp8_lm_head`` upgrades it to
    ``SpyreFp8LMHeadMethod`` post-load when ``SpyreFp8LinearKernel`` is detected
    in the body layers, covering both ``Fp8Config`` and ``compressed-tensors``
    FP8 models.
    """

    def initialize_fp8(self) -> None:
        """Upgrade this layer to FP8 projection post-load.

        Called by ``TorchSpyreModelRunner._initialize_fp8_lm_head`` after weights
        are on device. ``padded_weight_t`` already exists (built by
        ``SpyreUnquantizedLMHeadMethod`` during ``process_weights_after_loading``);
        this method computes ``weight_scale`` from it and swaps in
        ``SpyreFp8LMHeadMethod``.

        Note: the caller does not consult ``ignored_layers`` / ``ignore`` from the
        checkpoint quant config before calling this method — the head is always
        quantized to FP8 for any model where ``SpyreFp8LinearKernel`` is detected
        in the body layers.  See ``_initialize_fp8_lm_head`` docstring for details.
        """
        method = SpyreFp8LMHeadMethod()
        wt = getattr(self, method.WEIGHT_T_ATTR)
        amax = wt.data.abs().amax().clamp(min=1e-12)
        self.weight_scale = Parameter(
            (amax / FP8_E4M3FN_MAX).to(torch.float16).reshape(1),
            requires_grad=False,
        )
        self.quant_method = method

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

        # Only UnquantizedEmbeddingMethod is supported at construction time.
        # _initialize_fp8_lm_head may upgrade this to SpyreFp8LMHeadMethod post-load.
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                f"SpyreParallelLMHead does not support {type(self.quant_method).__name__}."
            )

        logger.debug("Building SpyreParallelLMHead with TP size %d", self.tp_size)
        self.quant_method = SpyreUnquantizedLMHeadMethod()
