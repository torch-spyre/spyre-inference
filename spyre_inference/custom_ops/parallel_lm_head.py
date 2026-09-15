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
    - Quantization: Fp8Config auto-selects FP8 LM head via tiled
      ``aten._scaled_mm``; other configs use the unquantized transposed-weight
      fast path. Unsupported quantization methods raise NotImplementedError.
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
    _n_tiles,
    _n_weight_splits,
    _pad_m,
)
from .lazy_compile import CompileOutermost, compile_when_outermost
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

    @compile_when_outermost
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
    is stored FP16 (online quantization): the compiled ``_scaled_mm`` graph
    re-quantizes both activation and weight to FP8 every forward call, matching
    the body ``SpyreFp8LinearKernel`` pattern.

    N-tiles split the padded vocab into SuperDSC-legal widths
    ``{4096, 1024, 128}``; M-tiles handle the batch dimension.
    """

    WEIGHT_T_ATTR = "padded_weight_t"
    ROW_ALIGN = 64 * 32

    def build_weight_t(self, layer: torch.nn.Module, w: torch.Tensor) -> None:
        """Pad, transpose, and compute a per-tensor FP8 weight scale."""
        super().build_weight_t(layer, w)
        wt = getattr(layer, self.WEIGHT_T_ATTR)
        amax = wt.data.abs().amax().clamp(min=1e-12)
        weight_scale = (amax / FP8_E4M3FN_MAX).to(torch.float16).reshape(1)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)

    @torch._dynamo.disable(recursive=False)
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w = getattr(layer, self.WEIGHT_T_ATTR)  # [K, N]

        if bias is not None:
            raise NotImplementedError("SpyreFp8LMHeadMethod does not yet support embedding_bias.")

        orig_shape = x.shape
        x2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        orig_m = x2d.shape[0]
        k, n = int(w.shape[0]), int(w.shape[1])

        # Rebuild cached weight splits when stale (first call or after device move).
        splits = getattr(layer, "_fp8_n_weight_splits", None)
        if splits is None or splits[0][0].device != w.device:
            splits = _n_weight_splits(w, cast(torch.Tensor, layer.weight_scale), _n_tiles(n))
            layer._fp8_n_weight_splits = splits

        # Cache m_parts: k and n are fixed; only orig_m varies between calls.
        # Decode overwhelmingly sends the same orig_m (typically 1) every step.
        cached = getattr(layer, "_fp8_cached_m_parts", None)
        if cached is None or cached[0] != orig_m:
            m_parts = _m_tiles(orig_m, k, n)
            layer._fp8_cached_m_parts = (orig_m, m_parts)
        else:
            m_parts = cached[1]

        x2d = _pad_m(x2d, sum(m_parts))

        row_outs: list[torch.Tensor] = []
        i = 0
        for mt in m_parts:
            xi = x2d[i : i + mt].clone()
            i += mt
            col_outs: list[torch.Tensor] = []
            for wj, sj in splits:
                col_outs.append(_fp8_mm(xi, wj, sj, None, per_token=False))
            row_outs.append(_join(col_outs, dim=-1))
        out = _join(row_outs, dim=0)[:orig_m]

        padding = cast(int, layer.spyre_row_padding)
        if padding:
            out = out[:, :-padding]

        if x.dim() > 2:
            out = out.reshape(*orig_shape[:-1], out.shape[-1])
        return out.clone()


def _is_fp8_config(quant_config: object) -> bool:
    """True when the quantization config is an ``Fp8Config``."""
    return quant_config is not None and type(quant_config).__name__ == "Fp8Config"


@ParallelLMHead.register_oot(name="ParallelLMHead")
class SpyreParallelLMHead(ParallelLMHead):
    """Out-of-tree (OOT) ParallelLMHead implementation for IBM's Spyre device.

    The projection lives in ``SpyreUnquantizedLMHeadMethod.apply`` (FP16) or
    ``SpyreFp8LMHeadMethod.apply`` (FP8), reached via
    ``LogitsProcessor._apply_head`` → ``lm_head.quant_method.apply``. The base
    ``ParallelLMHead.forward`` raises and is unused.
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

        # Only UnquantizedEmbeddingMethod supported. Fp8Config resolves to it;
        # other quantization methods are rejected.
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                f"SpyreParallelLMHead does not support {type(self.quant_method).__name__}."
            )

        if _is_fp8_config(self.quant_config):
            logger.debug("Building SpyreParallelLMHead with FP8 (TP size %d)", self.tp_size)
            self.quant_method = SpyreFp8LMHeadMethod()
        else:
            logger.debug("Building SpyreParallelLMHead with TP size %d", self.tp_size)
            self.quant_method = SpyreUnquantizedLMHeadMethod()
