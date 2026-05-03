# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre OOT replacement for ParallelLMHead.

Executes the lm_head matmul (hidden_states @ weight.T) on Spyre.

Architecture:
    - OOT Registration: @ParallelLMHead.register_oot() replaces upstream
      at instantiation
    - forward_oot(): Entry point for OOT dispatch, handles device conversion
      and runs the compiled F.linear on Spyre
    - Separate Compilation: forward_spyre is compiled independently via
      maybe_compile
    - quant_method override: SpyreUnquantizedLMHeadMethod.apply() calls
      forward_oot() so that LogitsProcessor._get_logits() routes through
      the Spyre path

Spyre Device Constraints:
    - Computations performed in torch.float16
    - No Tensor Parallelism (TP) support: tp_size > 1 raises NotImplementedError
    - No quantization support: only UnquantizedEmbeddingMethod is replaced

References:
    - Upstream ParallelLMHead:
      vllm/model_executor/layers/vocab_parallel_embedding.py
    - Pattern reference: spyre_inference/custom_ops/silu_and_mul.py
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

from .utils import convert

logger = init_logger(__name__)


class SpyreUnquantizedLMHeadMethod(UnquantizedEmbeddingMethod):
    """Routes lm_head computation through SpyreParallelLMHead.forward_oot()."""

    def apply(self, layer, x, bias=None):
        return layer.forward_oot(x, bias)


@ParallelLMHead.register_oot(name="ParallelLMHead")
class SpyreParallelLMHead(ParallelLMHead):
    """OOT ParallelLMHead that executes the lm_head matmul on Spyre.

    Weights reside on Spyre after model.to(spyre_device).
    The quant_method is replaced so that LogitsProcessor._get_logits()
    routes through forward_oot, which handles device/dtype conversion
    and runs F.linear on Spyre.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        quant_config = kwargs.get("quant_config")
        if quant_config is not None:
            raise NotImplementedError(
                "SpyreParallelLMHead does not support quantization "
                f"(quant_config={quant_config}). Only quant_config=None is supported."
            )

        if self.tp_size > 1:
            raise NotImplementedError(
                f"SpyreParallelLMHead does not support Tensor Parallelism "
                f"(tp_size={self.tp_size}). Only tp_size=1 is supported."
            )

        logger.debug("Building custom ParallelLMHead for Spyre")

        self.maybe_compiled_forward_spyre = self.maybe_compile(self.forward_spyre)

        if isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            self.quant_method = SpyreUnquantizedLMHeadMethod()

        logger.warning_once(
            "%s: no dtype promotion (torch-spyre limitation), "
            "expect numerical differences to upstream vLLM.",
            self.__class__.__name__,
        )

    def forward_oot(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """OOT forward pass — lm_head matmul on Spyre.

        Called by SpyreUnquantizedLMHeadMethod.apply() from within
        LogitsProcessor._get_logits(). Converts x to the weight device,
        runs the compiled F.linear, and converts back to the input device.

        Args:
            x: Hidden states tensor [num_tokens, hidden_dim]
            bias: Optional bias tensor

        Returns:
            Logits tensor [num_tokens, vocab_size] on the input device
        """
        x_device = x.device
        w_device = self.weight.data.device

        out = self.maybe_compiled_forward_spyre(
            convert(x, w_device),
            self.weight.data,
            bias if bias is not None else None,
        )

        return convert(out, x_device)

    @staticmethod
    def forward_spyre(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Spyre lm_head kernel compiled via maybe_compile."""
        return F.linear(x, weight, bias)