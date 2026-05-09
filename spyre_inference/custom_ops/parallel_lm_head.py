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

Executes the lm_head matmul (hidden_states @ weight.T) on Spyre.

Architecture:
    - OOT Registration: @ParallelLMHead.register_oot() replaces upstream
      at instantiation
    - forward_oot(): Entry point for OOT dispatch, handles device conversion
      and runs the compiled F.linear on Spyre
    - Separate Compilation: forward_spyre is compiled independently via
      maybe_compile (no opaque custom-op boundary)
    - quant_method override: SpyreUnquantizedLMHeadMethod.apply() calls
      forward_oot() so that LogitsProcessor._get_logits() routes through
      the Spyre path

Spyre Device Constraints:
    - No Tensor Parallelism (TP) support: tp_size > 1 raises NotImplementedError
    - No quantization support: only UnquantizedEmbeddingMethod is replaced

References:
    - Upstream ParallelLMHead:
      vllm/model_executor/layers/vocab_parallel_embedding.py
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

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)

        # torch-spyre currently has a limitation with the work division of larger
        # matmuls. The shapes needs to be a multiple of 64 * (k * 32), where k is
        # an integer.
        layer.padding = 0
        pad_1 = layer.weight.shape[0] % 64
        if pad_1 != 0:
            raise ValueError("The weight dimension must be a multiple of 64.")
        pad_2 = (layer.weight.shape[0] // 64) % 32
        if pad_2 > 0:
            pad_2 = 32 - pad_2
            layer.padding = pad_2 * 64
            layer.padded_weight = F.pad(layer.weight, (0, 0, 0, layer.padding))
            logger.warning_once(
                "%s: weights padded from %d to %d (torch-spyre limitation) "
                "expect numerical differences to upstream vLLM.",
                layer.__class__.__name__,
                layer.weight.shape[0],
                layer.padded_weight.shape[0],
            )
        else:
            layer.padded_weight = layer.weight


@ParallelLMHead.register_oot(name="ParallelLMHead")
class SpyreParallelLMHead(ParallelLMHead):
    """OOT ParallelLMHead that executes the lm_head matmul on Spyre.

    Weights reside on Spyre after model.to(spyre_device).
    The quant_method is replaced so that LogitsProcessor._get_logits()
    routes through forward_oot, which handles device conversion
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

        # Set the custom quantization method to route through spyre
        self.quant_method = SpyreUnquantizedLMHeadMethod()

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.padded_weight = fn(self.padded_weight)
        return self

    def forward_oot(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """OOT forward pass — lm_head matmul on Spyre.

        Called by SpyreUnquantizedLMHeadMethod.apply() from within
        LogitsProcessor._get_logits(). Converts x (arriving on cpu)
        to the weight device (residing on spyre), runs the compiled F.linear on spyre
        and converts back to the x device (cpu).

        Args:
            x: Hidden states tensor [num_tokens, hidden_dim]
            bias: Optional bias tensor

        Returns:
            Logits tensor [num_tokens, vocab_size] on the input device
        """
        x_device = x.device

        # Due to indexing operations inside the ModelRunner, which have
        # to be carried out on cpu due to a torch-spyre limitation,
        # the input to the SpyreParallelLMHead resides on CPU.
        # Due to a second limitation of torch-spyre regarding sizes that can be used
        # in a F.linear layer, the original weights need to be padded
        out = self.maybe_compiled_forward_spyre(
            convert(x, device=self.weight.device),
            self.padded_weight.data,
            bias,
        )

        return convert(out[:, : -self.padding] if self.padding > 0 else out, device=x_device)

    @staticmethod
    def forward_spyre(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Spyre lm_head kernel compiled via maybe_compile."""
        return F.linear(x, weight, bias)
