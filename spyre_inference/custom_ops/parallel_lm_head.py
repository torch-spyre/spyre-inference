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

Spyre Device Constraints:
    - Tensor Parallelism: TP>=1 supported with vocabulary sharding (each rank
      computes logits for its vocab partition)
    - No quantization support: only UnquantizedEmbeddingMethod is replaced
"""

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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
        # With TP>1, layer.weight.shape[0] is the per-rank vocab partition size.
        ALIGN = 64 * 32
        size = layer.weight.shape[0]
        layer.padding = (-size) % ALIGN

        if layer.padding > 0:
            layer.padded_weight = Parameter(F.pad(layer.weight.data, (0, 0, 0, layer.padding)))
            logger.warning_once(
                "%s: weights padded from %d to %d (torch-spyre limitation) "
                "expect numerical differences to upstream vLLM.",
                layer.__class__.__name__,
                size,
                layer.padded_weight.shape[0],
            )
        else:
            layer.padded_weight = layer.weight


@ParallelLMHead.register_oot(name="ParallelLMHead")
class SpyreParallelLMHead(ParallelLMHead):
    """Out-of-tree (OOT) ParallelLMHead implementation for IBM's Spyre device."""

    padding: int
    padded_weight: torch.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        quant_config = kwargs.get("quant_config")
        if quant_config is not None:
            raise NotImplementedError(
                "SpyreParallelLMHead does not support quantization "
                f"(quant_config={quant_config}). Only quant_config=None is supported."
            )

        logger.debug("Building SpyreParallelLMHead with TP size %d ", self.tp_size)

        # Set the custom quantization method to route through spyre
        self.quant_method = SpyreUnquantizedLMHeadMethod()

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.padded_weight = fn(self.padded_weight)
        return self

    def forward_oot(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """OOT forward pass.

        Args:
            x: Hidden states tensor [num_tokens, hidden_dim]
            bias: Optional bias tensor

        Returns:
            Logits tensor [num_tokens, num_embeddings_per_partition] on the input device
        """
        # x already resides on Spyre (moved in _SpyreModelWrapper.compute_logits),
        # so no conversion is needed here. Due to a limitation of torch-spyre
        # regarding sizes usable in F.linear, the weights are padded.
        out = F.linear(
            x,
            self.padded_weight.data,
            bias,
        )

        out_cpu = convert(out, device="cpu")
        out_cpu_no_pad = out_cpu[:, : -self.padding] if self.padding > 0 else out_cpu
        # Currently output has to remain on CPU because of all_gather in case of TP > 1
        return out_cpu_no_pad
