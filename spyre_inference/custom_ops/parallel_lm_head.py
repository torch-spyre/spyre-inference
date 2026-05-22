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
    - forward_oot(): Entry point for OOT dispatch, defers to shared
      forward_lm_head_oot which uses ``layer._spyre_matmul_state``
    - Chunked matmul: LmHeadMatmulState splits the weight along the vocab
      dim into N padded sub-weights (driven by SPYRE_LM_HEAD_NUM_CHUNKS),
      runs one F.linear per chunk on Spyre, and concatenates on CPU. This
      sidesteps torch-spyre's per-core 256 MB EAR span limit for large
      lm_head matmuls (e.g. Qwen3-8B 151936×4096).
    - quant_method override: SpyreUnquantizedLMHeadMethod.apply() calls
      forward_oot() so that LogitsProcessor._get_logits() routes through
      the Spyre path

Spyre Device Constraints:
    - No Tensor Parallelism (TP) support: tp_size > 1 raises NotImplementedError
    - No quantization support: only UnquantizedEmbeddingMethod is replaced

References:
    - Upstream ParallelLMHead:
      vllm/model_executor/layers/vocab_parallel_embedding.py
    - Chunking pattern: hf_adapters/hf_common.py chunk_lm_head
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
)

from .lm_head_common import (
    SpyreUnquantizedLMHeadMethod,
    forward_lm_head_oot,
)

__all__ = ["SpyreParallelLMHead", "SpyreUnquantizedLMHeadMethod"]

logger = init_logger(__name__)


@ParallelLMHead.register_oot(name="ParallelLMHead")
class SpyreParallelLMHead(ParallelLMHead):
    """OOT ParallelLMHead that executes the lm_head matmul on Spyre.

    Weights reside on Spyre after model.to(spyre_device). The shared
    LmHeadMatmulState (built in process_weights_after_loading) holds the
    padded weight chunks and compiled F.linear; forward_oot iterates over
    them and concatenates logits on CPU.
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

        # Set the custom quantization method to route through spyre.
        # The chunked LmHeadMatmulState is materialized in
        # SpyreUnquantizedLMHeadMethod.process_weights_after_loading once the
        # checkpoint values are loaded into self.weight.
        self.quant_method = SpyreUnquantizedLMHeadMethod()

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Move every padded chunk to the new device; chunks that alias
        # self.weight are already handled by super().
        state = getattr(self, "_spyre_matmul_state", None)
        if state is not None:
            new_chunks = []
            for chunk in state.chunks:
                if chunk is self.weight:
                    new_chunks.append(self.weight)
                else:
                    new_chunks.append(fn(chunk))
            state.chunks = new_chunks
        return self

    def forward_oot(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """OOT forward pass — chunked lm_head matmul on Spyre."""
        return forward_lm_head_oot(self, x, bias)
