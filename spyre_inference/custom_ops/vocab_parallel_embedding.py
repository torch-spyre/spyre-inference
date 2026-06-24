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

"""Spyre OOT replacement for VocabParallelEmbedding.

Inherits vocab sharding and weight loading from upstream. Overrides
`forward` only to compute the TP shard mask on CPU: the upstream helper
does int64 comparisons against Python int constants, which the Spyre
inductor backend rejects with `unexpected argument Constant(value=N,
dtype=torch.int64) to greaterequal`.

Also overrides `quant_method.apply` for the tied-weight lm_head path
(`config.tie_word_embeddings=True`, e.g. OPT). When the model assigns
`self.lm_head = self.model.decoder.embed_tokens`, the LogitsProcessor
calls `lm_head.quant_method.apply(layer, x, bias)` on this embedding.
The Spyre model wrapper has by then moved hidden_states to CPU while
the embedding weight stays on Spyre, so the upstream
`F.linear(x_cpu, weight_spyre, bias)` hits torch-spyre's decomposition
wrapper with mixed devices. The Spyre apply below converts x to the
weight device, runs the linear on Spyre, and returns the result on the
original input device — matching `SpyreParallelLMHead.forward_oot`.
"""

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)

from .utils import convert

logger = init_logger(__name__)


class SpyreUnquantizedEmbeddingMethod(UnquantizedEmbeddingMethod):
    """quant_method for SpyreVocabParallelEmbedding.

    Inherits the upstream embedding lookup unchanged (reached via
    `SpyreVocabParallelEmbedding.forward`). Overrides `apply` only —
    that is the tied-weight lm_head gemm path, where x arrives on CPU
    (from `_SpyreModelWrapper`) and the embedding weight is on Spyre.
    """

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_device = x.device
        weight_device = layer.weight.device
        if x_device == weight_device:
            # Both already on the same device — nothing for us to do.
            return torch.nn.functional.linear(x, layer.weight, bias)
        # Move x to where the weight lives (Spyre), run the linear there,
        # bring the result back to where x came from (CPU for the upstream
        # logits-processor path).
        out = torch.nn.functional.linear(
            convert(x, device=weight_device),
            layer.weight,
            bias,
        )
        return convert(out, device=x_device)


@VocabParallelEmbedding.register_oot(name="VocabParallelEmbedding")
class SpyreVocabParallelEmbedding(VocabParallelEmbedding):
    """Spyre OOT VocabParallelEmbedding.

    Mirrors upstream forward, but runs the mask helper on CPU (see module
    docstring) and zeroes out-of-shard rows by multiplication since
    torch-spyre lacks `masked_fill_`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                f"SpyreVocabParallelEmbedding does not support quantized "
                f"embeddings (got {type(self.quant_method).__name__})."
            )
        # Replace the apply() path so the tied-weight lm_head case (lm_head
        # is this same VocabParallelEmbedding) handles cpu-x / spyre-weight
        # device conversion. forward (embedding lookup) is unaffected.
        self.quant_method = SpyreUnquantizedEmbeddingMethod()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            masked_input, input_mask = get_masked_input_and_mask(
                input_.cpu(),
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
            masked_input = masked_input.to(input_.device)
        else:
            masked_input = input_

        output_parallel = self.quant_method.embedding(self, masked_input.long())

        if self.tp_size > 1:
            keep = (~input_mask).to(output_parallel.dtype).unsqueeze(-1)
            output_parallel = output_parallel * keep.to(output_parallel.device)

        return tensor_model_parallel_all_reduce(output_parallel)
