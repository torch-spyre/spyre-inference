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

"""Spyre OOT replacement for VocabParallelEmbedding."""

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
    """Embedding method whose logits projection lands on CPU.

    Tied-weight models in the Gemma (1/2/3n) and Cohere/Command-R families
    compute logits through the tied ``embed_tokens`` (a VocabParallelEmbedding)
    rather than a ParallelLMHead — vLLM passes ``self.model.embed_tokens`` to the
    LogitsProcessor, which calls ``quant_method.apply``. The base ``apply`` keeps
    the logits on the input (Spyre) device, so the sampler's downstream
    ``log_softmax`` (no Spyre kernel, no CPU fallback) raises NotImplementedError.

    Mirror SpyreParallelLMHead.forward_oot: return logits on CPU. The
    ``embedding()`` lookup path used by the normal embedding forward is inherited
    unchanged.
    """

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return convert(super().apply(layer, x, bias), device="cpu")


@VocabParallelEmbedding.register_oot(name="VocabParallelEmbedding")
class SpyreVocabParallelEmbedding(VocabParallelEmbedding):
    """Out-of-tree (OOT) VocabParallelEmbedding implementation for IBM's Spyre device."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                f"SpyreVocabParallelEmbedding does not support quantized "
                f"embeddings (got {type(self.quant_method).__name__})."
            )
        # Route the logits-projection path (used when this embedding is the tied
        # LM head) through the CPU-converting method so log_softmax runs on CPU.
        self.quant_method = SpyreUnquantizedEmbeddingMethod()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            masked_input, input_mask = get_masked_input_and_mask(
                convert(input_, device="cpu"),
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
            masked_input = convert(masked_input, device=input_.device)
        else:
            masked_input = input_

        output_parallel = self.quant_method.embedding(self, masked_input.long())

        if self.tp_size > 1:
            keep = (~input_mask).to(output_parallel.dtype).unsqueeze(-1)
            output_parallel = output_parallel * keep.to(output_parallel.device)

        return tensor_model_parallel_all_reduce(output_parallel)
