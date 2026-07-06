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
"""

from typing import cast

import torch
import torch.nn.functional as F

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)

from .utils import convert

logger = init_logger(__name__)


@VocabParallelEmbedding.register_oot(name="VocabParallelEmbedding")
class SpyreVocabParallelEmbedding(VocabParallelEmbedding):
    """Spyre OOT ``VocabParallelEmbedding``.

    torch-spyre has no embedding kernel, so the table stays on Spyre with the
    rest of the weights and the gather runs on CPU. The TP shard mask and
    ``tensor_model_parallel_all_reduce`` also run on CPU: tensors from
    ``convert()`` H2D cannot be the target of in-place Spyre ops (the manual
    TP=2 ``all_reduce`` does ``input_.add_(scratch)``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                f"SpyreVocabParallelEmbedding does not support quantized "
                f"embeddings (got {type(self.quant_method).__name__})."
            )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        weight = cast(torch.Tensor, self.weight.data)
        weight_cpu = weight.cpu()
        if self.tp_size > 1:
            masked_input, input_mask = get_masked_input_and_mask(
                input_.cpu(),
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_.cpu()

        output_cpu = F.embedding(masked_input.long(), weight_cpu)
        if self.tp_size > 1:
            keep = (~input_mask).to(output_cpu.dtype).unsqueeze(-1)
            output_cpu = output_cpu * keep
            output_cpu = tensor_model_parallel_all_reduce(output_cpu)

        return convert(output_cpu, device=weight.device)
