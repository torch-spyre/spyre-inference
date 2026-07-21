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

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Fallback for unit tests that call `layer.to("spyre")` directly and
        # skip the runner's CPU pin. Never hit from the runner path.
        if not torch.compiler.is_compiling() and self.weight.device.type == "spyre":
            self.weight = torch.nn.Parameter(self.weight.data.to("cpu"), requires_grad=False)

        cpu_input = convert(input_, device="cpu")
        if self.tp_size > 1:
            masked_input, input_mask = get_masked_input_and_mask(
                cpu_input,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = cpu_input

        output_parallel = self.quant_method.embedding(self, masked_input.long())
        output_parallel = convert(output_parallel, device=input_.device)

        if self.tp_size > 1:
            keep = (~input_mask).to(output_parallel.dtype).unsqueeze(-1)
            output_parallel = output_parallel * keep.to(output_parallel.device)

        return tensor_model_parallel_all_reduce(output_parallel)
