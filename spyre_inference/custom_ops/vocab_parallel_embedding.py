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

from functools import lru_cache

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)
from vllm.utils.torch_utils import direct_register_custom_op

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
            masked_input, keep = torch.ops.vllm.spyre_vocab_mask(
                cpu_input,  # ty: ignore[invalid-argument-type]
                self.shard_indices.org_vocab_start_index,  # ty: ignore[invalid-argument-type]
                self.shard_indices.org_vocab_end_index,  # ty: ignore[invalid-argument-type]
                self.shard_indices.num_org_vocab_padding,  # ty: ignore[invalid-argument-type]
                self.shard_indices.added_vocab_start_index,  # ty: ignore[invalid-argument-type]
                self.shard_indices.added_vocab_end_index,  # ty: ignore[invalid-argument-type]
                self.weight.data.dtype,  # ty: ignore[invalid-argument-type]
            )
        else:
            masked_input = cpu_input
            keep = None

        output = self.quant_method.embedding(self, masked_input.long())

        if self.tp_size > 1 and keep is not None:
            output = output * keep
            output = tensor_model_parallel_all_reduce(output)
        return output


def _vocab_mask_op_func(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = input_.device
    masked_input, input_mask = get_masked_input_and_mask(
        input_,
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
    )
    keep = (~input_mask).to(dtype=dtype).unsqueeze(-1)
    return masked_input.to(device), keep.to(device)


def _vocab_mask_op_fake(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_input = torch.empty(input_.shape, dtype=input_.dtype, device=input_.device)
    keep = torch.empty((*input_.shape, 1), dtype=dtype, device=input_.device)
    return masked_input, keep


@lru_cache(maxsize=1)
def register():
    """Register the spyre_vocab_mask custom op with vLLM."""
    direct_register_custom_op(
        op_name="spyre_vocab_mask",
        op_func=_vocab_mask_op_func,
        fake_impl=_vocab_mask_op_fake,
        mutates_args=[],
        dispatch_key=current_platform.dispatch_key,
    )
    logger.debug_once("Registered custom op: spyre_vocab_mask")
