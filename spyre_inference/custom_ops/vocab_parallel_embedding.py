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

from vllm.platforms import current_platform
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import convert, get_layer, register_layer

logger = init_logger(__name__)


@VocabParallelEmbedding.register_oot(name="VocabParallelEmbedding")
class SpyreVocabParallelEmbedding(VocabParallelEmbedding):
    """Out-of-tree (OOT) VocabParallelEmbedding implementation for Spyre."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                f"SpyreVocabParallelEmbedding does not support quantized "
                f"embeddings (got {type(self.quant_method).__name__})."
            )
        self._spyre_layer_name = register_layer(self, "spyre_vocab_parallel_embedding")

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # The whole forward is one opaque FallbackKernel as far as Inductor
        # is concerned. See module docstring for the why.
        return torch.ops.vllm.spyre_vocab_parallel_embedding_native(input_, self._spyre_layer_name)

    def _forward_impl(self, input_: torch.Tensor) -> torch.Tensor:
        """Inner body called by the opaque-wrapping custom op.

        The weight is already CPU-resident (see `_apply` override). We
        just need to land `input_` on CPU and send the result back to
        whatever device the caller provided.
        """
        input_cpu = convert(input_, device="cpu")

        if self.tp_size > 1:
            masked_input, input_mask = get_masked_input_and_mask(
                input_cpu,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
            masked_input = convert(masked_input, device=self.weight.data.device)
        else:
            masked_input = input_
            input_mask = None

        output = torch.nn.functional.embedding(masked_input, self.weight.data)

        if self.tp_size > 1 and input_mask is not None:
            input_mask = convert(~input_mask, device=self.weight.data.device, dtype=output.dtype)
            keep = input_mask.unsqueeze(-1)
            output = output * keep

        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output


def _vocab_parallel_embedding_native_op_func(
    input_: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_layer(layer_name)
    out = layer._forward_impl(input_)
    # Custom-op contract: fresh, non-aliasing storage for each returned tensor.
    return out.contiguous()


def _vocab_parallel_embedding_native_op_fake(
    input_: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_layer(layer_name)
    embed_weight = layer.weight
    hidden_size = embed_weight.shape[1]
    out_shape = list(input_.shape) + [hidden_size]
    return torch.empty(out_shape, dtype=embed_weight.dtype, device=input_.device)


@lru_cache(maxsize=1)
def register():
    """Register the spyre_vocab_parallel_embedding_native custom op."""
    direct_register_custom_op(
        op_name="spyre_vocab_parallel_embedding_native",
        op_func=_vocab_parallel_embedding_native_op_func,
        fake_impl=_vocab_parallel_embedding_native_op_fake,
        mutates_args=[],
        # dispatch_key="CompositeExplicitAutograd",
        dispatch_key=current_platform.dispatch_key,
    )
    logger.debug_once("Registered custom op: spyre_vocab_parallel_embedding_native")
