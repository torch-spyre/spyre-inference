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
import torch.nn.functional as F
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .lazy_compile import CompileOutermost, compile_when_outermost
from .utils import place_row_gathered

logger = init_logger(__name__)


@VocabParallelEmbedding.register_oot(name="VocabParallelEmbedding")
class SpyreVocabParallelEmbedding(CompileOutermost, VocabParallelEmbedding):
    """Out-of-tree (OOT) VocabParallelEmbedding implementation for IBM's Spyre device."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                f"SpyreVocabParallelEmbedding does not support quantized "
                f"embeddings (got {type(self.quant_method).__name__})."
            )

        if self.tp_size > 1:
            reindex_table, keep_table = self._build_reindex_and_keep_tables()
            self.register_buffer(
                "_spyre_reindex_table",
                reindex_table,
                persistent=False,
            )
            self.register_buffer(
                "_spyre_keep_table",
                keep_table.to(self.weight.data.dtype),  # ty: ignore[no-matching-overload]
                persistent=False,
            )
        else:
            self._spyre_reindex_table = None
            self._spyre_keep_table = None

    def _build_reindex_and_keep_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the reindex and keep lookup tables in one pass."""
        vocab_size = self.num_embeddings
        reindex_table = torch.zeros(vocab_size, 2, dtype=torch.int64)
        keep_table = torch.zeros(vocab_size, 2, dtype=torch.float16)
        for i in range(vocab_size):
            masked_input, input_mask = get_masked_input_and_mask(
                torch.tensor([i], dtype=torch.int64),
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
            reindex_table[i, 0] = masked_input.item()
            keep_table[i, 0] = 0.0 if input_mask.item() else 1.0
        return reindex_table, keep_table

    def _apply(self, fn, recurse=True):
        weight = self._parameters.get("weight")

        def place(tensor: torch.Tensor) -> torch.Tensor:
            if tensor is weight:
                return place_row_gathered(tensor.data, fn, "vocab table")
            return fn(tensor)

        return super()._apply(place, recurse)

    @compile_when_outermost
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            reindex_table = self._spyre_reindex_table
            keep_table = self._spyre_keep_table
            assert reindex_table is not None and keep_table is not None
            keep = F.embedding(input_, keep_table)[:, 0]
            masked_input = torch.index_select(reindex_table, 0, input_.flatten())[:, 0]
            masked_input = masked_input.view(input_.shape)
        else:
            masked_input = input_
            keep = None

        output = self.quant_method.embedding(self, masked_input.long())

        if keep is not None:
            output = output * keep.unsqueeze(-1)
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
        dispatch_key="CPU",
    )
    logger.debug_once("Registered custom op: spyre_vocab_mask")
