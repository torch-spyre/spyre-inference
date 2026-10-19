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

"""MRV2: pointer-table kernels (vllm/v1/worker/gpu/block_table.py:BlockTables).

gather_block_tables and compute_slot_mappings launch @triton.jit kernels
directly rather than through kernels.py, so they need method-level
overrides; may shrink to nothing if upstream lands a CPU BlockTables.
"""

import torch
from vllm.v1.worker.gpu.block_table import BlockTables


class SpyreBlockTables(BlockTables):
    """MRV2: BlockTables, replacing the Triton pointer-table kernels."""

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
        out: tuple[torch.Tensor, ...] | None = None,
        out_ptrs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
