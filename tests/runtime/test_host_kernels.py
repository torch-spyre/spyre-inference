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

"""The vLLM host kernels built from vLLM's csrc/ and the slot mapping route to them."""

import pytest
import torch

from spyre_inference.v1.host_kernels import has_host_kernels, register_slot_mapping_kernel

needs_kernels = pytest.mark.skipif(not has_host_kernels(), reason="host kernels are not built")


@needs_kernels
@pytest.mark.parametrize(
    "op", ["fused_gumbel_argmax", "greedy_argmax", "compute_slot_mapping_kernel_impl"]
)
def test_ops_are_registered_under_vllms_namespace(op: str) -> None:
    assert hasattr(torch.ops._C, op)


@needs_kernels
def test_slot_mapping_through_a_real_block_table():
    from vllm.v1.worker import block_table

    register_slot_mapping_kernel()

    block_size, max_tokens = 16, 8
    table = block_table.BlockTable(
        block_size=block_size,
        max_num_reqs=2,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=max_tokens,
        pin_memory=False,
        device=torch.device("cpu"),
        cp_kv_cache_interleave_size=1,
    )
    # Request 0 in physical blocks 7 and 3, request 1 in block 5.
    table.add_row([7, 3], row_idx=0)
    table.add_row([5], row_idx=1)
    table.commit_block_table(num_reqs=2)
    table.slot_mapping.gpu.fill_(12345)

    positions = torch.tensor([14, 15, 16, 17, 2], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 4, 5], dtype=torch.int32)
    table.compute_slot_mapping(2, query_start_loc, positions)

    expected = [7 * 16 + 14, 7 * 16 + 15, 3 * 16, 3 * 16 + 1, 5 * 16 + 2, -1, -1, -1]
    assert table.slot_mapping.gpu[:max_tokens].tolist() == expected
