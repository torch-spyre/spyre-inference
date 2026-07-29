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

"""Unit tests for _compute_slot_mapping_impl in spyre_model_runner.

This pure-PyTorch function replaces the Triton/C++ slot mapping kernel
used by the upstream GPU/CPU backends. Tests verify correctness against
hand-computed expected slot indices.
"""

import pytest
import torch

from spyre_inference.v1.worker.spyre_model_runner import (
    _compute_slot_mapping_impl,
    _PAD_SLOT_ID,
)


class TestComputeSlotMappingBasic:
    """Basic slot mapping correctness tests."""

    def test_single_request_single_block(self):
        """Single request with all tokens in one block.

        block_size=4, positions [0,1,2,3] → block_table[0][0]=block_num
        slot = block_num * block_size + position
        """
        num_tokens = 4
        max_num_tokens = 4
        block_size = 4
        block_table_stride = 1
        block_num = 5

        query_start_loc = torch.tensor([0, 4], dtype=torch.int64)
        positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        block_table = torch.tensor([[block_num]], dtype=torch.int64)
        slot_mapping = torch.full((max_num_tokens,), -1, dtype=torch.int64)

        _compute_slot_mapping_impl(
            num_tokens=num_tokens,
            max_num_tokens=max_num_tokens,
            query_start_loc=query_start_loc,
            positions=positions,
            block_table=block_table,
            block_table_stride=block_table_stride,
            block_size=block_size,
            slot_mapping=slot_mapping,
        )

        expected = torch.tensor(
            [block_num * block_size + 0,
             block_num * block_size + 1,
             block_num * block_size + 2,
             block_num * block_size + 3],
            dtype=torch.int64,
        )
        torch.testing.assert_close(slot_mapping, expected)

    def test_single_request_multiple_blocks(self):
        """Single request spanning two blocks.

        block_size=4, positions [0..7] → 2 blocks
        block_table = [[3, 7]]  (block 3 for first 4 positions, block 7 for next 4)
        """
        num_tokens = 8
        max_num_tokens = 8
        block_size = 4
        block_table = torch.tensor([[3, 7]], dtype=torch.int64)
        block_table_stride = 2

        query_start_loc = torch.tensor([0, 8], dtype=torch.int64)
        positions = torch.arange(8, dtype=torch.int64)
        slot_mapping = torch.full((max_num_tokens,), -1, dtype=torch.int64)

        _compute_slot_mapping_impl(
            num_tokens=num_tokens,
            max_num_tokens=max_num_tokens,
            query_start_loc=query_start_loc,
            positions=positions,
            block_table=block_table,
            block_table_stride=block_table_stride,
            block_size=block_size,
            slot_mapping=slot_mapping,
        )

        # Expected: positions 0-3 → block 3, positions 4-7 → block 7
        expected = torch.tensor(
            [3*4+0, 3*4+1, 3*4+2, 3*4+3, 7*4+0, 7*4+1, 7*4+2, 7*4+3],
            dtype=torch.int64,
        )
        torch.testing.assert_close(slot_mapping, expected)

    def test_multiple_requests(self):
        """Two requests batched together.

        Request 0: 3 tokens at positions [0,1,2], block_table row 0 = [10]
        Request 1: 2 tokens at positions [0,1], block_table row 1 = [20]
        block_size=4, block_table_stride=1
        """
        num_tokens = 5
        max_num_tokens = 5
        block_size = 4
        block_table_stride = 1

        query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int64)
        positions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64)
        block_table = torch.tensor([[10], [20]], dtype=torch.int64)
        slot_mapping = torch.full((max_num_tokens,), -1, dtype=torch.int64)

        _compute_slot_mapping_impl(
            num_tokens=num_tokens,
            max_num_tokens=max_num_tokens,
            query_start_loc=query_start_loc,
            positions=positions,
            block_table=block_table,
            block_table_stride=block_table_stride,
            block_size=block_size,
            slot_mapping=slot_mapping,
        )

        expected = torch.tensor(
            [10*4+0, 10*4+1, 10*4+2, 20*4+0, 20*4+1],
            dtype=torch.int64,
        )
        torch.testing.assert_close(slot_mapping, expected)


class TestComputeSlotMappingPadding:
    """Test padding behavior when max_num_tokens > num_tokens."""

    def test_padding_with_pad_slot_id(self):
        """Tokens beyond num_tokens get PAD_SLOT_ID."""
        num_tokens = 2
        max_num_tokens = 4
        block_size = 4
        block_table_stride = 1

        query_start_loc = torch.tensor([0, 2], dtype=torch.int64)
        positions = torch.tensor([0, 1, 0, 0], dtype=torch.int64)  # extra zeros don't matter
        block_table = torch.tensor([[5]], dtype=torch.int64)
        slot_mapping = torch.full((max_num_tokens,), 999, dtype=torch.int64)

        _compute_slot_mapping_impl(
            num_tokens=num_tokens,
            max_num_tokens=max_num_tokens,
            query_start_loc=query_start_loc,
            positions=positions,
            block_table=block_table,
            block_table_stride=block_table_stride,
            block_size=block_size,
            slot_mapping=slot_mapping,
        )

        assert slot_mapping[0] == 5 * 4 + 0
        assert slot_mapping[1] == 5 * 4 + 1
        assert slot_mapping[2] == _PAD_SLOT_ID
        assert slot_mapping[3] == _PAD_SLOT_ID

    def test_no_padding_when_exact_fit(self):
        """When max_num_tokens == num_tokens, no padding is added."""
        num_tokens = 3
        max_num_tokens = 3
        block_size = 8
        block_table_stride = 1

        query_start_loc = torch.tensor([0, 3], dtype=torch.int64)
        positions = torch.tensor([0, 1, 2], dtype=torch.int64)
        block_table = torch.tensor([[2]], dtype=torch.int64)
        slot_mapping = torch.full((max_num_tokens,), -1, dtype=torch.int64)

        _compute_slot_mapping_impl(
            num_tokens=num_tokens,
            max_num_tokens=max_num_tokens,
            query_start_loc=query_start_loc,
            positions=positions,
            block_table=block_table,
            block_table_stride=block_table_stride,
            block_size=block_size,
            slot_mapping=slot_mapping,
        )

        expected = torch.tensor([2*8+0, 2*8+1, 2*8+2], dtype=torch.int64)
        torch.testing.assert_close(slot_mapping, expected)


class TestComputeSlotMappingEdgeCases:
    """Edge cases and decode-style usage."""

    def test_decode_single_token_per_request(self):
        """Decode step: each request generates 1 new token.

        3 requests, each with 1 token at different positions (continuation).
        """
        num_tokens = 3
        max_num_tokens = 3
        block_size = 16
        block_table_stride = 4

        # Positions represent continuation within existing sequences
        positions = torch.tensor([10, 25, 5], dtype=torch.int64)
        # block_indices = positions // block_size = [0, 1, 0]
        # block_offsets = positions % block_size = [10, 9, 5]
        query_start_loc = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

        # block_table: 3 requests × 4 blocks each
        block_table = torch.tensor([
            [100, 101, 102, 103],  # req 0
            [200, 201, 202, 203],  # req 1
            [300, 301, 302, 303],  # req 2
        ], dtype=torch.int64)
        slot_mapping = torch.full((max_num_tokens,), -1, dtype=torch.int64)

        _compute_slot_mapping_impl(
            num_tokens=num_tokens,
            max_num_tokens=max_num_tokens,
            query_start_loc=query_start_loc,
            positions=positions,
            block_table=block_table,
            block_table_stride=block_table_stride,
            block_size=block_size,
            slot_mapping=slot_mapping,
        )

        # req 0: block_index=0, offset=10 → block_table[0*4 + 0]=100 → slot=100*16+10=1610
        # req 1: block_index=1, offset=9 → block_table[1*4 + 1]=201 → slot=201*16+9=3225
        # req 2: block_index=0, offset=5 → block_table[2*4 + 0]=300 → slot=300*16+5=4805
        expected = torch.tensor([1610, 3225, 4805], dtype=torch.int64)
        torch.testing.assert_close(slot_mapping, expected)

    def test_context_parallelism_assertion(self):
        """TOTAL_CP_WORLD_SIZE != 1 should trigger an assertion error."""
        with pytest.raises(AssertionError, match="Context Parallelism"):
            _compute_slot_mapping_impl(
                num_tokens=1,
                max_num_tokens=1,
                query_start_loc=torch.tensor([0, 1], dtype=torch.int64),
                positions=torch.tensor([0], dtype=torch.int64),
                block_table=torch.tensor([[0]], dtype=torch.int64),
                block_table_stride=1,
                block_size=4,
                slot_mapping=torch.zeros(1, dtype=torch.int64),
                TOTAL_CP_WORLD_SIZE=2,
            )
