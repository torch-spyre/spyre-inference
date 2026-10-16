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

"""Unit tests for _compute_slot_mapping_impl (spyre_model_runner.py).

This function is a pure-PyTorch reimplementation of the upstream Triton kernel
that maps token positions to flat indices in the paged KV cache. A bug here
silently corrupts KV cache lookups. All tests run on CPU — no Spyre device
needed.
"""

import pytest
import torch

from spyre_inference.v1.worker.spyre_model_runner import (
    _compute_slot_mapping_impl,
    _PAD_SLOT_ID,
)


class TestComputeSlotMapping:
    """Tests for the slot mapping kernel reimplementation."""

    def test_single_request_basic(self):
        """Single request with 4 tokens, block_size=4 → all in block 0."""
        num_tokens = 4
        max_num_tokens = 4
        block_size = 4
        num_blocks = 2

        positions = torch.arange(num_tokens, dtype=torch.int64)
        query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int64)
        # Block table: request 0 has blocks [0, 1]
        block_table = torch.tensor([[0, 1]], dtype=torch.int64)
        block_table_stride = block_table.shape[1]
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

        # Positions 0-3 → block index 0, offsets 0-3
        # Block number = block_table[0, 0] = 0
        # slot = 0 * 4 + offset = 0, 1, 2, 3
        expected = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        torch.testing.assert_close(slot_mapping, expected)

    def test_single_request_crosses_block_boundary(self):
        """Tokens spanning two blocks: positions [0..7], block_size=4."""
        num_tokens = 8
        max_num_tokens = 8
        block_size = 4

        positions = torch.arange(num_tokens, dtype=torch.int64)
        query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int64)
        # Block table: request 0 has blocks [5, 3]
        block_table = torch.tensor([[5, 3]], dtype=torch.int64)
        block_table_stride = block_table.shape[1]
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

        # Positions 0-3 → block_index=0, block_number=5 → slots 20,21,22,23
        # Positions 4-7 → block_index=1, block_number=3 → slots 12,13,14,15
        expected = torch.tensor([20, 21, 22, 23, 12, 13, 14, 15], dtype=torch.int64)
        torch.testing.assert_close(slot_mapping, expected)

    def test_multiple_requests(self):
        """Two requests with different lengths and block assignments."""
        # Request 0: 3 tokens at positions [0,1,2]
        # Request 1: 2 tokens at positions [0,1]
        num_tokens = 5
        max_num_tokens = 5
        block_size = 4

        positions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int64)
        # Block table: req0 gets block [2], req1 gets block [7]
        block_table = torch.tensor([[2], [7]], dtype=torch.int64)
        block_table_stride = block_table.shape[1]
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

        # Req 0: pos [0,1,2] → block_idx=0, block_num=2 → slots 8,9,10
        # Req 1: pos [0,1] → block_idx=0, block_num=7 → slots 28,29
        expected = torch.tensor([8, 9, 10, 28, 29], dtype=torch.int64)
        torch.testing.assert_close(slot_mapping, expected)

    def test_padding_fills_remaining_slots(self):
        """max_num_tokens > num_tokens pads with PAD_SLOT_ID."""
        num_tokens = 2
        max_num_tokens = 5
        block_size = 4

        positions = torch.tensor([0, 1, 0, 0, 0], dtype=torch.int64)  # only first 2 matter
        query_start_loc = torch.tensor([0, 2], dtype=torch.int64)
        block_table = torch.tensor([[1]], dtype=torch.int64)
        block_table_stride = block_table.shape[1]
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

        # Tokens: block_num=1, pos 0,1 → slots 4, 5
        assert slot_mapping[0].item() == 4
        assert slot_mapping[1].item() == 5
        # Padding: remaining slots = PAD_SLOT_ID
        assert slot_mapping[2].item() == _PAD_SLOT_ID
        assert slot_mapping[3].item() == _PAD_SLOT_ID
        assert slot_mapping[4].item() == _PAD_SLOT_ID

    def test_no_padding_when_equal(self):
        """max_num_tokens == num_tokens → no padding needed."""
        num_tokens = 3
        max_num_tokens = 3
        block_size = 2

        positions = torch.tensor([0, 1, 0], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int64)
        block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        block_table_stride = block_table.shape[1]
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

        # Req 0: pos [0,1] → block_idx [0,0], block_num=0 → slots 0,1
        # Req 1: pos [0] → block_idx [0], block_num=2 → slot 4
        expected = torch.tensor([0, 1, 4], dtype=torch.int64)
        torch.testing.assert_close(slot_mapping, expected)

    def test_large_block_size(self):
        """Block size=64 (Spyre default) - tokens stay in first block."""
        num_tokens = 4
        max_num_tokens = 4
        block_size = 64

        positions = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 4], dtype=torch.int64)
        block_table = torch.tensor([[3, 7]], dtype=torch.int64)
        block_table_stride = block_table.shape[1]
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

        # All positions < 64, so block_index=0, block_number=3
        # slots = 3*64 + offset = 192 + 10, 11, 12, 13
        expected = torch.tensor([202, 203, 204, 205], dtype=torch.int64)
        torch.testing.assert_close(slot_mapping, expected)

    def test_custom_pad_id(self):
        """Custom PAD_ID value is respected."""
        num_tokens = 1
        max_num_tokens = 3
        block_size = 4
        custom_pad = -99

        positions = torch.tensor([0, 0, 0], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int64)
        block_table = torch.tensor([[0]], dtype=torch.int64)
        block_table_stride = block_table.shape[1]
        slot_mapping = torch.full((max_num_tokens,), 0, dtype=torch.int64)

        _compute_slot_mapping_impl(
            num_tokens=num_tokens,
            max_num_tokens=max_num_tokens,
            query_start_loc=query_start_loc,
            positions=positions,
            block_table=block_table,
            block_table_stride=block_table_stride,
            block_size=block_size,
            slot_mapping=slot_mapping,
            PAD_ID=custom_pad,
        )

        assert slot_mapping[0].item() == 0  # pos 0, block 0, offset 0
        assert slot_mapping[1].item() == custom_pad
        assert slot_mapping[2].item() == custom_pad
