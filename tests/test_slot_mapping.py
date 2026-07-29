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

"""Unit tests for _compute_slot_mapping_impl.

This function is the pure-PyTorch replacement for the upstream Triton/C++
slot-mapping kernel. It maps each token position to its flat index in the
paged KV cache. A bug here silently corrupts KV lookups.

These tests run on CPU (no Spyre device needed).
"""

import pytest
import torch

from spyre_inference.v1.worker.spyre_model_runner import (
    _compute_slot_mapping_impl,
    _FuncWrapper,
    _PAD_SLOT_ID,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_slot_mapping_inputs(
    num_reqs: int,
    tokens_per_req: list[int],
    block_size: int,
    max_num_tokens: int | None = None,
):
    """Build synthetic inputs for _compute_slot_mapping_impl.

    Returns a dict of kwargs suitable for calling the function directly.
    """
    num_tokens = sum(tokens_per_req)
    if max_num_tokens is None:
        max_num_tokens = num_tokens

    # query_start_loc: cumulative token offsets per request
    offsets = [0]
    for t in tokens_per_req:
        offsets.append(offsets[-1] + t)
    query_start_loc = torch.tensor(offsets, dtype=torch.int32)

    # positions: per-token position index within each request's context
    positions = torch.zeros(num_tokens, dtype=torch.int64)
    for i, t in enumerate(tokens_per_req):
        start = offsets[i]
        for j in range(t):
            positions[start + j] = j

    # block_table: [num_reqs, blocks_per_req] with unique block numbers
    blocks_per_req = max(
        (max(tokens_per_req) + block_size - 1) // block_size, 1
    )
    block_table = torch.arange(
        num_reqs * blocks_per_req, dtype=torch.int32
    ).reshape(num_reqs, blocks_per_req)
    block_table_stride = blocks_per_req

    # slot_mapping: output buffer
    slot_mapping = torch.full((max_num_tokens,), -999, dtype=torch.int64)

    return {
        "num_tokens": num_tokens,
        "max_num_tokens": max_num_tokens,
        "query_start_loc": query_start_loc,
        "positions": positions,
        "block_table": block_table,
        "block_table_stride": block_table_stride,
        "block_size": block_size,
        "slot_mapping": slot_mapping,
    }


def _reference_slot_mapping(
    num_reqs: int,
    tokens_per_req: list[int],
    block_size: int,
    block_table: torch.Tensor,
    block_table_stride: int,
) -> torch.Tensor:
    """Golden reference: compute slot mapping via nested loops."""
    num_tokens = sum(tokens_per_req)
    result = torch.zeros(num_tokens, dtype=torch.int64)
    offset = 0
    for req_idx, t in enumerate(tokens_per_req):
        for tok in range(t):
            pos = tok  # position within the request
            block_idx = pos // block_size
            block_offset = pos % block_size
            block_number = block_table[req_idx, block_idx].item()
            result[offset] = block_number * block_size + block_offset
            offset += 1
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slot_mapping
@pytest.mark.parametrize("block_size", [64, 128, 256])
def test_single_request_single_block(block_size):
    """Single request fitting in one block — simplest case."""
    inputs = _make_slot_mapping_inputs(
        num_reqs=1, tokens_per_req=[block_size], block_size=block_size
    )
    _compute_slot_mapping_impl(**inputs)

    expected = _reference_slot_mapping(
        num_reqs=1,
        tokens_per_req=[block_size],
        block_size=block_size,
        block_table=inputs["block_table"],
        block_table_stride=inputs["block_table_stride"],
    )
    torch.testing.assert_close(inputs["slot_mapping"], expected)


@pytest.mark.slot_mapping
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("num_reqs", [1, 2, 4, 8])
def test_multiple_requests(block_size, num_reqs):
    """Multiple requests with varying token counts."""
    tokens_per_req = [(i + 1) * 10 for i in range(num_reqs)]
    inputs = _make_slot_mapping_inputs(
        num_reqs=num_reqs, tokens_per_req=tokens_per_req, block_size=block_size
    )
    _compute_slot_mapping_impl(**inputs)

    expected = _reference_slot_mapping(
        num_reqs=num_reqs,
        tokens_per_req=tokens_per_req,
        block_size=block_size,
        block_table=inputs["block_table"],
        block_table_stride=inputs["block_table_stride"],
    )
    torch.testing.assert_close(
        inputs["slot_mapping"][: sum(tokens_per_req)], expected
    )


@pytest.mark.slot_mapping
def test_multi_block_request():
    """Request spanning multiple blocks — verifies block boundary logic."""
    block_size = 64
    # 150 tokens spans 3 blocks (64, 64, 22)
    tokens_per_req = [150]
    inputs = _make_slot_mapping_inputs(
        num_reqs=1, tokens_per_req=tokens_per_req, block_size=block_size
    )
    _compute_slot_mapping_impl(**inputs)

    expected = _reference_slot_mapping(
        num_reqs=1,
        tokens_per_req=tokens_per_req,
        block_size=block_size,
        block_table=inputs["block_table"],
        block_table_stride=inputs["block_table_stride"],
    )
    torch.testing.assert_close(inputs["slot_mapping"][:150], expected)


@pytest.mark.slot_mapping
def test_padding_with_pad_slot_id():
    """Tokens beyond num_tokens get PAD_SLOT_ID."""
    block_size = 64
    num_tokens = 10
    max_num_tokens = 32
    inputs = _make_slot_mapping_inputs(
        num_reqs=1,
        tokens_per_req=[num_tokens],
        block_size=block_size,
        max_num_tokens=max_num_tokens,
    )
    _compute_slot_mapping_impl(**inputs)

    # Active tokens should NOT be PAD_SLOT_ID
    assert (inputs["slot_mapping"][:num_tokens] != _PAD_SLOT_ID).all()
    # Padding slots should be PAD_SLOT_ID
    assert (inputs["slot_mapping"][num_tokens:max_num_tokens] == _PAD_SLOT_ID).all()


@pytest.mark.slot_mapping
def test_no_padding_when_num_tokens_equals_max():
    """When num_tokens == max_num_tokens, no padding is written."""
    block_size = 64
    inputs = _make_slot_mapping_inputs(
        num_reqs=1, tokens_per_req=[20], block_size=block_size, max_num_tokens=20
    )
    sentinel = -999
    inputs["slot_mapping"][:] = sentinel
    _compute_slot_mapping_impl(**inputs)
    # All 20 slots should be valid (not the sentinel)
    assert (inputs["slot_mapping"] != sentinel).all()


@pytest.mark.slot_mapping
def test_custom_pad_id():
    """Custom PAD_ID propagates to padding slots."""
    block_size = 64
    custom_pad = -42
    inputs = _make_slot_mapping_inputs(
        num_reqs=1, tokens_per_req=[5], block_size=block_size, max_num_tokens=16
    )
    _compute_slot_mapping_impl(**inputs, PAD_ID=custom_pad)
    assert (inputs["slot_mapping"][5:16] == custom_pad).all()


@pytest.mark.slot_mapping
def test_block_boundary_exact():
    """Token at exactly block_size goes into the second block."""
    block_size = 64
    # 65 tokens: tokens 0..63 in block 0, token 64 in block 1
    inputs = _make_slot_mapping_inputs(
        num_reqs=1, tokens_per_req=[65], block_size=block_size
    )
    _compute_slot_mapping_impl(**inputs)

    # Token at position 64 should map to block 1, offset 0
    block_1_number = inputs["block_table"][0, 1].item()
    expected_slot_64 = block_1_number * block_size + 0
    assert inputs["slot_mapping"][64].item() == expected_slot_64


@pytest.mark.slot_mapping
def test_context_parallelism_assertion():
    """TOTAL_CP_WORLD_SIZE > 1 should raise AssertionError."""
    block_size = 64
    inputs = _make_slot_mapping_inputs(
        num_reqs=1, tokens_per_req=[10], block_size=block_size
    )
    with pytest.raises(AssertionError, match="Context Parallelism"):
        _compute_slot_mapping_impl(**inputs, TOTAL_CP_WORLD_SIZE=2)


@pytest.mark.slot_mapping
def test_func_wrapper_grid_launch():
    """_FuncWrapper mimics Triton's kernel[(grid,)](...) → kernel(...) syntax."""
    calls = []

    def fake_kernel(*args, **kwargs):
        calls.append((args, kwargs))

    wrapper = _FuncWrapper(fake_kernel)
    # Triton-style grid launch: kernel[(1,)](arg1, arg2)
    wrapper[(1,)](42, key="val")
    assert len(calls) == 1
    assert calls[0] == ((42,), {"key": "val"})


@pytest.mark.slot_mapping
def test_deterministic_across_calls():
    """Calling twice with the same inputs produces identical slot_mapping."""
    block_size = 64
    inputs1 = _make_slot_mapping_inputs(
        num_reqs=3, tokens_per_req=[20, 30, 15], block_size=block_size
    )
    inputs2 = _make_slot_mapping_inputs(
        num_reqs=3, tokens_per_req=[20, 30, 15], block_size=block_size
    )
    _compute_slot_mapping_impl(**inputs1)
    _compute_slot_mapping_impl(**inputs2)
    torch.testing.assert_close(inputs1["slot_mapping"], inputs2["slot_mapping"])
