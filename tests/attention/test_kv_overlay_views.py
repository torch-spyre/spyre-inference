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

"""SpyreAttentionImpl.view_pages on a card: gemma-4-26B's sliding [8, 128, 256] pages and
global [2, 128, 512] pages (a 256-token block as two kernel blocks) on one allocation,
as the hybrid KV cache manager overlays them."""

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl, SpyrePagedKVCache

pytestmark = pytest.mark.skipif(not spyre_available(), reason="Spyre device not available")

NUM_BLOCKS = 6
SLIDING = SlidingWindowSpec(
    block_size=128, num_kv_heads=8, head_size=256, dtype=torch.float16, sliding_window=1024
)
GLOBAL = FullAttentionSpec(block_size=128, num_kv_heads=2, head_size=512, dtype=torch.float16)


def _values(*shape: int) -> torch.Tensor:
    # Integers in [-256, 256) survive the device's fp16 format exactly.
    return torch.randint(-256, 256, shape).to(torch.float16)


@pytest.fixture
def overlay() -> tuple[torch.Tensor, torch.Tensor]:
    """The K pages of a sliding layer and of a global layer viewing them."""
    device = torch.device("spyre")
    torch.zeros(64, dtype=torch.float16, device=device)
    base = SpyreAttentionImpl.allocate_pages(NUM_BLOCKS, SLIDING, device)
    view = SpyreAttentionImpl.view_pages(base, 2 * NUM_BLOCKS, GLOBAL)
    return base.k_pages, view.k_pages


@torch.compile(dynamic=False)
def _store(slots: torch.Tensor, index: torch.Tensor, rows: torch.Tensor) -> None:
    slots.index_copy_(0, index, rows)


def test_both_shapes_address_the_same_row_major_bytes(overlay):
    base, view = overlay
    src = _values(*base.shape)
    base.copy_(src)
    assert torch.equal(view.cpu().flatten(), src.flatten())

    src = _values(*view.shape)
    view.copy_(src)
    assert torch.equal(base.cpu().flatten(), src.flatten())


@pytest.mark.parametrize("through", ["view", "base"])
def test_a_compiled_store_through_one_shape_writes_only_its_block(overlay, through):
    base, view = overlay
    base.copy_(_values(*base.shape))
    target = view if through == "view" else base
    before = target.cpu()
    block = 3
    # A manager block is kv_heads * pages_per_block rows of block_size tokens in either shape.
    rows_per_block = target.shape[1] * target.shape[0] // NUM_BLOCKS
    first = block * rows_per_block * target.shape[2]
    index = torch.arange(first, first + rows_per_block * target.shape[2], dtype=torch.int32)
    rows = _values(index.numel(), target.shape[3])

    _store(target.view(-1, target.shape[3]), index.to(target.device), rows.to(target.device))

    expected = before.clone()
    expected.view(-1, target.shape[3])[index.long()] = rows
    assert torch.equal(target.cpu(), expected)
    other = base if through == "view" else view
    assert torch.equal(other.cpu().flatten(), expected.flatten())


def test_pages_of_another_size_cannot_be_viewed(overlay):
    base, _ = overlay
    with pytest.raises(ValueError, match="cannot be viewed"):
        SpyreAttentionImpl.view_pages(SpyrePagedKVCache(base, base), NUM_BLOCKS, GLOBAL)
