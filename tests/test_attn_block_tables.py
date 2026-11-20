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

"""CPU checks for the attention tests' block-table helper."""

import pytest
import torch
from spyre_testing_plugin.attn_helpers import random_block_tables

# 1+2+...+8 = 36 real blocks of 64; every row but the last is padded out to width 8.
KV_LENS, BLOCK_SIZE, WIDTH = [64, 128, 192, 256, 320, 384, 448, 512], 64, 8


@pytest.mark.parametrize("num_blocks", [37, 256])
def test_real_pages_are_distinct_and_padding_avoids_them(num_blocks):
    # A shared page is written in a different order by the kernel and the reference.
    block_tables = random_block_tables(KV_LENS, WIDTH, BLOCK_SIZE, num_blocks)
    owned = torch.cat([row[: i + 1] for i, row in enumerate(block_tables)])
    padding = torch.cat([row[i + 1 :] for i, row in enumerate(block_tables)])
    assert owned.unique().numel() == owned.numel() == 36
    assert not torch.isin(padding, owned).any()


@pytest.mark.parametrize(
    ("kv_lens", "width", "num_blocks", "error"),
    [
        ([128, 128], 2, 4, None),
        (KV_LENS, WIDTH, 36, "plus a padding page"),
        (KV_LENS, WIDTH, 35, "do not fit"),
    ],
    ids=["exact_fit", "no_spare_page", "too_few_pages"],
)
def test_capacity(kv_lens, width, num_blocks, error):
    if error is None:
        block_tables = random_block_tables(kv_lens, width, BLOCK_SIZE, num_blocks)
        assert sorted(block_tables.flatten().tolist()) == list(range(num_blocks))
    else:
        with pytest.raises(AssertionError, match=error):
            random_block_tables(kv_lens, width, BLOCK_SIZE, num_blocks)


def test_draws_the_same_global_stream_as_a_plain_table():
    # Callers draw their K/V right after; a shifted stream would silently change it.
    torch.manual_seed(0)
    torch.randint(0, 256, (len(KV_LENS), WIDTH), dtype=torch.int32)
    expected = torch.randn(4)
    torch.manual_seed(0)
    random_block_tables(KV_LENS, WIDTH, BLOCK_SIZE, num_blocks=256)
    assert torch.equal(torch.randn(4), expected)
