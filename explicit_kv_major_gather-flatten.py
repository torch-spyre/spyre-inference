#!/usr/bin/env python3
# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Repro: gather from explicit KV-major K-cache device layout."""

import torch
import torch_spyre  # noqa: F401
from torch_spyre._C import SpyreTensorLayout, get_device_dtype

NUM_Q_TOKENS = 4
NUM_KV_BLOCKS = 8
NUM_BLOCKS_TOTAL = 257
BLOCK_SIZE = 128
NUM_KV_HEADS = 8
NUM_Q_HEADS = 32
NUM_QUERIES_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS
NUM_STAGING_TOKENS = 513
HEAD_SIZE = 128

def first_paged_attention_bmm(
    query_staging, row_index_tensor, k_pages, page_index
):
    q = query_staging.index_select(0, row_index_tensor).view(
        -1, NUM_QUERIES_PER_KV, HEAD_SIZE
    )
    k = k_pages.index_select(0, page_index).view(-1, BLOCK_SIZE, HEAD_SIZE)
    return torch.matmul(q, k.transpose(-2, -1))


def main():
    torch.manual_seed(0)
    page_ids = torch.randint(
        NUM_BLOCKS_TOTAL, (NUM_Q_TOKENS, NUM_KV_BLOCKS), dtype=torch.int32
    )
    page_index = (
        torch.arange(NUM_KV_HEADS, dtype=torch.int32)[:, None, None]
        * NUM_BLOCKS_TOTAL
        + page_ids[None, :, :]
    ).reshape(-1)
    k_cache = torch.randn(
        NUM_BLOCKS_TOTAL,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_SIZE,
        dtype=torch.float16,
    )
    k_host = k_cache.permute(2, 0, 1, 3).reshape(
        NUM_KV_HEADS * NUM_BLOCKS_TOTAL, BLOCK_SIZE, HEAD_SIZE
    ).contiguous()
    k_layout = SpyreTensorLayout(
        [NUM_KV_HEADS * NUM_BLOCKS_TOTAL, BLOCK_SIZE, HEAD_SIZE // 64, 64],
        [BLOCK_SIZE * HEAD_SIZE, HEAD_SIZE, 64, 1],
        get_device_dtype(k_host.dtype),
    )

    query_staging = torch.randn(
        NUM_KV_HEADS * NUM_STAGING_TOKENS,
        NUM_QUERIES_PER_KV,
        HEAD_SIZE,
        dtype=torch.float16,
    )
    query_layout = SpyreTensorLayout(
        [
            NUM_KV_HEADS * NUM_STAGING_TOKENS,
            NUM_QUERIES_PER_KV,
            HEAD_SIZE // 64,
            64,
        ],
        [NUM_QUERIES_PER_KV * HEAD_SIZE, HEAD_SIZE, 64, 1],
        get_device_dtype(query_staging.dtype),
    )
    row_index_tensor = (
        torch.arange(NUM_KV_HEADS, dtype=torch.int32)[:, None, None]
        * NUM_STAGING_TOKENS
        + torch.arange(NUM_Q_TOKENS, dtype=torch.int32)[None, :, None]
    ).expand(-1, -1, NUM_KV_BLOCKS).reshape(-1)
    expected_q = query_staging.index_select(0, row_index_tensor).view(
        -1, NUM_QUERIES_PER_KV, HEAD_SIZE
    )
    expected_k = k_host.index_select(0, page_index).view(-1, BLOCK_SIZE, HEAD_SIZE)
    expected = torch.matmul(expected_q, expected_k.transpose(-2, -1))
    got = torch.compile(first_paged_attention_bmm, dynamic=False)(
        query_staging.to("spyre", device_layout=query_layout),
        row_index_tensor.to("spyre"),
        k_host.to("spyre", device_layout=k_layout),
        page_index.to("spyre"),
    )
    torch.testing.assert_close(got.cpu(), expected, rtol=0.1, atol=0.1)
    print(f"ok: scores={tuple(got.shape)}")


if __name__ == "__main__":
    main()
