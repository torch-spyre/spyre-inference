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

"""Repro: KV-major paged attention with chunked online softmax."""

import argparse

import torch
import torch.nn.functional as F
import torch_spyre  # noqa: F401
from torch.profiler import ProfilerActivity, profile
from torch_spyre._C import SpyreTensorLayout, get_device_dtype

from spyre_inference.v1.attention.ops.layout import temporary_chunk_major_page_index_layout
from spyre_inference.v1.attention.ops.tile_loop import USE_FOR_EACH_TILE, walk_tiles

NUM_Q_TOKENS = 8
NUM_KV_BLOCKS = 8
NUM_BLOCKS_TOTAL = 257
BLOCKS_PER_CHUNK = 4
BLOCK_SIZE = 128
NUM_KV_HEADS = 8
NUM_Q_HEADS = 32
NUM_QUERIES_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS
NUM_STAGING_TOKENS = 513
HEAD_SIZE = 128
KV_LEN = NUM_KV_BLOCKS * BLOCK_SIZE
PROFILE_REPS = 10


def paged_attention(
    query_staging,
    row_index_tensor,
    k_pages,
    v_pages,
    chunk_page_ids,
    mask_by_chunk,
    scale,
):
    entries = NUM_KV_HEADS * NUM_Q_TOKENS
    num_chunks = NUM_KV_BLOCKS // BLOCKS_PER_CHUNK
    entries_per_chunk = entries * BLOCKS_PER_CHUNK
    q = query_staging.index_select(0, row_index_tensor).view(entries, NUM_QUERIES_PER_KV, HEAD_SIZE)

    def chunk_body(carry, tiles):
        page_ids, mask_rows, k_pages, v_pages, q = tiles
        page_ids = page_ids[0].reshape(-1)
        mask_rows = mask_rows[0]
        k_chunk = k_pages.index_select(0, page_ids).reshape(
            entries, BLOCKS_PER_CHUNK * BLOCK_SIZE, HEAD_SIZE
        )
        v_chunk = v_pages.index_select(0, page_ids)
        k_chunk = k_chunk.transpose(-2, -1)
        scores = torch.matmul(q, k_chunk) * scale
        mask = mask_rows.permute(1, 0, 2).reshape(entries, BLOCKS_PER_CHUNK * BLOCK_SIZE)
        scores = torch.clamp(scores + mask.unsqueeze(1), min=torch.finfo(scores.dtype).min)
        chunk_max = torch.amax(scores, dim=-1, keepdim=True)
        if carry is None:
            probs = torch.exp(scores - chunk_max)
            chunk_sum = torch.sum(probs, dim=-1, keepdim=True)
            chunk_out = (
                torch.matmul(
                    probs.view(entries, NUM_QUERIES_PER_KV, BLOCKS_PER_CHUNK, BLOCK_SIZE)
                    .permute(0, 2, 1, 3)
                    .reshape(entries_per_chunk, NUM_QUERIES_PER_KV, BLOCK_SIZE),
                    v_chunk,
                )
                .view(entries, BLOCKS_PER_CHUNK, NUM_QUERIES_PER_KV, HEAD_SIZE)
                .sum(dim=1)
            )
            return (chunk_max, chunk_sum, chunk_out), None

        tile_max, tile_sum, tile_out = carry
        rescale = torch.exp(-torch.relu(chunk_max - tile_max))
        new_max = torch.maximum(tile_max, chunk_max)
        probs = torch.exp(scores - new_max)
        chunk_sum = torch.sum(probs, dim=-1, keepdim=True)
        chunk_out = (
            torch.matmul(
                probs.view(entries, NUM_QUERIES_PER_KV, BLOCKS_PER_CHUNK, BLOCK_SIZE)
                .permute(0, 2, 1, 3)
                .reshape(entries_per_chunk, NUM_QUERIES_PER_KV, BLOCK_SIZE),
                v_chunk,
            )
            .view(entries, BLOCKS_PER_CHUNK, NUM_QUERIES_PER_KV, HEAD_SIZE)
            .sum(dim=1)
        )
        return (
            new_max,
            tile_sum * rescale + chunk_sum,
            tile_out * rescale + chunk_out,
        ), None

    if USE_FOR_EACH_TILE:
        state_shape = (entries, NUM_QUERIES_PER_KV, 1)
        out_shape = (entries, NUM_QUERIES_PER_KV, HEAD_SIZE)
        carry, _ = walk_tiles(
            chunk_body,
            (chunk_page_ids, mask_by_chunk, k_pages, v_pages, q),
            dims=(0, 0, None, None, None),
            tile_size=1,
            init=(
                torch.full(state_shape, float("-inf"), dtype=q.dtype, device=q.device),
                torch.zeros(state_shape, dtype=q.dtype, device=q.device),
                torch.zeros(out_shape, dtype=q.dtype, device=q.device),
            ),
        )
    else:
        carry = None
        for chunk in range(num_chunks):
            # A nonzero-offset int32 view reads as chunk zero in an indirect gather.
            index = torch.tensor([chunk], dtype=torch.int32, device=chunk_page_ids.device)
            page_ids = chunk_page_ids.index_select(0, index)
            mask_rows = mask_by_chunk.narrow(0, chunk, 1)
            carry, _ = chunk_body(carry, (page_ids, mask_rows, k_pages, v_pages, q))
    _, tile_sum, tile_out = carry
    out = tile_out / tile_sum
    return (
        out.view(NUM_KV_HEADS, NUM_Q_TOKENS, NUM_QUERIES_PER_KV, HEAD_SIZE)
        .permute(1, 0, 2, 3)
        .reshape(NUM_Q_TOKENS, NUM_Q_HEADS, HEAD_SIZE)
    )


def sdpa_reference(
    query_staging,
    row_index_tensor,
    k_pages,
    v_pages,
    chunk_page_ids,
    mask_by_chunk,
    scale,
):
    entries = NUM_KV_HEADS * NUM_Q_TOKENS
    q = query_staging.index_select(0, row_index_tensor).view(entries, NUM_QUERIES_PER_KV, HEAD_SIZE)
    page_index = (
        chunk_page_ids.reshape(NUM_KV_BLOCKS // BLOCKS_PER_CHUNK, entries, BLOCKS_PER_CHUNK)
        .permute(1, 0, 2)
        .reshape(-1)
    )
    k = k_pages.index_select(0, page_index).view(entries, KV_LEN, HEAD_SIZE)
    v = v_pages.index_select(0, page_index).view(entries, KV_LEN, HEAD_SIZE)
    mask = mask_by_chunk.permute(2, 0, 1, 3).reshape(entries, KV_LEN)
    out = F.scaled_dot_product_attention(
        q.unsqueeze(2),
        k.unsqueeze(1).expand(-1, NUM_QUERIES_PER_KV, -1, -1),
        v.unsqueeze(1).expand(-1, NUM_QUERIES_PER_KV, -1, -1),
        attn_mask=mask[:, None, None, :],
        dropout_p=0.0,
        scale=float(scale),
    ).squeeze(2)
    return (
        out.view(NUM_KV_HEADS, NUM_Q_TOKENS, NUM_QUERIES_PER_KV, HEAD_SIZE)
        .permute(1, 0, 2, 3)
        .reshape(NUM_Q_TOKENS, NUM_Q_HEADS, HEAD_SIZE)
    )


def main():
    global NUM_Q_TOKENS, NUM_KV_BLOCKS, BLOCKS_PER_CHUNK, KV_LEN, PROFILE_REPS

    parser = argparse.ArgumentParser()
    parser.add_argument("--qlen", type=int, default=NUM_Q_TOKENS)
    parser.add_argument("--kvlen", type=int, default=KV_LEN)
    parser.add_argument("--profile-reps", type=int, default=PROFILE_REPS)
    parser.add_argument("--blocks-per-chunk", type=int)
    args = parser.parse_args()
    if args.qlen <= 0 or args.qlen & (args.qlen - 1):
        parser.error("--qlen must be a positive power of two")
    if args.kvlen < BLOCK_SIZE or args.kvlen & (args.kvlen - 1):
        parser.error(f"--kvlen must be a power of two >= {BLOCK_SIZE}")
    num_blocks = args.kvlen // BLOCK_SIZE
    blocks_per_chunk = (
        min(BLOCKS_PER_CHUNK, num_blocks)
        if args.blocks_per_chunk is None
        else args.blocks_per_chunk
    )
    if (
        blocks_per_chunk <= 0
        or blocks_per_chunk & (blocks_per_chunk - 1)
        or num_blocks % blocks_per_chunk
    ):
        parser.error("--blocks-per-chunk must be a power of two dividing the page count")

    NUM_Q_TOKENS = args.qlen
    NUM_KV_BLOCKS = num_blocks
    BLOCKS_PER_CHUNK = blocks_per_chunk
    KV_LEN = args.kvlen
    PROFILE_REPS = args.profile_reps
    torch.manual_seed(0)

    entries = NUM_KV_HEADS * NUM_Q_TOKENS
    num_chunks = NUM_KV_BLOCKS // BLOCKS_PER_CHUNK
    page_ids = torch.randint(NUM_BLOCKS_TOTAL, (NUM_Q_TOKENS, NUM_KV_BLOCKS), dtype=torch.int32)
    page_index = (
        torch.arange(NUM_KV_HEADS, dtype=torch.int32)[:, None, None] * NUM_BLOCKS_TOTAL
        + page_ids[None, :, :]
    ).reshape(entries, NUM_KV_BLOCKS)
    chunk_page_ids = (
        page_index.view(entries, num_chunks, BLOCKS_PER_CHUNK)
        .permute(1, 0, 2)
        .reshape(num_chunks, entries * BLOCKS_PER_CHUNK, 1)
        .contiguous()
    )
    k_cache = torch.randn(
        NUM_BLOCKS_TOTAL,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_SIZE,
        dtype=torch.float16,
    )
    v_cache = torch.randn_like(k_cache)
    k_host = (
        k_cache.permute(2, 0, 1, 3)
        .reshape(NUM_KV_HEADS * NUM_BLOCKS_TOTAL, BLOCK_SIZE, HEAD_SIZE)
        .contiguous()
    )
    v_host = (
        v_cache.permute(2, 0, 1, 3)
        .reshape(NUM_KV_HEADS * NUM_BLOCKS_TOTAL, BLOCK_SIZE, HEAD_SIZE)
        .contiguous()
    )
    kv_layout = SpyreTensorLayout(
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
        torch.arange(NUM_KV_HEADS, dtype=torch.int32)[:, None] * NUM_STAGING_TOKENS
        + torch.arange(NUM_Q_TOKENS, dtype=torch.int32)[None, :]
    ).reshape(-1)
    mask_by_query = torch.zeros(NUM_Q_TOKENS, KV_LEN, dtype=torch.float16)
    mask_by_entry = (
        mask_by_query.unsqueeze(0)
        .expand(NUM_KV_HEADS, -1, -1)
        .reshape(entries, NUM_KV_BLOCKS, BLOCK_SIZE)
        .contiguous()
    )
    mask_by_chunk = (
        mask_by_entry.view(entries, num_chunks, BLOCKS_PER_CHUNK, BLOCK_SIZE)
        .permute(1, 2, 0, 3)
        .contiguous()
    )
    scale = torch.tensor(HEAD_SIZE**-0.5, dtype=torch.float16)

    reference = sdpa_reference(
        query_staging,
        row_index_tensor,
        k_host,
        v_host,
        chunk_page_ids,
        mask_by_chunk,
        scale,
    )
    device_args = (
        query_staging.to("spyre", device_layout=query_layout),
        row_index_tensor.to("spyre"),
        k_host.to("spyre", device_layout=kv_layout),
        v_host.to("spyre", device_layout=kv_layout),
        chunk_page_ids.to(
            "spyre",
            device_layout=temporary_chunk_major_page_index_layout(
                num_chunks, entries * BLOCKS_PER_CHUNK
            ),
        ),
        mask_by_chunk.to("spyre"),
        scale.to("spyre"),
    )
    compiled_paged_attention = torch.compile(
        paged_attention, dynamic=False, fullgraph=USE_FOR_EACH_TILE
    )
    got = compiled_paged_attention(*device_args)
    torch.testing.assert_close(got.cpu(), reference, rtol=0.1, atol=0.1)

    torch.spyre.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        for _ in range(PROFILE_REPS):
            got = compiled_paged_attention(*device_args)
            torch.spyre.synchronize()
    kernel_us = 0.0
    for event in prof.key_averages():
        device_us = getattr(event, "self_device_time_total", 0) or getattr(
            event, "self_cuda_time_total", 0
        )
        if device_us and "Memset" not in event.key and "Memcpy" not in event.key:
            kernel_us += device_us
    print(f"ok: output={tuple(got.shape)} matches CPU SDPA")
    print(
        f"device kernel time: {kernel_us:.3f} us total, "
        f"{kernel_us / PROFILE_REPS:.3f} us/run ({PROFILE_REPS} runs)"
    )
    print(
        f"RESULT chunked_online_softmax qlen={NUM_Q_TOKENS} kvlen={KV_LEN} "
        f"bpc={BLOCKS_PER_CHUNK} chunks={num_chunks} reps={PROFILE_REPS} "
        f"kernel_us_total={kernel_us:.3f} kernel_us_per_run={kernel_us / PROFILE_REPS:.3f}"
    )


if __name__ == "__main__":
    main()
