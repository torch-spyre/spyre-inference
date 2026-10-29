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

"""Repro: KV-major paged attention with exact flattened softmax."""

import argparse

import torch
import torch.nn.functional as F
import torch_spyre  # noqa: F401
from torch.profiler import ProfilerActivity, profile
from torch_spyre._C import SpyreTensorLayout, get_device_dtype

NUM_Q_TOKENS = 8
NUM_KV_BLOCKS = 8
NUM_BLOCKS_TOTAL = 257
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
    page_index,
    mask_by_entry,
    scale,
):
    entries = NUM_KV_HEADS * NUM_Q_TOKENS
    q = query_staging.index_select(0, row_index_tensor).view(entries, NUM_QUERIES_PER_KV, HEAD_SIZE)
    k = k_pages.index_select(0, page_index).view(entries, KV_LEN, HEAD_SIZE)
    v = v_pages.index_select(0, page_index).view(entries, KV_LEN, HEAD_SIZE)
    mask = mask_by_entry.view(entries, KV_LEN)
    # Materialize K's transposed layout before the score BMM so its restickified
    # form keeps a direct LX ownership mapping from gather through consumption.
    k = k.transpose(-2, -1) * 1.0
    scores = torch.matmul(q, k) * scale
    probs = torch.softmax(scores + mask.unsqueeze(1), dim=-1)
    page_probs = (
        probs.view(entries, NUM_QUERIES_PER_KV, NUM_KV_BLOCKS, BLOCK_SIZE)
        .permute(0, 2, 1, 3)
        .reshape(entries * NUM_KV_BLOCKS, NUM_QUERIES_PER_KV, BLOCK_SIZE)
    )
    page_v = v.view(entries * NUM_KV_BLOCKS, BLOCK_SIZE, HEAD_SIZE)
    out = (
        torch.matmul(page_probs, page_v)
        .view(entries, NUM_KV_BLOCKS, NUM_QUERIES_PER_KV, HEAD_SIZE)
        .sum(dim=1)
    )
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
    page_index,
    mask_by_entry,
    scale,
):
    entries = NUM_KV_HEADS * NUM_Q_TOKENS
    q = query_staging.index_select(0, row_index_tensor).view(entries, NUM_QUERIES_PER_KV, HEAD_SIZE)
    k = k_pages.index_select(0, page_index).view(entries, KV_LEN, HEAD_SIZE)
    v = v_pages.index_select(0, page_index).view(entries, KV_LEN, HEAD_SIZE)
    mask = mask_by_entry.view(entries, KV_LEN)
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
    global NUM_Q_TOKENS, NUM_KV_BLOCKS, KV_LEN, PROFILE_REPS

    parser = argparse.ArgumentParser()
    parser.add_argument("--qlen", type=int, default=NUM_Q_TOKENS)
    parser.add_argument("--kvlen", type=int, default=KV_LEN)
    parser.add_argument("--profile-reps", type=int, default=PROFILE_REPS)
    args = parser.parse_args()
    if args.qlen <= 0 or args.qlen & (args.qlen - 1):
        parser.error("--qlen must be a positive power of two")
    if args.kvlen < BLOCK_SIZE or args.kvlen & (args.kvlen - 1):
        parser.error(f"--kvlen must be a power of two >= {BLOCK_SIZE}")

    NUM_Q_TOKENS = args.qlen
    NUM_KV_BLOCKS = args.kvlen // BLOCK_SIZE
    KV_LEN = args.kvlen
    PROFILE_REPS = args.profile_reps
    torch.manual_seed(0)
    page_ids = torch.randint(NUM_BLOCKS_TOTAL, (NUM_Q_TOKENS, NUM_KV_BLOCKS), dtype=torch.int32)
    page_index = (
        torch.arange(NUM_KV_HEADS, dtype=torch.int32)[:, None, None] * NUM_BLOCKS_TOTAL
        + page_ids[None, :, :]
    ).reshape(-1)
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
    kv_lens = torch.full((NUM_Q_TOKENS,), KV_LEN, dtype=torch.int32)
    mask_by_query = torch.where(
        torch.arange(KV_LEN)[None, :] < kv_lens[:, None],
        0.0,
        float("-inf"),
    ).to(torch.float16)
    mask_by_entry = (
        mask_by_query.unsqueeze(0)
        .expand(NUM_KV_HEADS, -1, -1)
        .reshape(NUM_KV_HEADS * NUM_Q_TOKENS, NUM_KV_BLOCKS, BLOCK_SIZE)
        .contiguous()
    )
    mask_layout = SpyreTensorLayout(
        [
            NUM_KV_HEADS * NUM_Q_TOKENS,
            NUM_KV_BLOCKS,
            BLOCK_SIZE // 64,
            64,
        ],
        [NUM_KV_BLOCKS * BLOCK_SIZE, BLOCK_SIZE, 64, 1],
        get_device_dtype(mask_by_entry.dtype),
    )
    scale = torch.tensor(HEAD_SIZE**-0.5, dtype=torch.float16)

    expected = paged_attention(
        query_staging,
        row_index_tensor,
        k_host,
        v_host,
        page_index,
        mask_by_entry,
        scale,
    )
    reference = sdpa_reference(
        query_staging,
        row_index_tensor,
        k_host,
        v_host,
        page_index,
        mask_by_entry,
        scale,
    )
    torch.testing.assert_close(expected, reference, rtol=0.1, atol=0.1)
    device_args = (
        query_staging.to("spyre", device_layout=query_layout),
        row_index_tensor.to("spyre"),
        k_host.to("spyre", device_layout=kv_layout),
        v_host.to("spyre", device_layout=kv_layout),
        page_index.to("spyre"),
        mask_by_entry.to("spyre", device_layout=mask_layout),
        scale.to("spyre"),
    )
    compiled_paged_attention = torch.compile(paged_attention, dynamic=False)
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
        f"RESULT qlen={NUM_Q_TOKENS} kvlen={KV_LEN} reps={PROFILE_REPS} "
        f"kernel_us_total={kernel_us:.3f} "
        f"kernel_us_per_run={kernel_us / PROFILE_REPS:.3f}"
    )


if __name__ == "__main__":
    main()
