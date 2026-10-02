#!/usr/bin/env python3
"""Profile the head-major batched-decode kernel at the paged-attention repro shape."""

import argparse

import torch
import torch.nn.functional as F
import torch_spyre  # noqa: F401
from torch.profiler import ProfilerActivity, profile
from torch_spyre._C import SpyreTensorLayout, get_device_dtype

from spyre_inference.v1.attention.ops.batched_decode_head_major import (
    batched_decode_head_major_kernel,
)
from spyre_inference.v1.attention.ops.layout import (
    head_major_kv_layout,
    temporary_chunk_major_page_index_layout,
)
from spyre_inference.v1.attention.ops.tile_loop import USE_FOR_EACH_TILE

NUM_SEQS = 8
NUM_BLOCKS = 8
NUM_PAGES_TOTAL = 257
BLOCKS_PER_CHUNK = 4
NUM_CHUNKS = NUM_BLOCKS // BLOCKS_PER_CHUNK
BLOCK_SIZE = 128
NUM_KV_HEADS = 8
NUM_QUERIES_PER_KV = 4
NUM_Q_HEADS = NUM_KV_HEADS * NUM_QUERIES_PER_KV
HEAD_SIZE = 128
PROFILE_REPS = 10


def sdpa_reference(query, k_pages, v_pages, page_ids, scale):
    q = query.view(NUM_SEQS, NUM_KV_HEADS, NUM_QUERIES_PER_KV, HEAD_SIZE).reshape(
        NUM_SEQS, NUM_Q_HEADS, 1, HEAD_SIZE
    )
    k = (
        k_pages[page_ids]
        .permute(0, 2, 1, 3, 4)
        .reshape(NUM_SEQS, NUM_KV_HEADS, NUM_BLOCKS * BLOCK_SIZE, HEAD_SIZE)
    )
    v = (
        v_pages[page_ids]
        .permute(0, 2, 1, 3, 4)
        .reshape(NUM_SEQS, NUM_KV_HEADS, NUM_BLOCKS * BLOCK_SIZE, HEAD_SIZE)
    )
    k = k.repeat_interleave(NUM_QUERIES_PER_KV, dim=1)
    v = v.repeat_interleave(NUM_QUERIES_PER_KV, dim=1)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=float(scale)).squeeze(2)


def main():
    global NUM_SEQS, NUM_BLOCKS, BLOCKS_PER_CHUNK, NUM_CHUNKS, PROFILE_REPS

    parser = argparse.ArgumentParser()
    parser.add_argument("--qlen", type=int, default=NUM_SEQS)
    parser.add_argument("--kvlen", type=int, default=NUM_BLOCKS * BLOCK_SIZE)
    parser.add_argument("--blocks-per-chunk", type=int)
    parser.add_argument("--profile-reps", type=int, default=PROFILE_REPS)
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
    if args.profile_reps <= 0:
        parser.error("--profile-reps must be positive")
    NUM_SEQS = args.qlen
    NUM_BLOCKS = num_blocks
    BLOCKS_PER_CHUNK = blocks_per_chunk
    NUM_CHUNKS = num_blocks // blocks_per_chunk
    PROFILE_REPS = args.profile_reps
    if not USE_FOR_EACH_TILE:
        raise RuntimeError("this benchmark measures the production tiled batched-decode path")

    torch.manual_seed(0)
    page_ids = torch.randint(1, NUM_PAGES_TOTAL, (NUM_SEQS, NUM_BLOCKS), dtype=torch.int32)
    query = torch.randn(NUM_SEQS, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float16)
    k_pages = torch.randn(NUM_PAGES_TOTAL, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE, dtype=torch.float16)
    v_pages = torch.randn_like(k_pages)
    scale = torch.tensor(HEAD_SIZE**-0.5, dtype=torch.float16)

    entries = NUM_SEQS * BLOCKS_PER_CHUNK
    rep_row_ids = torch.arange(NUM_SEQS, dtype=torch.int32).repeat(BLOCKS_PER_CHUNK)
    chunk_page_ids = page_ids.t().reshape(NUM_CHUNKS, entries, 1).contiguous()
    mask_by_chunk = torch.zeros(
        NUM_CHUNKS,
        BLOCKS_PER_CHUNK,
        NUM_SEQS,
        NUM_KV_HEADS,
        NUM_QUERIES_PER_KV,
        BLOCK_SIZE,
        dtype=torch.float16,
    )
    reference = sdpa_reference(query, k_pages, v_pages, page_ids, scale)

    query_layout = SpyreTensorLayout(
        [NUM_SEQS, NUM_Q_HEADS, HEAD_SIZE // 64, 64],
        [NUM_Q_HEADS * HEAD_SIZE, HEAD_SIZE, 64, 1],
        get_device_dtype(query.dtype),
    )
    kv_layout = head_major_kv_layout(
        NUM_PAGES_TOTAL * NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE, k_pages.dtype
    )
    device_args = (
        query.to("spyre", device_layout=query_layout),
        rep_row_ids.to("spyre"),
        k_pages.to("spyre", device_layout=kv_layout),
        v_pages.to("spyre", device_layout=kv_layout),
        chunk_page_ids.to(
            "spyre",
            device_layout=temporary_chunk_major_page_index_layout(NUM_CHUNKS, entries),
        ),
        mask_by_chunk.to("spyre"),
        scale.to("spyre"),
        NUM_SEQS,
        BLOCKS_PER_CHUNK,
        NUM_KV_HEADS,
        NUM_QUERIES_PER_KV,
        BLOCK_SIZE,
        HEAD_SIZE,
    )
    compiled = torch.compile(
        batched_decode_head_major_kernel, dynamic=False, fullgraph=USE_FOR_EACH_TILE
    )
    got = compiled(*device_args)
    torch.testing.assert_close(got.cpu(), reference, rtol=0.1, atol=0.1)

    torch.spyre.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        for _ in range(PROFILE_REPS):
            got = compiled(*device_args)
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
        "RESULT head_major_batched_decode "
        f"seqs={NUM_SEQS} blocks={NUM_BLOCKS} bpc={BLOCKS_PER_CHUNK} "
        f"chunks={NUM_CHUNKS} reps={PROFILE_REPS} "
        f"kernel_us_total={kernel_us:.3f} kernel_us_per_run={kernel_us / PROFILE_REPS:.3f}"
    )


if __name__ == "__main__":
    main()
