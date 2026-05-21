#!/usr/bin/env python3
"""CPU-only validation of SpyreAttentionImpl using the real backend code.

Run without Spyre hardware or torch-spyre installed:
    SPYRE_USE_OVERWRITE_F=0 python tests/test_spyre_attn_cpu.py

This exercises the full SpyreAttentionImpl.forward path (metadata builder,
reshape_and_cache, online softmax) on CPU, comparing against a reference
standard-softmax implementation.
"""

import os
os.environ["SPYRE_USE_OVERWRITE_F"] = "0"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock

import torch
import torch.nn.functional as F

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyreAttentionMetadataBuilder,
)
from spyre_inference import envs

envs.clear_env_cache()


def _build_metadata(
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Use the real SpyreAttentionMetadataBuilder to construct metadata."""
    vllm_config = Mock()
    vllm_config.model_config.get_num_attention_heads.return_value = num_query_heads
    vllm_config.model_config.get_num_kv_heads.return_value = num_kv_heads

    kv_cache_spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
    )

    builder = SpyreAttentionMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layers.0.self_attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    max_query_len = int((query_start_loc[1:] - query_start_loc[:-1]).max().item())
    max_seq_len = int(seq_lens.max().item())
    num_actual_tokens = int(query_start_loc[-1].item())

    common_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        num_reqs=len(seq_lens),
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=True,
    )

    return builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_metadata,
    )


def ref_attn(
    query: torch.Tensor,
    k_pages: list[torch.Tensor],
    v_pages: list[torch.Tensor],
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    """Standard softmax reference (materializes full attention matrix)."""
    num_seqs = len(query_lens)
    outputs = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len] * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        k_blocks = [k_pages[int(block_tables[i, b])] for b in range(num_kv_blocks)]
        v_blocks = [v_pages[int(block_tables[i, b])] for b in range(num_kv_blocks)]

        k = torch.cat(k_blocks, dim=1).transpose(0, 1)[:kv_len]
        v = torch.cat(v_blocks, dim=1).transpose(0, 1)[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(torch.ones(query_len, kv_len), diagonal=kv_len - query_len + 1).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def run_test(
    seq_lens_list: list[tuple[int, int]],
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
):
    """Run a single test case using SpyreAttentionImpl.forward."""
    torch.manual_seed(42)
    scale = head_size**-0.5
    dtype = torch.float16

    num_seqs = len(seq_lens_list)
    query_lens = [x[0] for x in seq_lens_list]
    kv_lens = [x[1] for x in seq_lens_list]
    max_kv_len = max(kv_lens)

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=dtype)

    k_pages: list[torch.Tensor] = [
        torch.zeros(num_kv_heads, block_size, head_size, dtype=dtype)
        for _ in range(num_blocks)
    ]
    v_pages: list[torch.Tensor] = [
        torch.zeros(num_kv_heads, block_size, head_size, dtype=dtype)
        for _ in range(num_blocks)
    ]

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)

    # Pre-populate historical context
    for seq_idx in range(num_seqs):
        query_len = query_lens[seq_idx]
        kv_len = kv_lens[seq_idx]
        historical_len = kv_len - query_len
        if historical_len > 0:
            hist_k = torch.randn(historical_len, num_kv_heads, head_size, dtype=dtype)
            hist_v = torch.randn(historical_len, num_kv_heads, head_size, dtype=dtype)
            for t in range(historical_len):
                b_idx = block_tables[seq_idx, t // block_size].item()
                b_off = t % block_size
                k_pages[b_idx][:, b_off, :] = hist_k[t]
                v_pages[b_idx][:, b_off, :] = hist_v[t]

    # Slot mapping
    slot_mapping = []
    for seq_idx in range(num_seqs):
        query_len = query_lens[seq_idx]
        kv_len = kv_lens[seq_idx]
        for t in range(query_len):
            pos = kv_len - query_len + t
            b_idx = block_tables[seq_idx, pos // block_size].item()
            slot_mapping.append(b_idx * block_size + pos % block_size)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    # Build metadata using the real builder
    attn_metadata = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=kv_lens_tensor,
        query_start_loc=cu_query_lens,
        block_table=block_tables,
        slot_mapping=slot_mapping,
    )

    # Run through SpyreAttentionImpl.forward
    attn_impl = SpyreAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
    )

    output = torch.empty_like(query)
    kv_cache = (k_pages, v_pages)
    attn_impl.forward(
        layer=None,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
    )

    # Reference
    ref = ref_attn(
        query=query,
        k_pages=k_pages,
        v_pages=v_pages,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        block_size=block_size,
        scale=scale,
    )

    max_diff = (output - ref).abs().max().item()
    mean_diff = (output - ref).abs().mean().item()

    atol = 0.3 if max(query_lens) >= 32 else 0.2
    passed = max_diff < atol

    status = "PASS" if passed else "FAIL"
    seqs_str = ",".join(f"q={q}/kv={kv}" for q, kv in seq_lens_list)
    label = f"[{seqs_str}] heads={num_query_heads}/{num_kv_heads},d={head_size}"
    print(f"  [{status}] {label}  max_diff={max_diff:.4e} mean_diff={mean_diff:.4e}")
    return passed


if __name__ == "__main__":
    print("SpyreAttentionImpl — CPU Validation (full forward path)")
    print("=" * 70)

    test_cases = [
        # Single sequence cases
        ([(1, 256)], 32, 8, 128, 16, 64),
        ([(1, 1024)], 32, 8, 128, 16, 128),
        ([(1, 256)], 32, 32, 128, 16, 64),
        ([(1, 256)], 32, 1, 128, 16, 64),
        ([(32, 256)], 32, 8, 128, 16, 64),
        ([(64, 512)], 32, 8, 128, 16, 64),
        ([(100, 512)], 32, 8, 128, 16, 64),
        ([(1, 256)], 32, 8, 256, 16, 64),
        # Multi-sequence (varlen) cases
        ([(1, 256), (1, 512)], 32, 8, 128, 16, 64),
        ([(32, 256), (64, 512)], 32, 8, 128, 16, 64),
        ([(1, 256), (32, 256)], 32, 8, 128, 16, 64),
    ]

    all_passed = True
    for args in test_cases:
        if not run_test(*args):
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
        sys.exit(1)
