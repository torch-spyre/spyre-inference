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

from unittest.mock import Mock

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.utils.torch_utils import set_random_seed
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyreAttentionMetadataBuilder,
)
from spyre_inference import envs


def _spyre_available() -> bool:
    try:
        torch.randn(1, device=torch.device("spyre"))
        return True
    except Exception:
        return False


@pytest.fixture()
def configure_device(request, monkeypatch):
    """Configure overwrite_f and cache device based on the device_mode parameter.

    The spyre card check is done lazily here (not at import time) to avoid
    claiming the device before subprocess-based tests have a chance to run.
    """

    device_mode = request.param
    if device_mode == "spyre" and not _spyre_available():
        pytest.skip("Spyre device not available")
    envs.clear_env_cache()
    return device_mode


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
    key_cache: list[torch.Tensor],
    value_cache: list[torch.Tensor],
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    block_size: int,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
) -> torch.Tensor:
    """Reference implementation of attention for validation."""
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q = q * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        # Gather from page lists
        k_blocks = [key_cache[idx] for idx in block_indices]
        v_blocks = [value_cache[idx] for idx in block_indices]
        # Each block: [num_kv_heads, block_size, head_size]
        # cat along block_size dim → [num_kv_heads, total_tokens, head_size]
        k = torch.cat(k_blocks, dim=1).transpose(0, 1)[:kv_len]  # [kv_len, num_kv_heads, head_size]
        v = torch.cat(v_blocks, dim=1).transpose(0, 1)[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1)
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize(
    "configure_device",
    [
        pytest.param("cpu", id="device_cpu"),
        pytest.param("spyre", id="device_spyre"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([(1, 1024)], id="decode(q=1,kv=1024)"),
        pytest.param([(1, 256)], id="decode(q=1,kv=256)"),
        pytest.param([(32, 256)], id="prefill(q=32,kv=256)"),
        pytest.param([(64, 512)], id="prefill(q=64,kv=512)"),
        pytest.param([(100, 512)], id="prefill(q=100,kv=512)"),
        pytest.param([(1, 256), (1, 512)], id="batch_decode(2seqs)"),
        pytest.param([(32, 256), (64, 512)], id="batch_prefill(2seqs)"),
        pytest.param([(1, 256), (32, 256)], id="mixed(decode+prefill)"),
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [
        pytest.param((32, 8), id="GQA"),
        pytest.param((32, 32), id="MHA"),
        pytest.param((32, 1), id="MQA"),
    ],
)
@pytest.mark.parametrize(
    "head_size",
    [
        pytest.param(128, id="head_size(128)"),
        pytest.param(256, id="head_size(256)"),
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        pytest.param(16, id="block_size(16)"),
    ],
)
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="dtype(fp16)"),
    ],
)
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize(
    "num_blocks",
    [
        pytest.param(2048, id="num_blocks(2048)"),
        pytest.param(32768, id="num_blocks(32768)"),
    ],
)
@torch.inference_mode()
def test_spyre_attn(
    default_vllm_config,
    configure_device: str,
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
) -> None:
    """Validate SpyreAttentionImpl against a reference implementation."""
    num_query_heads, num_kv_heads = num_heads
    # only for preparation, actual device is set via `configure_device`
    torch.set_default_device("cpu")
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=dtype)

    cache_device = torch.device(configure_device)
    # list based creation here, update once this changes
    k_pages_cpu: list[torch.Tensor] = [
        torch.zeros(num_kv_heads, block_size, head_size, dtype=dtype) for _ in range(num_blocks)
    ]
    v_pages_cpu: list[torch.Tensor] = [
        torch.zeros(num_kv_heads, block_size, head_size, dtype=dtype) for _ in range(num_blocks)
    ]

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    # Pre-populate KV cache with historical context (on CPU)
    for seq_idx in range(num_seqs):
        query_len = query_lens[seq_idx]
        kv_len = kv_lens[seq_idx]
        historical_len = kv_len - query_len
        if historical_len > 0:
            historical_keys = torch.randn(historical_len, num_kv_heads, head_size, dtype=dtype)
            historical_values = torch.randn(historical_len, num_kv_heads, head_size, dtype=dtype)
            for token_idx in range(historical_len):
                block_idx = token_idx // block_size
                block_offset = token_idx % block_size
                actual_block = block_tables[seq_idx, block_idx].item()
                k_pages_cpu[actual_block][:, block_offset, :] = historical_keys[token_idx]
                v_pages_cpu[actual_block][:, block_offset, :] = historical_values[token_idx]

    # Transfer populated pages to device
    k_pages: list[torch.Tensor] = [p.to(cache_device) for p in k_pages_cpu]
    v_pages: list[torch.Tensor] = [p.to(cache_device) for p in v_pages_cpu]

    # Create slot mapping for new query tokens
    slot_mapping = []
    for seq_idx in range(num_seqs):
        query_len = query_lens[seq_idx]
        kv_len = kv_lens[seq_idx]
        for token_idx in range(query_len):
            pos = kv_len - query_len + token_idx
            actual_block = block_tables[seq_idx, pos // block_size].item()
            slot_mapping.append(actual_block * block_size + pos % block_size)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)

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

    attn_impl = SpyreAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
        logits_soft_cap=soft_cap,
    )

    output = torch.empty_like(query).to(cache_device)
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
    
    ref_output = ref_attn(
        query=query,
        key_cache=k_pages_cpu,
        value_cache=v_pages_cpu,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        block_size=block_size,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    if max(query_lens) >= 32:
        # TODO: what values are sensible here?
        atol, rtol = 0.5, 5.0
    else:
        atol, rtol = 0.2, 0.2

    if configure_device == "spyre":
        torch.testing.assert_close(output.to("cpu"), ref_output, atol=atol, rtol=rtol)
    else:
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)
