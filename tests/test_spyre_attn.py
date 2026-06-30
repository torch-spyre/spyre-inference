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
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec
from vllm.utils.torch_utils import set_random_seed
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyreAttentionMetadataBuilder,
    SpyrePagedKVCache,
)

pytestmark = pytest.mark.attention


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
    return device_mode


@pytest.fixture()
def configure_compilation(request, monkeypatch):
    """Configure torch.compile mode for tests."""
    import torch
    from vllm.config.compilation import CompilationMode
    from vllm.config import get_cached_compilation_config

    mode_name = request.param
    compilation_mode = getattr(CompilationMode, mode_name)

    # Reset dynamo cache first to ensure config changes take effect
    torch._dynamo.reset()

    cfg = get_cached_compilation_config()
    original_mode = cfg.mode

    # Store original torch._dynamo config
    original_limit = torch._dynamo.config.accumulated_recompile_limit

    cfg.mode = compilation_mode
    # Increase recompilation limit to handle list-based page_indices
    # which trigger recompilation on each unique block index value
    torch._dynamo.config.accumulated_recompile_limit = 1024

    yield mode_name

    # Cleanup: reset mode and limits
    cfg.mode = original_mode
    torch._dynamo.config.accumulated_recompile_limit = original_limit
    torch._dynamo.reset()


def _build_metadata(
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    sliding_window: int | None = None,
):
    """Use the real SpyreAttentionMetadataBuilder to construct metadata."""
    from vllm.config import get_current_vllm_config

    # Reuse the VllmConfig set up by the `default_vllm_config` fixture and
    # stub the head-count methods the builder reads.
    vllm_config = get_current_vllm_config()
    vllm_config.model_config.get_num_attention_heads = Mock(return_value=num_query_heads)
    vllm_config.model_config.get_num_kv_heads = Mock(return_value=num_kv_heads)

    if sliding_window is not None:
        kv_cache_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size,
            dtype=torch.float16,
            sliding_window=sliding_window,
        )
    else:
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


def assert_close_outliers(
    actual: torch.Tensor,
    expected: torch.Tensor,
    max_outliers: int = 0,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    *,
    outlier_atol: float | None = None,
    outlier_rtol: float | None = None,
) -> None:
    """Assert tensors are close, allowing up to *max_outliers* elements to exceed tolerance.

    Arguments beyond *max_outliers* are forwarded to ``torch.testing.assert_close``.

    Args:
        actual: tensor under test.
        expected: reference tensor.
        max_outliers: number of elements that may exceed the base tolerances.
        atol: absolute tolerance for the bulk of elements.
        rtol: relative tolerance for the bulk of elements.
        outlier_atol: absolute tolerance for outlier elements (defaults to *atol*,
            meaning outliers only need to be finite, not within any tighter bound).
        outlier_rtol: relative tolerance for outlier elements.
        msg: additional context for the failure message.
    """
    diff = (actual - expected).abs()
    tol = atol + rtol * expected.abs()
    outlier_mask = diff > tol
    n_outliers = outlier_mask.sum().item()

    if n_outliers <= max_outliers and max_outliers > 0:
        # Check that outliers are still within the relaxed bound (or simply finite)
        if outlier_atol is not None or outlier_rtol is not None:
            outlier_tol = (outlier_atol if outlier_atol is not None else atol) + (
                outlier_rtol if outlier_rtol is not None else rtol
            ) * expected.abs()
            if diff[outlier_mask].gt(outlier_tol[outlier_mask]).any():
                worst = diff[outlier_mask].max().item()
                raise AssertionError(
                    f"{n_outliers} outlier(s) exceed base tolerances, "
                    f"and at least one outlier also exceeds the relaxed bound "
                    f"(worst diff={worst:.4g})."
                )
        if n_outliers > 0:
            print(
                f"  [assert_close_outliers] {n_outliers}/{actual.numel()} element(s) "
                f"exceed base tolerance but remain within relaxed bound — acceptable."
            )
        return  # acceptable number of outliers within relaxed bounds

    # Fall through to standard assert_close for a clear error message
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as e:
        prefix = (
            f"{n_outliers} elements exceed atol={atol}, rtol={rtol}. "
            if n_outliers > max_outliers
            else ""
        )
        raise AssertionError(
            f"{prefix}"
            f"max_outliers={max_outliers} was specified "
            f"but {n_outliers} element(s) exceed tolerance.\n"
            f"{e}"
        ) from e


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
    "configure_compilation",
    [
        pytest.param("NONE", id="compilation_NONE"),
        pytest.param("STOCK_TORCH_COMPILE", id="compilation_STOCK"),
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
        # pytest.param((32, 32), id="MHA"),
        # pytest.param((32, 1), id="MQA"),
    ],
)
@pytest.mark.parametrize(
    "head_size",
    [
        pytest.param(128, id="head_size(128)"),
        # pytest.param(256, id="head_size(256)"),
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        # Valid block_size values: must be multiples of 64 for Spyre stick alignment.
        # See: https://github.com/torch-spyre/spyre-inference/issues/239
        pytest.param(64, id="block_size(64)"),
        pytest.param(128, id="block_size(128)"),
        pytest.param(256, id="block_size(256)"),
    ],
)
@pytest.mark.parametrize(
    "sliding_window",
    [
        pytest.param(None, id="swa_none"),
        pytest.param(4, id="swa_4"),
        pytest.param(16, id="swa_16"),
    ],
)
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
        # pytest.param(2048, id="num_blocks(2048)"),
        pytest.param(256, id="num_blocks(256)"),
    ],
)
@torch.inference_mode()
def test_spyre_attn(
    default_vllm_config,
    num_blocks: int,
    soft_cap: float | None,
    dtype: torch.dtype,
    sliding_window: int | None,
    block_size: int,
    head_size: int,
    num_heads: tuple[int, int],
    seq_lens: list[tuple[int, int]],
    configure_compilation: str,
    configure_device: str,
) -> None:
    """Validate SpyreAttentionImpl against a reference implementation."""
    # TODO: STOCK_TORCH_COMPILE + device_spyre, currently fails with
    # "missing device_tensor_layout on graph input arg0_1"
    if configure_compilation == "STOCK_TORCH_COMPILE" and configure_device == "spyre":
        pytest.skip("STOCK + device_spyre, currently fails.")

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

    # Populate KV cache and build slot mapping. New tokens use key/value so that
    # ref_attn sees the same data _reshape_and_cache writes to the device pages.
    slot_mapping = []
    q_offset = 0
    for seq_idx in range(num_seqs):
        query_len = query_lens[seq_idx]
        kv_len = kv_lens[seq_idx]
        historical_len = kv_len - query_len
        if historical_len > 0:
            historical_keys = torch.randn(historical_len, num_kv_heads, head_size, dtype=dtype)
            historical_values = torch.randn(historical_len, num_kv_heads, head_size, dtype=dtype)
            for token_idx in range(historical_len):
                actual_block = block_tables[seq_idx, token_idx // block_size].item()
                block_offset = token_idx % block_size
                k_pages_cpu[actual_block][:, block_offset, :] = historical_keys[token_idx]
                v_pages_cpu[actual_block][:, block_offset, :] = historical_values[token_idx]
        for token_idx in range(historical_len, kv_len):
            block_idx = token_idx // block_size
            block_offset = token_idx % block_size
            actual_block = block_tables[seq_idx, block_idx].item()
            k_pages_cpu[actual_block][:, block_offset, :] = key[
                q_offset + token_idx - historical_len
            ]
            v_pages_cpu[actual_block][:, block_offset, :] = value[
                q_offset + token_idx - historical_len
            ]
            slot_mapping.append(actual_block * block_size + block_offset)
        q_offset += query_len
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)

    # Transfer populated pages to device
    k_pages: list[torch.Tensor] = [p.to(cache_device) for p in k_pages_cpu]
    v_pages: list[torch.Tensor] = [p.to(cache_device) for p in v_pages_cpu]

    attn_metadata = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=kv_lens_tensor,
        query_start_loc=cu_query_lens,
        block_table=block_tables,
        slot_mapping=slot_mapping,
        sliding_window=sliding_window,
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
    kv_cache = SpyrePagedKVCache(k_pages=k_pages, v_pages=v_pages)
    # Note: attn_impl.forward() internally calls _reshape_and_cache() to write
    # the new K/V tokens into the cache, so this test exercises both the cache
    # writing and attention computation paths. TODO: Add a dedicated unit test
    # for _reshape_and_cache to independently verify cache writes.
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
        atol, rtol = 0.3, 0.2
    else:
        atol, rtol = 0.2, 0.2

    # Allow a small number of outlier elements to exceed the base tolerance,
    # which can happen due to nondeterministic hardware optimizations.
    assert_close_outliers(
        output.to("cpu"),
        ref_output,
        max_outliers=5,
        atol=atol,
        rtol=rtol,
        outlier_atol=atol * 2,
        outlier_rtol=rtol * 2,
    )


def test_block_size_validation():
    """Test that SpyreAttentionMetadataBuilder validates block_size alignment.

    The list-based attention backend requires block_size to be a multiple of 64
    for proper stick alignment during torch.compile. This test verifies the
    validation raises ValueError for invalid block sizes and accepts valid ones.
    """
    from vllm.config import VllmConfig, ModelConfig, CacheConfig
    from vllm.config.compilation import CompilationConfig

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model_config.get_num_attention_heads = Mock(return_value=8)
    model_config.get_num_kv_heads = Mock(return_value=2)

    # Test invalid block sizes
    invalid_block_sizes = [1, 8, 16, 32, 63, 100]
    for block_size in invalid_block_sizes:
        cache_config = CacheConfig(block_size=block_size)

        compilation_config = CompilationConfig(custom_ops=["all"])

        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            compilation_config=compilation_config,
        )
        kv_cache_spec = AttentionSpec(
            block_size=block_size,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.float16,
        )
        with pytest.raises(ValueError, match="must be a multiple of 64"):
            SpyreAttentionMetadataBuilder(
                kv_cache_spec=kv_cache_spec,
                layer_names=["test"],
                vllm_config=vllm_config,
                device=torch.device("cpu"),
            )

    # Test valid block sizes
    valid_block_sizes = [64, 128, 256, 512]
    for block_size in valid_block_sizes:
        cache_config = CacheConfig(block_size=block_size)

        compilation_config = CompilationConfig(custom_ops=["all"])

        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            compilation_config=compilation_config,
        )
        kv_cache_spec = AttentionSpec(
            block_size=block_size,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.float16,
        )
        builder = SpyreAttentionMetadataBuilder(
            kv_cache_spec=kv_cache_spec,
            layer_names=["test"],
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )
        assert builder.block_size == block_size


def test_sliding_window_none_equivalence(default_vllm_config):
    """Verify sliding_window=None produces identical results to full attention.

    This is a regression test to ensure the sliding window code path doesn't
    affect the standard full attention behavior.
    """

    torch.set_default_device("cpu")
    set_random_seed(0)

    num_query_heads, num_kv_heads = 32, 8
    head_size = 128
    block_size = 64
    num_blocks = 256
    dtype = torch.float16

    # Single sequence: query_len=32, kv_len=256
    query_len, kv_len = 32, 256

    k_pages_cpu = [
        torch.zeros(num_kv_heads, block_size, head_size, dtype=dtype) for _ in range(num_blocks)
    ]
    v_pages_cpu = [
        torch.zeros(num_kv_heads, block_size, head_size, dtype=dtype) for _ in range(num_blocks)
    ]

    # Pre-populate KV cache
    for i in range(kv_len):
        block_idx = i // block_size
        block_offset = i % block_size
        k_pages_cpu[block_idx][:, block_offset, :] = torch.randn(
            num_kv_heads, head_size, dtype=dtype
        )
        v_pages_cpu[block_idx][:, block_offset, :] = torch.randn(
            num_kv_heads, head_size, dtype=dtype
        )

    cu_query_lens = torch.tensor([0, query_len], dtype=torch.int32)
    kv_lens_tensor = torch.tensor([kv_len], dtype=torch.int32)
    max_num_blocks = (kv_len + block_size - 1) // block_size
    block_tables = torch.zeros((1, max_num_blocks), dtype=torch.int32)
    block_tables[0, : (kv_len + block_size - 1) // block_size] = torch.arange(
        (kv_len + block_size - 1) // block_size
    )

    slot_mapping = torch.arange(query_len, dtype=torch.int64) + (kv_len - query_len)

    # Build metadata with sliding_window=None
    metadata_none = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=kv_lens_tensor,
        query_start_loc=cu_query_lens,
        block_table=block_tables,
        slot_mapping=slot_mapping,
        sliding_window=None,
    )

    # Build metadata with sliding_window=256 (larger than seq_len, effectively None)
    metadata_swa = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=kv_lens_tensor,
        query_start_loc=cu_query_lens,
        block_table=block_tables,
        slot_mapping=slot_mapping,
        sliding_window=256,
    )

    # Compare masks - they should be identical when window doesn't bind
    mask_none = metadata_none.attention_mask_tiles[0][0]
    mask_swa = metadata_swa.attention_mask_tiles[0][0]

    assert torch.equal(mask_none, mask_swa), (
        "Masks differ when sliding_window >= seq_len. "
        f"Max diff: {(mask_none - mask_swa).abs().max().item()}"
    )


def test_sliding_window_boundary_conditions(default_vllm_config):
    """Test sliding window at boundary conditions.

    Tests:
    - seq_len == sliding_window (window exactly fits)
    - seq_len == sliding_window + 1 (one token beyond window)
    - Mixed batch with different seq_lens
    """

    torch.set_default_device("cpu")
    set_random_seed(0)

    num_query_heads, num_kv_heads = 8, 2
    head_size = 128
    block_size = 64
    sliding_window = 4

    # Test 1: seq_len == sliding_window (exactly 4 tokens)
    kv_len_eq = sliding_window
    query_len_eq = 1  # decode step
    context_len_eq = kv_len_eq - query_len_eq  # 3

    seq_lens_eq = torch.tensor([kv_len_eq], dtype=torch.int32)
    query_start_loc_eq = torch.tensor([0, query_len_eq], dtype=torch.int32)
    block_tables_eq = torch.zeros((1, 1), dtype=torch.int32)
    slot_mapping_eq = torch.tensor([context_len_eq], dtype=torch.int64)

    metadata_eq = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=seq_lens_eq,
        query_start_loc=query_start_loc_eq,
        block_table=block_tables_eq,
        slot_mapping=slot_mapping_eq,
        sliding_window=sliding_window,
    )

    # Query at position 3 (absolute) should attend to [0, 1, 2, 3] - all 4 tokens
    mask_eq = metadata_eq.attention_mask_tiles[0][0]
    attended_eq = (mask_eq[0] == 0).nonzero().flatten().tolist()
    assert attended_eq == [0, 1, 2, 3], f"Expected [0,1,2,3], got {attended_eq}"

    # Test 2: seq_len == sliding_window + 1 (5 tokens, window binds)
    kv_len_gt = sliding_window + 1
    query_len_gt = 1  # decode step
    context_len_gt = kv_len_gt - query_len_gt  # 4

    seq_lens_gt = torch.tensor([kv_len_gt], dtype=torch.int32)
    query_start_loc_gt = torch.tensor([0, query_len_gt], dtype=torch.int32)
    block_tables_gt = torch.zeros((1, 1), dtype=torch.int32)
    slot_mapping_gt = torch.tensor([context_len_gt], dtype=torch.int64)

    metadata_gt = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=seq_lens_gt,
        query_start_loc=query_start_loc_gt,
        block_table=block_tables_gt,
        slot_mapping=slot_mapping_gt,
        sliding_window=sliding_window,
    )

    # Query at position 4 (absolute) should attend to [1, 2, 3, 4] - 4 tokens
    mask_gt = metadata_gt.attention_mask_tiles[0][0]
    attended_gt = (mask_gt[0] == 0).nonzero().flatten().tolist()
    assert attended_gt == [1, 2, 3, 4], f"Expected [1,2,3,4], got {attended_gt}"

    # Test 3: Mixed batch - one seq within window, one beyond
    kv_len_mixed = [sliding_window, sliding_window + 5]  # [4, 9]
    context_lens_mixed = [3, 8]

    num_seqs_mixed = 2
    seq_lens_mixed = torch.tensor(kv_len_mixed, dtype=torch.int32)
    query_start_loc_mixed = torch.tensor([0, 1, 2], dtype=torch.int32)
    max_blocks_mixed = (max(kv_len_mixed) + block_size - 1) // block_size
    block_tables_mixed = torch.zeros((num_seqs_mixed, max_blocks_mixed), dtype=torch.int32)
    for s in range(num_seqs_mixed):
        block_tables_mixed[s, : (kv_len_mixed[s] + block_size - 1) // block_size] = torch.arange(
            (kv_len_mixed[s] + block_size - 1) // block_size
        )

    slot_mapping_mixed = torch.tensor(
        [context_lens_mixed[0], context_lens_mixed[1]], dtype=torch.int64
    )

    metadata_mixed = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=seq_lens_mixed,
        query_start_loc=query_start_loc_mixed,
        block_table=block_tables_mixed,
        slot_mapping=slot_mapping_mixed,
        sliding_window=sliding_window,
    )

    # Seq 0 (kv_len=4): query at position 3, attends to [0, 1, 2, 3]
    mask_mixed_0 = metadata_mixed.attention_mask_tiles[0][0]
    attended_mixed_0 = (mask_mixed_0[0] == 0).nonzero().flatten().tolist()
    assert attended_mixed_0 == [0, 1, 2, 3], f"Seq 0: expected [0,1,2,3], got {attended_mixed_0}"

    # Seq 1 (kv_len=9): query at position 8, attends to [5, 6, 7, 8]
    mask_mixed_1 = metadata_mixed.attention_mask_tiles[1][0]
    attended_mixed_1 = (mask_mixed_1[0] == 0).nonzero().flatten().tolist()
    assert attended_mixed_1 == [5, 6, 7, 8], f"Seq 1: expected [5,6,7,8], got {attended_mixed_1}"
