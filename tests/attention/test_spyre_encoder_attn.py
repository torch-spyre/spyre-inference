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
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec

from spyre_inference.v1.attention.backends import spyre_encoder_attn as encoder_attn
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionMetadataBuilder,
    SpyrePagedKVCache,
)
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    SpyreEncoderAttentionImpl,
    _content_query_lens,
    build_attention_mask,
    dummy_pack_row,
    gather_pack,
    gather_unpack,
    host_pack_indices,
    host_scatter_pack_dest,
    scatter_pack,
)

# extra `encoder_attention` mark so CI can split this into its own job
# because these tests are pretty slow.
pytestmark = [pytest.mark.attention, pytest.mark.encoder_attention]


@pytest.fixture()
def configure_device(request, monkeypatch):
    """Configure overwrite_f and cache device based on the device_mode parameter.

    The spyre card check is done lazily here (not at import time) to avoid
    claiming the device before subprocess-based tests have a chance to run.
    """

    device_mode = request.param
    if device_mode == "spyre" and not spyre_available():
        pytest.skip("Spyre device not available")
    return device_mode


@pytest.fixture()
def configure_compilation(request, monkeypatch):
    """Configure torch.compile mode for tests."""
    import torch
    from vllm.config import get_cached_compilation_config
    from vllm.config.compilation import CompilationMode

    mode_name = request.param
    compilation_mode = getattr(CompilationMode, mode_name)

    # Reset dynamo cache first to ensure config changes take effect
    torch._dynamo.reset()

    cfg = get_cached_compilation_config()
    original_mode = cfg.mode

    # Store original torch._dynamo config
    original_limit = torch._dynamo.config.accumulated_recompile_limit

    cfg.mode = compilation_mode
    # Increase recompilation limit: the page-attention kernel is specialized
    # (and so recompiled) per unique (num_blocks, padded_query_len)
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
):
    """Use the real SpyreAttentionMetadataBuilder to construct metadata."""
    from vllm.config import get_current_vllm_config

    # Reuse the VllmConfig set up by the `default_vllm_config` fixture and
    # stub the head-count methods the builder reads.
    vllm_config = get_current_vllm_config()
    vllm_config.model_config.get_num_attention_heads = Mock(return_value=num_query_heads)
    vllm_config.model_config.get_num_kv_heads = Mock(return_value=num_kv_heads)
    # The builder asserts these agree, and derives its padding buckets from the
    # cache_config one, so a test block_size has to be set in both places.
    vllm_config.cache_config.block_size = block_size

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
        causal=False,
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


def ref_encoder_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lens: list[int],
    scale: float,
) -> torch.Tensor:
    """Reference bidirectional self-attention (no causal mask, no KV cache)."""
    num_seqs = len(query_lens)
    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        q = query[start_idx : start_idx + query_len]
        q = q * scale
        k = key[start_idx : start_idx + query_len]
        v = value[start_idx : start_idx + query_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def _loop_attention_mask(
    num_seqs: int,
    aligned_len: int,
    query_lens: list[int],
    kv_lens: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Host slice-write reference for ``build_attention_mask``."""
    neg_inf = torch.finfo(dtype).min
    mask = torch.full((num_seqs, 1, aligned_len, aligned_len), neg_inf, dtype=dtype)
    for s in range(num_seqs):
        q_len = query_lens[s]
        kv_len = min(q_len, kv_lens[s])
        mask[s, 0, :q_len, :kv_len] = 0.0
    return mask


@pytest.mark.parametrize(
    "configure_device",
    [
        pytest.param("cpu", id="device_cpu"),
        pytest.param("spyre", id="device_spyre"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "query_lens,kv_lens,aligned_len",
    [
        pytest.param([32], [32], 64, id="single_32"),
        pytest.param([9, 70, 5], [9, 70, 5], 128, id="batch_unaligned"),
        pytest.param([16, 8], [8, 8], 64, id="kv_shorter_than_q"),
    ],
)
@torch.inference_mode()
def test_build_attention_mask_matches_loop(
    configure_device: str,
    query_lens: list[int],
    kv_lens: list[int],
    aligned_len: int,
) -> None:
    """Vectorized host mask must match the slice-write reference.

    Spyre ``convert`` of fp16 ``finfo.min`` (-65504) is not bit-exact (off by
    32). Attend slots must stay 0; pad slots must stay hugely negative.
    """
    dtype = torch.float16
    device = torch.device(configure_device)
    ref = _loop_attention_mask(len(query_lens), aligned_len, query_lens, kv_lens, dtype)
    got = build_attention_mask(
        len(query_lens),
        aligned_len,
        query_lens,
        kv_lens,
        dtype=dtype,
        device=device,
    ).cpu()
    attend = ref == 0
    torch.testing.assert_close(got[attend], ref[attend], atol=0, rtol=0)
    if configure_device == "cpu":
        torch.testing.assert_close(got, ref, atol=0, rtol=0)
        return
    assert bool((got[~attend] < -1e4).all()), "pad slots must stay a large negative"


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
        pytest.param([(32, 32)], id="prefill(q=32,kv=32)"),
        pytest.param([(64, 64)], id="prefill(q=64,kv=64)"),
        pytest.param([(100, 100)], id="prefill(q=100,kv=100)"),
        pytest.param([(16, 16), (32, 32)], id="batch_prefill(2seqs)"),
        pytest.param([(9, 9), (70, 70), (5, 5)], id="batch_unaligned(3seqs)"),
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [
        pytest.param((16, 4), id="GQA"),
        # pytest.param((16, 16), id="MHA"),
    ],
)
@pytest.mark.parametrize(
    "head_size",
    [
        # Product encoder models (Granite/E5/RoBERTa) use D=64; MiniLM uses 32.
        pytest.param(64, id="head_size(64)"),
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        # Valid block_size values: must be multiples of 64 for Spyre stick alignment.
        pytest.param(64, id="block_size(64)"),
        pytest.param(128, id="block_size(128)"),
        pytest.param(256, id="block_size(256)"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="dtype(fp16)"),
    ],
)
@torch.inference_mode()
def test_spyre_encoder_attn(
    default_vllm_config,
    dtype: torch.dtype,
    block_size: int,
    head_size: int,
    num_heads: tuple[int, int],
    seq_lens: list[tuple[int, int]],
    configure_compilation: str,
    configure_device: str,
) -> None:
    """Validate SpyreEncoderAttentionImpl against a bidirectional reference."""
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
    assert query_lens == kv_lens
    assert num_query_heads % num_kv_heads == 0
    scale = head_size**-0.5

    total_tokens = sum(query_lens)
    query = torch.randn(total_tokens, num_query_heads, head_size, dtype=dtype)
    key = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_query_len = max(query_lens)
    max_num_blocks_per_seq = (max_query_len + block_size - 1) // block_size
    block_table = torch.zeros(num_seqs, max_num_blocks_per_seq, dtype=torch.int32)
    slot_mapping = torch.arange(total_tokens, dtype=torch.int64)

    attn_metadata = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=kv_lens_tensor,
        query_start_loc=cu_query_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
    )

    attn_impl = SpyreEncoderAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
    )

    cache_device = torch.device(configure_device)
    output = torch.empty_like(query).to(cache_device)
    kv_cache = SpyrePagedKVCache(k_pages=torch.empty(0), v_pages=torch.empty(0))
    attn_impl.forward(
        layer=None,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
    )

    ref_output = ref_encoder_attn(
        query=query,
        key=key,
        value=value,
        query_lens=query_lens,
        scale=scale,
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


@torch.inference_mode()
def test_encoder_pack_cache_reused_across_layers(default_vllm_config) -> None:
    """Second layer must reuse the step's scatter dest tensors, not rebuild + H2D them."""
    torch.set_default_device("cpu")
    set_random_seed(0)
    query_lens = [32]
    total_tokens = 32
    num_heads, num_kv_heads, head_size, block_size = 16, 4, 64, 64
    dtype = torch.float16
    query = torch.randn(total_tokens, num_heads, head_size, dtype=dtype)
    key = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)
    cu = torch.tensor([0, 32], dtype=torch.int32)
    attn_metadata = _build_metadata(
        num_query_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=torch.tensor(query_lens, dtype=torch.int32),
        query_start_loc=cu,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        slot_mapping=torch.arange(total_tokens, dtype=torch.int64),
    )
    impl = SpyreEncoderAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
    )
    kv_cache = SpyrePagedKVCache(k_pages=torch.empty(0), v_pages=torch.empty(0))
    fwd = dict(
        layer=None,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
    )
    impl.forward(**fwd, output=torch.empty_like(query))
    cached_q = attn_metadata.encoder_q_pack_idx
    cached_kv = attn_metadata.encoder_kv_pack_idx
    cached_unpack = attn_metadata.encoder_unpack_idx
    cached_mask = attn_metadata.encoder_attn_mask
    cached_key_pad = attn_metadata.encoder_key_pad_mask
    assert cached_q is not None
    assert cached_kv is not None
    assert cached_unpack is not None
    assert cached_mask is not None
    assert cached_key_pad is not None
    assert cached_q.shape == (total_tokens,)
    assert cached_q.dtype == torch.int64
    impl.forward(**fwd, output=torch.empty_like(query))
    assert attn_metadata.encoder_q_pack_idx is cached_q
    assert attn_metadata.encoder_kv_pack_idx is cached_kv
    assert attn_metadata.encoder_unpack_idx is cached_unpack
    assert attn_metadata.encoder_attn_mask is cached_mask
    assert attn_metadata.encoder_key_pad_mask is cached_key_pad


def _b1_dense_forward_setup(total_tokens: int = 64):
    """B=1 body already at attention L so the fused SDPA path fires."""
    torch.set_default_device("cpu")
    set_random_seed(0)
    num_heads, num_kv_heads, head_size, block_size = 16, 4, 64, 64
    dtype = torch.float16
    query = torch.randn(total_tokens, num_heads, head_size, dtype=dtype)
    key = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)
    attn_metadata = _build_metadata(
        num_query_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=torch.tensor([total_tokens], dtype=torch.int32),
        query_start_loc=torch.tensor([0, total_tokens], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        slot_mapping=torch.arange(total_tokens, dtype=torch.int64),
    )
    impl = SpyreEncoderAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
    )
    kv_cache = SpyrePagedKVCache(k_pages=torch.empty(0), v_pages=torch.empty(0))
    fwd = dict(
        layer=None,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
    )
    return impl, fwd, query, attn_metadata


@torch.inference_mode()
def test_b1_dense_forward_skips_scatter_pack(monkeypatch, default_vllm_config) -> None:
    """Serve B=1 T==L must not eager-pack Q/K/V (that was the D2D clone tax)."""
    calls = {"n": 0}
    real = encoder_attn.scatter_pack

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(encoder_attn, "scatter_pack", counting)
    impl, fwd, query, _meta = _b1_dense_forward_setup()
    impl.forward(**fwd, output=torch.empty_like(query))
    assert calls["n"] == 0


@torch.inference_mode()
def test_b1_dense_pack_dest_stays_on_host(monkeypatch, default_vllm_config) -> None:
    """Fused B=1 path does not H2D dest, unpack, or a zeros L×L mask."""
    dest_calls = {"n": 0}
    idx_calls = {"n": 0}
    real_dest = encoder_attn._dest_for_device
    real_idx = encoder_attn._indices_for_device

    def count_dest(*args, **kwargs):
        dest_calls["n"] += 1
        return real_dest(*args, **kwargs)

    def count_idx(*args, **kwargs):
        idx_calls["n"] += 1
        return real_idx(*args, **kwargs)

    monkeypatch.setattr(encoder_attn, "_dest_for_device", count_dest)
    monkeypatch.setattr(encoder_attn, "_indices_for_device", count_idx)
    pad_calls = {"n": 0}
    real_pad = encoder_attn.host_key_pad_mask

    def count_pad(*args, **kwargs):
        pad_calls["n"] += 1
        return real_pad(*args, **kwargs)

    monkeypatch.setattr(encoder_attn, "host_key_pad_mask", count_pad)
    mask_calls = {"n": 0}
    real_mask = encoder_attn.build_attention_mask

    def count_mask(*args, **kwargs):
        mask_calls["n"] += 1
        return real_mask(*args, **kwargs)

    monkeypatch.setattr(encoder_attn, "build_attention_mask", count_mask)
    impl, fwd, query, meta = _b1_dense_forward_setup()
    impl.forward(**fwd, output=torch.empty_like(query))
    assert dest_calls["n"] == 0
    assert idx_calls["n"] == 0
    assert pad_calls["n"] == 0
    assert mask_calls["n"] == 0
    assert meta.encoder_q_pack_idx is not None
    assert meta.encoder_q_pack_idx.device.type == "cpu"
    assert meta.encoder_unpack_idx is not None
    assert meta.encoder_unpack_idx.device.type == "cpu"
    assert meta.encoder_attn_mask is not None
    assert meta.encoder_attn_mask.numel() == 0
    assert meta.encoder_key_pad_mask is meta.encoder_attn_mask


def _run_b1_padded_forward(monkeypatch, *, seq_len: int, qsl_end: int, actual: int):
    fused_calls = {"n": 0}
    scatter_calls = {"n": 0}
    packed_calls = {"n": 0}
    index_copy_calls = {"n": 0}
    real_fused = encoder_attn._b1_dense_attention
    real_scatter = encoder_attn.scatter_pack
    real_packed = encoder_attn._packed_masked_attention
    real_index = encoder_attn._index_copy

    def count_fused(*args, **kwargs):
        fused_calls["n"] += 1
        return real_fused(*args, **kwargs)

    def count_scatter(*args, **kwargs):
        scatter_calls["n"] += 1
        return real_scatter(*args, **kwargs)

    def count_packed(*args, **kwargs):
        packed_calls["n"] += 1
        return real_packed(*args, **kwargs)

    def count_index(*args, **kwargs):
        index_copy_calls["n"] += 1
        return real_index(*args, **kwargs)

    monkeypatch.setattr(encoder_attn, "_b1_dense_attention", count_fused)
    monkeypatch.setattr(encoder_attn, "scatter_pack", count_scatter)
    monkeypatch.setattr(encoder_attn, "_packed_masked_attention", count_packed)
    monkeypatch.setattr(encoder_attn, "_index_copy", count_index)
    impl, fwd, query, meta = _b1_dense_forward_setup(total_tokens=64)
    meta.query_start_loc = torch.tensor([0, qsl_end], dtype=torch.int32)
    meta.seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    meta.num_actual_tokens = actual
    impl.forward(**fwd, output=torch.empty_like(query))
    return fused_calls["n"], scatter_calls["n"], packed_calls["n"], index_copy_calls["n"], meta


@torch.inference_mode()
def test_b1_padded_body_uses_packed_mask(monkeypatch, default_vllm_config) -> None:
    """Short prompt padded to T=L=64 uses packed QK, not fused SDPA; no index_copy_."""
    fused, scatter, packed, index_copy, meta = _run_b1_padded_forward(
        monkeypatch, seq_len=5, qsl_end=64, actual=5
    )
    assert fused == 0
    assert packed == 1
    assert scatter == 3
    assert index_copy == 0
    _assert_pad_mask(meta, 5)


@torch.inference_mode()
def test_b1_padded_qsl_and_seq_use_actual_tokens(monkeypatch, default_vllm_config) -> None:
    """Upstream can pad both cu_seqlens and seq_lens; num_actual_tokens still marks pad."""
    fused, scatter, packed, index_copy, meta = _run_b1_padded_forward(
        monkeypatch, seq_len=64, qsl_end=64, actual=5
    )
    assert fused == 0
    assert packed == 1
    assert scatter == 3
    assert index_copy == 0
    _assert_pad_mask(meta, 5)


def _assert_pad_mask(meta, real_len: int) -> None:
    mask = meta.encoder_attn_mask
    assert mask is not None
    mask_cpu = mask.cpu() if mask.device.type != "cpu" else mask
    assert mask_cpu[0, 0, 0, 0].item() == 0.0
    assert mask_cpu[0, 0, 0, real_len].item() < -1.0e3
    key_pad = meta.encoder_key_pad_mask
    assert key_pad is not None
    key_cpu = key_pad.cpu() if key_pad.device.type != "cpu" else key_pad
    assert key_cpu.shape[-2] == key_cpu.shape[-1]
    assert key_cpu[0, 0, 0, 0].item() == 0.0
    assert key_cpu[0, 0, 0, real_len].item() < -1.0e3


def test_content_query_lens_prefers_seq_over_padded_qsl():
    assert _content_query_lens([64], [5]) == [5]
    assert _content_query_lens([5, 12], [5, 12]) == [5, 12]
    assert _content_query_lens([32, 32], [5, 12]) == [5, 12]


def test_content_query_lens_actual_tokens_beat_padded_qsl_and_seq():
    assert _content_query_lens([64], [64], num_actual_tokens=5) == [5]
    assert _content_query_lens([32, 32], [32, 32], num_actual_tokens=17) == [17, 0]
    assert _content_query_lens([5, 12], [5, 12], num_actual_tokens=17) == [5, 12]


@torch.inference_mode()
def test_b1_short_seq_scatter_uses_packed_mask(monkeypatch, default_vllm_config) -> None:
    """T != L (BGE short prompts) scatters; F.sdpa would drop the pad mask."""
    sdpa = {"n": 0}
    packed = {"n": 0}
    real_sdpa = encoder_attn.F.scaled_dot_product_attention
    real_packed = encoder_attn._packed_masked_attention

    def count_sdpa(*args, **kwargs):
        sdpa["n"] += 1
        return real_sdpa(*args, **kwargs)

    def count_packed(*args, **kwargs):
        packed["n"] += 1
        return real_packed(*args, **kwargs)

    monkeypatch.setattr(encoder_attn.F, "scaled_dot_product_attention", count_sdpa)
    monkeypatch.setattr(encoder_attn, "_packed_masked_attention", count_packed)
    impl, fwd, query, _meta = _b1_dense_forward_setup(total_tokens=5)
    impl.forward(**fwd, output=torch.empty_like(query))
    assert packed["n"] == 1
    assert sdpa["n"] == 0


def test_packed_masked_qk_matches_softmax_reference():
    torch.manual_seed(0)
    batch, heads, length, dim = 1, 4, 64, 64
    query = torch.randn(batch, heads, length, dim)
    key = torch.randn(batch, heads, length, dim)
    value = torch.randn(batch, heads, length, dim)
    real_len = 5
    mask = build_attention_mask(1, length, [real_len], [real_len], dtype=query.dtype)
    scale = dim**-0.5
    key_pad = encoder_attn.host_key_pad_mask(mask, heads)
    got = encoder_attn._packed_masked_attention(query, key, value, key_pad, scale, False)
    ref = torch.matmul(
        torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) * scale + mask[:, :, :1, :],
            dim=-1,
        ),
        value,
    )
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_packed_attention_compiles_matmul_without_mask(monkeypatch) -> None:
    """Serve must not compile ``matmul + mask`` (Inductor → SDPA drops pad)."""
    seen: list[str] = []
    real = encoder_attn._compile_if_spyre

    def rec(cache, kernel, enable_gqa, device_type):
        seen.append(kernel.__name__)
        return real(cache, kernel, enable_gqa, device_type)

    monkeypatch.setattr(encoder_attn, "_compile_if_spyre", rec)
    torch.manual_seed(0)
    batch, heads, length, dim = 2, 4, 64, 64
    query = torch.randn(batch, heads, length, dim)
    key = torch.randn(batch, heads, length, dim)
    value = torch.randn(batch, heads, length, dim)
    mask = build_attention_mask(batch, length, [5, 12], [5, 12], dtype=query.dtype)
    key_pad = encoder_attn.host_key_pad_mask(mask, heads)
    encoder_attn._packed_masked_attention(query, key, value, key_pad, dim**-0.5, False)
    assert seen == ["_packed_qk_matmul", "_packed_pv"]


def _assert_scatter_matches_gather(
    q_starts: list[int],
    query_lens: list[int],
    batch: int,
    aligned_len: int,
    heads: int,
    dim: int,
    extra_src_rows: int = 0,
) -> None:
    num_tokens = sum(query_lens)
    num_src = num_tokens + extra_src_rows
    torch.manual_seed(0)
    flat = torch.randn(num_src, heads, dim)
    pad_row = num_src
    padded_starts = list(q_starts) + [num_tokens] * (batch - len(q_starts))
    padded_lens = list(query_lens) + [0] * (batch - len(query_lens))
    pack_idx = host_pack_indices(padded_starts, padded_lens, aligned_len, pad_row)
    dest = host_scatter_pack_dest(
        padded_starts,
        padded_lens,
        aligned_len,
        num_src_rows=num_src,
        dummy_row=dummy_pack_row(padded_lens, aligned_len),
    )
    ref = gather_pack(flat, pack_idx, dim)
    got = scatter_pack(flat, dest, batch, aligned_len, dim)
    assert torch.equal(got, ref)


def test_dummy_pack_row_first_pad_or_zero_when_full():
    assert dummy_pack_row([62], 64) == 62
    assert dummy_pack_row([30, 12, 8, 0], 64) == 30
    assert dummy_pack_row([64, 64], 64) == 0


def test_scatter_pack_matches_gather_b1_pad():
    """62-token prompt on L=64 (vllm --random-input-len 64). T != L, still scatter."""
    _assert_scatter_matches_gather(
        q_starts=[0],
        query_lens=[62],
        batch=1,
        aligned_len=64,
        heads=2,
        dim=8,
    )


def test_scatter_pack_matches_gather_b4_pad():
    """3 real seqs padded to B=4, L=64."""
    _assert_scatter_matches_gather(
        q_starts=[0, 30, 42],
        query_lens=[30, 12, 8],
        batch=4,
        aligned_len=64,
        heads=2,
        dim=8,
    )


def test_scatter_pack_strided_qkv_source_matches_contiguous():
    """Fused QKV views are not contiguous; scatter must densify before index_copy."""
    torch.manual_seed(0)
    t, h, d = 8, 4, 8
    qkv = torch.randn(t, 3 * h * d)
    q = qkv[:, : h * d].view(t, h, d)
    assert not q.is_contiguous()
    dest = host_scatter_pack_dest([0, 4], [4, 4], 8, 8, dummy_row=16)
    got = scatter_pack(q, dest, batch=2, aligned_len=8, head_size_padded=d)
    ref = scatter_pack(q.contiguous(), dest, batch=2, aligned_len=8, head_size_padded=d)
    torch.testing.assert_close(got, ref)


def _count_select_rows(monkeypatch):
    """Wrap ``select_rows`` so tests can assert identity B=1 skips gather."""
    real = encoder_attn.select_rows
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(encoder_attn, "select_rows", counting)
    return calls


def _count_index_copy(monkeypatch):
    """Wrap ``_index_copy`` so tests can assert B=1 dense body skips scatter."""
    real = encoder_attn._index_copy
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(encoder_attn, "_index_copy", counting)
    return calls


def test_gather_pack_b1_identity_skips_index_select(monkeypatch):
    """B=1 with ``T == L`` is already dense; do not ``index_select`` the full pack."""
    calls = _count_select_rows(monkeypatch)
    length, heads, dim = 4, 2, 8
    flat = torch.arange(length * heads * dim, dtype=torch.float32).reshape(length, heads, dim)
    pack_idx = torch.arange(length, dtype=torch.int64).view(1, length)

    out = gather_pack(flat, pack_idx, dim)

    assert calls["n"] == 0
    expected = flat.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    assert torch.equal(out, expected)


def test_gather_unpack_b1_identity_skips_index_select(monkeypatch):
    calls = _count_select_rows(monkeypatch)
    batch, heads, length, dim = 1, 2, 4, 8
    attn_out = torch.arange(batch * heads * length * dim, dtype=torch.float32).reshape(
        batch, heads, length, dim
    )
    unpack_idx = torch.arange(length, dtype=torch.int64)

    out = gather_unpack(attn_out, unpack_idx, dim)

    assert calls["n"] == 0
    expected = attn_out.permute(0, 2, 1, 3).contiguous().reshape(length, heads, dim)
    assert torch.equal(out, expected)


def test_gather_pack_b1_pad_slots_still_index_select(monkeypatch):
    """Pad slots still gather the extra zero row; identity skip must not fire."""
    calls = _count_select_rows(monkeypatch)
    tokens, heads, dim = 3, 2, 8
    flat = torch.arange(tokens * heads * dim, dtype=torch.float32).reshape(tokens, heads, dim)
    # Last slot is the F.pad zero row (index == T).
    pack_idx = torch.tensor([[0, 1, 2, tokens]], dtype=torch.int64)

    out = gather_pack(flat, pack_idx, dim)

    assert calls["n"] == 1
    assert torch.equal(out[0, :, 3, :], torch.zeros(heads, dim))


def test_gather_pack_multi_seq_still_index_select(monkeypatch):
    """B>1 keeps per-layer gather even when rows are 0..T-1 (reverted pack-once)."""
    calls = _count_select_rows(monkeypatch)
    batch, length, heads, dim = 2, 4, 2, 8
    flat = torch.randn(batch * length, heads, dim)
    pack_idx = torch.arange(batch * length, dtype=torch.int64).view(batch, length)

    gather_pack(flat, pack_idx, dim)

    assert calls["n"] == 1


def test_scatter_pack_b1_body_bucket_skips_index_copy(monkeypatch):
    """Body-bucket T=64 with 62 real tokens: skip scatter; mask covers pad slots."""
    calls = _count_index_copy(monkeypatch)
    tokens, aligned_len, heads, dim = 62, 64, 2, 8
    extra = aligned_len - tokens
    torch.manual_seed(0)
    flat = torch.randn(tokens + extra, heads, dim)
    dest = host_scatter_pack_dest(
        q_starts=[0],
        lengths=[tokens],
        aligned_len=aligned_len,
        num_src_rows=tokens + extra,
        dummy_row=dummy_pack_row([tokens], aligned_len),
    )

    got = scatter_pack(flat, dest, batch=1, aligned_len=aligned_len, head_size_padded=dim)

    assert calls["n"] == 0
    expected = flat.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    assert torch.equal(got, expected)


def test_scatter_pack_b1_short_seq_still_index_copy(monkeypatch):
    """T=62, L=64: body is not dense; still scatter."""
    calls = _count_index_copy(monkeypatch)
    _assert_scatter_matches_gather(
        q_starts=[0],
        query_lens=[62],
        batch=1,
        aligned_len=64,
        heads=2,
        dim=8,
    )
    assert calls["n"] == 1


def test_scatter_pack_multi_seq_still_index_copy(monkeypatch):
    """B>1 still scatters even when T == B×L."""
    calls = _count_index_copy(monkeypatch)
    batch, length, heads, dim = 2, 4, 2, 8
    flat = torch.randn(batch * length, heads, dim)
    dest = torch.arange(batch * length, dtype=torch.int64)
    scatter_pack(flat, dest, batch, length, dim)
    assert calls["n"] == 1


def test_gather_unpack_b1_dense_body_skips_index_select(monkeypatch):
    """62-in-64 unpack is not identity (tail stays 0) but T==L still reshapes."""
    calls = _count_select_rows(monkeypatch)
    batch, heads, length, dim, real = 1, 2, 64, 8, 62
    attn_out = torch.randn(batch, heads, length, dim)
    unpack_idx = torch.zeros(length, dtype=torch.int64)
    unpack_idx[:real] = torch.arange(real, dtype=torch.int64)

    out = gather_unpack(attn_out, unpack_idx, dim)

    assert calls["n"] == 0
    expected = attn_out.permute(0, 2, 1, 3).contiguous().reshape(length, heads, dim)
    assert torch.equal(out, expected)
