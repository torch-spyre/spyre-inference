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
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec, EncoderOnlyAttentionSpec

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionMetadataBuilder,
    SpyrePagedKVCache,
)
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    ENCODER_LEN_ALIGNMENT,
    EncoderRectPlan,
    SpyreEncoderAttentionImpl,
    _alignment_units_for,
    _encoder_gather_kernel,
    _encoder_sdpa_kernel,
    build_encoder_plan,
    encoder_index_dtype,
    encoder_key_pad_mask,
    encoder_row_table,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import encoder_dense_row_indices


def encoder_mask(extent: int, kv_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Single-sequence mask, the shape these tests were written against."""
    return encoder_key_pad_mask(extent, [kv_len], dtype)


# extra `encoder_attention` mark so CI can split this into its own job
# because these tests are pretty slow.
pytestmark = [pytest.mark.attention, pytest.mark.encoder_attention]


def _create_dense_attn_kernel(num_heads: int, num_kv_heads: int, head_size: int):
    """Test helper: bind the non-tensor args the way a forward call would."""

    def specialized_dense_attn_kernel(q_rows, k_rows, v_rows, mask, scale):
        return _encoder_sdpa_kernel(
            q_rows, k_rows, v_rows, mask, scale, 1, num_heads, num_kv_heads, head_size
        )

    return specialized_dense_attn_kernel


def dense_sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lens: list[int],
    scale: float,
) -> torch.Tensor:
    """Per-sequence eager SDPA on the packed list (probe / unit-test reference)."""
    outs: list[torch.Tensor] = []
    start = 0
    for length in query_lens:
        q = query[start : start + length]
        k = key[start : start + length]
        v = value[start : start + length]
        qh = q.unsqueeze(0).transpose(1, 2)
        kh = k.unsqueeze(0).transpose(1, 2)
        vh = v.unsqueeze(0).transpose(1, 2)
        kwargs: dict = {"is_causal": False, "scale": scale}
        if q.shape[1] != k.shape[1]:
            kwargs["enable_gqa"] = True
        out = F.scaled_dot_product_attention(qh, kh, vh, **kwargs)
        outs.append(out.transpose(1, 2).squeeze(0))
        start += length
    return torch.cat(outs, dim=0)


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
    # Increase recompilation limit: the block kernel is specialized (and so
    # recompiled) per unique (group, extent, buffer_rows).
    torch._dynamo.config.accumulated_recompile_limit = 1024

    yield mode_name

    # Cleanup: reset mode and limits
    cfg.mode = original_mode
    torch._dynamo.config.accumulated_recompile_limit = original_limit
    torch._dynamo.reset()


def _vllm_style_output(query: torch.Tensor, device) -> torch.Tensor:
    """Allocate the attention output the way vLLM does.

    ``Attention.forward`` allocates ``[num_tokens, num_heads * head_size]`` and
    views it as 3-D, so its device layout is a flat one. A plain
    ``empty_like(query)`` allocates 3-D instead, and for a head size below one
    stick that layout differs in a way this backend is sensitive to -- a shape
    real traffic never produces.
    """
    tokens, heads, head_size = query.shape
    flat = torch.empty((tokens, heads * head_size), dtype=query.dtype)
    return flat.to(device).view(-1, heads, head_size)


def _build_metadata(
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    spec_cls: type[AttentionSpec] = EncoderOnlyAttentionSpec,
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

    # Defaults to the spec upstream hands an ENCODER_ONLY group, not a plain
    # AttentionSpec: build() branches on it to skip the KV-cache fields, so a plain
    # one would exercise a path production never takes.
    kv_cache_spec = spec_cls(
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


def test_alignment_units_for_rounds_up_to_powers_of_two():
    assert [_alignment_units_for(n) for n in (1, 64, 65, 128, 129, 200, 512)] == [
        1,
        1,
        2,
        2,
        4,
        4,
        8,
    ]


def test_encoder_mask_cuts_at_the_boundary_block():
    mask = encoder_mask(3 * ENCODER_LEN_ALIGNMENT, 100, torch.float32)
    masked = torch.finfo(torch.float32).min / 2
    # Head and query axes stay 1 and broadcast: an encoder mask depends only on
    # the KV column, so it need not be materialised per head.
    assert mask.shape == (1, 1, 1, 3 * ENCODER_LEN_ALIGNMENT)
    # Block 0 (cols 0:64) is all real keys, block 1 (64:128) straddles
    # kv_len=100, block 2 (128:192) is all padding.
    assert torch.equal(mask[..., :64], torch.zeros_like(mask[..., :64]))
    assert torch.equal(mask[..., 64:100], torch.zeros_like(mask[..., 64:100]))
    assert torch.equal(mask[..., 100:128], torch.full_like(mask[..., 100:128], masked))
    assert torch.equal(mask[..., 128:], torch.full_like(mask[..., 128:], masked))


def test_encoder_mask_at_one_alignment_unit_is_a_single_row():
    mask = encoder_mask(ENCODER_LEN_ALIGNMENT, 64, torch.float32)
    assert mask.shape == (1, 1, 1, ENCODER_LEN_ALIGNMENT)


def test_row_table_clamps_padding_lanes_to_the_last_real_row():
    rows = encoder_row_table(10, 3, ENCODER_LEN_ALIGNMENT, torch.int64)
    assert rows[:3].tolist() == [10, 11, 12]
    # Padding lanes repeat row 12 so the gather never reads the next request.
    assert rows[3:].unique().tolist() == [12]


def test_dense_attn_kernel_never_calls_arange(monkeypatch):
    """Regression guard: the original bug was building block indices inside
    the attention kernel, which falls back to CPU on real hardware (torch-spyre
    has no on-device ``arange``). The kernel takes already-gathered,
    fixed-shape tensors and a precomputed mask, and must not construct
    anything itself.
    """
    length = 70
    num_heads, num_kv_heads, head_size = 4, 4, 64
    extent = _alignment_units_for(length) * ENCODER_LEN_ALIGNMENT
    torch.manual_seed(0)
    q_rows = torch.randn(extent, num_heads, head_size, dtype=torch.float32)
    k_rows = torch.randn(extent, num_kv_heads, head_size, dtype=torch.float32)
    v_rows = torch.randn(extent, num_kv_heads, head_size, dtype=torch.float32)
    mask = convert(encoder_mask(extent, length, q_rows.dtype), q_rows.device)

    real_arange = torch.arange
    calls = {"n": 0}

    def counting_arange(*args, **kwargs):
        calls["n"] += 1
        return real_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", counting_arange)
    attn_fn = _create_dense_attn_kernel(num_heads, num_kv_heads, head_size)
    attn_fn(q_rows, k_rows, v_rows, mask, head_size**-0.5)
    assert calls["n"] == 0, "the attention kernel must not build any index tensor itself"


def test_gather_kernel_never_calls_arange(monkeypatch):
    length = 70
    num_heads, num_kv_heads, head_size = 4, 4, 64
    extent = _alignment_units_for(length) * ENCODER_LEN_ALIGNMENT
    torch.manual_seed(0)
    query = torch.randn(extent, num_heads, head_size, dtype=torch.float32)
    key = torch.randn(extent, num_kv_heads, head_size, dtype=torch.float32)
    value = torch.randn(extent, num_kv_heads, head_size, dtype=torch.float32)
    row_index = encoder_row_table(0, length, extent, torch.int64)

    real_arange = torch.arange
    calls = {"n": 0}

    def counting_arange(*args, **kwargs):
        calls["n"] += 1
        return real_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", counting_arange)
    _encoder_gather_kernel(query, key, value, row_index)
    assert calls["n"] == 0, "the gather kernel must not build any index tensor itself"


def _dense_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lens: list[int],
    scale: float,
) -> torch.Tensor:
    """Drive gather + dense attention over a packed list the way ``forward`` does."""
    num_heads, head_size = query.shape[1], query.shape[2]
    num_kv_heads = key.shape[1]
    index_dtype = encoder_index_dtype(query.device)
    attn_fn = _create_dense_attn_kernel(num_heads, num_kv_heads, head_size)
    out = torch.zeros_like(query)
    start = 0
    for length in query_lens:
        extent = _alignment_units_for(length) * ENCODER_LEN_ALIGNMENT
        row_index = encoder_row_table(start, length, extent, index_dtype)
        mask = convert(encoder_mask(extent, length, query.dtype), query.device)
        q_rows, k_rows, v_rows = _encoder_gather_kernel(query, key, value, row_index)
        attn = attn_fn(q_rows, k_rows, v_rows, mask, scale)
        out.index_copy_(0, row_index, attn)
        start += length
    return out


@pytest.mark.parametrize(
    "query_lens",
    [
        pytest.param([32], id="single_32"),
        pytest.param([9, 70, 5], id="batch_unaligned"),
        pytest.param([16, 8], id="two_seqs"),
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [pytest.param((4, 1), id="GQA"), pytest.param((4, 4), id="MHA")],
)
def test_dense_kernel_matches_dense_sdpa_reference(
    query_lens: list[int], num_heads: tuple[int, int]
) -> None:
    """The gather + dense-attention path must match a dense SDPA computation."""
    num_query_heads, num_kv_heads = num_heads
    head_size = 64
    scale = head_size**-0.5
    torch.manual_seed(0)
    total = sum(query_lens)
    query = torch.randn(total, num_query_heads, head_size, dtype=torch.float32)
    key = torch.randn(total, num_kv_heads, head_size, dtype=torch.float32)
    value = torch.randn(total, num_kv_heads, head_size, dtype=torch.float32)

    got = _dense_attn(query, key, value, query_lens, scale)
    ref = dense_sdpa_reference(query, key, value, query_lens, scale)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


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
        # Product encoder models (Granite/E5/RoBERTa) use D=64; MiniLM uses 32,
        # which is half a Spyre stick -- a reshape/transpose there can produce a
        # sub-stick interleaved index the backend rejects.
        pytest.param(64, id="head_size(64)"),
        pytest.param(32, id="head_size(32)"),
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
    output = _vllm_style_output(query, cache_device)
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


@pytest.mark.parametrize(
    "configure_device",
    [
        pytest.param("cpu", id="device_cpu"),
        pytest.param("spyre", id="device_spyre"),
    ],
    indirect=True,
)
@torch.inference_mode()
def test_single_sequence_exactly_filling_the_buffer_handles_a_fused_qkv_view(
    default_vllm_config, configure_device: str
) -> None:
    """Regression test: a single request whose length exactly equals the padded
    extent used to take a no-gather path, which handed the raw
    query/key/value straight to the attention math with no normalization. A
    model with a fused QKV projection (``qkv.split(...)``) hands out *strided*
    views there, not contiguous tensors -- on real Spyre hardware this crashed
    with "no mechanism to resolve stick incompatibility" the first time a
    request landed on this exact shape (a single sequence, no batching, exactly
    filling its body bucket). Build query/key/value the same way: slice them out
    of one fused buffer instead of allocating them independently.
    """
    num_heads, num_kv_heads, head_size, block_size = 4, 4, 64, 64
    total_tokens = 64  # == extent, so this is the single-sequence exact-fill case
    dtype = torch.float16
    torch.set_default_device("cpu")
    set_random_seed(0)

    fused = torch.randn(total_tokens, 3 * num_heads * head_size, dtype=dtype)
    query, key, value = fused.split([num_heads * head_size] * 3, dim=-1)
    query = query.view(total_tokens, num_heads, head_size)
    key = key.view(total_tokens, num_kv_heads, head_size)
    value = value.view(total_tokens, num_kv_heads, head_size)
    assert not query.is_contiguous(), "the fused-QKV slice must stay a strided view"

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
    device = torch.device(configure_device)
    output = _vllm_style_output(query, device)
    impl.forward(
        layer=None,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
    )

    ref = dense_sdpa_reference(
        query.contiguous(), key.contiguous(), value.contiguous(), [total_tokens], head_size**-0.5
    )
    torch.testing.assert_close(output.to("cpu"), ref, atol=0.2, rtol=0.2)


@torch.inference_mode()
def test_encoder_plan_built_once_and_reused_across_layers(default_vllm_config) -> None:
    """Second layer's forward() must reuse the first layer's plans, not rebuild them."""
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
    cached_plans = attn_metadata.encoder_plan
    assert cached_plans is not None
    assert len(cached_plans) == 1
    assert cached_plans[0].query_lens == [32]

    impl.forward(**fwd, output=torch.empty_like(query))
    assert attn_metadata.encoder_plan is cached_plans


def _profile_metadata(spec_cls, *, max_model_len: int, prompt_len: int, num_seqs: int):
    """Metadata for warmup's profiling batch: every sequence at the full body size.

    Upstream ``_dummy_run`` sets ``seq_lens = num_tokens`` for every request
    regardless of how it split the token budget, so this is what the builder sees
    when the pooling warmup loop runs its largest body bucket.
    """
    from vllm.config import get_current_vllm_config

    get_current_vllm_config().model_config.max_model_len = max_model_len
    return _build_metadata(
        num_query_heads=4,
        num_kv_heads=4,
        head_size=64,
        block_size=128,
        seq_lens=torch.full((num_seqs,), prompt_len, dtype=torch.int32),
        query_start_loc=torch.arange(0, (num_seqs + 1) * prompt_len, prompt_len).to(torch.int32),
        block_table=torch.zeros((num_seqs, 8), dtype=torch.int32),
        slot_mapping=torch.zeros(num_seqs * prompt_len, dtype=torch.int64),
        spec_cls=spec_cls,
    )


def test_encoder_build_skips_kv_cache_fields(default_vllm_config) -> None:
    meta = _profile_metadata(EncoderOnlyAttentionSpec, max_model_len=512, prompt_len=64, num_seqs=2)
    assert meta.padded_num_blocks is None
    assert not meta.attention_mask_stacks
    assert meta.page_index_tables_cpu is None
    assert meta.active_block_indices is None
    # Everything SpyreEncoderAttentionImpl.forward actually reads.
    assert meta.num_seqs == 2
    assert meta.num_actual_tokens == 128
    assert meta.seq_lens.tolist() == [64, 64]
    assert meta.query_start_loc.tolist() == [0, 64, 128]


def test_encoder_build_survives_a_body_bucket_past_max_model_len(default_vllm_config) -> None:
    """A pooling model whose max_model_len is under the top body bucket still builds.

    The two ladders are keyed on different quantities: the num_blocks buckets come
    from max_model_len, while the body buckets a pooling model warms come from
    max_num_batched_tokens. all-MiniLM-L6-v2 and all-roberta-large-v1 derive
    max_model_len=256 and get a 512-token top body bucket, so warmup profiled 4
    blocks against a 2-block ladder and the engine died at init.
    """
    meta = _profile_metadata(
        EncoderOnlyAttentionSpec, max_model_len=256, prompt_len=512, num_seqs=1
    )
    assert meta.padded_num_blocks is None

    # Same input on the paged builder still raises: the guard is what makes the
    # encoder case work, not a widened ladder.
    with pytest.raises(AssertionError, match="exceeds the largest recorded bucket"):
        _profile_metadata(AttentionSpec, max_model_len=256, prompt_len=512, num_seqs=1)


@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compilation_STOCK")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
@pytest.mark.parametrize("batched", [False, True], ids=["grouping_off", "grouping_on"])
@pytest.mark.parametrize(
    "seq_lens",
    [
        # All one extent: one group, no remainder.
        pytest.param([(64, 64)] * 4, id="4seqs_same_extent"),
        # Five members at one extent must split 4 + 1, not pad up to 8.
        pytest.param([(64, 64)] * 5, id="5seqs_splits_4plus1"),
        # Different extents must not be grouped together, and members sharing an
        # extent may still have different real kv_len.
        pytest.param([(40, 40), (50, 50), (300, 300), (310, 310), (20, 20)], id="mixed_extents"),
    ],
)
@torch.inference_mode()
def test_grouped_attention_matches_the_per_sequence_reference(
    default_vllm_config,
    batched: bool,
    seq_lens: list[tuple[int, int]],
    configure_compilation: str,
    configure_device: str,
) -> None:
    """Batching equal-extent requests into one kernel call must not change results.

    Runs the same batch with grouping off and on: both are checked against the
    bidirectional reference, so a grouping bug cannot hide behind a shared
    implementation. ``mixed_extents`` also covers members that share an extent
    while having different real ``kv_len`` -- only their masks differ, and those
    are concatenated in member order to match the kernel's folded batch dim.
    """
    num_heads, num_kv_heads, head_size, block_size = 12, 12, 64, 64
    dtype = torch.float16
    torch.set_default_device("cpu")
    set_random_seed(0)

    query_lens = [q for q, _ in seq_lens]
    kv_lens = [k for _, k in seq_lens]
    total_tokens = sum(query_lens)
    scale = head_size**-0.5

    query = torch.randn(total_tokens, num_heads, head_size, dtype=dtype)
    key = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )

    attn_metadata = _build_metadata(
        num_query_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32),
        query_start_loc=cu_query_lens,
        block_table=torch.zeros(
            len(seq_lens), (max(query_lens) + block_size - 1) // block_size, dtype=torch.int32
        ),
        slot_mapping=torch.arange(total_tokens, dtype=torch.int64),
    )
    impl = SpyreEncoderAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
    )
    output = _vllm_style_output(query, torch.device(configure_device))
    # Pre-built rather than left to the impl, which always groups under a compiled
    # config: `batched=False` is the shape the eager path builds.
    attn_metadata.encoder_plan = build_encoder_plan(
        attn_metadata,
        rectangles=(),
        width_cap_for=impl._width_caps,
        device=output.device,
        dtype=dtype,
        batched=batched,
    )
    impl.forward(
        layer=None,
        query=query,
        key=key,
        value=value,
        kv_cache=SpyrePagedKVCache(k_pages=torch.empty(0), v_pages=torch.empty(0)),
        attn_metadata=attn_metadata,
        output=output,
    )

    groups = [getattr(p, "group", 1) for p in attn_metadata.encoder_plan]
    if batched:
        assert any(g > 1 for g in groups), (
            "grouping was enabled but every plan stayed a single sequence"
        )
    else:
        assert groups == [1] * len(groups), "grouping was disabled but a plan batched"

    ref_output = ref_encoder_attn(
        query=query, key=key, value=value, query_lens=query_lens, scale=scale
    )
    assert_close_outliers(
        output.to("cpu"),
        ref_output,
        max_outliers=8,
        atol=0.3,
        rtol=0.2,
        outlier_atol=0.6,
        outlier_rtol=0.4,
    )


@pytest.mark.parametrize("configure_device", ["cpu", "spyre"], indirect=True)
@pytest.mark.parametrize("configure_compilation", ["STOCK_TORCH_COMPILE"], indirect=True)
@pytest.mark.parametrize(
    "query_lens",
    [
        pytest.param([64, 64, 64, 64], id="uniform_full_extent"),
        pytest.param([40, 64, 17, 64], id="ragged_within_one_extent"),
        pytest.param([100, 30, 128, 7], id="ragged_across_extents"),
        pytest.param([64], id="single_request"),
        # Fewer requests than the rectangle is wide, so batch-pad lanes are live.
        pytest.param([50, 60], id="batch_pad_lanes"),
    ],
)
@torch.inference_mode()
def test_fast_and_ragged_paths_agree(
    default_vllm_config,
    query_lens: list[int],
    configure_compilation: str,
    configure_device: str,
) -> None:
    """The same batch down both paths must give the same answer.

    This is what keeps the ragged path a fallback rather than a second implementation
    that silently diverges. The two see different inputs by construction -- the rectangular
    path a dense ``[B, L]`` grid, the ragged path the packed list -- so the test builds
    both from one set of activations and compares only the real token rows.
    """
    num_heads, num_kv_heads, head_size, block_size = 12, 12, 64, 64
    dtype = torch.float16
    device = torch.device(configure_device)
    torch.set_default_device("cpu")
    set_random_seed(0)

    extent = _alignment_units_for(max(query_lens)) * ENCODER_LEN_ALIGNMENT
    width = len(query_lens)
    scale = head_size**-0.5

    def make_impl():
        return SpyreEncoderAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
        )

    def metadata(lens):
        cu = torch.tensor([0] + list(lens), dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
        return _build_metadata(
            num_query_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            block_size=block_size,
            seq_lens=torch.tensor(list(lens), dtype=torch.int32),
            query_start_loc=cu,
            block_table=torch.zeros(len(lens), extent // block_size, dtype=torch.int32),
            slot_mapping=torch.arange(sum(lens), dtype=torch.int64),
        )

    kv_cache = SpyrePagedKVCache(k_pages=torch.empty(0), v_pages=torch.empty(0))

    # One set of activations, laid out both ways. Grid row `s * extent + i` is packed
    # row `cumsum(lens)[s] + i`; pad rows are zero and their outputs are dropped.
    packed_q = torch.randn(sum(query_lens), num_heads, head_size, dtype=dtype)
    packed_k = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    packed_v = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=dtype)

    rows = encoder_dense_row_indices(query_lens, extent)

    def to_grid(packed, heads):
        grid = torch.zeros(width * extent, heads, head_size, dtype=dtype)
        grid[rows] = packed
        return grid

    # Ragged path: metadata carries no plan, so the impl builds a packed one.
    slow_md = metadata(query_lens)
    slow_out = _vllm_style_output(packed_q, device)
    make_impl().forward(
        layer=None,
        query=convert(packed_q, device),
        key=convert(packed_k, device),
        value=convert(packed_v, device),
        kv_cache=kv_cache,
        attn_metadata=slow_md,
        output=slow_out,
    )
    assert not isinstance(slow_md.encoder_plan, EncoderRectPlan), "expected the packed path"

    # Rectangular path: the runner would have padded the body and laid out the grid, so the
    # plan is built here with the covering rectangle declared.
    grid_q = to_grid(packed_q, num_heads)
    fast_md = metadata(query_lens)
    fast_md.encoder_plan = build_encoder_plan(
        fast_md,
        rectangles=[(extent, width)],
        width_cap_for={},
        device=device,
        dtype=dtype,
        batched=True,
    )
    assert isinstance(fast_md.encoder_plan, EncoderRectPlan), "expected the rectangle path"
    fast_out = _vllm_style_output(grid_q, device)
    make_impl().forward(
        layer=None,
        query=convert(grid_q, device),
        key=convert(to_grid(packed_k, num_kv_heads), device),
        value=convert(to_grid(packed_v, num_kv_heads), device),
        kv_cache=kv_cache,
        attn_metadata=fast_md,
        output=fast_out,
    )

    # Only the real token rows: pad rows differ by construction and nothing reads them.
    assert_close_outliers(
        fast_out.to("cpu")[rows],
        slow_out.to("cpu"),
        max_outliers=8,
        atol=0.3,
        rtol=0.2,
        outlier_atol=0.6,
        outlier_rtol=0.4,
    )
