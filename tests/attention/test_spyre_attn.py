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

import gc
from unittest.mock import Mock

import pytest
import torch
from spyre_testing_plugin.attn_helpers import (
    _build_metadata,
    _fused_qkv_kv_views,
    _padded_mask_metadata,
    assert_close_outliers,
    random_block_tables,
    ref_attn,
)
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_cache_interface import AttentionSpec

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends import spyre_attn
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadataBuilder,
    SpyrePagedKVCache,
    _build_query_row_tables,
    _mirror_mask_stack,
)
from spyre_inference.v1.attention.backends.spyre_encoder_attn import SpyreEncoderAttentionImpl
from spyre_inference.v1.attention.ops import tile_loop
from spyre_inference.v1.attention.ops.batched_decode import (
    batched_decode_kernel,
)
from spyre_inference.v1.attention.ops.layout import INT32_ELEMS_PER_STICK
from spyre_inference.v1.attention.ops.page_attn_decode import (
    page_attn_decode_kernel,
)
from spyre_inference.v1.attention.ops.page_attn_prefill import (
    page_attn_prefill_kernel,
)
from spyre_inference.v1.attention.ops.reshape_and_cache import (
    quantize_kv,
    reshape_and_cache_kernel,
)
from spyre_inference.v1.attention.spyre_attn_bucketer import SpyreAttnBucketer

pytestmark = pytest.mark.attention


DTYPE = torch.float16


@pytest.fixture()
def enable_batched_decode(monkeypatch):
    """Pin ``SPYRE_BATCHED_DECODE`` on, so the batched tests cannot silently fall
    back to the per-seq loop if the default changes. The autouse cache-clearing
    fixture in ``tests/conftest.py`` makes the value visible to ``envs``.
    """
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")


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


def _fresh_pages(
    num_blocks: int,
    num_kv_heads: int,
    block_size: int,
    head_size: int,
    device: torch.device,
    dtype: torch.dtype = DTYPE,
    fp8: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zeroed pages, through the impl so they carry the production layout.

    Host-populating a cache with that layout pinned is numerically wrong, so every write
    below goes through the impl too.
    """
    if device.type != "spyre":
        shape = (num_blocks, num_kv_heads, block_size, head_size)
        page_dtype = torch.float8_e4m3fn if fp8 else dtype
        return torch.zeros(shape, dtype=page_dtype), torch.zeros(shape, dtype=page_dtype)
    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        # Upstream's spec dtype for an fp8 cache.
        dtype=torch.uint8 if fp8 else dtype,
    )
    cache = SpyreAttentionImpl.allocate_pages(num_blocks, spec, device)
    return cache.k_pages, cache.v_pages


def _write(impl, kv_cache, key, value, slots, device) -> None:
    impl.do_kv_cache_update(None, key, value, kv_cache, impl.kv_write_index(slots, device))


@torch.inference_mode()
def _run_spyre_attn_test(
    seq_lens: list[tuple[int, int]],
    block_size: int,
    sliding_window: int | None,
    configure_compilation: str,
    configure_device: str,
    soft_cap: float | None = None,
    use_sinks: bool = False,
    tolerance: tuple[float, float] | None = None,
    num_query_heads: int = 32,
    num_kv_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = DTYPE,
    kv_cache_dtype: str = "auto",
    kv_scales: tuple[float, float] = (1.0, 1.0),
    expect_fused_store: bool | None = None,
    expect_query_widths: set[int] | None = None,
) -> None:
    """Shared test body: validate SpyreAttentionImpl against a reference implementation.

    Seeds the cache entirely through the impl's store, history included: the device
    layout cannot be reproduced by a host-populated transfer.
    """
    # The compiled attention kernel targets the Spyre device. On CPU it routes through
    # Inductor's C++ backend, whose codegen for the kernels' indirect gathers is broken.
    # CPU is only the eager reference here, so skip the compiled+CPU combo.
    if configure_compilation == "STOCK_TORCH_COMPILE" and configure_device == "cpu":
        pytest.skip("Compiled attention targets Spyre; Inductor CPU codegen is unsupported here.")

    from vllm.config import get_current_vllm_config

    num_blocks = 256
    # SpyreAttentionImpl and the metadata builder both read the dtype off the VllmConfig.
    get_current_vllm_config().model_config.dtype = dtype
    torch.set_default_device("cpu")
    set_random_seed(0)

    # gpt-oss-20b's sinks span -2.45..8.19: from negligible to most of the softmax mass.
    sinks = torch.linspace(-2.5, 8.5, num_query_heads).to(dtype) if use_sinks else None
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    assert num_query_heads % num_kv_heads == 0
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=dtype)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    # Widened to the recorder's num_blocks buckets, as a real engine's block table is, so
    # build()'s padding isn't suppressed by an exactly-sized table. The extra entries point
    # at pages no sequence owns, so they cannot alias live KV.
    max_num_blocks_per_seq = (max(kv_lens) + block_size - 1) // block_size
    # The bucketer must see this test's block_size, not the fixture default.
    bucketer_config = get_current_vllm_config()
    bucketer_config.cache_config.block_size = block_size
    buckets = SpyreAttnBucketer(bucketer_config).num_blocks_buckets
    padded_width = SpyreAttnBucketer._round_up(max_num_blocks_per_seq, buckets)
    if padded_width is not None:
        max_num_blocks_per_seq = max(max_num_blocks_per_seq, padded_width)
    block_tables = random_block_tables(kv_lens, max_num_blocks_per_seq, block_size, num_blocks)

    k_expected = torch.zeros(num_blocks, num_kv_heads, block_size, head_size, dtype=dtype)
    v_expected = torch.zeros(num_blocks, num_kv_heads, block_size, head_size, dtype=dtype)
    hist_k, hist_v, hist_slots = [], [], []
    slot_mapping: list[int] = []
    q_offset = 0
    for seq_idx in range(num_seqs):
        query_len, kv_len = query_lens[seq_idx], kv_lens[seq_idx]
        historical_len = kv_len - query_len
        if historical_len > 0:
            hk = torch.randn(historical_len, num_kv_heads, head_size, dtype=dtype)
            hv = torch.randn(historical_len, num_kv_heads, head_size, dtype=dtype)
            for token_idx in range(historical_len):
                blk = int(block_tables[seq_idx, token_idx // block_size].item())
                off = token_idx % block_size
                k_expected[blk, :, off] = hk[token_idx]
                v_expected[blk, :, off] = hv[token_idx]
                hist_slots.append(blk * block_size + off)
            hist_k.append(hk)
            hist_v.append(hv)
        for token_idx in range(historical_len, kv_len):
            blk = int(block_tables[seq_idx, token_idx // block_size].item())
            off = token_idx % block_size
            k_expected[blk, :, off] = key[q_offset + token_idx - historical_len]
            v_expected[blk, :, off] = value[q_offset + token_idx - historical_len]
            slot_mapping.append(blk * block_size + off)
        q_offset += query_len
    slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int64)

    attn_metadata = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=kv_lens_tensor,
        query_start_loc=cu_query_lens,
        block_table=block_tables,
        slot_mapping=slot_mapping_t,
        sliding_window=sliding_window,
        dtype=dtype,
    )

    # After _build_metadata: the impl reads cache_config.block_size at construction.
    attn_impl = SpyreAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        sliding_window=sliding_window,
        kv_cache_dtype=kv_cache_dtype,
        logits_soft_cap=soft_cap,
        sinks=sinks,
    )
    assert attn_impl.block_size == block_size

    # The fused store is just whether the kernel was handed an `out` buffer.
    fused_calls: list[bool] = []
    dispatched_widths: set[int] = set()
    if expect_fused_store is not None or expect_query_widths is not None:
        real_run = attn_impl._run_page_attn

        def spy_run(query, row_table, k_pages, v_pages, index_table, mask_stack, nb, pql, out):
            fused_calls.append(out is not None)
            dispatched_widths.add(pql)
            return real_run(
                query, row_table, k_pages, v_pages, index_table, mask_stack, nb, pql, out
            )

        attn_impl._run_page_attn = spy_run

    cache_device = torch.device(configure_device)
    fp8 = attn_impl.kv_cache_fp8
    k_scale, v_scale = kv_scales
    attn_impl.prepare_kv_scales(Mock(_k_scale_float=k_scale, _v_scale_float=v_scale))
    k_pages, v_pages = _fresh_pages(
        num_blocks, num_kv_heads, block_size, head_size, cache_device, dtype, fp8
    )
    kv_cache = SpyrePagedKVCache(k_pages=k_pages, v_pages=v_pages)

    # The attention layer, not forward(), owns the KV write (see attn_layer.py).
    if hist_slots:
        _write(
            attn_impl,
            kv_cache,
            convert(torch.cat(hist_k), cache_device),
            convert(torch.cat(hist_v), cache_device),
            torch.tensor(hist_slots, dtype=torch.int64),
            cache_device,
        )
    key_src, value_src = _fused_qkv_kv_views(query, key, value, cache_device)
    _write(attn_impl, kv_cache, key_src, value_src, slot_mapping_t, cache_device)

    if fp8:
        # The cache holds K / k_scale and V / v_scale in e4m3: the reference reads what was
        # stored, and the stored bytes are checked first, so a bad store and a bad read fail
        # apart.
        k_expected = quantize_kv(k_expected, k_scale, torch.float8_e4m3fn)
        v_expected = quantize_kv(v_expected, v_scale, torch.float8_e4m3fn)
        for name, pages, expected in (("K", k_pages, k_expected), ("V", v_pages, v_expected)):
            # Through the device's fp8 -> fp16 conversion: the stored order is QFP8CH, so a
            # raw byte copy would read shuffled. The device conversion rounds a few values
            # one e4m3 step (2^-3 relative) from the host cast, mostly below the smallest
            # normal (2^-6); a misplaced or misordered store is off by far more.
            stored = pages.to(dtype).to("cpu").float()
            torch.testing.assert_close(
                stored,
                expected.float(),
                atol=2**-6,
                rtol=2**-3,
                msg=lambda m, name=name: f"fp8 {name} store: {m}",
            )
        k_expected = (k_expected.float() * k_scale).to(dtype)
        v_expected = (v_expected.float() * v_scale).to(dtype)

    # NaN, not empty: every row is expected to be written, so a store that lands nowhere
    # fails below instead of passing on whatever the allocator gave.
    output = torch.full_like(query, float("nan")).to(cache_device)
    attn_impl.forward(
        layer=None,
        query=convert(query, cache_device),
        key=key_src,
        value=value_src,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
    )

    if expect_fused_store is not None:
        assert fused_calls, "no attention kernel ran"
        assert set(fused_calls) == {expect_fused_store}, (
            f"fused output store: expected {expect_fused_store}, got {set(fused_calls)}"
        )
    if expect_query_widths is not None:
        assert dispatched_widths == expect_query_widths, (
            f"query widths dispatched: expected {expect_query_widths}, "
            f"got {sorted(dispatched_widths)}"
        )

    ref_output = ref_attn(
        query=query,
        # ref_attn indexes a page's token axis first.
        key_cache=k_expected.permute(0, 2, 1, 3),
        value_cache=v_expected.permute(0, 2, 1, 3),
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        block_size=block_size,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        sinks=sinks,
    )

    if tolerance is not None:
        atol, rtol = tolerance
    else:
        atol, rtol = (0.3, 0.2) if max(query_lens) >= 32 else (0.2, 0.2)
    assert_close_outliers(
        output.to("cpu"),
        ref_output,
        max_outliers=5,
        atol=atol,
        rtol=rtol,
        outlier_atol=atol * 2,
        outlier_rtol=rtol * 2,
    )

    # Release Spyre DMA mappings eagerly. Python doesn't free the KV-page tensors between
    # tests until GC runs, but the Spyre VFIO driver keeps DMA regions mapped until the
    # storage is actually released. Accumulated mappings across many tests in one pytest
    # process can exhaust the VFIO address-space table (RAS::VFIO::MapDMAFailed).
    if configure_device == "spyre":
        del k_pages, v_pages, kv_cache, output
        gc.collect()


_SHAPES = [
    pytest.param([(1, 128)], id="decode_single", marks=pytest.mark.attention_smoke),
    pytest.param([(1, 300), (1, 64), (1, 512)], id="decode_batch"),
    pytest.param([(64, 64)], id="prefill_single", marks=pytest.mark.attention_smoke),
    pytest.param([(128, 128), (64, 200)], id="prefill_batch"),
    pytest.param([(64, 300), (1, 128)], id="mixed_batch"),
    # Unbucketed kv_lens: build() appends fully-masked padded blocks, which must leave the
    # output unchanged.
    pytest.param([(32, 65)], id="kv_padded_prefill"),
]

# Spyre always compiles attention, even eager (torch-spyre#4334); CPU runs the same kernels
# uncompiled, which is what keeps the kernels checked on a card-free host.
_DEVICE_MODES = [
    pytest.param("cpu", "NONE", id="device_cpu-eager"),
    pytest.param("spyre", "STOCK_TORCH_COMPILE", id="device_spyre-compiled"),
]


# Residency is a property of the layout plan rather than a result, so it is not asserted.
@pytest.mark.parametrize("seq_lens", _SHAPES)
@pytest.mark.parametrize(
    ("configure_device", "configure_compilation"), _DEVICE_MODES, indirect=True
)
def test_spyre_attn_core(default_vllm_config, seq_lens, configure_compilation, configure_device):
    _run_spyre_attn_test(
        seq_lens=seq_lens,
        block_size=128,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
    )


@pytest.mark.parametrize(
    ("configure_device", "configure_compilation", "seq_lens", "expect_fused_store"),
    [
        pytest.param("spyre", "STOCK_TORCH_COMPILE", [(1, 512)], True, id="decode-fused"),
        pytest.param("spyre", "STOCK_TORCH_COMPILE", [(32, 256)], True, id="prefill-fused"),
        pytest.param("cpu", "NONE", [(1, 512)], False, id="cpu_eager-decode"),
    ],
    indirect=["configure_device", "configure_compilation"],
)
def test_spyre_attn_fused_output_store(
    default_vllm_config,
    monkeypatch,
    seq_lens,
    expect_fused_store,
    configure_compilation,
    configure_device,
):
    """Assert both the output and which store path ran, so a guard that stops engaging
    cannot leave these cases green on the unfused store alone."""
    # The per-seq kernels are the subject; batched decode would take a lone decode.
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "0")
    _run_spyre_attn_test(
        seq_lens=seq_lens,
        block_size=128,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
        expect_fused_store=expect_fused_store,
    )


@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize(
    ("configure_device", "configure_compilation"), _DEVICE_MODES, indirect=True
)
def test_spyre_attn_decode_head_size(
    default_vllm_config, monkeypatch, head_size, configure_compilation, configure_device
):
    """Single-sequence decode across head sizes (regression for #284)."""
    # The per-seq decode kernel is the subject; batched decode would take a lone decode.
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "0")
    _run_spyre_attn_test(
        seq_lens=[(1, 256)],
        block_size=128,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
        head_size=head_size,
    )


@pytest.mark.parametrize(
    "seq_lens",
    [
        # context_len on a block boundary vs. mid-block hits different boundary tiles.
        pytest.param([(64, 256)], id="chunk_on_block_boundary"),
        pytest.param([(48, 300)], id="unaligned_chunk_mid_block"),
    ],
)
@pytest.mark.parametrize(
    ("configure_device", "configure_compilation"), _DEVICE_MODES, indirect=True
)
def test_spyre_attn_chunked_prefill(
    default_vllm_config, seq_lens, configure_compilation, configure_device
):
    _run_spyre_attn_test(
        seq_lens=seq_lens,
        block_size=128,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
    )


@pytest.mark.parametrize(
    ("configure_device", "configure_compilation"), _DEVICE_MODES, indirect=True
)
def test_mixed_batch_dispatches_decode_at_query_width_one(
    default_vllm_config, monkeypatch, configure_compilation, configure_device
):
    """A decoding sequence in a mixed batch keeps the query_len=1 kernel."""
    from vllm.config import get_current_vllm_config

    # Batched decode would serve the decode prefix before the per-seq loop runs.
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "0")
    chunk_len = 64
    chunk_bucket = SpyreAttnBucketer(get_current_vllm_config()).find_query_bucket(chunk_len)
    assert chunk_bucket is not None and chunk_bucket > 1

    _run_spyre_attn_test(
        seq_lens=[(chunk_len, 256), (1, 256), (1, 512)],
        block_size=128,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
        expect_query_widths={1, chunk_bucket},
    )


@pytest.mark.parametrize(
    ("seq_lens", "sliding_window"),
    [
        pytest.param([(1, 256), (32, 256)], None, id="mixed"),
        pytest.param([(32, 256)], 64, id="sliding_window"),
    ],
)
@pytest.mark.parametrize(
    ("configure_device", "configure_compilation"), _DEVICE_MODES, indirect=True
)
def test_spyre_attn_bfloat16(
    default_vllm_config, seq_lens, sliding_window, configure_compilation, configure_device
):
    """The kernels at bf16: KV cache and masks follow the model dtype.

    TP1 only, matching the platform: bf16 with TP>1 is rejected as all_reduce is fp16-only.
    """
    _run_spyre_attn_test(
        seq_lens=seq_lens,
        block_size=128,
        sliding_window=sliding_window,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
        dtype=torch.bfloat16,
    )


def test_kernel_block_sizes_accept_the_layer_spec():
    """vLLM's attention layer passes its `kv_cache_spec` positionally (sliding-window
    layers included); every spec gets the same stick-aligned constraint."""
    from vllm.v1.attention.backend import MultipleOf
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionBackend

    spec = SlidingWindowSpec(
        block_size=64, num_kv_heads=2, head_size=64, dtype=torch.float16, sliding_window=128
    )
    for sizes in (
        SpyreAttentionBackend.get_supported_kernel_block_sizes(spec),
        SpyreAttentionBackend.get_supported_kernel_block_sizes(),
    ):
        assert [s.base for s in sizes if isinstance(s, MultipleOf)] == [64]


def test_block_size_validation():
    """Test that SpyreAttentionMetadataBuilder validates block_size alignment.

    The Spyre paged attention backend requires block_size to be a multiple of 64
    for proper stick alignment during torch.compile. This test verifies the
    validation raises ValueError for invalid block sizes and accepts valid ones.
    """
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
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
        # The platform's own check_and_update_config already rounds an invalid
        # block_size up to a multiple of 64 during VllmConfig construction, so
        # restore the invalid value here to exercise the builder's own check.
        vllm_config.cache_config.block_size = block_size
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


def test_each_layer_in_one_kv_cache_tensor_gets_its_own_pages(monkeypatch):
    """A KVCacheTensor packs *every* attention layer into one allocation.

    One shared page buffer per tensor would alias all those layers onto the same block,
    so a later layer's KV write clobbers an earlier layer's. Prefill still looks right --
    each layer reads back its own fresh write -- and only decode is corrupted, which is
    why this needs an allocation-level guard rather than an output check.

    Card-free on purpose: `allocate_pages` is stubbed with per-call sentinels, so the
    card-less CI shards run this instead of skipping past it.
    """
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.config.compilation import CompilationConfig
    from vllm.v1.kv_cache_interface import (
        AttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    from spyre_inference.v1.attention.backends import spyre_attn as spyre_attn_mod
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    block_size, num_kv_heads, head_size, num_blocks = 128, 8, 128, 16
    layer_names = [f"layers.{i}.self_attn" for i in range(4)]

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B",
            max_model_len=1,
            dtype=torch.float16,
            trust_remote_code=True,
        ),
        cache_config=CacheConfig(block_size=block_size),
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )
    runner = TorchSpyreModelRunner(vllm_config, torch.device("spyre"))

    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
    )
    # The shape under test: all four layers in a single KVCacheTensor, as vLLM packs them.
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec.page_size_bytes * num_blocks * len(layer_names),
                layers=layer_names,
                layer_stride=spec.page_size_bytes * num_blocks,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)],
    )

    for layer_name in layer_names:
        fake_layer = Mock()
        fake_layer.kv_cache = None
        runner.compilation_config.static_forward_context[layer_name] = fake_layer

    calls = []

    def _fake_allocate_pages(cls, blocks, allocate_spec, device):
        calls.append((blocks, allocate_spec, device))
        return Mock(name=f"pages-{len(calls)}")

    monkeypatch.setattr(
        spyre_attn_mod.SpyreAttentionImpl,
        "allocate_pages",
        classmethod(_fake_allocate_pages),
    )

    caches = runner.initialize_kv_cache_tensors(kv_cache_config, [block_size])

    assert set(caches) == set(layer_names)
    # The bug this guards against: one buffer handed to every layer.
    assert len({id(caches[name]) for name in layer_names}) == len(layer_names), (
        "layers packed into one KVCacheTensor must not share a page buffer"
    )
    # One allocation per layer, each sized for the full per-layer block count -- not one
    # allocation carved up by layer_stride.
    assert len(calls) == len(layer_names)
    assert all(blocks == num_blocks for blocks, _, _ in calls)


def test_supported_dtypes_includes_bfloat16():
    """Nothing selects bf16 by default, but `--dtype bfloat16` has to reach the kernels
    rather than be rejected during backend selection."""
    from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionBackend

    assert torch.float16 in SpyreAttentionBackend.supported_dtypes
    assert torch.bfloat16 in SpyreAttentionBackend.supported_dtypes
    assert "bfloat16" in SpyreAttentionBackend.supported_kv_cache_dtypes


def test_kv_cache_dtype_that_disagrees_with_the_model_is_rejected(default_vllm_config):
    """The kernels read a page at model dtype with no cast on the way in, so an explicit
    `--kv-cache-dtype` naming the other 2-byte dtype has to fail at construction."""
    from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl

    kwargs = dict(num_heads=8, head_size=64, scale=0.125, num_kv_heads=8)
    for accepted in ("auto", "float16"):
        assert SpyreAttentionImpl(kv_cache_dtype=accepted, **kwargs) is not None

    with pytest.raises(ValueError, match="does not match the model dtype"):
        SpyreAttentionImpl(kv_cache_dtype="bfloat16", **kwargs)


@pytest.mark.parametrize("kv_cache_dtype", ["fp8_e5m2", "fp8_inc", "nvfp4", "fp8_per_token_head"])
def test_unsupported_quantized_kv_cache_dtype_is_named(default_vllm_config, kv_cache_dtype):
    """Upstream sizes these as uint8 bytes like fp8, so the constructor is the only guard
    against storing them as e4m3; the message names the dtypes Spyre does take."""
    with pytest.raises(ValueError, match=f"{kv_cache_dtype} is not supported on Spyre.*fp8_e4m3"):
        SpyreAttentionImpl(
            num_heads=8, head_size=128, scale=1.0, num_kv_heads=2, kv_cache_dtype=kv_cache_dtype
        )


def test_attention_sinks_are_held_kv_major(default_vllm_config):
    """gpt-oss passes one sink logit per query head. The kernels index heads kv-major,
    ``[num_kv_heads, num_queries_per_kv]``, which is upstream's flat head order reshaped;
    a transposed reshape would pair each sink with another head and fail no shape check.
    Values are read only when first used, after the weight loader has filled them."""
    from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl

    kwargs = dict(num_heads=8, head_size=64, scale=0.125, num_kv_heads=2)
    assert SpyreAttentionImpl(sinks=None, **kwargs)._device_sinks(torch.device("cpu")) is None

    sinks = torch.nn.Parameter(torch.empty(8, dtype=torch.float32), requires_grad=False)
    impl = SpyreAttentionImpl(sinks=sinks, **kwargs)
    sinks.data.copy_(torch.arange(8, dtype=torch.float32))
    held = impl._device_sinks(torch.device("cpu"))
    assert held.dtype == torch.float16
    torch.testing.assert_close(held, torch.arange(8, dtype=torch.float16).reshape(2, 4, 1))

    with pytest.raises(ValueError, match="one logit per query head"):
        SpyreAttentionImpl(sinks=torch.zeros(2, dtype=torch.float16), **kwargs)


@pytest.mark.parametrize(
    "configure_device",
    [pytest.param("spyre", id="device_spyre")],
    indirect=True,
)
def test_attention_sinks_follow_the_parameter_moved_to_spyre(
    default_vllm_config, monkeypatch, configure_device
):
    """gpt-oss and granite register the sinks on their own attention module, so
    ``model.to("spyre")`` replaces that Parameter and a reload writes the replacement. A
    host-only test cannot see this: on CPU the move keeps the original."""
    from vllm.model_executor.layers.attention.attention import Attention

    from spyre_inference.v1.worker.model_load import move_model_to_spyre

    # move_model_to_spyre patches the class for good; restore it after the test.
    monkeypatch.setattr(Attention, "_apply", Attention._apply)
    model = torch.nn.Module()
    model.sinks = torch.nn.Parameter(torch.arange(8, dtype=torch.float16), requires_grad=False)
    model.attn = torch.nn.Module()
    model.attn.impl = SpyreAttentionImpl(
        num_heads=8, head_size=64, scale=0.125, num_kv_heads=2, sinks=model.sinks
    )
    device = torch.device("spyre")

    move_model_to_spyre(model, device)
    assert model.sinks.device.type == "spyre"

    model.sinks.data.copy_(torch.full((8,), 3.0, dtype=torch.float16))
    model.attn.impl.process_weights_after_loading(torch.float16)
    torch.testing.assert_close(
        model.attn.impl._device_sinks(device).cpu(),
        torch.full((2, 4, 1), 3.0, dtype=torch.float16),
    )


@pytest.mark.parametrize("impl_cls", [SpyreAttentionImpl, SpyreEncoderAttentionImpl])
def test_alibi_is_rejected(default_vllm_config, impl_cls):
    """The backstop behind the platform's `uses_alibi` check, which is a heuristic over HF
    config fields; without the raise a model it misses would run without its positional bias."""
    with pytest.raises(NotImplementedError, match="ALiBi is not supported"):
        impl_cls(num_heads=8, head_size=64, scale=0.125, num_kv_heads=8, alibi_slopes=[0.5] * 8)


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

    k_pages_cpu = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    v_pages_cpu = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)

    # Pre-populate KV cache
    for i in range(kv_len):
        block_idx = i // block_size
        block_offset = i % block_size
        k_pages_cpu[block_idx][block_offset] = torch.randn(num_kv_heads, head_size, dtype=dtype)
        v_pages_cpu[block_idx][block_offset] = torch.randn(num_kv_heads, head_size, dtype=dtype)

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
    mask_none = metadata_none.attention_mask_stacks[0][0]
    mask_swa = metadata_swa.attention_mask_stacks[0][0]

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
    mask_eq = metadata_eq.attention_mask_stacks[0][0]
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
    mask_gt = metadata_gt.attention_mask_stacks[0][0]
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
    mask_mixed_0 = metadata_mixed.attention_mask_stacks[0][0]
    attended_mixed_0 = (mask_mixed_0[0] == 0).nonzero().flatten().tolist()
    assert attended_mixed_0 == [0, 1, 2, 3], f"Seq 0: expected [0,1,2,3], got {attended_mixed_0}"

    # Seq 1 (kv_len=9): query at position 8, attends to [5, 6, 7, 8]
    mask_mixed_1 = metadata_mixed.attention_mask_stacks[1][0]
    attended_mixed_1 = (mask_mixed_1[0] == 0).nonzero().flatten().tolist()
    assert attended_mixed_1 == [5, 6, 7, 8], f"Seq 1: expected [5,6,7,8], got {attended_mixed_1}"


def test_mask_stacks_are_mirrored_lazily_per_sequence(default_vllm_config, monkeypatch):
    """Only requested stacks transfer, once each, with their exact byte volume."""
    torch.set_default_device("cpu")

    block_size = 64
    metadata = _padded_mask_metadata(
        [(1, 512), (1, 320), (7, 320), (33, 512)],
        block_size=block_size,
        max_num_blocks=_num_blocks_buckets(block_size)[-1],
    )

    stacks_cpu = metadata.attention_mask_stacks
    assert stacks_cpu is not None
    assert stacks_cpu[0].shape[0] > 1
    assert stacks_cpu[1].storage_offset() > 0
    assert len({stack.shape[1] for stack in stacks_cpu}) > 1

    calls: list[torch.Tensor] = []

    def counting_convert(tensor, device=None, dtype=None):
        calls.append(tensor)
        return tensor.clone(memory_format=torch.contiguous_format)

    monkeypatch.setattr(spyre_attn, "convert", counting_convert)

    # Equivalent to a fully batched decode returning before any per-sequence read.
    assert calls == []
    assert metadata.attention_mask_stacks_device is None

    # Equivalent to a mixed batch reading only its per-sequence suffix. A later
    # layer taking the same path reuses both mirrors.
    suffix = {
        seq_idx: _mirror_mask_stack(metadata, seq_idx, torch.device("cpu")) for seq_idx in (2, 3)
    }
    torch.testing.assert_close(calls, [stacks_cpu[2], stacks_cpu[3]], rtol=0, atol=0)
    assert sum(t.numel() * t.element_size() for t in calls) == sum(
        stacks_cpu[i].numel() * stacks_cpu[i].element_size() for i in (2, 3)
    )
    for seq_idx in (2, 3):
        assert _mirror_mask_stack(metadata, seq_idx, torch.device("cpu")) is suffix[seq_idx]
    torch.testing.assert_close(calls, [stacks_cpu[2], stacks_cpu[3]], rtol=0, atol=0)

    # A later layer that cannot use batched decode fills only the missing prefix.
    prefix = {
        seq_idx: _mirror_mask_stack(metadata, seq_idx, torch.device("cpu")) for seq_idx in (0, 1)
    }
    torch.testing.assert_close(
        calls, [stacks_cpu[2], stacks_cpu[3], stacks_cpu[0], stacks_cpu[1]], rtol=0, atol=0
    )
    assert metadata.attention_mask_stacks_device is not None
    for seq_idx, stack_device in (prefix | suffix).items():
        assert metadata.attention_mask_stacks_device[seq_idx] is stack_device
        assert torch.equal(stack_device, stacks_cpu[seq_idx])
        assert stack_device.is_contiguous()
        assert stack_device.storage_offset() == 0


@pytest.mark.parametrize(
    ("seq_lens", "builds_tables"),
    [
        pytest.param([(1, 256)] * 4, False, id="all_decode"),
        pytest.param([(1, 256), (1, 256), (7, 256)], True, id="mixed"),
    ],
)
def test_page_index_tables_skipped_on_a_fully_batched_step(
    default_vllm_config, monkeypatch, seq_lens, builds_tables
):
    """A step the batched kernel fully serves uploads no per-sequence index table."""
    torch.set_default_device("cpu")
    # Not the enable_batched_decode fixture: conftest skips its users, and no kernel runs here.
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
    metadata = _padded_mask_metadata(seq_lens, block_size=64)
    assert metadata.padded_num_seqs is not None
    assert metadata.num_decode_seqs == sum(q == 1 for q, _ in seq_lens)

    attn_impl = SpyreAttentionImpl(num_heads=32, head_size=128, scale=128**-0.5, num_kv_heads=8)
    dispatched: list[int] = []
    monkeypatch.setattr(attn_impl, "_batched_decode_preconditions_met", lambda *a: True)
    monkeypatch.setattr(attn_impl, "_run_batched_decode_dispatch", lambda *a: dispatched.append(1))

    class _TablesBuilt(Exception):
        pass

    def built(*args):
        # Stop before the per-sequence kernels, which this CPU test cannot run.
        raise _TablesBuilt

    monkeypatch.setattr(attn_impl, "build_index_tables", built)

    query = torch.zeros(len(seq_lens), 32, 128, dtype=torch.float16)
    output = torch.empty_like(query)
    pages = torch.empty(0)
    args = (query, pages, pages, metadata, output, torch.device("cpu"))
    if builds_tables:
        with pytest.raises(_TablesBuilt):
            attn_impl._online_softmax_attention(*args)
    else:
        assert attn_impl._online_softmax_attention(*args) is output
        assert metadata.kernel_index_tables is None
    assert dispatched == [1]


def test_empty_mask_stack_cannot_be_mirrored(default_vllm_config):
    torch.set_default_device("cpu")
    metadata = _padded_mask_metadata([(1, 0), (1, 65)], max_num_blocks=4)

    with pytest.raises(AssertionError, match="empty mask stack"):
        _mirror_mask_stack(metadata, 0, torch.device("cpu"))


class _StubAttentionLayer:
    """Enough of `Attention` for `attn_layer.install` to decide and patch."""

    def __init__(self, attn_type: str):
        self.attn_type = attn_type
        self.impl = Mock(spec=["do_kv_cache_update", "kv_slot_views", "kv_write_index"])
        self.kv_sharing_target_layer_name = None
        self.query_quant = None
        self.kv_cache: list[torch.Tensor] = []


def test_install_patches_layers_not_the_attention_class():
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.backend import AttentionType

    from spyre_inference.v1.attention import attn_layer

    class_forward = Attention.forward
    decoder = _StubAttentionLayer(AttentionType.DECODER)
    encoder = _StubAttentionLayer(AttentionType.ENCODER_ONLY)

    holder = attn_layer.install([decoder, encoder])

    assert Attention.forward is class_forward
    assert decoder.spyre_slots is holder
    assert decoder.forward.__func__ is attn_layer._spyre_attention_forward
    assert not hasattr(encoder, "spyre_slots")
    assert not hasattr(encoder, "forward")

    # No cache bound, so there is no device to mirror onto and nothing to publish.
    holder.publish_null(8)
    assert holder.slots is None


@pytest.mark.parametrize("max_num_seqs", [4, 5, 6, 8, 10, 16, 32])
@pytest.mark.parametrize("max_model_len", [512, 1536, 2048, 4096, 10000])
def test_batched_decode_chunking_covers_every_block(
    default_vllm_config,
    enable_batched_decode,
    max_num_seqs: int,
    max_model_len: int,
) -> None:
    """Every block of every sequence reaches a chunk, for any bucket pair.

    blocks_per_chunk is capped, not chosen as a divisor, so the block axis has to
    be padded up to a multiple of it. Neither bucket lattice is all powers of two
    -- _powers_of_two_up_to appends n itself -- so an uneven pair is reachable
    from ordinary engine args (max_model_len=1536 gives 12 blocks, and 8 does not
    divide 12). Getting this wrong drops the tail blocks and then raises on the
    mask reshape, i.e. crashes a decode step. Card-free on purpose: the
    integration tests all land on power-of-two buckets, where it cannot fire.
    """
    from vllm.config import get_current_vllm_config

    torch.set_default_device("cpu")
    block_size = 128
    num_kv_heads, head_size = 2, 64

    vllm_config = get_current_vllm_config()
    vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    vllm_config.model_config.max_model_len = max_model_len

    max_blocks = (max_model_len + block_size - 1) // block_size
    # Longest sequence the config allows: the block bucket is picked off the
    # real block count, so this is what reaches the lattice's top entry.
    ctx = max_model_len

    for num_seqs in range(1, max_num_seqs + 1):
        seq_lens = torch.full((num_seqs,), ctx, dtype=torch.int32)
        query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32)
        block_table = torch.arange(num_seqs * max_blocks, dtype=torch.int32).reshape(
            num_seqs, max_blocks
        )
        slot_mapping = (seq_lens.to(torch.int64) - 1) + torch.arange(num_seqs) * ctx

        md = _build_metadata(
            num_query_heads=num_kv_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            block_size=block_size,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )

        assert md.blocks_per_chunk is not None, (
            f"batched decode declined num_seqs={num_seqs}, which it should accept"
        )
        bpc = md.blocks_per_chunk
        padded = md.padded_batch_blocks
        assert padded is not None
        assert padded % bpc == 0, (
            f"max_num_seqs={max_num_seqs} max_model_len={max_model_len} "
            f"num_seqs={num_seqs}: {padded} blocks is not a multiple of "
            f"blocks_per_chunk={bpc}"
        )
        assert padded >= max_blocks, (
            f"max_num_seqs={max_num_seqs} max_model_len={max_model_len} "
            f"num_seqs={num_seqs}: chunks cover {padded} of {max_blocks} blocks"
        )
        # The mask carries the same axis, so a mismatch here is the reshape that
        # would have raised inside build(). Its KV axis follows the walk in force:
        # broadcast for the plain loop, materialized for the tiled one.
        assert md.mask_by_chunk_cpu is not None
        assert md.mask_by_chunk_cpu.shape == (
            padded,
            md.padded_num_seqs,
            num_kv_heads if tile_loop.USE_FOR_EACH_TILE else 1,
            1,
            block_size,
        )


def test_batched_decode_mask_follows_the_layers_num_kv_heads(
    default_vllm_config,
    enable_batched_decode,
) -> None:
    """The decode mask's KV axis follows the layer's head count.

    A model with per-layer head counts (gemma-4) has attention layers whose
    num_kv_heads is not `model_config.get_num_kv_heads()`. Taking the model-level
    count asks for the wrong number of elements, so every batched-decode variant
    fails to compile.
    """
    from vllm.config import get_current_vllm_config

    torch.set_default_device("cpu")
    block_size = 128
    # This group's own counts; get_num_kv_heads() reports 8 below, the 4x-too-wide
    # broadcast this guards against.
    num_kv_heads, num_query_heads, head_size = 2, 4, 64

    vllm_config = get_current_vllm_config()
    vllm_config.scheduler_config.max_num_seqs = 1
    vllm_config.model_config.max_model_len = 2048

    num_seqs = 1
    ctx = 256
    blocks_per_seq = ctx // block_size
    seq_lens = torch.full((num_seqs,), ctx, dtype=torch.int32)
    query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32)
    block_table = torch.arange(num_seqs * blocks_per_seq, dtype=torch.int32).reshape(
        num_seqs, blocks_per_seq
    )
    slot_mapping = (seq_lens.to(torch.int64) - 1) + torch.arange(num_seqs) * ctx

    md = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        block_table=block_table,
        slot_mapping=slot_mapping,
        model_num_kv_heads=8,
    )

    assert md.blocks_per_chunk is not None, "batched decode declined this batch"
    assert md.mask_by_chunk_cpu is not None
    assert md.mask_by_chunk_cpu.shape[2] == num_kv_heads, (
        f"mask carries {md.mask_by_chunk_cpu.shape[2]} KV heads; this group has {num_kv_heads}"
    )
    # This layer is GQA (4 query heads over 2 KV heads), so a materialized
    # query-group axis would double the transfer for nothing; it broadcasts instead.
    assert md.mask_by_chunk_cpu.shape[3] == 1, (
        f"query-group axis is {md.mask_by_chunk_cpu.shape[3]} wide; it should broadcast "
        f"in the kernel, not be transferred {num_query_heads // num_kv_heads} times"
    )
    # The add the kernel actually performs, one chunk at a time: the mask tile must
    # broadcast against the score tile.
    scores = torch.zeros(
        md.blocks_per_chunk,
        md.padded_num_seqs,
        num_kv_heads,
        num_query_heads // num_kv_heads,
        block_size,
    )
    masked = scores + md.mask_by_chunk_cpu.narrow(0, 0, md.blocks_per_chunk)
    assert masked.shape == scores.shape


def _decode_reference_fp32(
    query: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    page_ids: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    qpk: int,
    head_size: int,
) -> torch.Tensor:
    """Per-sequence softmax over each sequence's own blocks, no chunking.

    k/v_pages: [pages, KV, block_size, D]. page_ids: [num_seqs, num_blocks].
    mask: [num_seqs, num_blocks, block_size].
    """
    num_seqs, num_blocks = page_ids.shape
    out = torch.zeros(num_seqs, num_kv_heads * qpk, head_size, dtype=torch.float32)
    for s in range(num_seqs):
        q = query[s].reshape(num_kv_heads, qpk, head_size)
        k = torch.cat([k_pages[page_ids[s, b]].transpose(0, 1) for b in range(num_blocks)], dim=0)
        v = torch.cat([v_pages[page_ids[s, b]].transpose(0, 1) for b in range(num_blocks)], dim=0)
        # [KV, qpk, kv_len]
        scores = torch.einsum("hqd,thd->hqt", q, k) * scale + mask[s].reshape(-1)
        probs = torch.softmax(scores, dim=-1)
        out[s] = torch.einsum("hqt,thd->hqd", probs, v).reshape(num_kv_heads * qpk, head_size)
    return out.reshape(num_seqs, num_kv_heads * qpk, head_size)


@pytest.mark.parametrize(
    ("kv_lens", "sliding_window"),
    [
        pytest.param([256, 512, 128, 384, 256, 512, 128, 384], None, id="no_window_ragged"),
        pytest.param([64, 512, 512, 512], None, id="no_window_short_first"),
        pytest.param([1, 128, 128, 128], None, id="no_window_zero_full_blocks"),
        pytest.param([512, 512, 512, 512], 256, id="window_uniform"),
    ],
)
def test_bucketed_block_ids_match_scalar_fill(
    default_vllm_config, enable_batched_decode, kv_lens: list[int], sliding_window: int | None
) -> None:
    block_size = 64
    seq_lens = [(1, kv) for kv in kv_lens]
    metadata = _padded_mask_metadata(seq_lens, block_size=block_size, sliding_window=sliding_window)
    assert metadata.chunk_page_ids_cpu is not None
    assert metadata.blocks_per_chunk is not None
    assert metadata.padded_num_seqs is not None

    got = metadata.chunk_page_ids_cpu
    assert got.shape[0] == metadata.padded_batch_blocks
    bt = metadata.block_table
    active = metadata.active_block_indices
    b_blocks = got.shape[0]
    for s, kv in enumerate(kv_lens):
        abs_blocks = (
            active[s] if active is not None else list(range((kv + block_size - 1) // block_size))
        )
        n_use = min(len(abs_blocks), b_blocks)
        for b in range(n_use):
            assert got[b, s].item() == bt[s, abs_blocks[b]].item(), (
                f"seq={s} block={b}: got {got[b, s].item()}, expected {bt[s, abs_blocks[b]].item()}"
            )
        for b in range(n_use, b_blocks):
            assert got[b, s].item() == 0, f"seq={s} block={b} (past end): got {got[b, s].item()}"


@pytest.mark.parametrize("batched_decode", ["0", "1"])
def test_batched_decode_metadata_follows_env(
    default_vllm_config, monkeypatch, batched_decode: str
) -> None:
    """build() computes the batched-decode metadata only when the path is enabled.

    With the env off the fields stay None at any batch size, so anything reading them
    must gate on the same flag — `_batched_decode_preconditions_met` does.
    """
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", batched_decode)
    metadata = _padded_mask_metadata([(1, 256)] * 4, block_size=64)

    if batched_decode == "1":
        assert metadata.padded_num_seqs is not None
        assert metadata.padded_batch_blocks is not None
        assert metadata.rep_row_ids_cpu is not None
        assert metadata.chunk_page_ids_cpu is not None
    else:
        assert metadata.padded_num_seqs is None
        assert metadata.padded_batch_blocks is None
        assert metadata.rep_row_ids_cpu is None
        assert metadata.chunk_page_ids_cpu is None


def _seq_mask(metadata, seq_idx: int) -> torch.Tensor:
    """Concatenate a sequence's per-block mask tiles into [aligned_q, num_blocks*block]."""
    return torch.cat(list(metadata.attention_mask_stacks[seq_idx]), dim=-1)


@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([(7, 256)], id="prefill_q7_pads_to_32"),
        pytest.param([(1, 256)], id="decode_q1_clamp_lower_bound"),
        pytest.param([(33, 512)], id="prefill_q33_pads_to_64"),
        pytest.param([(32, 256)], id="prefill_q32_exact_no_padding"),
    ],
)
def test_padded_mask_rows_equal_last_real_row(default_vllm_config, seq_lens):
    """Padded query rows carry row query_len-1's mask, not a fully-masked row."""
    torch.set_default_device("cpu")
    query_len = seq_lens[0][0]
    metadata = _padded_mask_metadata(seq_lens)

    aligned = metadata.aligned_query_lens[0]
    if query_len == 1:
        assert aligned == 1
        return
    assert aligned >= query_len

    mask = _seq_mask(metadata, 0)
    last_real = mask[query_len - 1]
    for row in range(query_len, aligned):
        assert torch.equal(mask[row], last_real), (
            f"padded row {row} differs from last real row {query_len - 1}"
        )


@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([(7, 256)], id="prefill_q7"),
        pytest.param([(1, 320)], id="decode_q1"),
        pytest.param([(40, 512)], id="prefill_q40"),
        # Unbucketed kv_lens, so build() appends fully-masked padded blocks
        # (given a table with headroom -- see max_num_blocks above): 65 -> 2
        # real blocks, already a bucket; 300 -> 5 real padded to 8; 513 -> 9
        # real padded to 16.
        pytest.param([(7, 65)], id="prefill_q7_padded_blocks"),
        pytest.param([(1, 300)], id="decode_q1_padded_blocks"),
        pytest.param([(40, 513)], id="prefill_q40_padded_blocks"),
    ],
)
def test_padded_mask_rows_are_not_fully_masked(default_vllm_config, seq_lens):
    """No mask row is fully masked: attn = tile_output / tile_sum would be NaN."""
    torch.set_default_device("cpu")
    metadata = _padded_mask_metadata(seq_lens, max_num_blocks=_num_blocks_buckets()[-1])
    mask = _seq_mask(metadata, 0)
    mask_min = torch.finfo(torch.float16).min

    open_per_row = (mask > mask_min).sum(dim=-1)
    assert (open_per_row > 0).all(), f"fully-masked row(s) at {(open_per_row == 0).nonzero()}"


def test_padded_mask_rows_isolated_across_sequences(default_vllm_config):
    """A packed batch's padded rows never take a neighbour's mask."""
    torch.set_default_device("cpu")
    seq_lens = [(7, 256), (33, 512), (1, 128)]
    metadata = _padded_mask_metadata(seq_lens)
    aligned = metadata.aligned_query_lens
    assert aligned[2] == 1
    assert aligned[0] > 1 and aligned[1] > 1

    for seq_idx, (query_len, _) in enumerate(seq_lens):
        mask = _seq_mask(metadata, seq_idx)
        assert mask.shape[0] == aligned[seq_idx]
        last_real = mask[query_len - 1]
        for row in range(query_len, aligned[seq_idx]):
            assert torch.equal(mask[row], last_real), (
                f"seq {seq_idx} padded row {row} does not match its own row {query_len - 1}"
            )


@pytest.mark.parametrize(
    "sliding_window",
    [pytest.param(None, id="full_attention"), pytest.param(128, id="sliding_window128")],
)
def test_per_sequence_masks_match_a_solo_build(default_vllm_config, monkeypatch, sliding_window):
    """Each sequence's mask must not depend on who else is in its batch."""
    monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "1,8,64,512")
    torch.set_default_device("cpu")
    seq_lens = [(7, 256), (33, 512), (1, 128), (40, 300)]
    # Headroom for the padded block counts, as a real engine's table has.
    table_width = _num_blocks_buckets()[-1]

    batched = _padded_mask_metadata(
        seq_lens, sliding_window=sliding_window, max_num_blocks=table_width
    )
    assert sorted(set(batched.aligned_query_lens)) == [1, 8, 64]

    for seq_idx, one in enumerate(seq_lens):
        solo = _padded_mask_metadata(
            [one], sliding_window=sliding_window, max_num_blocks=table_width
        )
        assert batched.aligned_query_lens[seq_idx] == solo.aligned_query_lens[0]
        assert torch.equal(_seq_mask(batched, seq_idx), _seq_mask(solo, 0))


def test_query_row_table_clamp_matches_mask_clamp(default_vllm_config):
    """The gather's row table and the mask clamp padded rows to the same row."""
    torch.set_default_device("cpu")
    seq_lens = [(7, 256), (33, 512)]
    metadata = _padded_mask_metadata(seq_lens)

    row_tables = _build_query_row_tables(metadata, torch.device("cpu"))
    starts = metadata.query_start_loc[:-1].tolist()

    for seq_idx, (query_len, _) in enumerate(seq_lens):
        aligned = metadata.aligned_query_lens[seq_idx]
        rows = row_tables[seq_idx][:aligned].tolist()
        expected = [starts[seq_idx] + min(q, query_len - 1) for q in range(aligned)]
        assert rows == expected, f"seq {seq_idx} row table {rows} != {expected}"
        assert row_tables[seq_idx].storage_offset() == 0


@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([(7, 256)], id="prefill_q7"),
        pytest.param([(33, 512)], id="prefill_q33"),
    ],
)
def test_sliding_window_padded_mask_rows_equal_last_real_row(default_vllm_config, seq_lens):
    """Same padded-row invariant on the sliding-window path (_build_single_tile)."""
    torch.set_default_device("cpu")
    query_len = seq_lens[0][0]
    metadata = _padded_mask_metadata(seq_lens, sliding_window=128)

    aligned = metadata.aligned_query_lens[0]
    mask = _seq_mask(metadata, 0)
    last_real = mask[query_len - 1]
    for row in range(query_len, aligned):
        assert torch.equal(mask[row], last_real), (
            f"padded row {row} differs from last real row {query_len - 1}"
        )


def test_sliding_window_block_skip_unaffected_by_clamp(default_vllm_config):
    """Clamping padded rows forward must not change which blocks stay active."""
    torch.set_default_device("cpu")
    block_size, window = 64, 128
    query_len, kv_len = 7, 512
    metadata = _padded_mask_metadata(
        [(query_len, kv_len)], block_size=block_size, sliding_window=window
    )

    context_len = kv_len - query_len
    first_active = max(0, context_len - window + 1) // block_size
    num_blocks = (kv_len + block_size - 1) // block_size
    real_active = list(range(first_active, num_blocks))
    assert metadata.active_block_indices is not None
    active = metadata.active_block_indices[0]
    # This case does pad, so assert the padded length first: the tail checks below
    # are vacuous on an empty tail and would pass a regression to no padding.
    assert len(active) == num_blocks > len(real_active)
    # The real prefix is what the clamp could have disturbed; the tail is bucket
    # padding, which repeats the last real block.
    assert active[: len(real_active)] == real_active
    assert set(active[len(real_active) :]) == {real_active[-1]}


def test_sliding_window_mask_and_page_rows_share_active_block_order(default_vllm_config):
    """Mask row i and page-table row i describe the same active logical block."""
    torch.set_default_device("cpu")
    block_size, window = 64, 128
    query_len, kv_len = 1, 512
    num_blocks = kv_len // block_size
    # Physical pages deliberately run opposite to logical blocks, so accidentally
    # treating either row number as a page id cannot pass.
    block_table = torch.arange(100, 100 + num_blocks, dtype=torch.int32).flip(0).unsqueeze(0)
    metadata = _build_metadata(
        num_query_heads=8,
        num_kv_heads=2,
        head_size=64,
        block_size=block_size,
        seq_lens=torch.tensor([kv_len], dtype=torch.int32),
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
        block_table=block_table,
        slot_mapping=torch.tensor(
            [int(block_table[0, -1]) * block_size + block_size - 1], dtype=torch.int64
        ),
        sliding_window=window,
    )

    active = metadata.active_block_indices
    stacks = metadata.attention_mask_stacks
    tables = metadata.page_index_tables_cpu
    assert active is not None and stacks is not None and tables is not None
    assert active[0][0] > 0
    assert stacks[0].shape[0] == tables[0].shape[0] == len(active[0])

    mask_min = torch.finfo(stacks[0].dtype).min
    open_positions = []
    for row, logical_block in enumerate(active[0]):
        assert tables[0][row, 0] == block_table[0, logical_block]
        open_offsets = (stacks[0][row, 0] > mask_min).nonzero().flatten().tolist()
        open_positions.extend(logical_block * block_size + offset for offset in open_offsets)
    assert open_positions == list(range(kv_len - window, kv_len))


def _num_blocks_buckets(block_size: int = 64) -> list[int]:
    """The recorder's num_blocks buckets for the fixture's config."""
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    vllm_config.cache_config.block_size = block_size
    return SpyreAttnBucketer(vllm_config).num_blocks_buckets


@pytest.mark.parametrize(
    ("kv_len", "expected"),
    [
        pytest.param(65, 2, id="kv65_to_2"),
        pytest.param(300, 8, id="kv300_to_8"),
        pytest.param(256, 4, id="kv256_exact_noop"),
        pytest.param(1025, 32, id="kv1025_to_32"),
    ],
)
def test_padded_num_blocks_lands_on_a_bucket(default_vllm_config, kv_len, expected):
    torch.set_default_device("cpu")
    buckets = _num_blocks_buckets()
    assert expected in buckets

    metadata = _padded_mask_metadata([(1, kv_len)], max_num_blocks=buckets[-1])

    assert metadata.padded_num_blocks == [expected]
    assert metadata.attention_mask_stacks[0].shape[0] == expected
    # One table per sequence, sized to that sequence's own active-block count.
    assert [t.shape[0] for t in metadata.page_index_tables_cpu] == [expected]


def test_padded_tiles_are_finfo_min_and_prefix_is_unchanged(default_vllm_config):
    """Padded blocks are exactly finfo.min; the real prefix is bit-identical
    regardless of how much headroom the block table has beyond the bucket
    build() actually pads to -- build() always pads to a bucket (no unpadded
    fallback), so both tables here must have at least that much headroom."""
    torch.set_default_device("cpu")
    block_size = 64
    kv_len, query_len = 300, 32
    real_blocks = (kv_len + block_size - 1) // block_size
    buckets = _num_blocks_buckets()

    narrow = _padded_mask_metadata(
        [(query_len, kv_len)], block_size=block_size, max_num_blocks=buckets[-1]
    )
    assert narrow.padded_num_blocks[0] > real_blocks

    mask_min = torch.finfo(torch.float16).min
    for b in range(real_blocks, narrow.padded_num_blocks[0]):
        tile = narrow.attention_mask_stacks[0][b]
        assert torch.equal(tile, torch.full_like(tile, mask_min)), f"block {b} is not finfo.min"

    # The real prefix must be bit-identical regardless of extra table
    # headroom past the bucket build() actually uses.
    wide = _padded_mask_metadata(
        [(query_len, kv_len)], block_size=block_size, max_num_blocks=buckets[-1] * 2
    )
    assert wide.padded_num_blocks == narrow.padded_num_blocks
    for b in range(real_blocks):
        assert torch.equal(narrow.attention_mask_stacks[0][b], wide.attention_mask_stacks[0][b]), (
            f"real block {b} changed"
        )


def test_zero_kv_len_stays_at_zero_blocks(default_vllm_config):
    """Zero real blocks must not be padded: a fully-masked tile would divide by zero."""
    torch.set_default_device("cpu")
    metadata = _padded_mask_metadata([(1, 0), (1, 65)], max_num_blocks=_num_blocks_buckets()[-1])

    assert metadata.padded_num_blocks[0] == 0
    assert metadata.attention_mask_stacks[0].shape[0] == 0
    assert metadata.padded_num_blocks[1] == 2


def test_sliding_window_is_left_unpadded(default_vllm_config):
    torch.set_default_device("cpu")
    metadata = _padded_mask_metadata(
        [(7, 300)], block_size=64, sliding_window=128, max_num_blocks=_num_blocks_buckets()[-1]
    )
    assert metadata.padded_num_blocks is None


def test_spyre_attn_write_index(default_vllm_config):
    """kv_write_index maps a slot to the row holding each of that token's heads."""
    from vllm.config import get_current_vllm_config

    block_size, num_kv_heads = 64, 4
    get_current_vllm_config().cache_config.block_size = block_size
    impl = SpyreAttentionImpl(num_heads=8, head_size=64, scale=1.0, num_kv_heads=num_kv_heads)

    slots = torch.tensor([0, 3, block_size, 5 * block_size + 63], dtype=torch.int64)
    per_head = impl.kv_write_index(slots, torch.device("cpu"))

    assert len(per_head) == num_kv_heads
    for h, rows in enumerate(per_head):
        assert rows.shape == slots.shape
        # Its own allocation, not a row of a [KV, T] tensor (torch-spyre#3851).
        assert rows.storage_offset() == 0
        for t, slot in enumerate(slots.tolist()):
            block, offset = divmod(slot, block_size)
            assert rows[t] == (block * num_kv_heads + h) * block_size + offset
    # Every destination distinct: two writes sharing a row would silently drop one.
    stacked = torch.stack(per_head)
    assert stacked.unique().numel() == stacked.numel()


def test_fp8_kv_cache_dtype(default_vllm_config):
    assert "fp8" in SpyreAttentionBackend.supported_kv_cache_dtypes

    impl = SpyreAttentionImpl(
        num_heads=8, head_size=128, scale=1.0, num_kv_heads=2, kv_cache_dtype="fp8"
    )
    assert impl.kv_cache_fp8
    # A float8 stick is 128 elements wide.
    with pytest.raises(ValueError, match="head_size % 128"):
        SpyreAttentionImpl(
            num_heads=8, head_size=64, scale=1.0, num_kv_heads=2, kv_cache_dtype="fp8"
        )


def test_encoder_layers_ignore_an_fp8_cache_dtype(default_vllm_config):
    """Upstream hands encoder layers the engine's cache dtype, but they keep no cache."""
    from vllm.v1.attention.backend import AttentionType

    from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
        SpyreEncoderAttentionImpl,
    )

    impl = SpyreEncoderAttentionImpl(
        num_heads=8,
        head_size=64,
        scale=1.0,
        num_kv_heads=2,
        kv_cache_dtype="fp8",
        attn_type=AttentionType.ENCODER_ONLY,
    )
    assert not impl.kv_cache_fp8


def test_fp8_kv_scales_mirror_the_layer_floats(default_vllm_config):
    impl = SpyreAttentionImpl(
        num_heads=8, head_size=128, scale=1.0, num_kv_heads=2, kv_cache_dtype="fp8"
    )
    layer = Mock(_k_scale_float=0.25, _v_scale_float=0.5)
    impl.prepare_kv_scales(layer)
    assert impl._kernel_kv_scales() == (0.25, 0.5)
    assert impl._read_scales() == (0.25, 0.5)
    # A model-dtype cache passes v_scale=None, so its kernels trace as before.
    plain = SpyreAttentionImpl(num_heads=8, head_size=128, scale=1.0, num_kv_heads=2)
    plain.prepare_kv_scales(layer)
    assert plain._kernel_kv_scales() == (None, None)
    assert plain._read_scales() == (1.0, None)


def test_fp8_quantize_saturates():
    """The float8 cast maps an overflow to NaN; the store must clamp first."""
    x = torch.tensor([1e4, -1e4, 3.0], dtype=DTYPE)
    stored = quantize_kv(x, 0.05, torch.float8_e4m3fn)
    assert stored.float().tolist() == [448.0, -448.0, 60.0]


def _fp8_pages(num_pages, num_kv_heads, block_size, head_size, k_scale, v_scale):
    """Random head-major pages stored as float8, and the model-dtype pages they decode to."""
    k = (torch.randn(num_pages, num_kv_heads, block_size, head_size) * 3).to(DTYPE)
    v = torch.randn(num_pages, num_kv_heads, block_size, head_size).to(DTYPE)
    k8 = quantize_kv(k, k_scale, torch.float8_e4m3fn)
    v8 = quantize_kv(v, v_scale, torch.float8_e4m3fn)
    return (
        k8,
        v8,
        (k8.float() * k_scale).to(DTYPE),
        (v8.float() * v_scale).to(DTYPE),
    )


def _fp8_layer(k_scale, v_scale, **attrs):
    """An fp8-cache attention layer whose upstream-resolved scales are (k_scale, v_scale)."""
    from vllm.model_executor.layers.attention.attention import Attention

    layer = Mock(spec=Attention)
    layer.impl = SpyreAttentionImpl(
        num_heads=8, head_size=128, scale=1.0, num_kv_heads=2, kv_cache_dtype="fp8"
    )
    layer._k_scale_float, layer._v_scale_float = k_scale, v_scale
    for name, value in attrs.items():
        setattr(layer, name, value)
    return layer


def _resolve(layers, static_forward_context=None):
    from spyre_inference.v1.worker.model_load import resolve_kv_scales

    model = Mock(modules=Mock(return_value=layers))
    return resolve_kv_scales(model, static_forward_context or {})


def _warnings(monkeypatch):
    from spyre_inference.v1.worker import model_load

    warnings = []
    monkeypatch.setattr(model_load.logger, "warning_once", lambda msg, *args: warnings.append(msg))
    return warnings


def test_distinct_kv_scales_warn_about_warmup(default_vllm_config, monkeypatch):
    """Each distinct (k_scale, v_scale) pair is its own compile, so startup says so."""
    first, other = _fp8_layer(0.25, 0.5), _fp8_layer(0.25, 0.5)
    warnings = _warnings(monkeypatch)

    _resolve([first, other])
    assert warnings == [], "one shared pair compiles once"
    other._k_scale_float = 0.125
    _resolve([first, other])
    assert len(warnings) == 1 and "distinct" in warnings[0]


def test_per_head_kv_scales_warn(default_vllm_config, monkeypatch):
    """Upstream reduces a per-head scheme to its max scale; the cache cannot do better."""
    layer = _fp8_layer(0.25, 0.5, _k_scale=torch.tensor([0.25, 0.025]))
    warnings = _warnings(monkeypatch)

    _resolve([layer])

    assert layer.impl._kernel_kv_scales() == (0.25, 0.5)
    assert len(warnings) == 1 and "per-head" in warnings[0]


def test_kv_sharing_layer_reads_with_its_target_scales(default_vllm_config, monkeypatch):
    """A layer reading another's cache dequantizes with the scales that cache was written at."""
    target = _fp8_layer(0.05, 0.02)
    # A sharing layer stores nothing, so a calibrated checkpoint leaves its own at 1.0.
    reader = _fp8_layer(1.0, 1.0, kv_sharing_target_layer_name="layers.0.attn")
    _resolve([reader, target], {"layers.0.attn": target})

    assert reader.impl._kernel_kv_scales() == (0.05, 0.02)


def test_kv_sharing_layer_and_target_must_agree_on_fp8(default_vllm_config, monkeypatch):
    """A reader whose target skipped fp8 through kv_cache_dtype_skip_layers would decode
    fp16 pages as fp8, so it fails at load and names the setting."""
    target = _fp8_layer(1.0, 1.0)
    target.impl.kv_cache_fp8 = False
    reader = _fp8_layer(
        1.0, 1.0, layer_name="layers.1.attn", kv_sharing_target_layer_name="layers.0.attn"
    )
    with pytest.raises(ValueError, match="kv_cache_dtype_skip_layers"):
        _resolve([reader, target], {"layers.0.attn": target})


def test_reload_weights_refreshes_the_kv_scales(default_vllm_config, monkeypatch):
    """A reload passes its arguments through, re-reads the scales, and re-warms only on a
    change, so an armed compile guard never sees the new scales mid-request."""
    import contextlib
    from types import SimpleNamespace

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from spyre_inference.v1.worker import compile_guard
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    layer = _fp8_layer(0.25, 0.5)
    runner = object.__new__(TorchSpyreModelRunner)
    monkeypatch.setattr(runner, "get_model", lambda: Mock(modules=Mock(return_value=[layer])))
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    _resolve([layer])
    calls, warmups, allowed = [], [], []
    new_k_scale = 0.25

    @contextlib.contextmanager
    def allow_compile(reason):
        allowed.append(True)
        yield
        allowed.pop()

    def reprocess(self, *args, **kwargs):
        calls.append((args, kwargs))
        layer._k_scale_float = new_k_scale

    monkeypatch.setattr(GPUModelRunner, "reload_weights", reprocess)
    monkeypatch.setattr(compile_guard, "allow_compile", allow_compile)
    monkeypatch.setattr(runner, "warming_up_model", lambda: warmups.append(bool(allowed)))

    runner.reload_weights(weights_path="/ckpt", is_checkpoint_format=False)
    assert calls == [((), {"weights_path": "/ckpt", "is_checkpoint_format": False})]
    assert warmups == [], "unchanged scales leave every graph valid"

    new_k_scale = 0.125
    runner.reload_weights()
    assert layer.impl._kernel_kv_scales() == (0.125, 0.5)
    assert warmups == [True], "a change re-warms once, with the compile guard suspended"


def test_fp8_prefill_reads_match_the_dequantized_cache(monkeypatch):
    """A float8 cache read through the scales equals the model-dtype read of its values."""

    monkeypatch.setattr(tile_loop, "USE_FOR_EACH_TILE", False)
    set_random_seed(0)
    num_kv_heads, qpk, head_size, block_size, num_blocks, padded_q = 2, 4, 128, 64, 3, 4
    num_heads = num_kv_heads * qpk
    k_scale, v_scale = 0.05, 0.02
    k8, v8, k_dq, v_dq = _fp8_pages(6, num_kv_heads, block_size, head_size, k_scale, v_scale)

    query = torch.randn(padded_q, num_heads, head_size).to(DTYPE)
    rows = torch.arange(padded_q, dtype=torch.int32)
    index_table = torch.zeros(num_blocks, INT32_ELEMS_PER_STICK, dtype=torch.int32)
    index_table[:, 0] = torch.tensor([4, 1, 3])
    mask = torch.zeros(num_blocks, padded_q, block_size, dtype=DTYPE)
    mask[-1, :, 40:] = float("-inf")
    common = dict(
        query=query,
        query_row_index=rows,
        page_index_table=index_table,
        mask_stack=mask,
        num_blocks=num_blocks,
        padded_query_len=padded_q,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )
    scale = head_size**-0.5

    got = page_attn_prefill_kernel(
        **common, k_pages=k8, v_pages=v8, scale=scale * k_scale, v_scale=v_scale
    )
    want = page_attn_prefill_kernel(**common, k_pages=k_dq, v_pages=v_dq, scale=scale)
    torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)


# Named clear of the conftest batched-decode skip: this is the untiled walk on CPU.
def test_fp8_decode_batch_reads_match_the_dequantized_cache(monkeypatch):
    from spyre_inference.v1.attention.ops import batched_decode

    monkeypatch.setattr(tile_loop, "USE_FOR_EACH_TILE", False)
    monkeypatch.setattr(batched_decode, "USE_FOR_EACH_TILE", False)
    set_random_seed(0)
    num_kv_heads, qpk, head_size, block_size = 2, 4, 128, 64
    num_seqs, bpc, num_chunks = 2, 2, 3
    k_scale, v_scale = 0.05, 0.02
    k8, v8, k_dq, v_dq = _fp8_pages(6, num_kv_heads, block_size, head_size, k_scale, v_scale)

    query = torch.randn(num_seqs, num_kv_heads * qpk, head_size).to(DTYPE)
    rep_row_ids = torch.arange(num_seqs, dtype=torch.int64).repeat(bpc)
    page_ids = torch.randint(0, 6, (num_chunks * bpc, num_seqs), dtype=torch.int64)
    mask = torch.zeros(num_chunks * bpc, num_seqs, 1, 1, block_size, dtype=DTYPE)
    mask[-1, ..., 32:] = float("-inf")
    scale = head_size**-0.5
    common = dict(
        query=query,
        rep_row_ids=rep_row_ids,
        chunk_page_ids=page_ids,
        mask_by_chunk=mask,
        num_seqs=num_seqs,
        blocks_per_chunk=bpc,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=qpk,
        block_size=block_size,
        head_size=head_size,
    )

    got = batched_decode_kernel(
        **common, k_pages=k8, v_pages=v8, scale=scale * k_scale, v_scale=v_scale
    )
    want = batched_decode_kernel(**common, k_pages=k_dq, v_pages=v_dq, scale=scale)
    torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)


_SLOT_CASES = [
    # One token is the decode shape and its own failure mode: a fused-QKV source at
    # T == 1 is flagged contiguous at a nonzero offset, which the kernel works around.
    ("single_token", [2], [17]),
    ("single_block", [0, 0, 0], [0, 1, 63]),
    ("across_blocks", [0, 2, 5], [7, 0, 63]),
    ("repeat_block", [3, 3, 3, 1], [0, 31, 63, 5]),
]


@pytest.mark.parametrize(
    "configure_device",
    [
        pytest.param("cpu", id="device_cpu"),
        pytest.param("spyre", id="device_spyre"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "label,block_indices,block_offsets", _SLOT_CASES, ids=[c[0] for c in _SLOT_CASES]
)
@pytest.mark.parametrize("source_layout", ["contiguous", "qkv_split"])
def test_spyre_attn_scatter(
    default_vllm_config, configure_device, label, block_indices, block_offsets, source_layout
):
    """The store writes exactly the mapped rows and leaves every other row zero.

    Zero rather than a sentinel fill: the fill would go through this same store.
    """
    import warnings

    from spyre_testing_plugin.attn_helpers import _fused_qkv_kv_views
    from torch_spyre.ops.fallbacks import FallbackWarning
    from vllm.config import get_current_vllm_config

    set_random_seed(0)
    num_tokens = len(block_indices)
    num_kv_heads, head_size, block_size = 8, 128, 64
    num_blocks = max(block_indices) + 1
    cache_device = torch.device(configure_device)
    slots = torch.tensor(
        [b * block_size + o for b, o in zip(block_indices, block_offsets)], dtype=torch.int64
    )

    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=DTYPE)
    value = torch.randn(num_tokens, num_kv_heads, head_size, dtype=DTYPE)

    get_current_vllm_config().cache_config.block_size = block_size
    impl = SpyreAttentionImpl(
        num_heads=num_kv_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
    )

    k_actual, v_actual = _fresh_pages(num_blocks, num_kv_heads, block_size, head_size, cache_device)
    kv_cache = SpyrePagedKVCache(k_pages=k_actual, v_pages=v_actual)

    k_expected = torch.zeros((num_blocks, num_kv_heads, block_size, head_size), dtype=DTYPE)
    v_expected = k_expected.clone()
    for t, (block, offset) in enumerate(zip(block_indices, block_offsets)):
        k_expected[block, :, offset] = key[t]
        v_expected[block, :, offset] = value[t]

    if source_layout == "qkv_split":
        query = torch.randn(num_tokens, num_kv_heads, head_size, dtype=DTYPE)
        key_src, value_src = _fused_qkv_kv_views(query, key, value, cache_device)
    else:
        key_src, value_src = convert(key, cache_device), convert(value, cache_device)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FallbackWarning)
        _write(impl, kv_cache, key_src, value_src, slots, cache_device)

    fallback_msgs = [str(w.message) for w in caught if issubclass(w.category, FallbackWarning)]
    assert not any("index_copy" in m for m in fallback_msgs), (
        f"the head-major KV scatter fell back to CPU: {fallback_msgs}"
    )

    torch.testing.assert_close(k_actual.to("cpu"), k_expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v_actual.to("cpu"), v_expected, atol=1e-2, rtol=1e-2)

    if configure_device == "spyre":
        del k_actual, v_actual, kv_cache
        gc.collect()


@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
def test_spyre_attn_store_fuses_to_one_kernel(default_vllm_config, num_kv_heads, configure_device):
    """A partial fusion would leave the V writes and later heads outside the returned K
    view ``attn_layer`` orders the read against: a race, not a slowdown."""
    import torch._inductor.metrics as inductor_metrics

    set_random_seed(0)
    block_size, head_size, num_blocks, num_tokens = 64, 128, 32, 8
    cache_device = torch.device(configure_device)

    k_pages, v_pages = _fresh_pages(num_blocks, num_kv_heads, block_size, head_size, cache_device)
    key = convert(torch.randn(num_tokens, num_kv_heads, head_size, dtype=DTYPE), cache_device)
    value = convert(torch.randn(num_tokens, num_kv_heads, head_size, dtype=DTYPE), cache_device)
    slots = torch.arange(num_tokens, dtype=torch.int64)
    base = torch.div(slots, block_size, rounding_mode="floor") * num_kv_heads * block_size
    rows = [
        convert(base + slots % block_size + h * block_size, cache_device)
        for h in range(num_kv_heads)
    ]

    # Its own compile, so the count covers this store alone rather than an artifact a
    # previous test already built.
    torch._dynamo.reset()
    store = torch.compile(reshape_and_cache_kernel, dynamic=False)
    inductor_metrics.reset()
    store(key, value, k_pages.view(-1, head_size), v_pages.view(-1, head_size), rows)

    assert inductor_metrics.generated_kernel_count == 1, (
        f"the head-major store lowered to {inductor_metrics.generated_kernel_count} kernels "
        f"at num_kv_heads={num_kv_heads}; only a fully fused store keeps the returned K "
        "view a dependency for every write"
    )

    del k_pages, v_pages
    gc.collect()
    torch._dynamo.reset()


@pytest.mark.parametrize("seq_lens", _SHAPES)
# Unit scales are a checkpoint without a kv_cache_scheme; the others a calibrated one.
@pytest.mark.parametrize("kv_scales", [(1.0, 1.0), (0.05, 0.02)], ids=["unit", "scaled"])
@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
def test_spyre_attn_fp8_kv_cache(
    default_vllm_config, seq_lens, kv_scales, configure_compilation, configure_device
):
    _run_spyre_attn_test(
        seq_lens=seq_lens,
        block_size=128,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
        kv_cache_dtype="fp8",
        kv_scales=kv_scales,
    )


@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
def test_spyre_attn_dispatches_by_query_width(
    default_vllm_config, monkeypatch, configure_compilation, configure_device
):
    """A decode takes the LX-resident kernel and a wider query the batched one. Sending a
    wide query to the decode kernel pays for residency the query width already amortises."""
    # Batched decode now covers every batch size, and it serves a mixed batch's decode
    # prefix before the per-seq loop runs, so the per-seq kernels are the subject here.
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "0")
    envs.clear_env_cache()

    called = []

    def spy(name, fn):
        def wrapper(*args, **kwargs):
            called.append(name)
            return fn(*args, **kwargs)

        return wrapper

    # Patched before the impl is built: __init__ reads these globals into the impl.
    monkeypatch.setattr(
        spyre_attn,
        "_page_attn_decode_compiled",
        spy("decode", spyre_attn._page_attn_decode_compiled),
    )
    monkeypatch.setattr(
        spyre_attn,
        "_page_attn_prefill_compiled",
        spy("prefill", spyre_attn._page_attn_prefill_compiled),
    )

    _run_spyre_attn_test(
        seq_lens=[(1, 300), (64, 200)],
        block_size=128,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
    )

    assert sorted(called) == ["decode", "prefill"], called


# Not named "batched_decode": tests/conftest.py skips those under the tiled walk.
@pytest.mark.parametrize(
    ("seq_lens", "use_sinks"),
    [
        pytest.param([(1, 7000), (1, 200), (1, 200), (1, 200)], False, id="long_and_shorts"),
        pytest.param([(1, 6000), (1, 3000), (1, 1000), (1, 300)], False, id="graded"),
        pytest.param(
            [(1, 7000), (1, 200), (1, 200), (1, 200), (64, 300)], False, id="with_prefill"
        ),
        # Sinks barely move the 7000-token row, so the short group is what catches a miss.
        pytest.param([(1, 7000), (1, 200), (1, 200), (1, 200)], True, id="long_and_shorts_sinks"),
    ],
)
@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
def test_spyre_attn_grouped_decode_matches_reference(
    default_vllm_config, monkeypatch, seq_lens, use_sinks, configure_compilation, configure_device
):
    from vllm.config import get_current_vllm_config

    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
    monkeypatch.setenv("SPYRE_JAGGED_DECODE", "1")
    envs.clear_env_cache()
    cfg = get_current_vllm_config()
    cfg.scheduler_config.max_num_seqs = 8
    cfg.model_config.max_model_len = 8192

    groups_run = []
    run_grouped = SpyreAttentionImpl._run_grouped_decode

    def spy(self, query_dev, k_pages, v_pages, attn_metadata, output, device):
        groups_run.append(len(attn_metadata.decode_groups))
        return run_grouped(self, query_dev, k_pages, v_pages, attn_metadata, output, device)

    monkeypatch.setattr(SpyreAttentionImpl, "_run_grouped_decode", spy)

    _run_spyre_attn_test(
        seq_lens=seq_lens,
        block_size=128,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
        use_sinks=use_sinks,
    )

    assert groups_run == [2], f"expected one grouped step of two groups, got {groups_run}"


@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
def test_spyre_attn_block_sizes(
    default_vllm_config, block_size, configure_compilation, configure_device
):
    """block_size sets the row stride between a token's heads, so the store's index
    arithmetic changes with it."""
    _run_spyre_attn_test(
        seq_lens=[(1, 300), (64, 200)],
        block_size=block_size,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
    )


@pytest.mark.parametrize("seq_lens", [pytest.param([(1, 300), (64, 200)], id="mixed_batch")])
@pytest.mark.parametrize("sliding_window", [64, 256])
@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
def test_spyre_attn_sliding_window(
    default_vllm_config, seq_lens, sliding_window, configure_compilation, configure_device
):
    _run_spyre_attn_test(
        seq_lens=seq_lens,
        block_size=64,
        sliding_window=sliding_window,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
    )


@pytest.mark.parametrize("seq_lens", [pytest.param([(1, 300), (64, 200)], id="mixed_batch")])
@pytest.mark.parametrize("soft_cap", [30.0])
@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
def test_spyre_attn_soft_cap(
    default_vllm_config, seq_lens, soft_cap, configure_compilation, configure_device
):
    _run_spyre_attn_test(
        seq_lens=seq_lens,
        block_size=64,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
        soft_cap=soft_cap,
    )


@pytest.mark.parametrize(
    ("seq_lens", "sliding_window", "multi_seq_decode"),
    [
        # The per-seq decode (LX-resident) and prefill kernels.
        pytest.param([(1, 300), (64, 200)], None, False, id="per_seq_mixed"),
        pytest.param([(1, 300), (64, 200)], 128, False, id="per_seq_sliding_window(128)"),
        # The multi-sequence decode kernel, which serves decode by default: kv 512 at block
        # 64 spans several chunks, so its per-slot merge runs and must count each sink
        # once. Named apart from "batched_decode", which tests/conftest.py skips wholesale.
        pytest.param([(1, 512), (1, 300), (1, 64)], None, True, id="multi_seq_decode"),
    ],
)
# One KV head cannot tell the kv-major sink order from a transposed one; two can.
@pytest.mark.parametrize(
    "num_kv_heads", [pytest.param(1, id="8q_1kv"), pytest.param(2, id="8q_2kv")]
)
@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
def test_spyre_attn_sinks(
    default_vllm_config,
    monkeypatch,
    seq_lens,
    sliding_window,
    multi_seq_decode,
    num_kv_heads,
    configure_compilation,
    configure_device,
):
    """Attention sinks (gpt-oss: 64 q / 8 kv heads, head_dim 64; 8 / 1 per rank at TP8)."""
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1" if multi_seq_decode else "0")
    envs.clear_env_cache()
    calls: list[str] = []
    real_decode_fn = spyre_attn._batched_decode_compiled

    def spy(*args, **kwargs):
        calls.append("multi_seq")
        return real_decode_fn(*args, **kwargs)

    # Patched before the impl is built: __init__ reads this global into the impl.
    monkeypatch.setattr(spyre_attn, "_batched_decode_compiled", spy)
    _run_spyre_attn_test(
        seq_lens=seq_lens,
        block_size=64,
        sliding_window=sliding_window,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
        use_sinks=True,
        # Measured within 0.007 of the reference; dropping the sinks moves the output by up
        # to 0.3 here, which the default 0.2 + 20% bound would absorb.
        tolerance=(0.05, 0.05),
        num_query_heads=8,
        num_kv_heads=num_kv_heads,
        head_size=64,
    )
    assert bool(calls) == multi_seq_decode, calls


@pytest.mark.parametrize(
    "num_query_heads,num_kv_heads",
    [pytest.param(8, 8, id="mha"), pytest.param(8, 1, id="mqa")],
)
@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
def test_spyre_attn_head_configs(
    default_vllm_config, num_query_heads, num_kv_heads, configure_compilation, configure_device
):
    _run_spyre_attn_test(
        seq_lens=[(1, 300), (64, 200)],
        block_size=64,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
    )


def test_kv_cache_shape_matches_runner_allocation():
    """The advertised shape, the runner's allocation and the kernels share one layout.

    If get_kv_cache_shape drifts, vLLM code that allocates from the contract (KV
    transfer, future tests) allocates a transposed cache."""
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.config.compilation import CompilationConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    if not spyre_available():
        pytest.skip("Spyre device not available")

    block_size, num_kv_heads, head_size, num_blocks = 128, 8, 128, 16
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B", max_model_len=1, dtype=torch.float16, trust_remote_code=True
        ),
        cache_config=CacheConfig(block_size=block_size),
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )
    runner = TorchSpyreModelRunner(vllm_config, torch.device("spyre"))
    spec = AttentionSpec(
        block_size=block_size, num_kv_heads=num_kv_heads, head_size=head_size, dtype=DTYPE
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec.page_size_bytes * num_blocks,
                layers=["layers.0.self_attn"],
                layer_stride=spec.page_size_bytes * num_blocks,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layers.0.self_attn"], kv_cache_spec=spec)],
    )

    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        impl = SpyreAttentionImpl(
            num_heads=32, head_size=head_size, scale=head_size**-0.5, num_kv_heads=num_kv_heads
        )
    fake_layer = Mock()
    fake_layer.kv_cache = None
    fake_layer.impl = impl
    runner.compilation_config.static_forward_context["layers.0.self_attn"] = fake_layer

    caches = runner.initialize_kv_cache_tensors(kv_cache_config, [block_size])
    k_pages = caches["layers.0.self_attn"].k_pages
    v_pages = caches["layers.0.self_attn"].v_pages

    expected = SpyreAttentionBackend.get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size
    )
    assert k_pages.shape == expected == (num_blocks, num_kv_heads, block_size, head_size)
    assert v_pages.shape == expected

    # The kernel gathers a (page, kv_head) row, which has to stay whole at device
    # position 0 (torch-spyre#3705) or the gather costs the whole cache, not one page.
    for pages in (k_pages, v_pages):
        device_size = pages.device_tensor_layout().device_size
        assert device_size[0] == num_blocks * num_kv_heads
        assert device_size[1] == block_size

    del caches, k_pages, v_pages
    gc.collect()


def test_page_attn_decode_matches_fp32_reference():
    """The per-sequence decode kernel matches the suite's CPU reference."""
    from spyre_testing_plugin.attn_helpers import ref_attn

    torch.set_default_device("cpu")
    set_random_seed(0)
    kv, qpk, d, block, blocks = 8, 4, 128, 64, 5
    heads, query_len = kv * qpk, 1
    # A ragged tail, so the last page is half masked rather than whole.
    kv_len = (blocks - 1) * block + block // 2
    k = torch.randn(blocks * kv, block, d)
    v = torch.randn(blocks * kv, block, d)
    # The kernel reads its row out of a wider staging buffer, and not the first one.
    query = torch.randn(query_len + 2, heads, d)
    rows = (torch.arange(query_len, dtype=torch.int32) + 2) % (query_len + 2)
    # The base's page table: stick-wide rows, page id in column 0.
    page_ids = torch.tensor([3, 1, 4, 0, 2], dtype=torch.int32)
    page_table = torch.zeros(blocks, INT32_ELEMS_PER_STICK, dtype=torch.int32)
    page_table[:, 0] = page_ids
    kv_row_pool = torch.arange(blocks * kv, dtype=torch.int32).reshape(blocks, kv, 1)
    # Causal: query row q sits at absolute position kv_len - query_len + q.
    q_abs = kv_len - query_len + torch.arange(query_len).unsqueeze(1)
    pos = torch.arange(blocks * block).reshape(blocks, 1, block)
    allow = (pos <= q_abs) & (pos < kv_len)
    masks = torch.where(allow, 0.0, torch.finfo(torch.float32).min)

    for soft_cap in (0.0, 30.0):
        got = page_attn_decode_kernel(
            query=query,
            query_row_index=rows,
            k_pages=k,
            v_pages=v,
            page_index_table=page_table,
            kv_row_pool=kv_row_pool,
            mask_stack=masks,
            scale=d**-0.5,
            num_blocks=blocks,
            padded_query_len=query_len,
            num_heads=heads,
            num_kv_heads=kv,
            head_size=d,
            block_size=block,
            logits_soft_cap=soft_cap,
        )
        expected = ref_attn(
            query=query.index_select(0, rows.to(torch.int64)),
            # ref_attn indexes a page's token axis first.
            key_cache=k.reshape(blocks, kv, block, d).permute(0, 2, 1, 3),
            value_cache=v.reshape(blocks, kv, block, d).permute(0, 2, 1, 3),
            query_lens=[query_len],
            kv_lens=[kv_len],
            block_tables=page_ids.unsqueeze(0),
            block_size=block,
            scale=d**-0.5,
            soft_cap=soft_cap,
        )
        assert got.shape == expected.shape
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "num_seqs,b_seqs,num_blocks,bpc,num_kv_heads,qpk,ragged",
    [
        pytest.param(4, 4, 8, 8, 2, 1, False, id="one_chunk"),
        pytest.param(4, 4, 8, 2, 2, 1, False, id="four_chunks"),
        pytest.param(4, 4, 8, 1, 2, 1, False, id="bpc_1"),
        pytest.param(3, 4, 8, 4, 2, 1, False, id="padded_batch_rows"),
        pytest.param(4, 4, 8, 2, 2, 4, True, id="gqa_ragged"),
        pytest.param(5, 8, 12, 4, 1, 2, True, id="uneven_buckets_ragged"),
        pytest.param(4, 4, 12, 8, 2, 1, True, id="padded_block_axis_ragged"),
    ],
)
def test_multi_seq_decode_matches_fp32_reference(
    num_seqs: int,
    b_seqs: int,
    num_blocks: int,
    bpc: int,
    num_kv_heads: int,
    qpk: int,
    ragged: bool,
    monkeypatch,
) -> None:
    """The head-major page read and reduction agree with a full-sequence reference.

    Card-free and fp32, so it pins the reduction, the read and the entry-major, kv-minor
    row order the mask is broadcast in, rather than the fp16 tolerances.
    """
    from spyre_inference.v1.attention.ops import batched_decode

    monkeypatch.setattr(tile_loop, "USE_FOR_EACH_TILE", False)
    monkeypatch.setattr(batched_decode, "USE_FOR_EACH_TILE", False)

    torch.set_default_device("cpu")
    set_random_seed(0)

    block_size, head_size = 16, 8
    num_heads = num_kv_heads * qpk
    padded_blocks = ((num_blocks + bpc - 1) // bpc) * bpc
    scale = 0.5

    n_pages = padded_blocks * b_seqs + 1
    query = torch.randn(num_seqs, num_heads * head_size, dtype=torch.float32)
    k_pages = torch.randn(n_pages, num_kv_heads, block_size, head_size, dtype=torch.float32)
    v_pages = torch.randn(n_pages, num_kv_heads, block_size, head_size, dtype=torch.float32)

    # Page 0 is the padding page, exactly as the builder leaves it.
    page_ids = torch.zeros(b_seqs, padded_blocks, dtype=torch.int64)
    mask = torch.full((b_seqs, padded_blocks, block_size), float("-inf"), dtype=torch.float32)
    for s in range(num_seqs):
        n_use = max(1, num_blocks - s) if ragged else num_blocks
        tail = (block_size // 2) if ragged else block_size
        kv_len = (n_use - 1) * block_size + tail
        for b in range(n_use):
            page_ids[s, b] = 1 + s * padded_blocks + b
            mask[s, b, : min(block_size, kv_len - b * block_size)] = 0.0
    mask[num_seqs:, 0] = torch.finfo(torch.float16).min
    mask.masked_fill_(torch.isneginf(mask), torch.finfo(torch.float32).min)

    rep_row_ids = torch.arange(b_seqs, dtype=torch.int64).clamp(max=num_seqs - 1)
    rep_row_ids = rep_row_ids.repeat(bpc)
    # One index row per page, in int64 for eager CPU indexing.
    chunk_page_ids = page_ids.t().contiguous()
    mask_by_chunk = (
        mask.transpose(0, 1)
        .unsqueeze(2)
        .unsqueeze(3)
        .expand(padded_blocks, b_seqs, num_kv_heads, qpk, block_size)
        .contiguous()
    )

    query_padded = torch.zeros(b_seqs, num_heads * head_size, dtype=torch.float32)
    query_padded[:num_seqs] = query

    actual = batched_decode_kernel(
        query=query_padded,
        rep_row_ids=rep_row_ids,
        k_pages=k_pages,
        v_pages=v_pages,
        chunk_page_ids=chunk_page_ids,
        mask_by_chunk=mask_by_chunk,
        scale=scale,
        num_seqs=b_seqs,
        blocks_per_chunk=bpc,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=qpk,
        block_size=block_size,
        head_size=head_size,
    )
    expected = _decode_reference_fp32(
        query_padded,
        k_pages,
        v_pages,
        page_ids,
        mask,
        scale,
        num_kv_heads,
        qpk,
        head_size,
    )

    assert torch.isfinite(actual[:num_seqs]).all()
    torch.testing.assert_close(actual[:num_seqs], expected[:num_seqs], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
def test_spyre_attn_batched_decode_uses_plain_page_ids(default_vllm_config, configure_compilation):
    """The batched kernel gathers whole pages, so its index is the builder's page ids.

    Guards against folding them onto ``page * KV + kv`` again: that moves the same bytes
    with num_kv_heads times the gather entries, which measured ~2x the kernel time.

    The fold's premise is that it buys LX residency, as it does for the per-sequence kernel.
    It does not here: the whole-page gather is already LX-resident, so the fold is a dead end
    rather than a tuning gap.
    """
    from spyre_testing_plugin.attn_helpers import _build_metadata

    torch.set_default_device("cpu")
    num_query_heads, num_kv_heads, head_size, block_size = 8, 4, 64, 64
    num_seqs, blocks_per_seq = 8, 4

    query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32)
    block_table = 1 + torch.arange(num_seqs * blocks_per_seq, dtype=torch.int32).reshape(
        num_seqs, blocks_per_seq
    )
    attn_metadata = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=torch.full((num_seqs,), blocks_per_seq * block_size, dtype=torch.int32),
        query_start_loc=query_start_loc,
        block_table=block_table,
        slot_mapping=torch.zeros(num_seqs, dtype=torch.int64),
    )
    impl = SpyreAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
    )

    assert impl._batched_decode_supported()
    assert impl._batched_decode_preconditions_met(attn_metadata, 1 + num_seqs * blocks_per_seq), (
        "an all-decode batch of 8 must reach the batched path, or the test proves nothing"
    )
    assert attn_metadata.chunk_page_ids_cpu is not None
    assert attn_metadata.padded_num_seqs is not None
    assert attn_metadata.blocks_per_chunk is not None
    table = attn_metadata.chunk_page_ids_cpu
    assert table.shape == (
        attn_metadata.padded_batch_blocks,
        attn_metadata.padded_num_seqs,
    )
    assert table.dtype == torch.int32


@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
def test_dispatch_declines_a_gather_spanning_the_cache(default_vllm_config, configure_compilation):
    """Dispatch keeps the recorder's page bound, so it never reaches a variant warmup skipped.

    Two decode seqs of two blocks each over a three-page cache is the TP2 gemma-4 shape
    that compiled the skipped variant mid-serving and hit a torch-spyre codegen assert.
    """
    from spyre_testing_plugin.attn_helpers import _build_metadata

    torch.set_default_device("cpu")
    num_query_heads, num_kv_heads, head_size, block_size = 8, 2, 64, 64
    num_seqs, blocks_per_seq = 2, 2

    attn_metadata = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=torch.full((num_seqs,), blocks_per_seq * block_size, dtype=torch.int32),
        query_start_loc=torch.arange(num_seqs + 1, dtype=torch.int32),
        block_table=torch.ones(num_seqs, blocks_per_seq, dtype=torch.int32),
        slot_mapping=torch.zeros(num_seqs, dtype=torch.int64),
    )
    impl = SpyreAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
    )
    assert attn_metadata.padded_num_seqs is not None
    assert attn_metadata.blocks_per_chunk is not None
    entries = attn_metadata.padded_num_seqs * attn_metadata.blocks_per_chunk

    assert impl._batched_decode_preconditions_met(attn_metadata, entries + 1)
    assert not impl._batched_decode_preconditions_met(attn_metadata, entries)
    assert not impl._batched_decode_preconditions_met(attn_metadata, entries - 1)


@pytest.fixture()
def batched_decode_calls(monkeypatch):
    """Pin ``SPYRE_BATCHED_DECODE`` on and count the batched dispatches.

    A fall back to the per-seq loop still matches the reference, so without the count the
    tests below would pass while testing nothing.
    """
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
    calls: list[bool] = []
    original = SpyreAttentionImpl._run_batched_decode

    def counting(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SpyreAttentionImpl, "_run_batched_decode", counting)
    return calls


@pytest.mark.parametrize(
    "seq_lens,soft_cap",
    [
        pytest.param([(1, 256), (1, 512), (1, 128), (1, 384)] * 2, None, id="bucket_exact(N=8)"),
        pytest.param(
            [(1, 128), (1, 256), (1, 384), (1, 512), (1, 128)], None, id="bucket_pad(N=5_bucket=8)"
        ),
        # Leading decode prefix batched, trailing prefill through the per-seq loop.
        pytest.param([(1, 256)] * 4 + [(64, 256)], None, id="mixed_decode_prefix(N=4+1)"),
        pytest.param([(1, 256), (1, 512), (1, 128), (1, 384)], 50.0, id="soft_cap(N=4)"),
    ],
)
@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
def test_spyre_attn_batched_decode_correctness(
    default_vllm_config,
    batched_decode_calls,
    seq_lens,
    soft_cap,
    configure_compilation,
    configure_device,
):
    """The batched decode kernel over folded pages, against the CPU reference.

    Not bit-exact with the per-seq kernel and cannot be: the chunked reduction sums in a
    different order under one shared max.
    """
    _run_spyre_attn_test(
        seq_lens=seq_lens,
        block_size=128,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
        soft_cap=soft_cap,
    )
    assert batched_decode_calls, "the batch fell back to the per-seq loop"


@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
def test_spyre_attn_warmup_records_a_batched_decode_variant(
    default_vllm_config, batched_decode_calls, configure_compilation, configure_device
):
    """Warmup can trace this layout's batched kernel, so serving does not compile it.

    One bucket, not the whole enumeration: what matters is that the recorder reaches the
    folded gather through builder-produced metadata.
    """
    import sys
    from unittest.mock import MagicMock

    from vllm.config import get_current_vllm_config

    from spyre_inference.v1.attention.backends.spyre_attn import (
        SpyreAttentionMetadataBuilder,
    )
    from spyre_inference.v1.attention.spyre_attn_bucketer import (
        SpyreAttnBatchedDecodeBucket,
        batched_decode_chunking,
    )

    torch.set_default_device("cpu")
    num_query_heads, num_kv_heads, head_size, block_size = 8, 2, 64, 64
    num_seqs, num_blocks, num_pages = 4, 4, 64

    vllm_config = get_current_vllm_config()
    vllm_config.model_config.get_num_attention_heads = Mock(return_value=num_query_heads)
    vllm_config.model_config.get_num_kv_heads = Mock(return_value=num_kv_heads)
    vllm_config.cache_config.block_size = block_size
    spec = AttentionSpec(
        block_size=block_size, num_kv_heads=num_kv_heads, head_size=head_size, dtype=DTYPE
    )
    builder = SpyreAttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["layers.0.self_attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )
    impl = SpyreAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
    )
    assert impl._batched_decode_supported()

    device = torch.device(configure_device)
    kv_cache = SpyreAttentionImpl.allocate_pages(num_pages, spec, device)
    blocks_per_chunk, num_chunks = batched_decode_chunking(num_seqs, num_blocks)
    bucket = SpyreAttnBatchedDecodeBucket(
        num_seqs=num_seqs,
        num_blocks=num_blocks,
        blocks_per_chunk=blocks_per_chunk,
        num_chunks=num_chunks,
    )

    realized = impl._record_batched_one(bucket, MagicMock(), kv_cache, builder, set(), sys.maxsize)

    assert realized == (num_seqs, blocks_per_chunk, num_chunks)
    assert batched_decode_calls, "the recorder traced the per-seq loop, not the batched kernel"

    del kv_cache
    gc.collect()
