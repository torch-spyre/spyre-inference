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

"""Head-major KV cache backend (``SPYRE_ATTN_KV_LAYOUT=head_major``).

Both backends share everything above the cache layout, which ``test_spyre_attn.py``
already covers, so this file exercises only what the layout changes.
"""

import gc
from unittest.mock import Mock

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_cache_interface import AttentionSpec

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert
from spyre_inference.platform import TorchSpyrePlatform
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyrePagedKVCache,
)
from spyre_inference.v1.attention.backends.spyre_head_major_attn import (
    SpyreHeadMajorAttentionBackend,
    SpyreHeadMajorAttentionImpl,
)
from spyre_inference.v1.attention.ops.batched_decode_head_major import (
    batched_decode_head_major_kernel,
)
from spyre_inference.v1.attention.ops.page_attn_head_major import (
    page_attn_head_major_decode_kernel,
    page_attn_head_major_kernel,
)
from spyre_inference.v1.attention.ops.reshape_and_cache_head_major import (
    reshape_and_cache_head_major_kernel,
)
from spyre_inference.v1.attention.spyre_attn_bucketer import SpyreAttnBucketer

# The token-major suite's helpers are imported inside each user, not here: the upstream
# job's rootdir spans two trees, so `tests` is not importable at collection time.

pytestmark = pytest.mark.attention

DTYPE = torch.float16


# Per-module in this directory, as in test_spyre_attn.py and test_spyre_encoder_attn.py.
@pytest.fixture()
def configure_device(request, monkeypatch):
    """Device for the cache and the kernels. The card check is lazy so it does not
    claim the device before subprocess-based tests can run."""
    device_mode = request.param
    if device_mode == "spyre" and not spyre_available():
        pytest.skip("Spyre device not available")
    return device_mode


@pytest.fixture()
def configure_compilation(request, monkeypatch):
    """Configure torch.compile mode for tests."""
    from vllm.config import get_cached_compilation_config
    from vllm.config.compilation import CompilationMode

    mode_name = request.param
    torch._dynamo.reset()

    cfg = get_cached_compilation_config()
    original_mode = cfg.mode
    original_limit = torch._dynamo.config.accumulated_recompile_limit

    cfg.mode = getattr(CompilationMode, mode_name)
    # The page-attention kernel is specialized per (num_blocks, padded_query_len).
    torch._dynamo.config.accumulated_recompile_limit = 1024

    yield mode_name

    cfg.mode = original_mode
    torch._dynamo.config.accumulated_recompile_limit = original_limit
    torch._dynamo.reset()


def _fresh_pages(
    num_blocks: int,
    num_kv_heads: int,
    block_size: int,
    head_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zeroed head-major pages, through the impl so they carry the production layout.

    Host-populating a cache with that layout pinned is numerically wrong, so every write
    below goes through the impl too.
    """
    if device.type != "spyre":
        shape = (num_blocks, num_kv_heads, block_size, head_size)
        return torch.zeros(shape, dtype=DTYPE), torch.zeros(shape, dtype=DTYPE)
    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=DTYPE,
    )
    cache = SpyreHeadMajorAttentionImpl.allocate_pages(num_blocks, spec, device)
    return cache.k_pages, cache.v_pages


def _write(impl, kv_cache, key, value, slots, device) -> None:
    impl.do_kv_cache_update(None, key, value, kv_cache, impl.kv_write_index(slots, device))


@torch.inference_mode()
def _run_head_major_attn_test(
    seq_lens: list[tuple[int, int]],
    block_size: int,
    sliding_window: int | None,
    configure_compilation: str,
    configure_device: str,
    soft_cap: float | None = None,
    num_query_heads: int = 32,
    num_kv_heads: int = 8,
    head_size: int = 128,
) -> None:
    """Validate against the same CPU reference the token-major suite uses.

    Seeds the cache entirely through the impl's store, history included: the head-major
    device layout cannot be reproduced by a host-populated transfer.
    """
    if configure_compilation == "STOCK_TORCH_COMPILE" and configure_device == "cpu":
        pytest.skip("Compiled attention targets Spyre; Inductor CPU codegen is unsupported here.")

    from spyre_testing_plugin.attn_helpers import (
        _build_metadata,
        _fused_qkv_kv_views,
        assert_close_outliers,
        ref_attn,
    )

    num_blocks = 256
    torch.set_default_device("cpu")
    set_random_seed(0)

    # This layout does not carry ALiBi; the impl rejects slopes at construction.
    alibi_slopes = None
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    assert num_query_heads % num_kv_heads == 0
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=DTYPE)
    key = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=DTYPE)
    value = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=DTYPE)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    # Widened onto the recorder's buckets, as a real engine's block table is; the extra
    # entries point at garbage pages on purpose (padded blocks must stay inert).
    from vllm.config import get_current_vllm_config

    max_num_blocks_per_seq = (max(kv_lens) + block_size - 1) // block_size
    buckets = SpyreAttnBucketer(get_current_vllm_config()).num_blocks_buckets
    padded_width = SpyreAttnBucketer._round_up(max_num_blocks_per_seq, buckets)
    if padded_width is not None:
        max_num_blocks_per_seq = max(max_num_blocks_per_seq, padded_width)
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    k_expected = torch.zeros(num_blocks, num_kv_heads, block_size, head_size, dtype=DTYPE)
    v_expected = torch.zeros(num_blocks, num_kv_heads, block_size, head_size, dtype=DTYPE)
    hist_k, hist_v, hist_slots = [], [], []
    slot_mapping: list[int] = []
    q_offset = 0
    for seq_idx in range(num_seqs):
        query_len, kv_len = query_lens[seq_idx], kv_lens[seq_idx]
        historical_len = kv_len - query_len
        if historical_len > 0:
            hk = torch.randn(historical_len, num_kv_heads, head_size, dtype=DTYPE)
            hv = torch.randn(historical_len, num_kv_heads, head_size, dtype=DTYPE)
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
    )

    # After _build_metadata: the impl reads cache_config.block_size at construction.
    attn_impl = SpyreHeadMajorAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=alibi_slopes,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
        logits_soft_cap=soft_cap,
    )
    assert attn_impl.block_size == block_size

    cache_device = torch.device(configure_device)
    k_pages, v_pages = _fresh_pages(num_blocks, num_kv_heads, block_size, head_size, cache_device)
    kv_cache = SpyrePagedKVCache(k_pages=k_pages, v_pages=v_pages)

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

    # NaN, not empty: a store that lands nowhere has to fail here.
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
        alibi_slopes=alibi_slopes,
    )

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

    # Release Spyre DMA mappings eagerly (see _run_spyre_attn_test).
    if configure_device == "spyre":
        del k_pages, v_pages, kv_cache, output
        gc.collect()


def test_head_major_kv_cache_shape():
    """The backend advertises head-major, and vLLM can still find the block dim."""
    shape = SpyreHeadMajorAttentionBackend.get_kv_cache_shape(16, 128, 8, 64)
    assert shape == (16, 8, 128, 64)
    assert SpyreHeadMajorAttentionBackend.get_kv_cache_block_dim(128, 8, 64) == 0


def test_head_major_write_index(default_vllm_config):
    """kv_write_index maps a slot to the row holding each of that token's heads."""
    from vllm.config import get_current_vllm_config

    block_size, num_kv_heads = 64, 4
    get_current_vllm_config().cache_config.block_size = block_size
    impl = SpyreHeadMajorAttentionImpl(
        num_heads=8, head_size=64, scale=1.0, num_kv_heads=num_kv_heads
    )

    slots = torch.tensor([0, 3, block_size, 5 * block_size + 63], dtype=torch.int64)
    per_head = impl.kv_write_index(slots, torch.device("cpu"))

    assert len(per_head) == num_kv_heads
    for h, rows in enumerate(per_head):
        assert rows.shape == slots.shape
        # Its own allocation, not a row of a [KV, T] tensor (torch-spyre#3770).
        assert rows.storage_offset() == 0
        for t, slot in enumerate(slots.tolist()):
            block, offset = divmod(slot, block_size)
            assert rows[t] == (block * num_kv_heads + h) * block_size + offset
    # Every destination distinct: two writes sharing a row would silently drop one.
    stacked = torch.stack(per_head)
    assert stacked.unique().numel() == stacked.numel()


def test_token_major_write_index_is_the_slot_mapping(default_vllm_config):
    """The shared publish path must leave the token-major index untouched."""
    impl = SpyreAttentionImpl(num_heads=8, head_size=64, scale=1.0, num_kv_heads=4)
    slots = torch.tensor([0, 7, 129], dtype=torch.int64)
    torch.testing.assert_close(impl.kv_write_index(slots, torch.device("cpu")), slots)


@pytest.mark.parametrize(
    "layout,expected_suffix",
    [
        ("token_major", "spyre_attn.SpyreAttentionBackend"),
        ("head_major", "spyre_head_major_attn.SpyreHeadMajorAttentionBackend"),
    ],
)
def test_platform_selects_backend_by_kv_layout(monkeypatch, layout, expected_suffix):
    monkeypatch.setenv("SPYRE_ATTN_KV_LAYOUT", layout)
    envs.clear_env_cache()
    assert TorchSpyrePlatform._decoder_backend_path().endswith(expected_suffix)


def test_platform_rejects_unknown_kv_layout(monkeypatch):
    monkeypatch.setenv("SPYRE_ATTN_KV_LAYOUT", "page_major")
    envs.clear_env_cache()
    with pytest.raises(ValueError, match="SPYRE_ATTN_KV_LAYOUT"):
        TorchSpyrePlatform._decoder_backend_path()


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
def test_head_major_scatter(
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
    impl = SpyreHeadMajorAttentionImpl(
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
def test_head_major_store_fuses_to_one_kernel(default_vllm_config, num_kv_heads, configure_device):
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
    store = torch.compile(reshape_and_cache_head_major_kernel, dynamic=False)
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


_SHAPES = [
    pytest.param([(1, 128)], id="decode_single"),
    pytest.param([(1, 300), (1, 64), (1, 512)], id="decode_batch"),
    pytest.param([(64, 64)], id="prefill_single"),
    pytest.param([(128, 128), (64, 200)], id="prefill_batch"),
    pytest.param([(64, 300), (1, 128)], id="mixed_batch"),
]


# Spyre and compiled only: the kernel's page gather lowers to aten.index, which fails
# eager, so the impl always compiles attention. Residency itself is a property of the
# layout plan rather than a result, and is measured by
# scripts/probes/lx_head_major_residency.py.
@pytest.mark.parametrize("seq_lens", _SHAPES)
@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
def test_head_major_attn_core(
    default_vllm_config, seq_lens, configure_compilation, configure_device
):
    _run_head_major_attn_test(
        seq_lens=seq_lens,
        block_size=128,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
    )


@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
def test_head_major_dispatches_by_query_width(
    default_vllm_config, monkeypatch, configure_compilation, configure_device
):
    """A decode takes the LX-resident kernel and a wider query the batched one. Sending a
    wide query to the unrolled kernel pays for residency the query width already amortises."""
    from spyre_inference.v1.attention.backends import spyre_head_major_attn as hm

    called = []

    def spy(name, fn):
        def wrapper(*args, **kwargs):
            called.append(name)
            return fn(*args, **kwargs)

        return wrapper

    # Patched before the impl is built: __init__ reads these globals into the impl.
    monkeypatch.setattr(
        hm, "_page_attn_decode_compiled", spy("decode", hm._page_attn_decode_compiled)
    )
    monkeypatch.setattr(
        hm, "_page_attn_prefill_compiled", spy("prefill", hm._page_attn_prefill_compiled)
    )

    _run_head_major_attn_test(
        seq_lens=[(1, 300), (64, 200)],
        block_size=128,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
    )

    assert sorted(called) == ["decode", "prefill"], called


@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
def test_head_major_attn_block_sizes(
    default_vllm_config, block_size, configure_compilation, configure_device
):
    """block_size sets the row stride between a token's heads, so the store's index
    arithmetic changes with it."""
    _run_head_major_attn_test(
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
def test_head_major_attn_sliding_window(
    default_vllm_config, seq_lens, sliding_window, configure_compilation, configure_device
):
    _run_head_major_attn_test(
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
def test_head_major_attn_soft_cap(
    default_vllm_config, seq_lens, soft_cap, configure_compilation, configure_device
):
    _run_head_major_attn_test(
        seq_lens=seq_lens,
        block_size=64,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
        soft_cap=soft_cap,
    )


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
def test_head_major_attn_head_configs(
    default_vllm_config, num_query_heads, num_kv_heads, configure_compilation, configure_device
):
    _run_head_major_attn_test(
        seq_lens=[(1, 300), (64, 200)],
        block_size=64,
        sliding_window=None,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
    )


@pytest.mark.parametrize(
    "configure_compilation",
    [pytest.param("STOCK_TORCH_COMPILE", id="compiled")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_device", [pytest.param("spyre", id="device_spyre")], indirect=True
)
@torch.inference_mode()
def test_head_major_matches_token_major(
    default_vllm_config, configure_compilation, configure_device
):
    """Both backends over the same logical KV must agree far more tightly than
    either agrees with the fp16 CPU reference: the layout is the only difference."""
    from spyre_testing_plugin.attn_helpers import _build_metadata, _fused_qkv_kv_views

    torch.set_default_device("cpu")
    set_random_seed(0)

    num_blocks, block_size = 64, 64
    num_query_heads, num_kv_heads, head_size = 32, 8, 128
    seq_lens = [(64, 300), (1, 128)]
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    scale = head_size**-0.5
    cache_device = torch.device(configure_device)

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=DTYPE)
    key = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=DTYPE)
    value = torch.randn(sum(query_lens), num_kv_heads, head_size, dtype=DTYPE)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )

    # Widened onto the padded block count, as in _run_head_major_attn_test.
    from vllm.config import get_current_vllm_config

    blocks_per_seq = (max(kv_lens) + block_size - 1) // block_size
    buckets = SpyreAttnBucketer(get_current_vllm_config()).num_blocks_buckets
    padded_width = SpyreAttnBucketer._round_up(blocks_per_seq, buckets)
    if padded_width is not None:
        blocks_per_seq = max(blocks_per_seq, padded_width)
    block_tables = torch.arange(len(seq_lens) * blocks_per_seq, dtype=torch.int32).view(
        len(seq_lens), blocks_per_seq
    )

    # Seed every KV position of every sequence, current tokens last.
    hist_k, hist_v, hist_slots = [], [], []
    slot_mapping: list[int] = []
    for s, (ql, kvl) in enumerate(seq_lens):
        hist = kvl - ql
        if hist:
            hist_k.append(torch.randn(hist, num_kv_heads, head_size, dtype=DTYPE))
            hist_v.append(torch.randn(hist, num_kv_heads, head_size, dtype=DTYPE))
            for t in range(hist):
                blk = int(block_tables[s, t // block_size].item())
                hist_slots.append(blk * block_size + t % block_size)
        for t in range(hist, kvl):
            blk = int(block_tables[s, t // block_size].item())
            slot_mapping.append(blk * block_size + t % block_size)
    slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int64)
    hist_slots_t = torch.tensor(hist_slots, dtype=torch.int64)
    hk = convert(torch.cat(hist_k), cache_device)
    hv = convert(torch.cat(hist_v), cache_device)

    attn_metadata = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32),
        query_start_loc=cu_query_lens,
        block_table=block_tables,
        slot_mapping=slot_mapping_t,
    )
    key_src, value_src = _fused_qkv_kv_views(query, key, value, cache_device)
    query_dev = convert(query, cache_device)

    outputs = {}
    for name, impl_cls in (
        ("token_major", SpyreAttentionImpl),
        ("head_major", SpyreHeadMajorAttentionImpl),
    ):
        impl = impl_cls(
            num_heads=num_query_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
        )
        spec = AttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=DTYPE,
        )
        kv_cache = impl_cls.allocate_pages(num_blocks, spec, cache_device)
        _write(impl, kv_cache, hk, hv, hist_slots_t, cache_device)
        _write(impl, kv_cache, key_src, value_src, slot_mapping_t, cache_device)

        output = torch.full_like(query, float("nan")).to(cache_device)
        # Each impl builds its own device tables onto the metadata; clear the first one's
        # so the second is not handed the wrong shapes.
        attn_metadata.kernel_index_tables = None
        attn_metadata.attention_mask_tiles_device = None
        attn_metadata.query_row_tables = None
        impl.forward(
            layer=None,
            query=query_dev,
            key=key_src,
            value=value_src,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )
        outputs[name] = output.to("cpu")
        del kv_cache, output
        gc.collect()

    assert not outputs["head_major"].isnan().any()
    torch.testing.assert_close(outputs["head_major"], outputs["token_major"], atol=1e-2, rtol=1e-2)


def test_runner_allocates_head_major_for_a_head_major_layer():
    """The worker's allocation follows the layer's impl, not a hardcoded shape."""
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
            KVCacheTensor(size=spec.page_size_bytes * num_blocks, shared_by=["layers.0.self_attn"])
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layers.0.self_attn"], kv_cache_spec=spec)],
    )

    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        impl = SpyreHeadMajorAttentionImpl(
            num_heads=32, head_size=head_size, scale=head_size**-0.5, num_kv_heads=num_kv_heads
        )
    fake_layer = Mock()
    fake_layer.kv_cache = None
    fake_layer.impl = impl
    runner.compilation_config.static_forward_context["layers.0.self_attn"] = fake_layer

    caches = runner.initialize_kv_cache_tensors(kv_cache_config, [block_size])
    k_pages = caches["layers.0.self_attn"].k_pages
    v_pages = caches["layers.0.self_attn"].v_pages

    expected = SpyreHeadMajorAttentionBackend.get_kv_cache_shape(
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


def test_decode_fold_matches_unrolled():
    """The folded decode kernel is the unrolled one rebatched, so it must agree exactly."""
    set_random_seed(0)
    kv, qpk, d, block, blocks = 8, 4, 128, 64, 5
    heads = kv * qpk
    k = torch.randn(blocks * kv, block, d)
    v = torch.randn(blocks * kv, block, d)
    query = torch.randn(3, heads, d)
    rows = torch.tensor([2, 0, 1], dtype=torch.int32)
    kv_tables = [
        torch.tensor([[p * kv + h] for h in range(kv)], dtype=torch.int32) for p in range(blocks)
    ]
    head_tables = [
        torch.tensor([h * qpk + g for h in range(kv)], dtype=torch.int32) for g in range(qpk)
    ]
    masks = [torch.zeros(1, block) for _ in range(blocks)]
    masks[-1][0, block // 2 :] = torch.finfo(torch.float32).min

    for soft_cap in (0.0, 30.0):
        args = (
            query,
            rows,
            k,
            v,
            kv_tables,
            head_tables,
            masks,
            d**-0.5,
            blocks,
            1,
            heads,
            kv,
            d,
            block,
            soft_cap,
            None,
        )
        unrolled = page_attn_head_major_kernel(*args)
        # The folded kernel needs no head gather, so it takes no head index tables.
        folded = page_attn_head_major_decode_kernel(*args[:5], *args[6:])
        assert folded.shape == unrolled.shape
        torch.testing.assert_close(folded, unrolled, atol=1e-5, rtol=1e-5)


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
def test_head_major_batched_decode_matches_fp32_reference(
    num_seqs: int,
    b_seqs: int,
    num_blocks: int,
    bpc: int,
    num_kv_heads: int,
    qpk: int,
    ragged: bool,
) -> None:
    """The head-major page read feeds the same reduction the token-major kernel gets.

    Card-free and fp32, as its token-major twin: it pins the read and the entry-major,
    page-row mask broadcast, not the fp16 tolerances.
    """
    from tests.attention.test_spyre_attn import _decode_reference_fp32

    torch.set_default_device("cpu")
    set_random_seed(0)

    block_size, head_size = 16, 8
    num_heads = num_kv_heads * qpk
    padded_blocks = ((num_blocks + bpc - 1) // bpc) * bpc
    num_chunks = padded_blocks // bpc
    entries = b_seqs * bpc
    scale = 0.5

    n_pages = padded_blocks * b_seqs + 1
    query = torch.randn(num_seqs, num_heads * head_size, dtype=torch.float32)
    k_pages = torch.randn(n_pages, block_size, num_kv_heads, head_size, dtype=torch.float32)
    v_pages = torch.randn(n_pages, block_size, num_kv_heads, head_size, dtype=torch.float32)

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

    rep_row_ids = torch.arange(b_seqs, dtype=torch.int64).clamp(max=num_seqs - 1)
    rep_row_ids = rep_row_ids.repeat_interleave(bpc)
    # One index row per page, in int64 for eager CPU indexing.
    chunk_page_ids = [
        page_ids[:, c * bpc : (c + 1) * bpc].reshape(entries, 1).contiguous()
        for c in range(num_chunks)
    ]
    mask_by_chunk = (
        mask.reshape(b_seqs, num_chunks, bpc, block_size)
        .permute(1, 0, 2, 3)
        .reshape(num_chunks, entries, 1, block_size)
        .contiguous()
    )

    query_padded = torch.zeros(b_seqs, num_heads * head_size, dtype=torch.float32)
    query_padded[:num_seqs] = query

    # The cache as this layout stores it: [pages, KV, block, D].
    k_hm = k_pages.permute(0, 2, 1, 3).contiguous()
    v_hm = v_pages.permute(0, 2, 1, 3).contiguous()
    actual = batched_decode_head_major_kernel(
        query_padded,
        rep_row_ids,
        k_hm,
        v_hm,
        chunk_page_ids,
        mask_by_chunk,
        scale,
        b_seqs,
        bpc,
        num_kv_heads,
        qpk,
        block_size,
        head_size,
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
def test_head_major_batched_decode_uses_plain_page_ids(default_vllm_config, configure_compilation):
    """The batched kernel gathers whole pages, so its index is the builder's page ids.

    Guards against folding them onto ``page * KV + kv`` again: that moves the same bytes
    with num_kv_heads times the gather entries, which measured ~2x the kernel time.
    """
    from tests.attention.test_spyre_attn import _build_metadata

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
    impl = SpyreHeadMajorAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
    )

    assert impl._batched_decode_supported()
    assert impl._batched_decode_preconditions_met(attn_metadata), (
        "an all-decode batch of 8 must reach the batched path, or the test proves nothing"
    )
    assert attn_metadata.chunk_page_ids_cpu is not None
    assert attn_metadata.padded_num_seqs is not None
    assert attn_metadata.blocks_per_chunk is not None
    entries = attn_metadata.padded_num_seqs * attn_metadata.blocks_per_chunk

    tables = impl.build_chunk_index_tables(attn_metadata, torch.device("cpu"))
    assert len(tables) == len(attn_metadata.chunk_page_ids_cpu)
    for pages, table in zip(attn_metadata.chunk_page_ids_cpu, tables, strict=True):
        assert table.shape == (entries, 1)
        torch.testing.assert_close(table, pages)


@pytest.fixture()
def batched_decode_calls(monkeypatch):
    """Pin ``SPYRE_BATCHED_DECODE`` on and count the batched dispatches.

    A fall back to the per-seq loop still matches the reference, so without the count the
    tests below would pass while testing nothing.
    """
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
    calls: list[bool] = []
    original = SpyreHeadMajorAttentionImpl._run_batched_decode

    def counting(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SpyreHeadMajorAttentionImpl, "_run_batched_decode", counting)
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
def test_head_major_batched_decode_correctness(
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
    _run_head_major_attn_test(
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
def test_head_major_warmup_records_a_batched_decode_variant(
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
    impl = SpyreHeadMajorAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
    )
    assert impl._batched_decode_supported()

    device = torch.device(configure_device)
    kv_cache = SpyreHeadMajorAttentionImpl.allocate_pages(num_pages, spec, device)
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
