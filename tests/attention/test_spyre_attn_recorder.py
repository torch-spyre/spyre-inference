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

"""Tests for the attention graph recorder.

CPU-only: these check that recording compiles a graph for every variant the
bucketer enumerates, and that a subsequent dispatch compiles nothing more. The
count comes from Dynamo's own ``unique_graphs`` counter, since Dynamo is what
decides whether a dispatch reuses a graph. The kernels run on CPU here (no
Spyre), which is enough to exercise the metadata the recorder builds and the
guards it trips.
"""

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch._dynamo.utils import counters
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.logger import _print_warning_once
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec

from spyre_inference import envs
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadataBuilder,
    SpyrePagedKVCache,
    _build_query_row_tables,
)
from spyre_inference.v1.attention.backends.spyre_head_major_attn import (
    SpyreHeadMajorAttentionBackend,
)
from spyre_inference.v1.attention.ops import tile_loop
from spyre_inference.v1.attention.ops.page_attn import page_attn_kernel
from spyre_inference.v1.attention.ops.page_attn_head_major_prefill import (
    page_attn_head_major_prefill_kernel,
)
from spyre_inference.v1.attention.spyre_attn_bucketer import (
    SpyreAttnBucket,
    SpyreAttnBucketer,
    batched_decode_chunking,
)
from spyre_inference.v1.attention.warmup import record_attention_graphs

pytestmark = pytest.mark.attention

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_SIZE = 64
BLOCK_SIZE = 64
NUM_PAGES = 8


def compiles() -> int:
    """Graphs Dynamo has compiled so far, process-wide."""
    return counters["stats"]["unique_graphs"]


@pytest.fixture()
def impl(default_vllm_config):
    # Dynamo caches on the kernel's code object, shared by every impl in the
    # process, so an earlier test's graphs would hide a recorder that compiled none.
    torch._dynamo.reset()
    # The fixture's bare CompilationConfig leaves mode unset, which resolves to
    # eager. __init__ reads the mode to pick its kernel, so set it before building.
    get_current_vllm_config().compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
    return SpyreAttentionImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        scale=1.0 / (HEAD_SIZE**0.5),
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
    )


@pytest.fixture()
def kv_cache():
    shape = (NUM_PAGES, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    return SpyrePagedKVCache(
        k_pages=torch.zeros(shape, dtype=torch.float16),
        v_pages=torch.zeros(shape, dtype=torch.float16),
    )


def _make_builder(kv_cache_spec, backend=SpyreAttentionBackend) -> SpyreAttentionMetadataBuilder:
    vllm_config = get_current_vllm_config()
    vllm_config.model_config.get_num_attention_heads = MagicMock(return_value=NUM_HEADS)
    vllm_config.model_config.get_num_kv_heads = MagicMock(return_value=NUM_KV_HEADS)
    vllm_config.cache_config.block_size = BLOCK_SIZE
    return backend.get_builder_cls()(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layers.0.self_attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )


@pytest.fixture()
def builder(default_vllm_config):
    """The real metadata builder the recorder records through, built as
    ``tests.attention.test_spyre_attn._build_metadata`` does."""
    return _make_builder(
        AttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            dtype=torch.float16,
        )
    )


@pytest.fixture()
def sliding_window_builder(default_vllm_config):
    """A builder whose window leaves the active-block count unpadded, so several
    requested buckets realize onto one kernel."""
    return _make_builder(
        FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            head_size_v=HEAD_SIZE,
            dtype=torch.float16,
            sliding_window=BLOCK_SIZE,
        )
    )


def make_bucketer(max_model_len=256, max_num_batched_tokens=64, max_num_seqs=8):
    config = MagicMock()
    config.cache_config.block_size = BLOCK_SIZE
    config.model_config.max_model_len = max_model_len
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    config.scheduler_config.max_num_seqs = max_num_seqs
    return SpyreAttnBucketer(config)


def _recordable(bucketer, pages: int = NUM_PAGES) -> list[SpyreAttnBucket]:
    return [v for v in bucketer.variants() if v.num_blocks <= pages]


@pytest.mark.parametrize(
    "layout,backend,kernel",
    [
        ("token_major", SpyreAttentionBackend, page_attn_kernel),
        ("head_major", SpyreHeadMajorAttentionBackend, page_attn_head_major_prefill_kernel),
    ],
)
def test_non_causal_draft_variant_keeps_future_block_tokens_visible(
    default_vllm_config, monkeypatch, layout, backend, kernel
):
    monkeypatch.setenv("SPYRE_ATTN_KV_LAYOUT", layout)
    assert backend.supports_non_causal()
    builder = _make_builder(
        AttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            dtype=torch.float16,
        ),
        backend=backend,
    )
    bucket = SpyreAttnBucket(1, 16)
    bidirectional = builder.build_for_variant(bucket, causal=False)
    causal = builder.build_for_variant(bucket, causal=True)
    assert bidirectional.causal is False
    assert not bidirectional.apply_causal_mask
    assert causal.apply_causal_mask
    visible = bidirectional.attention_mask_stacks[0][0, : bidirectional.max_query_len]
    assert torch.all(visible == 0)
    mask = causal.attention_mask_stacks[0]
    assert mask[0, 0, -1] == torch.finfo(mask.dtype).min

    # Exercise the kernel bodies on CPU without the Spyre tiled-loop lowering.
    monkeypatch.setattr(tile_loop, "USE_FOR_EACH_TILE", False)
    shape = backend.get_kv_cache_shape(1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    k_pages, v_pages = (torch.zeros(shape, dtype=torch.float16) for _ in range(2))
    if layout == "head_major":
        v_pages[0, :, -1] = 1
    else:
        v_pages[0, -1] = 1
    for metadata, expected in ((bidirectional, 1 / BLOCK_SIZE), (causal, 0.0)):
        padded_query_len = metadata.aligned_query_lens[0]
        result = kernel(
            torch.zeros(padded_query_len, NUM_HEADS, HEAD_SIZE, dtype=torch.float16),
            torch.arange(padded_query_len, dtype=torch.int32),
            k_pages,
            v_pages,
            torch.zeros(1, 32, dtype=torch.int32),
            metadata.attention_mask_stacks[0],
            1 / HEAD_SIZE**0.5,
            1,
            padded_query_len,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_SIZE,
        )
        torch.testing.assert_close(result[0], torch.full_like(result[0], expected), rtol=0, atol=0)


@pytest.mark.parametrize(
    "causal,expected",
    [
        ([True, False], [True, False]),
        ([False, True], [False, True]),
        ([True, True], [True]),
        ([False, False], [False]),
    ],
)
def test_attention_recording_shares_only_layers_with_matching_causality(
    impl, kv_cache, builder, monkeypatch, causal, expected
):
    layers = {str(i): SimpleNamespace(impl=impl) for i in range(len(causal))}
    record = MagicMock(return_value=1)
    monkeypatch.setattr(impl, "record_graphs", record)
    count = record_attention_graphs(
        layers,
        dict.fromkeys(layers, kv_cache),
        dict.fromkeys(layers, builder),
        causal_by_layer=dict(zip(layers, causal, strict=True)),
    )
    assert count == len(expected)
    assert [call.kwargs["causal"] for call in record.call_args_list] == expected


def test_attention_recording_requires_causality_for_every_layer(impl, kv_cache, builder):
    with pytest.raises(KeyError, match="draft"):
        record_attention_graphs(
            {"draft": SimpleNamespace(impl=impl)},
            {"draft": kv_cache},
            {"draft": builder},
            causal_by_layer={},
        )


@pytest.mark.parametrize("causal", [False, True])
def test_recorder_passes_explicit_causality_to_metadata(
    impl, kv_cache, builder, monkeypatch, causal
):
    builder._attn_bucketer = make_bucketer()
    forward = MagicMock()
    monkeypatch.setattr(impl, "forward", forward)
    monkeypatch.setattr(impl, "_batched_decode_supported", lambda: False)
    impl.record_graphs(SimpleNamespace(), kv_cache, builder, causal=causal)
    assert forward.call_count > 0
    assert all(call.args[5].causal is causal for call in forward.call_args_list)


def test_attention_sinks_are_rejected(default_vllm_config):
    with pytest.raises(NotImplementedError, match="attention sinks"):
        SpyreAttentionImpl(NUM_HEADS, HEAD_SIZE, 1.0, NUM_KV_HEADS, sinks=torch.zeros(NUM_HEADS))


def _record(impl, kv_cache, builder) -> int:
    """``record_graphs`` as the runner calls it; ``forward`` ignores the layer."""
    return impl.record_graphs(MagicMock(), kv_cache, builder)


def _dispatch(impl, builder, kv_cache, num_blocks, padded_query_len):
    """Invoke the kernel the way a batch of this shape would; the empty ``recorded``
    set forces a re-trace, so an already-traced variant must hit its cached graph."""
    impl._record_one(
        SpyreAttnBucket(num_blocks, padded_query_len), MagicMock(), kv_cache, builder, set()
    )


def _dispatch_batched(impl, builder, kv_cache, bucket):
    """``_dispatch`` for the batched decode kernel.

    The recorder's page budget is unbounded so this traces whatever the bucket realizes
    onto, the way a request would; ``forward`` still applies dispatch's own page bound.
    """
    impl._record_batched_one(bucket, MagicMock(), kv_cache, builder, set(), sys.maxsize)


class TestRecordGraphs:
    @pytest.fixture(autouse=True)
    def _per_seq_only(self, monkeypatch):
        """These counts are the per-seq recorder's; the batched ones are
        TestRecordBatchedDecode's."""
        monkeypatch.setenv("SPYRE_BATCHED_DECODE", "0")
        envs.clear_env_cache()
        yield
        envs.clear_env_cache()

    def test_records_every_enumerated_variant(self, impl, kv_cache, builder):
        bucketer = builder._attn_bucketer = make_bucketer()

        recorded = _record(impl, kv_cache, builder)

        assert recorded == len(_recordable(bucketer)) > 0

    def test_dispatch_after_recording_compiles_nothing(self, impl, kv_cache, builder):
        """The acceptance criterion: no request compiles a new variant.

        Rounds sizes the way production does, so a bucket the enumeration misses
        shows up here.
        """
        bucketer = builder._attn_bucketer = make_bucketer()
        before = compiles()
        _record(impl, kv_cache, builder)
        assert compiles() > before, "recording compiled nothing"

        snapshot = compiles()
        for kv_len in (1, 60, 64, 200, 256):
            for query_len in (1, 5, 32, 64):
                if query_len > kv_len:
                    continue
                padded_query_len = bucketer.find_query_bucket(query_len)
                num_blocks = bucketer._round_up(
                    (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE, bucketer.num_blocks_buckets
                )
                assert padded_query_len is not None and num_blocks is not None
                if num_blocks > NUM_PAGES:
                    continue
                _dispatch(impl, builder, kv_cache, num_blocks, padded_query_len)

        assert compiles() == snapshot

    def test_re_recording_compiles_nothing(self, impl, kv_cache, builder):
        builder._attn_bucketer = make_bucketer()
        first = _record(impl, kv_cache, builder)

        snapshot = compiles()
        assert _record(impl, kv_cache, builder) == first
        assert compiles() == snapshot

    def test_layer_with_equal_graph_key_shares_the_recorded_graphs(self, impl, kv_cache, builder):
        """Another layer's impl and pages, with an equal key, compile nothing new."""
        bucketer = builder._attn_bucketer = make_bucketer()
        _record(impl, kv_cache, builder)
        other = SpyreAttentionImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
            scale=impl.scale,
            num_kv_heads=NUM_KV_HEADS,
        )
        other_cache = SpyrePagedKVCache(*(torch.zeros_like(t) for t in kv_cache))
        assert other.graph_key() == impl.graph_key()

        snapshot = compiles()
        for bucket in _recordable(bucketer):
            _dispatch(other, builder, other_cache, bucket.num_blocks, bucket.padded_query_len)
        assert compiles() == snapshot

    @pytest.mark.parametrize("attr", ["num_heads", "scale"])
    def test_graph_key_tells_apart_per_layer_kernel_scalars(self, impl, attr):
        base = impl.graph_key()
        setattr(impl, attr, getattr(impl, attr) * 2)
        assert impl.graph_key() != base

    def test_buckets_collapsing_onto_one_kernel_record_once(
        self, impl, kv_cache, sliding_window_builder
    ):
        """Deduping on the realized bucket must not drop a graph dispatch needs."""
        bucketer = sliding_window_builder._attn_bucketer = make_bucketer()

        recorded = _record(impl, kv_cache, sliding_window_builder)

        requested = _recordable(bucketer)
        assert 0 < recorded < len(requested), "no buckets collapsed; nothing deduped"

        snapshot = compiles()
        for bucket in requested:
            _dispatch(
                impl,
                sliding_window_builder,
                kv_cache,
                bucket.num_blocks,
                bucket.padded_query_len,
            )
        assert compiles() == snapshot

    def test_collapsing_without_a_sliding_window_warns(
        self, impl, kv_cache, builder, caplog, monkeypatch
    ):
        """Without a window every bucket must realize onto itself; drift is a bug."""
        builder._attn_bucketer = make_bucketer()
        monkeypatch.setattr(builder, "_pad_num_blocks", lambda n: min(n * 2, NUM_PAGES) if n else 0)

        with caplog.at_level(logging.WARNING):
            _record(impl, kv_cache, builder)

        assert "bucketer and build() have diverged" in caplog.text

    def test_collapsing_with_a_sliding_window_is_quiet(
        self, impl, kv_cache, sliding_window_builder, caplog
    ):
        sliding_window_builder._attn_bucketer = make_bucketer()

        with caplog.at_level(logging.WARNING):
            _record(impl, kv_cache, sliding_window_builder)

        assert "diverged" not in caplog.text

    def test_skips_variants_exceeding_the_page_allocation(self, impl, kv_cache, builder):
        """Buckets sized from max_model_len can outrun a small KV cache."""
        bucketer = builder._attn_bucketer = make_bucketer(max_model_len=4096)

        recorded = _record(impl, kv_cache, builder)

        assert 0 < recorded == len(_recordable(bucketer)) < len(bucketer.variants())

    def test_real_metadata_dispatch_compiles_nothing(self, impl, kv_cache, builder):
        """The acceptance criterion, driven from real builder metadata.

        Unlike ``test_dispatch_after_recording_compiles_nothing``, this builds
        metadata for unbucketed kv_lens and dispatches on the block counts
        ``build()`` actually produced for them.
        """
        from spyre_testing_plugin.attn_helpers import _padded_mask_metadata

        # The builder's own bucketer, derived from the live config, since
        # _padded_mask_metadata builds its metadata from that same config.
        bucketer = builder.attn_bucketer
        _record(impl, kv_cache, builder)

        snapshot = compiles()
        for query_len, kv_len in [(1, 1), (1, 65), (1, 200), (7, 65), (32, 300), (33, 300)]:
            metadata = _padded_mask_metadata(
                [(query_len, kv_len)],
                block_size=BLOCK_SIZE,
                num_query_heads=NUM_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_size=HEAD_SIZE,
                max_num_blocks=NUM_PAGES,
            )
            assert metadata.padded_num_blocks is not None
            num_blocks = metadata.padded_num_blocks[0]
            assert num_blocks in bucketer.num_blocks_buckets, (
                f"kv_len={kv_len} produced an unrecorded block count {num_blocks}"
            )
            _dispatch(impl, builder, kv_cache, num_blocks, metadata.aligned_query_lens[0])

        assert compiles() == snapshot

    def test_mixed_batch_dispatch_compiles_nothing(self, impl, kv_cache, builder):
        """A mixed batch dispatches two query widths; both must be recorded."""
        from spyre_testing_plugin.attn_helpers import _padded_mask_metadata

        _record(impl, kv_cache, builder)

        snapshot = compiles()
        metadata = _padded_mask_metadata(
            [(32, 300), (1, 200), (1, 65)],
            block_size=BLOCK_SIZE,
            num_query_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            max_num_blocks=NUM_PAGES,
        )
        assert metadata.aligned_query_lens[0] > 1
        assert metadata.aligned_query_lens[1:] == [1, 1]
        assert metadata.padded_num_blocks is not None

        for seq_idx, aligned in enumerate(metadata.aligned_query_lens):
            num_blocks = metadata.padded_num_blocks[seq_idx]
            assert num_blocks <= NUM_PAGES, "variant would have been skipped when recording"
            _dispatch(impl, builder, kv_cache, num_blocks, aligned)

        assert compiles() == snapshot

    def test_wide_chunk_beside_short_decode_stays_on_recorded_keys(
        self, default_vllm_config, monkeypatch
    ):
        """The case that makes variants()' pruning load-bearing.

        Pruning drops a (num_blocks, query bucket) pair on query_len <= kv_len,
        which only holds within one sequence. It removes nothing until there are
        three query buckets, and only bites when a chunk is wider than another
        sequence's padded KV, so the other recorder tests never reach it.
        """
        from spyre_testing_plugin.attn_helpers import _padded_mask_metadata

        cfg = get_current_vllm_config()
        monkeypatch.setattr(cfg.scheduler_config, "max_num_batched_tokens", 2048)

        chunk_len, chunk_kv, decode_kv = 600, 700, 200
        metadata = _padded_mask_metadata(
            [(chunk_len, chunk_kv), (1, decode_kv)],
            block_size=128,
            num_query_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            max_num_blocks=16,
        )

        bucketer = SpyreAttnBucketer(cfg)
        recorded = set(bucketer.variants())
        assert len(recorded) < len(bucketer.query_buckets) * len(bucketer.num_blocks_buckets), (
            "config prunes nothing, so this test would pass vacuously"
        )

        assert metadata.padded_num_blocks is not None
        chunk_width = bucketer.find_query_bucket(chunk_len)
        assert chunk_width is not None
        assert metadata.aligned_query_lens == [chunk_width, 1]

        # The variant a batch-wide width would have produced for the decode
        # sequence. Pruned, so dispatching it means an Inductor compile in the
        # serving path; asserted absent so this test fails loudly if the
        # bucketer stops pruning it and the case goes uncovered.
        assert SpyreAttnBucket(metadata.padded_num_blocks[1], chunk_width) not in recorded

        for seq_idx, aligned in enumerate(metadata.aligned_query_lens):
            variant = SpyreAttnBucket(metadata.padded_num_blocks[seq_idx], aligned)
            assert variant in recorded, f"sequence {seq_idx} dispatches unrecorded {variant}"

    def test_mixed_batch_row_tables_keep_their_own_width(self, impl, kv_cache):
        """The recorded key is not enough: the row table's width is a guard too."""
        from spyre_testing_plugin.attn_helpers import _padded_mask_metadata

        metadata = _padded_mask_metadata(
            [(32, 300), (1, 200), (1, 65)],
            block_size=BLOCK_SIZE,
            num_query_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            max_num_blocks=NUM_PAGES,
        )
        aligned = metadata.aligned_query_lens
        assert aligned[0] > 1 and aligned[1:] == [1, 1], "not a mixed batch"

        row_tables = _build_query_row_tables(metadata, torch.device("cpu"))

        widths = [(t.shape[-1], al) for t, al in zip(row_tables, aligned)]
        assert all(got == want for got, want in widths), (
            f"row-table widths {widths} (got, want) differ from the recorder's, so these "
            "sequences dispatch to an unrecorded graph"
        )

    def test_a_real_step_through_forward_compiles_nothing(self, impl, kv_cache, builder):
        """Record, then run a real step: a runtime batch — its own block table, mask
        tiles and row tables, none synthesized — must reuse the recorded graphs."""
        from spyre_testing_plugin.attn_helpers import _padded_mask_metadata

        _record(impl, kv_cache, builder)

        # A 4-block table over two sequences keeps every page id inside the
        # cache, so the kernel's gather reads real pages.
        metadata = _padded_mask_metadata(
            [(32, 200), (1, 65)],
            block_size=BLOCK_SIZE,
            num_query_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            max_num_blocks=4,
        )
        aligned = metadata.aligned_query_lens
        assert aligned[0] > 1 and aligned[1] == 1, "not a mixed batch"

        # The buffers attn_layer hands forward() in production.
        q_staging, out_staging = impl._staging_buffers(torch.device("cpu"))

        snapshot = compiles()
        impl.forward(MagicMock(), q_staging, q_staging, q_staging, kv_cache, metadata, out_staging)

        assert compiles() == snapshot

    def test_eager_records_nothing(self, impl, kv_cache, builder):
        impl._compile_attn = False
        builder._attn_bucketer = make_bucketer()
        snapshot = compiles()
        assert _record(impl, kv_cache, builder) == 0
        assert compiles() == snapshot

    def test_a_failing_variant_does_not_abort_the_pass(self, impl, kv_cache, builder, monkeypatch):
        """One bad variant must not take down engine startup."""
        bucketer = builder._attn_bucketer = make_bucketer()
        calls = {"n": 0}
        real = impl._record_one

        def flaky(bucket, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("synthetic lowering failure")
            return real(bucket, *args, **kwargs)

        monkeypatch.setattr(impl, "_record_one", flaky)
        recorded = _record(impl, kv_cache, builder)

        assert recorded == calls["n"] - 1 == len(_recordable(bucketer)) - 1

    def test_recording_nothing_warns(self, impl, kv_cache, builder, monkeypatch, caplog):
        """A pass that records nothing degrades to first-use compiles; say so loudly."""
        builder._attn_bucketer = make_bucketer()

        def always_fails(*args, **kwargs):
            raise RuntimeError("synthetic lowering failure")

        monkeypatch.setattr(impl, "_record_one", always_fails)
        _print_warning_once.cache_clear()

        with caplog.at_level(logging.WARNING):
            assert _record(impl, kv_cache, builder) == 0

        assert "every shape will compile on first use" in caplog.text


@pytest.mark.parametrize("num_blocks", [1, 2, 4])
def test_one_token_variant_records_the_per_seq_kernel(builder, monkeypatch, num_blocks):
    """Per-seq variants are built with the batched path off, so recording the one-token
    variant runs the per-seq loop rather than the batched decode kernel."""
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
    envs.clear_env_cache()

    metadata = builder.build_for_variant(SpyreAttnBucket(num_blocks, 1))

    assert metadata.padded_num_seqs is None


class TestRecompileLimit:
    def test_limit_is_raised_during_recording_and_restored(self, impl, kv_cache, builder):
        """Dynamo's accumulated limit is global, so more buckets than it allows would
        otherwise stop compiling partway through and fall back to eager."""
        bucketer = builder._attn_bucketer = make_bucketer()
        before = torch._dynamo.config.accumulated_recompile_limit
        seen = []

        real = impl._record_one

        def spy(*args, **kwargs):
            seen.append(torch._dynamo.config.accumulated_recompile_limit)
            return real(*args, **kwargs)

        impl._record_one = spy
        _record(impl, kv_cache, builder)

        assert seen and min(seen) >= len(bucketer.variants())
        assert torch._dynamo.config.accumulated_recompile_limit == before

    def test_limit_is_restored_even_when_recording_raises(
        self, impl, kv_cache, builder, monkeypatch
    ):
        before = torch._dynamo.config.accumulated_recompile_limit
        builder._attn_bucketer = make_bucketer()
        monkeypatch.setattr(
            impl, "_record_all", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        with pytest.raises(RuntimeError):
            _record(impl, kv_cache, builder)
        assert torch._dynamo.config.accumulated_recompile_limit == before


class TestRecordBatchedDecode:
    """Recording the batched decode kernel's variants."""

    # entries = num_seqs * blocks_per_chunk tops out at the core count, so a cache
    # this wide clears the recorder's page-budget skip for every variant. The
    # per-seq recorder is still bounded by NUM_PAGES, hence the two page counts.
    PAGES = 64

    @pytest.fixture(autouse=True)
    def _enabled(self, monkeypatch):
        monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
        envs.clear_env_cache()
        yield
        envs.clear_env_cache()

    @pytest.fixture()
    def wide_cache(self):
        shape = (self.PAGES, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        return SpyrePagedKVCache(
            k_pages=torch.zeros(shape, dtype=torch.float16),
            v_pages=torch.zeros(shape, dtype=torch.float16),
        )

    @classmethod
    def _recordable_batched(cls, bucketer, pages: int | None = None) -> list:
        pages = cls.PAGES if pages is None else pages
        return [
            v for v in bucketer.batched_decode_variants() if v.num_seqs * v.blocks_per_chunk < pages
        ]

    def _expected(self, bucketer, batched: int) -> int:
        return len(_recordable(bucketer, self.PAGES)) + batched

    def test_records_every_enumerated_batched_variant(self, impl, wide_cache, builder):
        bucketer = builder._attn_bucketer = make_bucketer()

        recorded = _record(impl, wide_cache, builder)

        batched = self._recordable_batched(bucketer)
        assert len(batched) == len(bucketer.batched_decode_variants()), (
            "the cache is too small to record the whole enumeration, so this would not cover it"
        )
        assert recorded == self._expected(bucketer, len(batched))

    def test_records_nothing_batched_when_the_flag_is_off(
        self, impl, wide_cache, builder, monkeypatch
    ):
        monkeypatch.setenv("SPYRE_BATCHED_DECODE", "0")
        envs.clear_env_cache()
        bucketer = builder._attn_bucketer = make_bucketer()

        assert _record(impl, wide_cache, builder) == self._expected(bucketer, 0)

    def test_alibi_layer_records_nothing_batched(self, default_vllm_config, wide_cache, builder):
        """The batched kernel doesn't implement ALiBi, so those layers never dispatch to it."""
        torch._dynamo.reset()
        get_current_vllm_config().compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
        alibi_impl = SpyreAttentionImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
            scale=1.0 / (HEAD_SIZE**0.5),
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=[0.5] * NUM_HEADS,
            sliding_window=None,
        )
        bucketer = builder._attn_bucketer = make_bucketer()

        assert _record(alibi_impl, wide_cache, builder) == self._expected(bucketer, 0)

    def test_skips_batched_variants_exceeding_the_page_allocation(self, impl, kv_cache, builder):
        """The gather's entry axis, not the block count, is what a small cache bounds."""
        bucketer = builder._attn_bucketer = make_bucketer()

        recorded = _record(impl, kv_cache, builder)

        batched = self._recordable_batched(bucketer, pages=NUM_PAGES)
        assert len(batched) < len(bucketer.batched_decode_variants())
        assert recorded == len(_recordable(bucketer)) + len(batched)

    def test_window_variants_over_the_requested_budget_still_record(
        self, impl, sliding_window_builder
    ):
        """A window moves the realized entry axis, so the skip must key on that.

        Keying on the bucket's window-agnostic ``blocks_per_chunk`` would drop
        variants whose realized gather fits the cache, putting their compile back
        in the serving path.
        """
        bucketer = sliding_window_builder._attn_bucketer = make_bucketer()
        # Wide enough for what the window realizes, too narrow for what the buckets
        # ask for. NUM_PAGES no longer straddles any variant once padding lifts the
        # realized axis back to the bucket maximum.
        pages = 12
        shape = (pages, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        cache = SpyrePagedKVCache(
            k_pages=torch.zeros(shape, dtype=torch.float16),
            v_pages=torch.zeros(shape, dtype=torch.float16),
        )
        # Whether the realized gather fits is a question about build()'s output.
        reachable = []
        for v in bucketer.batched_decode_variants():
            if v.num_seqs * v.blocks_per_chunk < pages:
                continue
            md = sliding_window_builder.build_for_batched_decode_variant(v)
            assert md.padded_num_seqs is not None and md.blocks_per_chunk is not None
            if md.padded_num_seqs * md.blocks_per_chunk < pages:
                reachable.append(v)
        assert reachable, "no over-budget variant realizes a gather that fits; nothing under test"

        _record(impl, cache, sliding_window_builder)

        # Each one must have been traced during recording, so dispatch compiles nothing.
        snapshot = compiles()
        for bucket in reachable:
            _dispatch_batched(impl, sliding_window_builder, cache, bucket)
        assert compiles() == snapshot

    def test_re_recording_compiles_nothing(self, impl, wide_cache, builder):
        builder._attn_bucketer = make_bucketer()
        first = _record(impl, wide_cache, builder)

        snapshot = compiles()
        assert _record(impl, wide_cache, builder) == first
        assert compiles() == snapshot

    def test_batched_buckets_collapsing_onto_one_kernel_record_once(
        self, impl, wide_cache, sliding_window_builder
    ):
        """Deduping on the realized key must not drop a graph dispatch needs."""
        bucketer = sliding_window_builder._attn_bucketer = make_bucketer()
        requested = self._recordable_batched(bucketer)

        recorded = _record(impl, wide_cache, sliding_window_builder)
        # The window collapses both axes, so the total is below what either
        # enumeration asks for on its own.
        assert recorded < self._expected(bucketer, len(requested)), (
            "nothing collapsed; the dedupe path is untested"
        )

        # Every requested bucket must still reach a traced graph.
        snapshot = compiles()
        for bucket in requested:
            _dispatch_batched(impl, sliding_window_builder, wide_cache, bucket)
        assert compiles() == snapshot

    def test_batched_dispatch_after_recording_compiles_nothing(self, impl, wide_cache, builder):
        """The acceptance criterion: no batched decode batch compiles a new variant.

        Dispatch goes through ``build()`` and ``forward()``, so a drift between the
        builder's chunking and what the bucketer enumerates shows up here.
        """
        bucketer = builder._attn_bucketer = make_bucketer()
        _record(impl, wide_cache, builder)

        snapshot = compiles()
        for bucket in self._recordable_batched(bucketer):
            _dispatch_batched(impl, builder, wide_cache, bucket)
        assert compiles() == snapshot

    def test_real_decode_batch_lands_on_a_recorded_key(self, builder):
        """A batch the scheduler could hand over must realize a key warmup enumerated."""
        from spyre_testing_plugin.attn_helpers import _padded_mask_metadata

        bucketer = builder._attn_bucketer = make_bucketer()
        keys = {
            (v.num_seqs, v.blocks_per_chunk, v.num_chunks)
            for v in bucketer.batched_decode_variants()
        }

        for num_seqs in (1, 2):
            for kv_len in (65, 200):
                metadata = _padded_mask_metadata(
                    [(1, kv_len)] * num_seqs,
                    block_size=BLOCK_SIZE,
                    num_query_heads=NUM_HEADS,
                    num_kv_heads=NUM_KV_HEADS,
                    head_size=HEAD_SIZE,
                    max_num_blocks=NUM_PAGES,
                )
                assert metadata.padded_num_seqs is not None, "builder declined the batched path"
                assert metadata.chunk_page_ids_cpu is not None
                key = (
                    metadata.padded_num_seqs,
                    metadata.blocks_per_chunk,
                    metadata.chunk_page_ids_cpu.shape[0] // metadata.blocks_per_chunk,
                )
                assert key in keys, f"num_seqs={num_seqs} kv_len={kv_len} realized {key}"


def test_large_batch_ladder_variants_build_onto_their_own_kernel(builder, monkeypatch):
    """Warmup builds a large-batch bucket below the base bucket build() pads each sequence onto."""
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
    monkeypatch.setenv("SPYRE_ATTN_KV_LADDER", "pow2")
    monkeypatch.setenv("SPYRE_ATTN_KV_LADDER_LARGE_BATCH", "9_8")
    monkeypatch.setenv("SPYRE_ATTN_LARGE_BATCH_MIN_SEQS", "4")
    envs.clear_env_cache()
    bucketer = builder._attn_bucketer = make_bucketer(max_model_len=1024)
    assert set(bucketer.batched_blocks_buckets(8)) > set(bucketer.num_blocks_buckets)
    for v in bucketer.batched_decode_variants():
        md = builder.build_for_batched_decode_variant(v)
        realized = (
            md.padded_num_seqs,
            md.blocks_per_chunk,
            md.chunk_page_ids_cpu.shape[0] // md.blocks_per_chunk,
        )
        assert realized == (v.num_seqs, v.blocks_per_chunk, v.num_chunks), v


@pytest.mark.parametrize("num_seqs", [4, 8])
def test_large_batch_dispatch_lands_on_a_recorded_key(builder, monkeypatch, num_seqs):
    """build() must dispatch onto the large-batch kernels warmup records, not the base ones."""
    from vllm.v1.attention.backend import CommonAttentionMetadata

    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
    monkeypatch.setenv("SPYRE_ATTN_KV_LADDER", "pow2")
    monkeypatch.setenv("SPYRE_ATTN_KV_LADDER_LARGE_BATCH", "9_8")
    monkeypatch.setenv("SPYRE_ATTN_LARGE_BATCH_MIN_SEQS", "4")
    envs.clear_env_cache()
    bucketer = builder._attn_bucketer = make_bucketer(max_model_len=1024)
    keys = {
        (v.num_seqs, v.blocks_per_chunk, v.num_chunks) for v in bucketer.batched_decode_variants()
    }
    max_blocks = 1024 // BLOCK_SIZE

    realized = set()
    for kv_len in range(65, 1025, 37):
        seq_lens = torch.tensor([kv_len - 7 * i for i in range(num_seqs)], dtype=torch.int32)
        md = builder.build(
            common_prefix_len=0,
            common_attn_metadata=CommonAttentionMetadata(
                query_start_loc=torch.arange(num_seqs + 1, dtype=torch.int32),
                query_start_loc_cpu=torch.arange(num_seqs + 1, dtype=torch.int32),
                seq_lens=seq_lens,
                num_reqs=num_seqs,
                num_actual_tokens=num_seqs,
                max_query_len=1,
                max_seq_len=int(seq_lens.max()),
                block_table_tensor=torch.arange(num_seqs * max_blocks, dtype=torch.int32).reshape(
                    num_seqs, max_blocks
                ),
                slot_mapping=torch.zeros(num_seqs, dtype=torch.int64),
                causal=True,
                is_prefilling=torch.zeros(num_seqs, dtype=torch.bool),
            ),
        )
        assert md.padded_num_seqs is not None, "builder declined the batched path"
        key = (
            md.padded_num_seqs,
            md.blocks_per_chunk,
            md.chunk_page_ids_cpu.shape[0] // md.blocks_per_chunk,
        )
        assert key in keys, f"num_seqs={num_seqs} kv_len={kv_len} realized {key}"
        realized.add(key)

    base = {
        (num_seqs, *batched_decode_chunking(num_seqs, nb)) for nb in bucketer.num_blocks_buckets
    }
    assert realized - base, "dispatch never used a large-batch bucket"
