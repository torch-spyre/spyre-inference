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
from unittest.mock import MagicMock

import pytest
import torch
from torch._dynamo.utils import counters
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.logger import _print_warning_once
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec

from spyre_inference import envs
from spyre_inference.v1.attention.backends import spyre_attn
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyreAttentionMetadataBuilder,
    SpyrePagedKVCache,
    _build_query_row_tables,
)
from spyre_inference.v1.attention.ops.layout import (
    stick_aligned_len,
)
from spyre_inference.v1.attention.spyre_attn_bucketer import (
    _MIN_BATCHED_SEQS,
    SpyreAttnBucket,
    SpyreAttnBucketer,
    batched_decode_chunking,
    sliding_active_blocks,
)

pytestmark = pytest.mark.attention

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_SIZE = 64
BLOCK_SIZE = 64
NUM_PAGES = 8
# The whole KV cache, so a sequence can outgrow the wide window and leave blocks behind it.
WIDE_MODEL_LEN = NUM_PAGES * BLOCK_SIZE


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


def _make_builder(kv_cache_spec) -> SpyreAttentionMetadataBuilder:
    vllm_config = get_current_vllm_config()
    vllm_config.model_config.get_num_attention_heads = MagicMock(return_value=NUM_HEADS)
    vllm_config.model_config.get_num_kv_heads = MagicMock(return_value=NUM_KV_HEADS)
    vllm_config.cache_config.block_size = BLOCK_SIZE
    return SpyreAttentionMetadataBuilder(
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


def _window_builder(sliding_window: int) -> SpyreAttentionMetadataBuilder:
    return _make_builder(
        FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            head_size_v=HEAD_SIZE,
            dtype=torch.float16,
            sliding_window=sliding_window,
        )
    )


@pytest.fixture()
def sliding_window_builder(default_vllm_config):
    """A one-block window, which shrinks the batched kernel's gather to one block
    per chunk -- what the batched-decode budget tests turn on."""
    return _window_builder(BLOCK_SIZE)


@pytest.fixture()
def wide_window_builder(default_vllm_config):
    """A window several blocks wide, so a sequence has *interior* blocks.

    Those are the ones handed a shared all-zero mask tile instead of a real one, and
    which tile object lands at which position is part of the kernel's identity guards.
    A one-block window never produces one -- every active block is a boundary block --
    so it cannot exercise that at all.
    """
    return _window_builder(3 * BLOCK_SIZE)


def make_bucketer(
    max_model_len=256, max_num_batched_tokens=64, max_num_seqs=8, sliding_window=None
):
    config = MagicMock()
    config.cache_config.block_size = BLOCK_SIZE
    config.model_config.max_model_len = max_model_len
    config.model_config.runner_type = "generate"
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    config.scheduler_config.max_num_seqs = max_num_seqs
    return SpyreAttnBucketer(config, sliding_window=sliding_window)


def sliding_bucketer(builder, **kwargs):
    """Swap in a bucketer carrying ``builder``'s own window.

    Passing a windowless one would model a config that cannot occur -- the builder and
    its bucketer read the same attention group's spec -- and would make every windowed
    variant realize onto the wrong kernel.
    """
    builder._attn_bucketer = make_bucketer(sliding_window=builder.sliding_window, **kwargs)
    return builder._attn_bucketer


def _dispatch_shape(impl, builder, kv_cache, kv_len, query_len):
    """Invoke the kernel a real sequence of exactly this shape reaches."""
    attn_metadata = builder.build_for_shape(kv_len, query_len)
    q_staging, out_staging = impl._staging_buffers(kv_cache[0].device)
    impl.forward(MagicMock(), q_staging, q_staging, q_staging, kv_cache, attn_metadata, out_staging)


def _recordable(bucketer, pages: int = NUM_PAGES) -> list[SpyreAttnBucket]:
    return [v for v in bucketer.variants() if v.num_blocks <= pages]


def _record(impl, kv_cache, builder) -> int:
    """``record_graphs`` as the runner calls it; ``forward`` ignores the layer."""
    return impl.record_graphs(MagicMock(), kv_cache, builder)


def _dispatch(impl, builder, kv_cache, num_blocks, padded_query_len):
    """Invoke the kernel the way a batch of this shape would; the empty ``recorded``
    set forces a re-trace, so an already-traced variant must hit its cached graph."""
    impl._record_one(
        SpyreAttnBucket(num_blocks, padded_query_len), MagicMock(), kv_cache, builder, set()
    )


def _dispatch_decode_batch(impl, builder, kv_cache, num_seqs, kv_len):
    """Invoke the batched kernel a real all-decode batch at ``kv_len`` reaches.

    Unlike ``_dispatch_batched`` this takes a LENGTH, not a bucket, so it exercises what
    ``build()`` rounds that length onto rather than what the recorder chose to trace.
    """
    query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32)
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc,
            seq_lens=torch.full((num_seqs,), kv_len, dtype=torch.int32),
            num_reqs=num_seqs,
            num_actual_tokens=num_seqs,
            max_query_len=1,
            max_seq_len=kv_len,
            block_table_tensor=torch.zeros(
                num_seqs, (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE, dtype=torch.int32
            ),
            slot_mapping=torch.zeros(num_seqs, dtype=torch.int64),
            causal=True,
            is_prefilling=torch.zeros(num_seqs, dtype=torch.bool),
        ),
    )
    q_staging, out_staging = impl._staging_buffers(kv_cache[0].device)
    impl.forward(MagicMock(), q_staging, q_staging, q_staging, kv_cache, attn_metadata, out_staging)


def _dispatch_batched(impl, builder, kv_cache, bucket):
    """``_dispatch`` for the batched decode kernel.

    The page budget is unbounded: real dispatch has no such check, so this traces
    whatever the bucket realizes onto, the way a request would.
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

    def test_every_windowed_variant_records_its_own_kernel(
        self, impl, kv_cache, wide_window_builder
    ):
        """Under a window, nothing pads: each enumerated active count is its own kernel."""
        bucketer = sliding_bucketer(wide_window_builder, max_model_len=WIDE_MODEL_LEN)

        recorded = _record(impl, kv_cache, wide_window_builder)

        requested = _recordable(bucketer)
        assert recorded == len(requested) > 0

        snapshot = compiles()
        for bucket in requested:
            _dispatch(
                impl,
                wide_window_builder,
                kv_cache,
                bucket.num_blocks,
                bucket.padded_query_len,
            )
        assert compiles() == snapshot

    def test_windowed_lengths_sharing_a_block_count_share_a_kernel(
        self, impl, kv_cache, wide_window_builder
    ):
        """No length may compile after warmup, whatever its interior blocks look like.

        The regression this guards: interior blocks used to share ONE all-zero tile, and
        Dynamo guards which mask-tile arguments are the same object, so two lengths with
        the same active count and query width still compiled separately when their
        boundary blocks fell in different places. Recording the block counts alone did
        not cover them.
        """
        sliding_bucketer(wide_window_builder, max_model_len=WIDE_MODEL_LEN)
        _record(impl, kv_cache, wide_window_builder)

        snapshot = compiles()
        for query_len in (1, 2, BLOCK_SIZE // 2, BLOCK_SIZE - 1, BLOCK_SIZE):
            for kv_len in range(query_len, WIDE_MODEL_LEN + 1):
                _dispatch_shape(impl, wide_window_builder, kv_cache, kv_len, query_len)
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

    def test_windowed_buckets_realize_themselves(self, impl, kv_cache, wide_window_builder, caplog):
        """A window is no excuse for drift either: the witness realizes the count exactly."""
        sliding_bucketer(wide_window_builder, max_model_len=WIDE_MODEL_LEN)

        with caplog.at_level(logging.WARNING):
            _record(impl, kv_cache, wide_window_builder)

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

        widths = [(t.shape[-1], stick_aligned_len(al)) for t, al in zip(row_tables, aligned)]
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


class TestSlidingVariants:
    """The enumeration itself, without a device: does it reach every length?

    Cheap enough to sweep exhaustively, which is the point -- the bug this path fixes was
    an enumeration that looked complete and was not, and no device test can cover every
    (kv_len, query_len) a long context admits.
    """

    WINDOWS = (BLOCK_SIZE, 2 * BLOCK_SIZE, 3 * BLOCK_SIZE + 1, 500, 1024)

    @pytest.mark.parametrize("sliding_window", WINDOWS)
    @pytest.mark.parametrize("max_model_len", (128, 256, 512, 1536, 2048))
    @pytest.mark.parametrize("max_num_batched_tokens", (1, 2, 64, 512))
    def test_every_length_dispatches_to_an_enumerated_variant(
        self, default_vllm_config, sliding_window, max_model_len, max_num_batched_tokens
    ):
        """No sequence may reach a (block count, query width) warmup did not record.

        Scans all the way to ``max_model_len``, past the bucketer's own scan limit, so it
        also pins the claim that the tile plan repeats with period ``block_size`` beyond
        one window plus one query bucket.
        """
        if max_num_batched_tokens > max_model_len:
            pytest.skip("the scheduler cannot batch more tokens than the model holds")
        bucketer = make_bucketer(
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            sliding_window=sliding_window,
        )
        enumerated = set(bucketer.variants())

        unreached = set()
        for query_len in range(1, min(max_num_batched_tokens, max_model_len) + 1):
            width = 1 if query_len == 1 else bucketer.find_query_bucket(query_len)
            if width is None:
                continue
            for kv_len in range(query_len, max_model_len + 1):
                active = sliding_active_blocks(
                    kv_len, kv_len - query_len, sliding_window, BLOCK_SIZE
                )
                if not active:
                    continue
                bucket = SpyreAttnBucket(num_blocks=len(active), padded_query_len=width)
                if bucket not in enumerated:
                    unreached.add((bucket, kv_len, query_len))
        assert not unreached, (
            f"{len(unreached)} lengths reach no recorded kernel: {sorted(unreached)[:5]}"
        )

    @pytest.mark.parametrize("sliding_window", WINDOWS)
    def test_each_variant_has_a_sequence_that_realizes_it(
        self, default_vllm_config, sliding_window
    ):
        """Warmup records through ``witness``, so a variant it cannot realize is a variant
        nothing records -- and the recorder would silently leave it to the serving path."""
        bucketer = make_bucketer(
            max_model_len=2048, max_num_batched_tokens=512, sliding_window=sliding_window
        )
        for bucket in bucketer.variants():
            kv_len, query_len = bucketer.witness(bucket)
            active = sliding_active_blocks(kv_len, kv_len - query_len, sliding_window, BLOCK_SIZE)
            assert len(active) == bucket.num_blocks, (bucket, kv_len, query_len)
            assert query_len <= kv_len <= 2048
            reached = 1 if query_len == 1 else bucketer.find_query_bucket(query_len)
            assert reached == bucket.padded_query_len, (bucket, query_len, reached)


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


def _toy_kernel(x, n):
    """`n` is a plain int, so ``dynamic=False`` gives one graph per value, like num_blocks."""
    for _ in range(n):
        x = x + 1
    return x


class TestLateCompileWarning:
    """The runtime half of the acceptance criterion, for what the tests above cannot see:
    a real config whose buckets miss something. ``backend="eager"`` suffices since the
    counter is Dynamo's.
    """

    @pytest.fixture(autouse=True)
    def _isolated(self, monkeypatch):
        torch._dynamo.reset()
        # warning_once is lru_cached process-wide, so a prior emit would mask ours.
        _print_warning_once.cache_clear()
        monkeypatch.setattr(spyre_attn, "_warmup_complete", False)
        yield
        _print_warning_once.cache_clear()

    def test_quiet_before_warmup_is_marked(self, caplog):
        fn = torch.compile(_toy_kernel, dynamic=False, backend="eager")
        with caplog.at_level(logging.WARNING):
            spyre_attn._call_kernel("page attention", fn, torch.ones(4), 1)
        assert "outside warmup" not in caplog.text

    def test_warns_when_an_unrecorded_variant_compiles(self, caplog):
        fn = torch.compile(_toy_kernel, dynamic=False, backend="eager")
        spyre_attn._call_kernel("page attention", fn, torch.ones(4), 1)
        spyre_attn.mark_warmup_complete()

        with caplog.at_level(logging.WARNING):
            spyre_attn._call_kernel("page attention", fn, torch.ones(4), 2)

        assert "page attention compiled outside warmup" in caplog.text

    def test_quiet_when_the_variant_was_already_recorded(self, caplog):
        fn = torch.compile(_toy_kernel, dynamic=False, backend="eager")
        spyre_attn._call_kernel("page attention", fn, torch.ones(4), 1)
        spyre_attn.mark_warmup_complete()

        with caplog.at_level(logging.WARNING):
            spyre_attn._call_kernel("page attention", fn, torch.ones(4), 1)

        assert "outside warmup" not in caplog.text

    def test_mark_warmup_complete_arms_the_check(self):
        assert spyre_attn._warmup_complete is False
        spyre_attn.mark_warmup_complete()
        assert spyre_attn._warmup_complete is True


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

    def test_unreachable_windowed_block_buckets_are_skipped(
        self, impl, wide_cache, sliding_window_builder
    ):
        """The window caps the decode block count, so the upper ladder entries are dead.

        Recording them is worse than wasteful: a block-aligned length realizes the same low
        count for every one of them, so they look like coverage while the bucket a real
        decode asks for goes untraced.
        """
        bucketer = sliding_bucketer(sliding_window_builder)

        enumerated = {v.num_blocks for v in bucketer.batched_decode_variants()}
        assert enumerated < set(bucketer.num_blocks_buckets), (
            "every ladder entry is reachable; nothing under test"
        )
        assert all(bucketer.decode_witness_kv_len(n) is not None for n in enumerated)

    def test_every_decode_length_hits_a_recorded_batched_kernel(
        self, impl, wide_cache, sliding_window_builder
    ):
        """The regression: no decode length may compile the batched kernel after warmup.

        A block-aligned witness puts the window boundary on a block edge, so it realizes
        ``ceil(W / block_size)`` active blocks for EVERY requested bucket -- and the bucket a
        real, non-block-aligned decode rounds onto was therefore never traced.
        """
        bucketer = sliding_bucketer(sliding_window_builder)
        window = sliding_window_builder.sliding_window
        _record(impl, wide_cache, sliding_window_builder)

        snapshot = compiles()
        for num_seqs in bucketer.num_seqs_buckets:
            for kv_len in range(2, WIDE_MODEL_LEN + 1):
                active = sliding_active_blocks(kv_len, kv_len - 1, window, BLOCK_SIZE)
                if (
                    num_seqs
                    * batched_decode_chunking(num_seqs, bucketer.find_blocks_bucket(len(active)))[0]
                    >= self.PAGES
                ):
                    continue  # the gather would outrun the cache; dispatch declines it too
                _dispatch_decode_batch(impl, sliding_window_builder, wide_cache, num_seqs, kv_len)
        assert compiles() == snapshot

    def test_re_recording_compiles_nothing(self, impl, wide_cache, builder):
        builder._attn_bucketer = make_bucketer()
        first = _record(impl, wide_cache, builder)

        snapshot = compiles()
        assert _record(impl, wide_cache, builder) == first
        assert compiles() == snapshot

    def test_every_batched_variant_realizes_its_own_kernel(
        self, impl, wide_cache, sliding_window_builder
    ):
        """Under a window each enumerated batched bucket must trace the kernel it names."""
        bucketer = sliding_bucketer(sliding_window_builder)
        requested = self._recordable_batched(bucketer)

        recorded = _record(impl, wide_cache, sliding_window_builder)
        assert recorded == self._expected(bucketer, len(requested)) > 0

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

        for num_seqs in (_MIN_BATCHED_SEQS, _MIN_BATCHED_SEQS + 1):
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
                    len(metadata.chunk_page_ids_cpu),
                )
                assert key in keys, f"num_seqs={num_seqs} kv_len={kv_len} realized {key}"
