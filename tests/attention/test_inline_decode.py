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

"""Card-free tests for pure-decode attention traced into the block graph."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.attention.backend import AttentionType

from spyre_inference import envs
from spyre_inference.v1.attention import attn_layer
from spyre_inference.v1.attention.ops import batched_decode_head_major as bdhm
from spyre_inference.v1.attention.ops import tile_loop

pytestmark = pytest.mark.attention

BLOCK, HEAD, KV, QPK = 4, 8, 2, 2
NUM_PAGES = 64


@pytest.fixture(autouse=True)
def fullgraph_blocks(monkeypatch):
    monkeypatch.setattr(attn_layer, "outer_graph_fullgraph", True)


def _impl(**overrides):
    impl = SimpleNamespace(
        do_kv_cache_update=Mock(),
        kv_write_index=Mock(),
        kv_slot_views=lambda kv_cache: (kv_cache[0].view(-1, HEAD), kv_cache[1].view(-1, HEAD)),
        inline_decode_kernel=bdhm.batched_decode_head_major_kernel,
        _batched_decode_supported=lambda: True,
        _batched_decode_preconditions_met=lambda md: md.padded_num_seqs is not None,
        inline_batched_decode=lambda b_seqs, bpc: True,
        scale=0.3,
        num_kv_heads=KV,
        num_queries_per_kv=QPK,
        block_size=BLOCK,
        head_size=HEAD,
        logits_soft_cap=0.0,
    )
    for key, value in overrides.items():
        setattr(impl, key, value)
    return impl


def _layer(impl, pages=NUM_PAGES):
    torch.manual_seed(0)
    return SimpleNamespace(
        attn_type=AttentionType.DECODER,
        impl=impl,
        kv_sharing_target_layer_name=None,
        query_quant=None,
        kv_cache=(torch.randn(pages, KV, BLOCK, HEAD), torch.randn(pages, KV, BLOCK, HEAD)),
    )


class TestEligibility:
    def test_eligible_layer_gets_the_grid(self):
        layer = _layer(_impl())
        _, grid = attn_layer.install([layer])
        assert grid.inline
        assert layer.spyre_decode_grid is grid

    @pytest.mark.parametrize(
        "refuse",
        ["no_kernel", "batched_off", "max_cores", "not_fullgraph"],
    )
    def test_refusals_keep_the_opaque_path(self, refuse, monkeypatch):
        impl = _impl()
        if refuse == "no_kernel":
            impl.inline_decode_kernel = None
        elif refuse == "batched_off":
            impl._batched_decode_supported = lambda: False
        elif refuse == "max_cores":
            monkeypatch.setenv("SPYRE_ATTN_MAX_CORES", "8")
        else:
            monkeypatch.setattr(attn_layer, "outer_graph_fullgraph", False)
        envs.clear_env_cache()
        layer = _layer(impl)
        _, grid = attn_layer.install([layer])
        assert not grid.inline
        assert layer.spyre_decode_grid is None


def _metadata(num_seqs=4, num_decode_seqs=4, padded=4, bpc=2):
    return SimpleNamespace(
        num_seqs=num_seqs,
        num_decode_seqs=num_decode_seqs,
        padded_num_seqs=padded,
        blocks_per_chunk=bpc,
        rep_row_ids_dev=None,
        chunk_page_ids_dev=None,
        mask_by_chunk_dev=None,
    )


def _mirroring(impl):
    def mirror(md, device):
        md.rep_row_ids_dev, md.chunk_page_ids_dev, md.mask_by_chunk_dev = (
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        )

    impl._mirror_batched_decode_indices = mirror
    return impl


class TestPublish:
    def _grid(self, pages=NUM_PAGES, **impl_overrides):
        layer = _layer(_mirroring(_impl(**impl_overrides)), pages=pages)
        _, grid = attn_layer.install([layer])
        return grid

    def test_pure_decode_publishes(self):
        grid = self._grid()
        grid.publish(_metadata())
        assert grid.mask is not None and grid.rep_row_ids is not None

    @pytest.mark.parametrize(
        "md,overrides,pages",
        [
            pytest.param(_metadata(num_seqs=5, num_decode_seqs=4), {}, NUM_PAGES, id="mixed"),
            pytest.param(_metadata(num_seqs=3, num_decode_seqs=0), {}, NUM_PAGES, id="prefill"),
            pytest.param(_metadata(padded=None), {}, NUM_PAGES, id="no_bucket"),
            pytest.param(
                _metadata(), {"inline_batched_decode": lambda b, bpc: False}, NUM_PAGES, id="capped"
            ),
            pytest.param(_metadata(padded=4, bpc=2), {}, 8, id="entries_cover_every_page"),
        ],
    )
    def test_other_steps_stay_opaque(self, md, overrides, pages):
        """Mixed steps in particular must never trace in: their structure varies per step."""
        grid = self._grid(pages=pages, **overrides)
        grid.publish(md)
        assert grid.mask is None and grid.rep_row_ids is None and grid.page_ids is None

    def test_a_later_step_clears_the_last_grid(self):
        grid = self._grid()
        grid.publish(_metadata())
        grid.publish(_metadata(num_seqs=5, num_decode_seqs=4))
        assert grid.mask is None

    def test_null_slots_clear_the_grid_unless_warmup_keeps_it(self):
        grid = self._grid()
        grid.publish(_metadata())
        attn_layer.publish_null_slots(4, keep_decode_grids=True)
        assert grid.mask is not None
        attn_layer.publish_null_slots(4)
        assert grid.mask is None


def _upload(b, bpc, c, split_index, monkeypatch):
    """Head-major upload of synthetic metadata, with the device transfer stubbed out."""
    from spyre_inference.v1.attention.backends import spyre_head_major_attn as backend

    torch.manual_seed(1)
    page_ids = (torch.arange(c * bpc * b).reshape(c * bpc, b) % (NUM_PAGES - 1) + 1).to(torch.int32)
    mask = torch.zeros(c * bpc, b, KV if split_index else 1, 1, BLOCK)
    mask[0, 0, ..., 2:] = float("-inf")
    metadata = SimpleNamespace(
        padded_num_seqs=b,
        blocks_per_chunk=bpc,
        rep_row_ids_cpu=torch.arange(b, dtype=torch.int32).repeat(bpc),
        chunk_page_ids_cpu=page_ids,
        mask_by_chunk_cpu=mask,
    )
    original_to = torch.Tensor.to

    def cpu_transfer(tensor, *args, **kwargs):
        kwargs.pop("device_layout", None)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_transfer)
    monkeypatch.setattr(tile_loop, "USE_FOR_EACH_TILE", False)
    monkeypatch.setattr(backend, "USE_FOR_EACH_TILE", split_index)
    monkeypatch.setattr(bdhm, "USE_FOR_EACH_TILE", split_index)
    object.__new__(backend.SpyreHeadMajorAttentionImpl)._mirror_batched_decode_indices(
        metadata, torch.device("cpu")
    )
    return metadata, page_ids, mask


def _reference(query, k_pages, v_pages, page_ids, mask, b):
    """Independent per-sequence softmax over each sequence's pages."""
    out = torch.empty(b, KV, QPK, HEAD)
    q = query[:b].reshape(b, KV, QPK, HEAD)
    for s in range(b):
        k = torch.cat([k_pages[int(p)] for p in page_ids[:, s]], dim=1)
        v = torch.cat([v_pages[int(p)] for p in page_ids[:, s]], dim=1)
        m = torch.cat([mask[j, s, 0, 0] for j in range(page_ids.shape[0])])
        for h in range(KV):
            out[s, h] = torch.softmax(q[s, h] @ k[h].T * 0.3 + m, dim=-1) @ v[h]
    return out.reshape(b, KV * QPK, HEAD)


class TestInlineKernel:
    @pytest.mark.parametrize("split_index", [False, True], ids=["python_walk", "split_index"])
    @pytest.mark.parametrize("b,bpc,c", [(1, 4, 1), (2, 2, 3), (4, 1, 2)])
    def test_static_arguments_come_from_the_grid_shapes(self, b, bpc, c, split_index, monkeypatch):
        """The traced branch reads (num_seqs, blocks_per_chunk) off the uploaded shapes."""
        metadata, _, _ = _upload(b, bpc, c, split_index, monkeypatch)
        mask, rep = metadata.mask_by_chunk_dev, metadata.rep_row_ids_dev
        assert mask.shape[-4] == b
        assert rep.shape[0] // mask.shape[-4] == bpc
        chunks = metadata.chunk_page_ids_dev.shape[0]
        assert (chunks if split_index else chunks // bpc) == c

    @pytest.mark.parametrize("split_index", [False, True], ids=["python_walk", "split_index"])
    @pytest.mark.parametrize("rows", [2, 1, 3], ids=["rows_eq_seqs", "rows_below", "rows_above"])
    def test_matches_reference_and_fits_the_body_rows(self, rows, split_index, monkeypatch):
        b, bpc, c = 2, 2, 2
        metadata, page_ids, mask = _upload(b, bpc, c, split_index, monkeypatch)
        staging = torch.zeros(9, KV * QPK, HEAD)
        impl = _impl(staging_buffers=lambda device: (staging, torch.zeros_like(staging)))
        layer = _layer(impl)
        grid = SimpleNamespace(
            mask=metadata.mask_by_chunk_dev,
            rep_row_ids=metadata.rep_row_ids_dev,
            page_ids=metadata.chunk_page_ids_dev,
        )
        query = torch.randn(rows, KV * QPK, HEAD)

        got = attn_layer._inline_batched_decode(layer, grid, query)

        assert got.shape == (rows, KV * QPK, HEAD)
        padded_query = torch.cat([query, torch.zeros(max(0, b - rows), *query.shape[1:])])
        want = _reference(padded_query, *layer.kv_cache, page_ids, mask[:, :, :1], b)
        n = min(rows, b)
        torch.testing.assert_close(got[:n], want[:n], atol=1e-4, rtol=1e-4)
        # Rows past the batch must be finite: MoE routing reads them.
        assert torch.equal(got[n:], torch.zeros_like(got[n:]))


class TestWarmup:
    def test_every_reachable_body_bucket_and_key_is_traced_once(self, monkeypatch):
        from spyre_inference.v1.attention.spyre_attn_bucketer import SpyreAttnBatchedDecodeBucket
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        variants = [
            SpyreAttnBatchedDecodeBucket(num_seqs=s, num_blocks=n, blocks_per_chunk=1, num_chunks=1)
            for n in (2, 1)
            for s in (4, 2, 1)
        ]
        attn_bucketer = SimpleNamespace(
            batched_decode_variants=lambda: variants,
            find_sequence_bucket=lambda n: 1 if n == 1 else 2 if n == 2 else 4,
        )
        published = []

        def publish(bucket):
            published.append(bucket)
            # num_blocks 1 and 2 realize one key here, as a sliding window can make them.
            return None if bucket.num_seqs == 1 else (bucket.num_seqs, 1, 1)

        class Builder:  # hashable, as the runner dedupes builders shared by layers
            inlines_decode = True

        builder = Builder()
        builder.attn_bucketer = attn_bucketer
        builder.publish_decode_warmup_variant = publish
        from vllm.config.compilation import CompilationMode

        runner = object.__new__(TorchSpyreModelRunner)
        runner.compilation_config = SimpleNamespace(mode=CompilationMode.STOCK_TORCH_COMPILE)
        runner._spyre_kv_caches = {"layers.0.self_attn": None}
        runner.max_num_reqs = 4
        runner.spyre_shape_bucketer = SimpleNamespace(
            find_bucket=lambda n: 1 << (n - 1).bit_length()
        )
        runner._attn_metadata_builders = lambda: {"a": builder, "b": builder}
        runs = []

        def dummy_run(num_tokens):
            assert runner._warming_inline_decode
            runs.append(num_tokens)

        runner._dummy_run = dummy_run
        cleared = []
        monkeypatch.setattr(attn_layer, "clear_decode_grids", lambda: cleared.append(True))

        runner._warm_inline_decode()

        # num_seqs 1 stays opaque, and each (T, key) pair is traced once.
        assert sorted(runs) == [2, 4]
        assert not runner._warming_inline_decode
        assert cleared


@pytest.mark.parametrize(
    "b_seqs,bpc,inline",
    [
        pytest.param(1, 8, True, id="bs1"),
        pytest.param(2, 4, True, id="bs2"),
        pytest.param(4, 8, True, id="bs4"),
        pytest.param(8, 4, False, id="bs8"),
        pytest.param(1, 2, False, id="bs1_capped"),
    ],
)
def test_head_major_inlines_only_small_uncapped_batches(b_seqs, bpc, inline):
    from spyre_inference.v1.attention.backends.spyre_head_major_attn import (
        SpyreHeadMajorAttentionImpl,
    )

    impl = object.__new__(SpyreHeadMajorAttentionImpl)
    impl.num_kv_heads = 8
    assert impl.inline_batched_decode(b_seqs, bpc) is inline
