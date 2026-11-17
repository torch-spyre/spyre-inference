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

"""Card-free tests for SPYRE_ATTN_DECODE_SHAPES, the exact batched decode shape list."""

import json
import logging
import random

import pytest
import torch
from spyre_testing_plugin.attn_helpers import _build_metadata

from spyre_inference import envs
from spyre_inference.v1.attention.spyre_attn_bucketer import (
    SpyreAttnBucketer,
    parse_decode_shapes,
)

BLOCK = 128
KV_HEADS, HEAD = 2, 64

# Powers of two within a 131072-token budget, plus non-power-of-two extents and batches and
# a long shape that a short context clamps away.
LISTED_SHAPES = [
    (s, kv)
    for s in (1, 2, 4, 8, 16, 32)
    for kv in (2048, 4096, 8192, 16384, 32768)
    if s * kv <= 131072
] + [(32, 3584), (20, 4096), (20, 8192), (3, 32768), (8, 20480)]


@pytest.fixture()
def shapes_json(tmp_path):
    path = tmp_path / "shapes.json"
    path.write_text(json.dumps([list(p) for p in LISTED_SHAPES]))
    return str(path)


def _config(max_num_seqs: int, max_model_len: int):
    from vllm.config import get_current_vllm_config

    cfg = get_current_vllm_config()
    cfg.scheduler_config.max_num_seqs = max_num_seqs
    cfg.model_config.max_model_len = max_model_len
    cfg.cache_config.block_size = BLOCK
    return cfg


@pytest.fixture()
def shapes_4k(default_vllm_config, monkeypatch, shapes_json):
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
    monkeypatch.setenv("SPYRE_ATTN_DECODE_SHAPES", shapes_json)
    envs.clear_env_cache()
    return _config(32, 4096)


@pytest.fixture()
def shapes_32k(default_vllm_config, monkeypatch, shapes_json):
    monkeypatch.setenv("SPYRE_BATCHED_DECODE", "1")
    monkeypatch.setenv("SPYRE_JAGGED_DECODE", "1")
    monkeypatch.setenv("SPYRE_ATTN_DECODE_SHAPES", shapes_json)
    envs.clear_env_cache()
    return _config(32, 32768)


def _bucketer(cfg) -> SpyreAttnBucketer:
    cfg.model_config.get_num_kv_heads = lambda *a, **k: KV_HEADS
    return SpyreAttnBucketer(cfg)


def _build(kv_lens: list[int]):
    width = 256
    block_table = torch.arange(len(kv_lens) * width, dtype=torch.int32).reshape(-1, width)
    query_start_loc = torch.arange(len(kv_lens) + 1, dtype=torch.int32)
    return _build_metadata(
        num_query_heads=KV_HEADS,
        num_kv_heads=KV_HEADS,
        head_size=HEAD,
        block_size=BLOCK,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32),
        query_start_loc=query_start_loc,
        block_table=block_table,
        slot_mapping=torch.arange(len(kv_lens), dtype=torch.int64),
    )


def _dispatched(md) -> list[tuple[int, int, int]]:
    """Every (num_seqs, blocks_per_chunk, num_chunks) kernel key a step dispatches."""
    if md.decode_groups is not None:
        return [
            (g.padded_num_seqs, g.blocks_per_chunk, g.padded_batch_blocks // g.blocks_per_chunk)
            for g in md.decode_groups
        ]
    assert md.padded_num_seqs is not None, "batched decode declined the batch"
    return [
        (md.padded_num_seqs, md.blocks_per_chunk, md.padded_batch_blocks // md.blocks_per_chunk)
    ]


def _shapes(md) -> list[tuple[int, int]]:
    return [(g.padded_num_seqs, g.padded_batch_blocks * BLOCK) for g in md.decode_groups]


def test_inline_and_json_forms_agree(shapes_json):
    inline = ",".join(f"{s}x{kv}" for s, kv in LISTED_SHAPES)
    assert parse_decode_shapes(inline) == parse_decode_shapes(shapes_json)
    assert len(parse_decode_shapes(inline)) == len(LISTED_SHAPES)
    assert parse_decode_shapes(None) is None


@pytest.mark.parametrize("raw", ["32", "32x", "0x2048", "4x-1"])
def test_malformed_entries_are_rejected(raw):
    with pytest.raises(ValueError):
        parse_decode_shapes(raw)


def test_a_malformed_json_pair_is_rejected(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps([[32, 4096], [8]]))
    with pytest.raises(ValueError):
        parse_decode_shapes(str(path))


@pytest.mark.parametrize("var", ["SPYRE_ATTN_NUM_SEQS_BUCKETS", "SPYRE_ATTN_KV_LADDER_LARGE_BATCH"])
def test_shapes_reject_the_axis_knobs_they_replace(shapes_4k, monkeypatch, var):
    monkeypatch.setenv(var, "1,2" if var.endswith("BUCKETS") else "9_8")
    envs.clear_env_cache()
    with pytest.raises(ValueError, match="SPYRE_ATTN_DECODE_SHAPES"):
        _bucketer(shapes_4k)


def test_at_4k_the_shapes_are_exactly_the_listed_ones_up_to_4096(shapes_4k):
    """The longer shapes shrink onto 4096 but keep their listed volume, so each loses
    to a shape already there (3x32768 to 4x4096) and is dropped."""
    shapes = _bucketer(shapes_4k).decode_shapes
    assert {(s, b * BLOCK) for s, b in shapes} == {(s, kv) for s, kv in LISTED_SHAPES if kv <= 4096}
    assert len(shapes) == 14


def test_a_clamped_shape_keeps_its_listed_rank(default_vllm_config, monkeypatch):
    """6x32768 at an 8k context covers 6 seqs at 8k, where only 8x8192 also does: it
    ranks by its listed volume, so the 8-seq shape wins and the 6x8192 shape is dropped."""
    monkeypatch.setenv("SPYRE_ATTN_DECODE_SHAPES", "6x32768,8x8192")
    envs.clear_env_cache()
    bucketer = _bucketer(_config(32, 8192))
    assert bucketer.decode_shapes == [(8, 64)]
    monkeypatch.setenv("SPYRE_ATTN_DECODE_SHAPES", "6x32768,8x8192,4x2048")
    envs.clear_env_cache()
    assert _bucketer(_config(32, 8192)).find_decode_shape(6, 64) == (8, 64)


def test_variants_are_exactly_the_shapes(shapes_4k):
    bucketer = _bucketer(shapes_4k)
    variants = {(v.num_seqs, v.num_blocks) for v in bucketer.batched_decode_variants()}
    assert variants == set(bucketer.decode_shapes)


@pytest.mark.parametrize(
    ("num_seqs", "kv_len", "shape"),
    [
        (32, 2049, (32, 3584)),  # an extent a power-of-two ladder lacks
        (32, 3585, (32, 4096)),
        (20, 3000, (20, 4096)),  # a batch the power-of-two buckets lack
        (17, 4000, (20, 4096)),
        (5, 1000, (8, 2048)),
        (3, 4000, (4, 4096)),  # 3x4096 is only a clamped 3x32768 at 4k
        (1, 1, (1, 2048)),
    ],
)
def test_smallest_shape_that_holds_the_batch(shapes_4k, num_seqs, kv_len, shape):
    found = _bucketer(shapes_4k).find_decode_shape(num_seqs, -(-kv_len // BLOCK))
    assert found is not None
    s, b = found
    assert (s, b * BLOCK) == shape


def test_a_volume_tie_goes_to_the_shorter_kv_len(default_vllm_config, monkeypatch):
    monkeypatch.setenv("SPYRE_ATTN_DECODE_SHAPES", "32x2560,20x4096")
    envs.clear_env_cache()
    bucketer = _bucketer(_config(32, 4096))
    assert bucketer.find_decode_shape(20, 2400 // BLOCK + 1) == (32, 2560 // BLOCK)


def test_recorded_variants_build_onto_their_own_kernel(shapes_4k):
    """A warmup batch at a listed shape must dispatch that shape, not a neighbour."""
    bucketer = _bucketer(shapes_4k)
    for v in bucketer.batched_decode_variants():
        md = _build([v.num_blocks * BLOCK] * v.num_seqs)
        assert _dispatched(md) == [(v.num_seqs, v.blocks_per_chunk, v.num_chunks)], v


@pytest.mark.parametrize("fixture", ["shapes_4k", "shapes_32k"])
def test_dispatch_only_reaches_recorded_kernels(request, fixture):
    cfg = request.getfixturevalue(fixture)
    max_len = cfg.model_config.max_model_len
    bucketer = _bucketer(cfg)
    keys = {
        (v.num_seqs, v.blocks_per_chunk, v.num_chunks) for v in bucketer.batched_decode_variants()
    }
    rng = random.Random(0)
    for _ in range(60):
        n = rng.randint(1, 32)
        kv_lens = [rng.randint(1, max_len) for _ in range(n)]
        for key in _dispatched(_build(kv_lens)):
            assert key in keys, f"{n} seqs up to {max(kv_lens)} dispatched unrecorded {key}"


def test_a_long_non_power_of_two_extent_holds_the_batch_whole(shapes_32k):
    """8 seqs near 18000 tokens: no power-of-two shape holds 8 that long, 8x20480 does."""
    md = _build([18000 + 10 * i for i in range(8)])
    assert md.decode_groups is None
    assert (md.padded_num_seqs, md.padded_batch_blocks * BLOCK) == (8, 20480)


def test_a_batch_no_shape_holds_splits_by_length(shapes_32k):
    """32 seqs with one at 8000 tokens: batch 32 stops at 4096, so no shape holds the
    batch. The long one runs alone and the short ones keep 32x2048, rather than padding
    short ones up to 8192 next to it."""
    md = _build([1000] * 31 + [8000])
    assert md.padded_num_seqs is None
    groups = md.decode_groups
    assert groups is not None
    assert [g.rows_cpu.tolist() for g in groups] == [[31], list(range(31))]
    assert _shapes(md) == [(1, 8192), (32, 2048)]


def test_a_length_group_past_the_batch_cap_runs_in_pieces(shapes_32k):
    """30 seqs near 7000 tokens: 20 is the largest batch at 8192, so the longest 20 run
    there and the other 10 on the smallest shape that holds them."""
    kv_lens = [7000 + 10 * i for i in range(30)]
    md = _build(kv_lens)
    assert md.padded_num_seqs is None
    groups = md.decode_groups
    assert groups is not None
    assert [g.num_seqs for g in groups] == [20, 10]
    blocks = [-(-kv // BLOCK) for kv in kv_lens]
    long_rows, short_rows = (g.rows_cpu.tolist() for g in groups)
    assert min(blocks[r] for r in long_rows) >= max(blocks[r] for r in short_rows)
    assert _shapes(md) == [(20, 8192), (16, 8192)]


def test_without_jagged_decode_an_uncovered_batch_takes_the_per_seq_loop(shapes_32k, monkeypatch):
    monkeypatch.setenv("SPYRE_JAGGED_DECODE", "0")
    envs.clear_env_cache()
    md = _build([1000] * 31 + [8000])
    assert md.padded_num_seqs is None and md.decode_groups is None


def test_a_sequence_past_every_shape_takes_the_per_seq_loop(shapes_32k, monkeypatch):
    monkeypatch.setenv("SPYRE_ATTN_DECODE_SHAPES", "8x2048,4x4096")
    envs.clear_env_cache()
    md = _build([1000, 9000])
    assert md.padded_num_seqs is None and md.decode_groups is None


def test_a_list_short_of_max_model_len_warns(shapes_32k, monkeypatch, caplog):
    monkeypatch.setenv("SPYRE_ATTN_DECODE_SHAPES", "8x2048,4x4096")
    envs.clear_env_cache()
    with caplog.at_level(logging.WARNING):
        _bucketer(shapes_32k)
    assert "reaches 4096 tokens, below max_model_len=32768" in caplog.text
