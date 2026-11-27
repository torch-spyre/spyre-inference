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

"""Card-free tests for head-major batched decode and its automatic metadata layout."""

from types import SimpleNamespace

import pytest
import torch
from spyre_testing_plugin.attn_helpers import split_by_chunk, use_python_for_each_tile

from spyre_inference.v1.attention.ops import batched_decode as bdhm

BLOCK, HEAD, SCALE = 4, 8, 0.3


def _inputs(b, bpc, c, kv):
    """Synthetic #876 metadata: pages [J, B] and mask [J, B, 1, 1, block]."""
    j_total = c * bpc
    torch.manual_seed(0)
    num_pages = 64
    k_pages = torch.randn(num_pages, kv, BLOCK, HEAD)
    v_pages = torch.randn(num_pages, kv, BLOCK, HEAD)
    # Distinct, in-range page ids (e = j*B + s) so a wrong read is observable.
    page_ids = (torch.arange(j_total * b).reshape(j_total, b) + 1).to(torch.int32)
    mask = torch.zeros(j_total, b, 1, 1, BLOCK, dtype=torch.float32)
    return k_pages, v_pages, page_ids, mask, j_total


def _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, kv, q, cap, sinks=None):
    """Independent per-sequence softmax; pages concatenated along the token axis.

    ``sinks`` ([kv, q, 1]) adds each head's logit to the denominator once."""
    out = torch.empty(b, kv, q, HEAD)
    for s in range(b):
        k = torch.cat([k_pages[int(page_ids[jj, s])] for jj in range(j_total)], dim=1)
        v = torch.cat([v_pages[int(page_ids[jj, s])] for jj in range(j_total)], dim=1)
        m = torch.cat([mask[jj, s, 0, 0, :] for jj in range(j_total)])
        for h in range(kv):
            for qq in range(q):
                sc = (query[s, h, qq] @ k[h].transpose(0, 1)) * SCALE
                if cap > 0.0:
                    sc = torch.tanh(sc / cap) * cap
                sc = sc + m
                if sinks is None:
                    out[s, h, qq] = torch.softmax(sc, dim=-1) @ v[h]
                else:
                    p = torch.softmax(torch.cat([sc, sinks[h, qq]]), dim=-1)[:-1]
                    out[s, h, qq] = p @ v[h]
    return out.reshape(b, kv * q, HEAD)


def _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc, kv, q, cap=0.0, sinks=None):
    """Run the kernel on #876 metadata, split per chunk as the upload does."""
    page_ids, mask = split_by_chunk(page_ids, mask, bpc)
    return _kernel(query, rep, k_pages, v_pages, page_ids, mask, b, bpc, kv, q, cap, sinks)


def _kernel(
    query,
    rep,
    k_pages,
    v_pages,
    page_ids,
    mask,
    b,
    bpc,
    kv,
    q,
    cap=0.0,
    sinks=None,
    entry_local=None,
):
    # The kernel's own rule: per-slot state past one chunk.
    if entry_local is None:
        entry_local = page_ids.shape[0] > 1
    return bdhm.batched_decode_kernel(
        entry_local=entry_local,
        query=query,
        rep_row_ids=rep,
        k_pages=k_pages,
        v_pages=v_pages,
        chunk_page_ids=page_ids,
        mask_by_chunk=mask,
        scale=SCALE,
        num_seqs=b,
        blocks_per_chunk=bpc,
        num_kv_heads=kv,
        num_queries_per_kv=q,
        block_size=BLOCK,
        head_size=HEAD,
        logits_soft_cap=cap,
        sinks=sinks,
    )


def _meta(b, bpc, kv, q):
    query = torch.randn(b, kv, q, HEAD)
    rep = torch.arange(b).repeat(bpc)
    return query, rep


@pytest.fixture(autouse=True)
def python_walk(monkeypatch):
    # The body math is under test, not the tiled op.
    use_python_for_each_tile(monkeypatch)


@pytest.mark.parametrize(("c", "kv", "q"), [(1, 1, 1), (2, 1, 1), (2, 2, 4)])
def test_default_body_matches_reference(c, kv, q):
    b, bpc = 2, 2
    query, rep = _meta(b, bpc, kv, q)
    k_pages, v_pages, page_ids, mask, j_total = _inputs(b, bpc, c, kv)
    got = _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc, kv, q)
    torch.testing.assert_close(
        got, _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, kv, q, 0.0)
    )


def test_one_chunk_keeps_chunk_reduction(monkeypatch):
    b, bpc, c, kv, q = 2, 2, 1, 2, 4
    query, rep = _meta(b, bpc, kv, q)
    k_pages, v_pages, page_ids, mask, j_total = _inputs(b, bpc, c, kv)

    def unexpected_merge(**kwargs):
        pytest.fail("A single chunk must not build and merge per-slot state")

    monkeypatch.setattr(bdhm, "_merge_entry_local", unexpected_merge)
    got = _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc, kv, q)
    torch.testing.assert_close(
        got, _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, kv, q, 0.0)
    )


def test_per_slot_state_at_one_chunk_matches_the_chunk_reduction(monkeypatch):
    """A dynamic chunk count keeps per-slot state at one chunk too, which must give what the
    single-chunk reduction gives, a wholly masked padding slot included."""
    b, bpc, c, kv, q = 2, 2, 1, 2, 4
    query, rep = _meta(b, bpc, kv, q)
    k_pages, v_pages, page_ids, mask, j_total = _inputs(b, bpc, c, kv)
    mask[1, 0] = torch.finfo(mask.dtype).min
    page_ids_c, mask_c = split_by_chunk(page_ids, mask, bpc)
    merges = []
    merge = bdhm._merge_entry_local

    def counting_merge(**kwargs):
        merges.append(True)
        return merge(**kwargs)

    monkeypatch.setattr(bdhm, "_merge_entry_local", counting_merge)
    args = (query, rep, k_pages, v_pages, page_ids_c, mask_c, b, bpc, kv, q)
    per_slot = _kernel(*args, entry_local=True)
    assert merges, "entry_local=True did not take the per-slot path"
    torch.testing.assert_close(per_slot, _kernel(*args))
    torch.testing.assert_close(
        per_slot, _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, kv, q, 0.0)
    )


def test_one_slot_per_chunk_matches_reference():
    """A one-slot carry has no cross-slot merge to defer (B32/1024 uses this)."""
    b, bpc, c, kv, q = 2, 1, 3, 2, 4
    query, rep = _meta(b, bpc, kv, q)
    k_pages, v_pages, page_ids, mask, j_total = _inputs(b, bpc, c, kv)
    mask[-1, 0, 0, 0, 2:] = float("-inf")
    got = _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc, kv, q)
    torch.testing.assert_close(
        got, _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, kv, q, 0.0)
    )


@pytest.mark.parametrize("c", [2, 3])
def test_multi_chunk_masked(c):
    b, bpc, kv, q, cap = 2, 2, 2, 4, 2.0
    query, rep = _meta(b, bpc, kv, q)
    k_pages, v_pages, page_ids, mask, j_total = _inputs(b, bpc, c, kv)
    # Partial page (s=0, slot 0); a slot padded in EVERY chunk for s=1; a slot valid
    # early and masked in a later chunk.
    mask[0, 0, 0, 0, 2:] = float("-inf")
    for jj in range(0, j_total, bpc):
        mask[jj, 1, 0, 0, :] = float("-inf")
    mask[1 + bpc, 0, 0, 0, :] = float("-inf")
    got = _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc, kv, q, cap=cap)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(
        got,
        _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, kv, q, cap),
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize(("c", "bpc"), [(1, 2), (3, 1), (3, 2)])
def test_sinks_count_once_per_sequence(c, bpc):
    """A sink joins the denominator once per sequence: with several rows per chunk the
    per-slot merge must not add it per slot. Some sinks dominate the scores, some vanish,
    and one sequence has a fully masked slot in every chunk."""
    b, kv, q = 2, 2, 4
    query, rep = _meta(b, bpc, kv, q)
    k_pages, v_pages, page_ids, mask, j_total = _inputs(b, bpc, c, kv)
    if bpc > 1:
        for jj in range(0, j_total, bpc):
            mask[jj, 1, 0, 0, :] = float("-inf")
    sinks = torch.randn(kv, q, 1) * 3
    sinks[0, 0] = 20.0
    sinks[-1, -1] = -20.0
    got = _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc, kv, q, sinks=sinks)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(
        got,
        _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, kv, q, 0.0, sinks),
        atol=1e-4,
        rtol=1e-4,
    )


def test_body_matches_reference_with_init_state():
    """Thread the real FET init ``(-inf, 0, 0)`` through every trip, with slot 0 of the
    first chunk fully masked."""
    b, bpc, c, kv, q = 2, 2, 2, 2, 4
    query, rep = _meta(b, bpc, kv, q)
    k_pages, v_pages, page_ids, mask, j_total = _inputs(b, bpc, c, kv)
    mask[0, :, 0, 0, :] = float("-inf")
    entries = b * bpc
    q_flat = query.index_select(0, rep).reshape(entries, kv, q, HEAD)

    state_shape = (bpc, b, kv, q, 1)
    carry = (
        torch.full(state_shape, float("-inf")),
        torch.zeros(state_shape),
        torch.zeros(bpc, b, kv, q, HEAD),
    )
    for chunk in range(c):
        tile_ids = page_ids[chunk * bpc : (chunk + 1) * bpc]
        tile_mask = mask[chunk * bpc : (chunk + 1) * bpc]
        k_page = k_pages[tile_ids].reshape(entries, kv, BLOCK, HEAD)
        v_page = v_pages[tile_ids].reshape(entries, kv, BLOCK, HEAD)
        scores = torch.matmul(q_flat, k_page.transpose(-2, -1)) * SCALE
        sc = scores.reshape(bpc, b, kv, q, BLOCK) + tile_mask
        carry = bdhm._entry_local_update(carry, sc, v_page, entries, bpc, b, kv, q, BLOCK, HEAD)
    tile_max, tile_sum, tile_output = carry
    got = bdhm._merge_entry_local(
        tile_max=tile_max,
        tile_sum=tile_sum,
        tile_output=tile_output,
        num_seqs=b,
        num_kv_heads=kv,
        num_queries_per_kv=q,
        head_size=HEAD,
    )
    torch.testing.assert_close(
        got,
        _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, kv, q, 0.0),
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize("c", [1, 3])
def test_uploaded_metadata_matches_kernel(monkeypatch, c):
    """Feed the production upload's metadata to the whole kernel on CPU.

    Only the device transfer is replaced. The Python walk emulates chunk
    boundaries; this does not test compiled for_each_tile execution or the
    device's physical index layout.
    """
    from spyre_inference.v1.attention.backends import spyre_attn as backend

    b, bpc, kv, q = 2, 2, 2, 4
    query, rep = _meta(b, bpc, kv, q)
    k_pages, v_pages, page_ids, mask, j_total = _inputs(b, bpc, c, kv)
    mask[0, 0, 0, 0, 2:] = float("-inf")
    mask[::bpc, 1] = float("-inf")
    # The builder materializes the KV-head axis.
    mask = mask.expand(-1, -1, kv, -1, -1).contiguous()
    original_ids, original_mask = page_ids.clone(), mask.clone()
    metadata = SimpleNamespace(
        padded_num_seqs=b,
        blocks_per_chunk=bpc,
        rep_row_ids_cpu=rep,
        chunk_page_ids_cpu=page_ids,
        mask_by_chunk_cpu=mask,
        chunk_range=None,
        shared_slot_seqs=None,
    )
    layouts = []
    original_to = torch.Tensor.to

    def cpu_transfer(tensor, *args, **kwargs):
        if "device_layout" in kwargs:
            layouts.append(kwargs.pop("device_layout"))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_transfer)
    impl = object.__new__(backend.SpyreAttentionImpl)
    impl._mirror_batched_decode_indices(metadata, torch.device("cpu"))
    assert len(layouts) == 1
    assert list(layouts[0].device_size) == [c, 1, bpc * b, 32]
    got = _kernel(
        query,
        metadata.rep_row_ids_dev,
        k_pages,
        v_pages,
        metadata.chunk_page_ids_dev,
        metadata.mask_by_chunk_dev,
        b,
        bpc,
        kv,
        q,
    )
    torch.testing.assert_close(
        got,
        _reference(query, k_pages, v_pages, original_ids, original_mask, b, j_total, kv, q, 0.0),
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.equal(metadata.chunk_page_ids_cpu, original_ids)
    assert torch.equal(metadata.mask_by_chunk_cpu, original_mask)


def test_temp_split_layout_fields():
    """The proven host [C, E, 1] / device [C, 1, E, 32] fields."""
    from spyre_inference.v1.attention.ops.layout import (
        temporary_chunk_major_page_index_layout,
    )

    layout = temporary_chunk_major_page_index_layout(3, 8)
    assert list(layout.device_size) == [3, 1, 8, 32]
    assert list(layout.stride_map) == [8, 0, 1, 0]


def _shared_slot_case(
    blocks, rows, kv, num_chunks=None, first_block_offset=0.0, slots=32, dtype=torch.float32
):
    """Distinct pages per sequence and a part-filled last block, laid out by the builder.

    ``first_block_offset`` is added to sequence 0's block-0 scores through its mask.
    Returns the shared-slot kernel's inputs and each sequence's own pages and mask tiles.
    """
    from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadataBuilder
    from spyre_inference.v1.attention.spyre_attn_bucketer import SharedSlotPlan, share_block_slots

    chunks, seq_slots = share_block_slots(blocks, slots)
    block_table = torch.zeros(len(blocks), max(blocks), dtype=torch.int32)
    stacks = []
    first = 1
    for s, b in enumerate(blocks):
        block_table[s, :b] = torch.arange(first, first + b)
        first += b
        tiles = torch.zeros(b, 1, BLOCK, dtype=dtype)
        tiles[-1, 0, 2:] = float("-inf")
        stacks.append(tiles)
    stacks[0][0] += first_block_offset
    owner = SimpleNamespace(
        block_size=BLOCK,
        model_dtype=dtype,
        num_kv_heads=kv,
        _attn_bucketer=SimpleNamespace(shared_slot_seqs=rows, shared_slots=slots),
    )
    fields = SpyreAttentionMetadataBuilder._shared_slot_tensors(
        owner,
        list(range(len(blocks))),
        SharedSlotPlan(num_chunks or chunks, None, seq_slots),
        blocks,
        block_table,
        torch.arange(len(blocks) + 1),
        None,
        stacks,
    )
    return fields, block_table, stacks, first


def _slot_pages(num_pages, kv):
    torch.manual_seed(1)
    return torch.randn(num_pages, kv, BLOCK, HEAD), torch.randn(num_pages, kv, BLOCK, HEAD)


def _run_shared_slots(query, k_pages, v_pages, fields, kv, q, **kwargs):
    (_, slots, _, rep, page_ids, mask), onehot = fields
    page_ids, mask = split_by_chunk(page_ids, mask, slots)
    return bdhm.batched_decode_kernel(
        query=query,
        rep_row_ids=rep,
        k_pages=k_pages,
        v_pages=v_pages,
        chunk_page_ids=page_ids,
        mask_by_chunk=mask,
        scale=SCALE,
        num_seqs=1,
        blocks_per_chunk=slots,
        num_kv_heads=kv,
        num_queries_per_kv=q,
        block_size=BLOCK,
        head_size=HEAD,
        entry_local=True,
        slot_onehot=onehot,
        **kwargs,
    )


def test_idle_shared_slots_stay_out_of_every_row_when_their_sums_overflow():
    """An idle block slot reads page 0 fully masked, and after the clamp each of its masked
    positions counts with weight 1, so its fp16 sums grow with every chunk. Large values in
    page 0 overflow them to inf; the merge must still leave that slot out of every row
    instead of multiplying it by zero."""
    kv, q, blocks = 2, 4, [3, 2]
    # 24 chunks of 4 positions at 1000 per value: 96000, past fp16's 65504.
    fields, block_table, stacks, num_pages = _shared_slot_case(
        blocks, 2, kv, num_chunks=24, dtype=torch.float16
    )
    k_pages, v_pages = _slot_pages(num_pages, kv)
    # No sequence owns page 0, so only the idle slots read it.
    v_pages[0] = 1000.0
    query = torch.randn(len(blocks), kv, q, HEAD)
    got = _run_shared_slots(query.half(), k_pages.half(), v_pages.half(), fields, kv, q).float()
    assert torch.isfinite(got).all()
    want = _per_seq_reference(
        query, k_pages, v_pages, block_table, [t.float() for t in stacks], kv, q
    )
    # fp16 inputs and accumulation against an fp32 reference.
    torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)


def _per_seq_reference(query, k_pages, v_pages, block_table, stacks, kv, q, cap=0.0, sinks=None):
    return torch.cat(
        [
            _reference(
                query[s : s + 1],
                k_pages,
                v_pages,
                block_table[s, : len(tiles)].reshape(-1, 1),
                tiles.reshape(-1, 1, 1, 1, BLOCK),
                1,
                len(tiles),
                kv,
                q,
                cap,
                sinks,
            )
            for s, tiles in enumerate(stacks)
        ]
    )


@pytest.mark.parametrize(
    ("blocks", "rows", "num_chunks"),
    [
        pytest.param([3], 4, None, id="one_seq_one_chunk_spare_slots"),
        pytest.param([40, 7], 4, None, id="two_seqs_two_chunks"),
        pytest.param([64, 64, 64], 4, None, id="three_uniform_seven_chunks"),
        pytest.param([9, 1, 30, 2, 17], 8, None, id="five_jagged"),
        pytest.param([1] * 31 + [5], 32, None, id="thirty_two_one_slot_each"),
        pytest.param([235, 5, 5, 5, 5], 8, None, id="one_long_four_short_nine_chunks"),
        pytest.param([40, 7], 4, 4, id="later_chunks_all_padding"),
    ],
)
@pytest.mark.parametrize("slots", [32, 64])
def test_shared_slots_match_reference(blocks, rows, num_chunks, slots):
    """Each row's block slots merge into its sequence's attention, and a padding row repeats
    row 0."""
    kv, q, n = 2, 4, len(blocks)
    fields, block_table, stacks, num_pages = _shared_slot_case(
        blocks, rows, kv, num_chunks, slots=slots
    )
    k_pages, v_pages = _slot_pages(num_pages, kv)
    query = torch.randn(n, kv, q, HEAD)
    got = _run_shared_slots(query, k_pages, v_pages, fields, kv, q)
    assert got.shape == (rows, kv * q, HEAD)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(
        got[:n],
        _per_seq_reference(query, k_pages, v_pages, block_table, stacks, kv, q),
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.equal(got[n:], got[:1].expand(rows - n, -1, -1))


@pytest.mark.parametrize("cap", [0.0, 2.0])
def test_shared_slot_sinks_count_once_per_sequence(cap):
    """A sink joins a sequence's denominator once, however many block slots it spans."""
    kv, q, blocks = 2, 4, [40, 7]
    fields, block_table, stacks, num_pages = _shared_slot_case(blocks, 4, kv)
    k_pages, v_pages = _slot_pages(num_pages, kv)
    query = torch.randn(len(blocks), kv, q, HEAD)
    sinks = torch.randn(kv, q, 1) * 3
    sinks[0, 0] = 20.0
    sinks[-1, -1] = -20.0
    got = _run_shared_slots(
        query, k_pages, v_pages, fields, kv, q, logits_soft_cap=cap, sinks=sinks
    )
    torch.testing.assert_close(
        got[: len(blocks)],
        _per_seq_reference(query, k_pages, v_pages, block_table, stacks, kv, q, cap, sinks),
        atol=1e-4,
        rtol=1e-4,
    )


def test_shared_slot_padding_rows_stay_finite_beside_a_dominant_slot():
    """Block slot 0's scores sit far below its row's maximum, so its own weight underflows
    to 0. A padding row built from block slot 0 alone would divide 0 by 0."""
    kv, q, blocks = 2, 4, [3]
    fields, block_table, stacks, num_pages = _shared_slot_case(
        blocks, 4, kv, first_block_offset=-500.0
    )
    k_pages, v_pages = _slot_pages(num_pages, kv)
    query = torch.randn(1, kv, q, HEAD)
    got = _run_shared_slots(query, k_pages, v_pages, fields, kv, q)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(
        got[:1],
        _per_seq_reference(query, k_pages, v_pages, block_table, stacks, kv, q),
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.equal(got[1:], got[:1].expand(3, -1, -1))


def test_shared_slot_store_writes_its_rows_from_row_zero():
    """``out`` takes exactly the slot rows, scaled by v_scale; rows past them are untouched."""
    kv, q, blocks, rows = 2, 4, [40, 7], 4
    fields, _, _, num_pages = _shared_slot_case(blocks, rows, kv)
    k_pages, v_pages = _slot_pages(num_pages, kv)
    query = torch.randn(9, kv, q, HEAD)
    out = torch.full((9, kv * q, HEAD), float("nan"))
    expected = _run_shared_slots(query, k_pages, v_pages, fields, kv, q) * 0.5
    assert _run_shared_slots(query, k_pages, v_pages, fields, kv, q, out=out, v_scale=0.5) is out
    torch.testing.assert_close(out[:rows], expected)
    assert out[rows:].isnan().all()


def test_shared_slots_match_the_per_bucket_kernel_on_a_uniform_batch():
    """Four sequences of 16 blocks: 8 block slots each over 2 chunks, the per-bucket kernel's
    B=4, J=8 split in another entry order."""
    kv, q, b, nb = 2, 4, 4, 16
    fields, block_table, stacks, num_pages = _shared_slot_case([nb] * b, b, kv)
    assert fields[0][4].shape[0] == 2 * 32
    k_pages, v_pages = _slot_pages(num_pages, kv)
    query = torch.randn(b, kv, q, HEAD)
    page_ids = block_table.t().contiguous()
    mask = torch.stack([tiles[:, 0] for tiles in stacks], dim=1).reshape(nb, b, 1, 1, BLOCK)
    bucketed = _call(
        query, torch.arange(b).repeat(8), k_pages, v_pages, page_ids, mask, b, 8, kv, q
    )
    torch.testing.assert_close(
        _run_shared_slots(query, k_pages, v_pages, fields, kv, q), bucketed, atol=1e-5, rtol=1e-5
    )


def test_shared_slots_walk_the_active_blocks():
    """Under a window a block slot reads the i-th active block's page and tile, not block i's."""
    from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadataBuilder
    from spyre_inference.v1.attention.spyre_attn_bucketer import SharedSlotPlan

    block_table = torch.arange(10, 20, dtype=torch.int32).reshape(1, 10)
    active = [[6, 7, 8, 9]]
    tiles = torch.arange(4 * BLOCK, dtype=torch.float32).reshape(4, 1, BLOCK)
    owner = SimpleNamespace(
        block_size=BLOCK,
        model_dtype=torch.float32,
        num_kv_heads=1,
        _attn_bucketer=SimpleNamespace(shared_slot_seqs=1, shared_slots=32),
    )
    (_, _, _, _, page_ids, mask), _ = SpyreAttentionMetadataBuilder._shared_slot_tensors(
        owner,
        [0],
        SharedSlotPlan(1, None, [4]),
        [4],
        block_table,
        torch.arange(2),
        active,
        [tiles],
    )
    assert page_ids[:4, 0].tolist() == [16, 17, 18, 19]
    assert torch.equal(mask[:4, 0, 0, 0], tiles[:, 0])
    assert mask[4:].isneginf().all()
