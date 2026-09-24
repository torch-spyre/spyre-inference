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

from spyre_inference.v1.attention.ops import batched_decode_head_major as bdhm
from spyre_inference.v1.attention.ops import tile_loop

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


def _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, kv, q, cap):
    """Independent per-sequence softmax; pages concatenated along the token axis."""
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
                out[s, h, qq] = torch.softmax(sc + m, dim=-1) @ v[h]
    return out.reshape(b, kv * q, HEAD)


def _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc, kv, q, cap=0.0):
    return bdhm.batched_decode_head_major_kernel(
        query,
        rep,
        k_pages,
        v_pages,
        page_ids,
        mask,
        SCALE,
        b,
        bpc,
        kv,
        q,
        BLOCK,
        HEAD,
        logits_soft_cap=cap,
    )


def _meta(b, bpc, kv, q):
    query = torch.randn(b, kv, q, HEAD)
    rep = torch.arange(b).repeat(bpc)
    return query, rep


@pytest.fixture(autouse=True)
def python_walk(monkeypatch):
    # The body math is under test; drive the Python walk (carry=None).
    monkeypatch.setattr(tile_loop, "USE_FOR_EACH_TILE", False)
    # The kernel captures the walk mode separately at import time.
    monkeypatch.setattr(bdhm, "USE_FOR_EACH_TILE", False)


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

    def unexpected_merge(*args):
        pytest.fail("A single chunk must not build and merge per-slot state")

    monkeypatch.setattr(bdhm, "_merge_entry_local", unexpected_merge)
    got = _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc, kv, q)
    torch.testing.assert_close(
        got, _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, kv, q, 0.0)
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
    got = bdhm._merge_entry_local(*carry, b, kv, q, HEAD)
    torch.testing.assert_close(
        got,
        _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, kv, q, 0.0),
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize("c", [1, 3])
@pytest.mark.parametrize("split_index", [False, True])
def test_uploaded_metadata_matches_kernel(monkeypatch, c, split_index):
    """Feed the production upload's metadata to the whole kernel on CPU.

    Only the device transfer is replaced. The Python walk emulates chunk
    boundaries, including the split-index shape branch; this does not test
    compiled for_each_tile execution or the device's physical index layout.
    """
    from spyre_inference.v1.attention.backends import spyre_head_major_attn as backend

    b, bpc, kv, q = 2, 2, 2, 4
    query, rep = _meta(b, bpc, kv, q)
    k_pages, v_pages, page_ids, mask, j_total = _inputs(b, bpc, c, kv)
    mask[0, 0, 0, 0, 2:] = float("-inf")
    mask[::bpc, 1] = float("-inf")
    if split_index:
        # The tiled walk's builder materializes the KV-head axis.
        mask = mask.expand(-1, -1, kv, -1, -1).contiguous()
    original_ids, original_mask = page_ids.clone(), mask.clone()
    metadata = SimpleNamespace(
        padded_num_seqs=b,
        blocks_per_chunk=bpc,
        rep_row_ids_cpu=rep,
        chunk_page_ids_cpu=page_ids,
        mask_by_chunk_cpu=mask,
    )
    layouts = []
    original_to = torch.Tensor.to

    def cpu_transfer(tensor, *args, **kwargs):
        if "device_layout" in kwargs:
            layouts.append(kwargs.pop("device_layout"))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_transfer)
    monkeypatch.setattr(backend, "USE_FOR_EACH_TILE", split_index)
    monkeypatch.setattr(bdhm, "USE_FOR_EACH_TILE", split_index)
    impl = object.__new__(backend.SpyreHeadMajorAttentionImpl)
    impl._mirror_batched_decode_indices(metadata, torch.device("cpu"))
    assert len(layouts) == int(split_index)
    if split_index:
        assert list(layouts[0].device_size) == [c, 1, bpc * b, 32]
    got = _call(
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
    assert list(layout.stride_map) == [8, -1, 1, -1]
