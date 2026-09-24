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

"""Card-free tests for the token-major split-index walk."""

from types import SimpleNamespace

import pytest
import torch

from spyre_inference.v1.attention.ops import batched_decode as bd
from spyre_inference.v1.attention.ops import tile_loop

BLOCK, HEAD, SCALE = 4, 8, 0.3


def _inputs(b, bpc, c, kv):
    """Token-major pages [P, block, KV, D]; mask [J, B, 1, 1, block]."""
    j_total = c * bpc
    torch.manual_seed(0)
    k_pages = torch.randn(64, BLOCK, kv, HEAD)
    v_pages = torch.randn(64, BLOCK, kv, HEAD)
    page_ids = (torch.arange(j_total * b).reshape(j_total, b) + 1).to(torch.int32)
    mask = torch.zeros(j_total, b, 1, 1, BLOCK, dtype=torch.float32)
    return k_pages, v_pages, page_ids, mask, j_total


def _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, kv, q):
    """Independent per-sequence softmax; pages concatenated along the token axis."""
    out = torch.empty(b, kv, q, HEAD)
    for s in range(b):
        # Token-major page is [block, KV, D]; take the head's tokens.
        k = torch.cat([k_pages[int(page_ids[jj, s])] for jj in range(j_total)], dim=0)
        v = torch.cat([v_pages[int(page_ids[jj, s])] for jj in range(j_total)], dim=0)
        m = torch.cat([mask[jj, s, 0, 0, :] for jj in range(j_total)])
        for h in range(kv):
            for qq in range(q):
                sc = (query[s, h, qq] @ k[:, h].transpose(0, 1)) * SCALE
                out[s, h, qq] = torch.softmax(sc + m, dim=-1) @ v[:, h]
    return out.reshape(b, kv * q, HEAD)


def _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc, kv, q):
    return bd.batched_decode_kernel(
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
    )


def _meta(b, bpc, kv, q):
    return torch.randn(b, kv, q, HEAD), torch.arange(b).repeat(bpc)


@pytest.fixture(autouse=True)
def python_walk(monkeypatch):
    # The body math is under test; drive the Python walk (carry=None).
    monkeypatch.setattr(tile_loop, "USE_FOR_EACH_TILE", False)
    monkeypatch.setattr(bd, "USE_FOR_EACH_TILE", False)


@pytest.mark.parametrize(("c", "kv", "q"), [(1, 1, 1), (2, 1, 1), (2, 2, 4)])
def test_split_index_off_matches_reference(monkeypatch, c, kv, q):
    b, bpc = 2, 2
    query, rep = _meta(b, bpc, kv, q)
    k_pages, v_pages, page_ids, mask, j_total = _inputs(b, bpc, c, kv)
    got = _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc, kv, q)
    torch.testing.assert_close(
        got,
        _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, kv, q),
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize("c", [1, 3])
@pytest.mark.parametrize("split_index", [False, True])
def test_uploaded_metadata_matches_kernel(monkeypatch, c, split_index):
    """Feed the production upload's metadata to the whole kernel on CPU.

    Only the device transfer is replaced. This is the test that pins the
    upload/kernel contract: with split_index the kernel's mask reshape needs the
    chunk-major 6-D tile, and against a 5-D one it silently folds the KV axis
    into the block slots whenever blocks_per_chunk == num_kv_heads.
    """
    from spyre_inference.v1.attention.backends import spyre_attn as backend

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
    monkeypatch.setattr(backend.tile_loop, "USE_FOR_EACH_TILE", split_index)
    monkeypatch.setattr(bd, "USE_FOR_EACH_TILE", split_index)
    impl = object.__new__(backend.SpyreAttentionImpl)
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
        _reference(query, k_pages, v_pages, original_ids, original_mask, b, j_total, kv, q),
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.equal(metadata.chunk_page_ids_cpu, original_ids)
    assert torch.equal(metadata.mask_by_chunk_cpu, original_mask)


def test_token_major_supports_tiled_batched_decode(monkeypatch):
    from spyre_inference.v1.attention.backends import spyre_attn as backend

    impl = object.__new__(backend.SpyreAttentionImpl)
    impl.alibi_slopes = None
    impl._compile_attn = True
    # SPYRE_BATCHED_DECODE is read at import and is off by default; the gate checks
    # it before the walk-mode branch this test is about.
    monkeypatch.setattr(backend.envs, "SPYRE_BATCHED_DECODE", True)
    monkeypatch.setattr(backend.tile_loop, "USE_FOR_EACH_TILE", True)
    assert impl._batched_decode_supported() is True
