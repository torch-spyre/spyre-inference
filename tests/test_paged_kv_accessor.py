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

"""Unit tests for SpyrePagedKVCacheAccessor with fake CPU page lists.

These run without vLLM, NIXL, or AIU hardware: the accessor only needs
torch tensors shaped like the Spyre list-of-pages KV cache
([num_kv_heads, block_size, head_size] per page).
"""

import importlib.util
import pathlib
from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch", reason="torch required for paged accessor tests")

REPO = pathlib.Path(__file__).resolve().parents[1]
CONNECTOR_DIR = REPO / "spyre_inference" / "distributed" / "kv_transfer" / "kv_connector" / "v1"

# Load by file path: the accessor module is vLLM-free, but importing it through
# the spyre_inference package would pull vLLM in via spyre_inference/__init__.py.
_spec = importlib.util.spec_from_file_location(
    "spyre_paged_kv_accessor", CONNECTOR_DIR / "spyre_paged_kv_accessor.py"
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SpyrePagedKVCacheAccessor = _mod.SpyrePagedKVCacheAccessor
is_paged_layer_cache = _mod.is_paged_layer_cache

NUM_KV_HEADS = 2
BLOCK_SIZE = 4
HEAD_DIM = 8
NUM_PAGES = 3
LAYERS = ["model.layers.0.attn", "model.layers.1.attn"]


class FakeSpyrePagedKVCache(NamedTuple):
    """Tuple-compatible stand-in for spyre_attn.SpyrePagedKVCache."""

    k_pages: list[torch.Tensor]
    v_pages: list[torch.Tensor]


def make_pages(fill: float = 0.0) -> list[torch.Tensor]:
    return [
        torch.full((NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM), fill, dtype=torch.float16)
        for _ in range(NUM_PAGES)
    ]


def make_kv_caches() -> dict[str, FakeSpyrePagedKVCache]:
    return {
        name: FakeSpyrePagedKVCache(k_pages=make_pages(), v_pages=make_pages()) for name in LAYERS
    }


def test_is_paged_layer_cache_detects_page_lists():
    assert is_paged_layer_cache((make_pages(), make_pages()))
    assert is_paged_layer_cache(FakeSpyrePagedKVCache(make_pages(), make_pages()))


def test_is_paged_layer_cache_rejects_other_layouts():
    staging = torch.zeros(2, NUM_PAGES, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    assert not is_paged_layer_cache(staging)
    assert not is_paged_layer_cache(None)
    assert not is_paged_layer_cache(([], []))
    assert not is_paged_layer_cache((make_pages(), make_pages()[:-1]))
    assert not is_paged_layer_cache((make_pages()[0], make_pages()[0]))


def test_try_from_kv_caches_infers_geometry():
    accessor = SpyrePagedKVCacheAccessor.try_from_kv_caches(make_kv_caches())
    assert accessor is not None
    assert accessor.num_layers == len(LAYERS)
    assert accessor.layer_names == sorted(LAYERS)
    assert accessor.num_pages == NUM_PAGES
    assert accessor.num_kv_heads == NUM_KV_HEADS
    assert accessor.block_size == BLOCK_SIZE
    assert accessor.head_dim == HEAD_DIM
    assert accessor.dtype == torch.float16
    assert accessor.page_shape == (NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    assert accessor.block_shape == (BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    assert accessor.page_nbytes == NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM * 2


def test_try_from_kv_caches_returns_none_for_staging_tensors():
    staging = {"layer": torch.zeros(2, NUM_PAGES, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)}
    assert SpyrePagedKVCacheAccessor.try_from_kv_caches(staging) is None
    assert SpyrePagedKVCacheAccessor.try_from_kv_caches({}) is None


def test_inconsistent_geometry_rejected():
    caches = make_kv_caches()
    caches[LAYERS[1]].k_pages[0] = torch.zeros(
        NUM_KV_HEADS, BLOCK_SIZE * 2, HEAD_DIM, dtype=torch.float16
    )
    with pytest.raises(ValueError, match="Inconsistent page geometry"):
        SpyrePagedKVCacheAccessor(caches)


def test_write_then_read_block_roundtrip():
    caches = make_kv_caches()
    accessor = SpyrePagedKVCacheAccessor(caches)
    block = torch.arange(BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM, dtype=torch.float16).reshape(
        BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM
    )

    accessor.write_block(layer_name=LAYERS[0], kv_kind="k", page_id=1, values=block)

    assert torch.equal(accessor.read_block(layer_name=LAYERS[0], kv_kind="k", page_id=1), block)
    # The write landed in the registered page tensor, permuted to page layout.
    assert torch.equal(caches[LAYERS[0]].k_pages[1], block.permute(1, 0, 2))
    # Other pages, kinds, and layers stay untouched.
    assert caches[LAYERS[0]].k_pages[0].abs().sum() == 0
    assert caches[LAYERS[0]].v_pages[1].abs().sum() == 0
    assert caches[LAYERS[1]].k_pages[1].abs().sum() == 0


def test_write_block_validates_shape_dtype_kind_and_range():
    accessor = SpyrePagedKVCacheAccessor(make_kv_caches())
    good = torch.zeros(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)

    with pytest.raises(ValueError, match="block tensor shape"):
        accessor.write_block(
            layer_name=LAYERS[0], kv_kind="k", page_id=0, values=good.permute(1, 0, 2)
        )
    with pytest.raises(ValueError, match="block dtype"):
        accessor.write_block(layer_name=LAYERS[0], kv_kind="k", page_id=0, values=good.float())
    with pytest.raises(ValueError, match="kv_kind"):
        accessor.write_block(layer_name=LAYERS[0], kv_kind="q", page_id=0, values=good)
    with pytest.raises(IndexError):
        accessor.write_block(layer_name=LAYERS[0], kv_kind="k", page_id=NUM_PAGES, values=good)
    with pytest.raises(KeyError):
        accessor.write_block(layer_name="nope", kv_kind="k", page_id=0, values=good)


def test_transfer_units_accounting():
    accessor = SpyrePagedKVCacheAccessor(make_kv_caches())
    units = accessor.transfer_units([0, 1, NUM_PAGES + 5])
    assert units["unit"] == "page"
    assert units["unit_shape"] == (NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    assert units["units_total"] == len(LAYERS) * 2 * 2
    assert units["total_nbytes"] == units["units_total"] * units["unit_nbytes"]
    assert units["invalid_block_ids"] == [NUM_PAGES + 5]


def test_describe_reports_layout():
    desc = SpyrePagedKVCacheAccessor(make_kv_caches()).describe()
    assert desc["layout"] == "list_of_pages"
    assert desc["num_layers"] == len(LAYERS)
    assert desc["num_pages"] == NUM_PAGES


def test_accessor_module_has_no_runtime_imports():
    """The accessor must stay importable without vLLM, NIXL, or torch_spyre."""
    src = (CONNECTOR_DIR / "spyre_paged_kv_accessor.py").read_text()
    for forbidden in (
        "import vllm",
        "from vllm",
        "import nixl",
        "from nixl",
        "import torch_spyre",
        "from torch_spyre",
    ):
        assert forbidden not in src


def test_connector_dispatches_paged_path():
    """Structural check: connector wires the paged accessor into load/save."""
    src = (CONNECTOR_DIR / "inmemory_spyre_connector.py").read_text()
    assert "SpyrePagedKVCacheAccessor.try_from_kv_caches" in src
    assert "_load_via_paged_accessor" in src
    assert "def paged_kv_active" in src
    assert "paged.read_block" in src
