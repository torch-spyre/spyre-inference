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

"""CPU-only tests for the Spyre paged -> canonical KV cache adapter."""

import inspect

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from spyre_inference.v1.kv_offload.connector import (
    SpyreOffloadingConnectorWorker,
    spyre_paged_to_canonical,
)

NUM_BLOCKS = 4
BLOCK_SIZE = 8
NUM_KV_HEADS = 2
HEAD_SIZE = 64


def _spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
    )


def _paged_cache() -> tuple[torch.Tensor, torch.Tensor]:
    """A stand-in for SpyrePagedKVCache: a 2-tuple of dense page tensors."""
    shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    return (torch.zeros(shape, dtype=torch.float16), torch.zeros(shape, dtype=torch.float16))


def _config(layer_names: list[str], spec: FullAttentionSpec) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)],
    )


def test_canonicalizes_paged_cache_per_layer():
    spec = _spec()
    layers = ["layer.0", "layer.1"]
    kv_caches = {name: _paged_cache() for name in layers}

    canonical = spyre_paged_to_canonical(kv_caches, _config(layers, spec))

    # Two tensors (K and V) per distinct paged cache.
    assert len(canonical.tensors) == 2 * len(layers)
    half_page = spec.page_size_bytes // 2
    for entry in canonical.tensors:
        assert entry.page_size_bytes == half_page
        assert entry.tensor.shape == (NUM_BLOCKS, BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE)
        # The canonical page must describe the whole device page.
        assert entry.tensor[0].numel() * entry.tensor.element_size() == half_page

    (group_refs,) = canonical.group_data_refs
    assert [ref.tensor_idx for ref in group_refs] == [0, 1, 2, 3]
    assert all(ref.page_size_bytes == half_page for ref in group_refs)
    assert all(ref.mapping is None for ref in group_refs)


def test_shared_cache_reuses_tensor_indices():
    spec = _spec()
    layers = ["layer.0", "layer.1"]
    shared = _paged_cache()

    canonical = spyre_paged_to_canonical(dict.fromkeys(layers, shared), _config(layers, spec))

    # One physical cache -> one K tensor and one V tensor, referenced twice.
    assert len(canonical.tensors) == 2
    (group_refs,) = canonical.group_data_refs
    assert [ref.tensor_idx for ref in group_refs] == [0, 1, 0, 1]


def test_canonical_tensor_aliases_device_pages():
    """The canonical view must be metadata-only, not a copy."""
    spec = _spec()
    k_pages, v_pages = _paged_cache()

    canonical = spyre_paged_to_canonical(
        {"layer.0": (k_pages, v_pages)}, _config(["layer.0"], spec)
    )

    canonical.tensors[0].tensor[2, 5] = 1.5
    assert k_pages.flatten()[2 * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE + 5] == 1.5
    assert canonical.tensors[0].tensor.data_ptr() == k_pages.data_ptr()
    assert canonical.tensors[1].tensor.data_ptr() == v_pages.data_ptr()


def test_no_storage_reinterpretation(monkeypatch):
    """Guard the whole point of this adapter: no untyped_storage()/set_()."""

    def _banned(name):
        def fail(*args, **kwargs):
            pytest.fail(f"spyre_paged_to_canonical called torch.Tensor.{name}")

        return fail

    for name in ("untyped_storage", "set_", "as_strided"):
        monkeypatch.setattr(torch.Tensor, name, _banned(name), raising=True)

    spec = _spec()
    spyre_paged_to_canonical({"layer.0": _paged_cache()}, _config(["layer.0"], spec))


def test_rejects_non_paged_cache():
    spec = _spec()
    with pytest.raises(TypeError, match="SpyrePagedKVCache 2-tuple"):
        spyre_paged_to_canonical(
            {"layer.0": torch.zeros(NUM_BLOCKS, 8, dtype=torch.float16)},
            _config(["layer.0"], spec),
        )


def test_rejects_mismatched_page_size():
    spec = _spec()
    wrong = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE // 2, dtype=torch.float16)
    with pytest.raises(ValueError, match="device page is"):
        spyre_paged_to_canonical({"layer.0": (wrong, wrong.clone())}, _config(["layer.0"], spec))


def test_worker_constructor_tolerates_extra_positional_args():
    """Upstream added `vllm_config` as a 2nd positional arg once already; a
    further signature change should fail here, not at serve time."""
    params = inspect.signature(SpyreOffloadingConnectorWorker.__init__).parameters.values()
    kinds = {p.kind for p in params}
    assert inspect.Parameter.VAR_POSITIONAL in kinds, "must accept *args"
    assert inspect.Parameter.VAR_KEYWORD in kinds, "must accept **kwargs"
    assert not any(
        p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params if p.name != "self"
    ), "must not pin upstream's positional signature"
