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

"""attn_groups.init: upstream's groups, split by window, each with its own builder."""

import types

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from spyre_inference.v1.worker import kv_cache_setup
from spyre_inference.v1.worker.spyre import attn_groups

_SPEC = dict(block_size=64, num_kv_heads=1, head_size=64, dtype=torch.float16)


class _Builder:
    def __init__(self, spec, layer_names, vllm_config, device):
        self.spec, self.layer_names = spec, layer_names


class _Group:
    """An AttentionGroup whose builders record what they were built for."""

    def __init__(self, backend, layer_names, kv_cache_spec, kv_cache_group_id):
        self.backend = backend
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.kv_cache_group_id = kv_cache_group_id
        self.metadata_builders: list[_Builder] = []

    def create_metadata_builders(
        self, vllm_config, device, kernel_block_size, num_metadata_builders
    ):
        self.metadata_builders = [
            _Builder(self.kv_cache_spec, self.layer_names, vllm_config, device)
            for _ in range(num_metadata_builders)
        ]


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch):
    """Stub each layer's sliding window; the split builds _Group for its new groups."""
    monkeypatch.setattr(kv_cache_setup, "AttentionGroup", _Group)

    def install(windows_by_layer: dict[str, int | None]) -> None:
        layers = {
            name: types.SimpleNamespace(sliding_window=w) for name, w in windows_by_layer.items()
        }
        monkeypatch.setattr(attn_groups, "get_layers_from_vllm_config", lambda *a, **k: layers)

    return install


def _config() -> types.SimpleNamespace:
    return types.SimpleNamespace(parallel_config=types.SimpleNamespace(use_ubatching=False))


def test_each_window_gets_a_group_with_its_own_builders(windows, monkeypatch) -> None:
    names = [f"layers.{i}" for i in range(6)]
    windows({n: (None if i % 3 == 2 else 512) for i, n in enumerate(names)})
    merged = _Group(object, list(names), FullAttentionSpec(sliding_window=512, **_SPEC), 0)
    merged.create_metadata_builders(None, None, None, 1)

    def upstream(kv_cache_config, vllm_config, device, *args, **kwargs):
        return [[merged]], "pre-split cg-support", [64]

    recomputed = []
    monkeypatch.setattr(
        attn_groups,
        "get_attn_cg_support",
        lambda groups, vllm_config: recomputed.append(groups) or "post-split cg-support",
    )

    groups, cg_support, kernel_block_sizes = attn_groups.init(
        upstream, None, _config(), torch.device("cpu")
    )

    assert (cg_support, kernel_block_sizes) == ("post-split cg-support", [64])
    assert recomputed == [groups], "cg support must describe the groups after the split"
    by_window = {g.kv_cache_spec.sliding_window: g for g in groups[0]}
    assert set(by_window) == {512, None}
    assert sorted(by_window[None].layer_names) == ["layers.2", "layers.5"]
    for group in groups[0]:
        assert group.metadata_builders, "a split group has no metadata builder"
        (builder,) = group.metadata_builders
        assert builder.layer_names == group.layer_names
        assert builder.spec.sliding_window == group.kv_cache_spec.sliding_window


def test_an_unsplit_group_keeps_upstreams_builders(windows) -> None:
    names = ["layers.0", "layers.1"]
    windows(dict.fromkeys(names))
    group = _Group(object, names, FullAttentionSpec(**_SPEC), 0)
    group.create_metadata_builders(None, None, None, 1)
    upstream_builders = group.metadata_builders

    groups, cg_support, _ = attn_groups.init(
        lambda *a, **k: ([[group]], "upstream cg-support", [64]),
        None,
        _config(),
        torch.device("cpu"),
    )

    assert groups == [[group]]
    assert cg_support == "upstream cg-support"
    assert group.metadata_builders is upstream_builders
