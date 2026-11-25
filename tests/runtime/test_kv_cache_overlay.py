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

"""allocate_spyre_kv_caches follows upstream's KVCacheTensor placement: layer l of every
cache group shares one K/V pair, which each layer views in its own page shape.

Card-free: upstream's real grouping and placement run on CPU, and the impl's
allocate_pages / view_pages are stubbed to record what they were asked for."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
from vllm.v1.core.kv_cache_planning import get_kv_cache_groups_from_builder
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec

from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl
from spyre_inference.v1.worker import kv_cache_setup
from spyre_inference.v1.worker.spyre.worker import _kv_block_layout

MiB = 1 << 20


def _gemma4_26b() -> dict:
    """gemma-4-26B-A4B: 30 layers, every sixth global, as vLLM specs them at TP=1."""
    specs = {}
    for i in range(30):
        if i % 6 == 5:
            spec = FullAttentionSpec(
                block_size=128, num_kv_heads=2, head_size=512, dtype=torch.float16
            )
        else:
            spec = SlidingWindowSpec(
                block_size=128,
                num_kv_heads=8,
                head_size=256,
                dtype=torch.float16,
                sliding_window=1024,
            )
        specs[f"layers.{i}.attn"] = spec
    return specs


def _nemotron_h_4b() -> dict:
    """Nemotron-H-4B-Instruct at TP=1: 24 Mamba2 and 4 attention layers, 1024-token block."""
    attn = FullAttentionSpec(block_size=1024, num_kv_heads=8, head_size=96, dtype=torch.float16)
    mamba = MambaSpec(
        shapes=((3, 2 * 3072 + 2 * 8 * 128), (112, 64, 128)),
        dtypes=(torch.float16, torch.float16),
        block_size=8192,
        page_size_padded=attn.page_size_bytes,
        mamba_cache_mode="none",
    )
    pattern = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    specs: dict = {}
    for i, kind in enumerate(pattern):
        if kind == "M":
            specs[f"layers.{i}.mixer"] = mamba
        elif kind == "*":
            specs[f"layers.{i}.attn"] = attn
    return specs


def _config(hybrid: bool = True, block_size: int = 128) -> VllmConfig:
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        max_model_len=4096, multimodal_config=None, supports_kv_cache_grouping=False
    )
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    vllm_config.cache_config.block_size = block_size
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = not hybrid
    return vllm_config


def _kv_cache_config(vllm_config: VllmConfig, specs: dict, num_blocks: int):
    builder = get_kv_cache_config_builder(vllm_config)
    groups = get_kv_cache_groups_from_builder(builder, vllm_config, specs)
    return builder.get_kv_cache_config_from_groups(vllm_config, groups, num_blocks)


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def allocate_pages(cls, num_pages, spec, device):
        pages = (object(), object())
        calls.append(("allocate", num_pages, spec, pages))
        return pages

    def view_pages(cls, base, num_pages, spec):
        pages = (object(), object())
        calls.append(("view", num_pages, spec, base))
        return pages

    def allocate_mamba_states(num_blocks, spec, device):
        states = (object(), object())
        calls.append(("mamba", num_blocks, spec, states))
        return states

    monkeypatch.setattr(SpyreAttentionImpl, "allocate_pages", classmethod(allocate_pages))
    monkeypatch.setattr(SpyreAttentionImpl, "view_pages", classmethod(view_pages))
    monkeypatch.setattr(kv_cache_setup, "_allocate_mamba_states", allocate_mamba_states)
    return calls


def test_gemma4_keeps_upstreams_groups_and_block_sizes():
    """Upstream grows the global group's blocks to 256 tokens so every page is 1 MiB."""
    config = _kv_cache_config(_config(), _gemma4_26b(), num_blocks=100)

    sizes = sorted(g.kv_cache_spec.block_size for g in config.kv_cache_groups)
    assert sizes == [128] * 5 + [256]
    assert {g.kv_cache_spec.page_size_bytes for g in config.kv_cache_groups} == {1 * MiB}


def test_gemma4_layers_of_one_slot_share_one_allocation(recorded):
    """Five slots, one K/V pair each; the global layer views it as 128-token pages."""
    num_blocks = 100
    config = _kv_cache_config(_config(), _gemma4_26b(), num_blocks)
    global_group = next(
        i for i, g in enumerate(config.kv_cache_groups) if g.kv_cache_spec.block_size == 256
    )
    kernel_block_sizes = [128] * len(config.kv_cache_groups)

    kv_caches = kv_cache_setup.allocate_spyre_kv_caches(
        config, {}, {}, torch.device("cpu"), kernel_block_sizes
    )

    allocations = [c for c in recorded if c[0] == "allocate"]
    views = [c for c in recorded if c[0] == "view"]
    assert len(allocations) == 5
    assert len(views) == 25
    assert len({id(kv_caches[name]) for name in kv_caches}) == 30
    bases = {id(c[3]) for c in allocations}
    assert {id(c[3]) for c in views} == bases
    for kind, num_pages, spec, _ in recorded:
        assert spec.block_size == 128
        if spec.head_size == 512:
            assert (kind, num_pages, spec.num_kv_heads) == ("view", 2 * num_blocks, 2)
        else:
            assert (num_pages, spec.num_kv_heads) == (num_blocks, 8)
    global_layers = [n for n, s in _gemma4_26b().items() if isinstance(s, FullAttentionSpec)]
    assert sorted(config.kv_cache_groups[global_group].layer_names) == sorted(global_layers)


def test_without_the_hybrid_manager_every_layer_gets_its_own_pages(recorded):
    """One group: no layer overlays another, so nothing is viewed, as on main."""
    config = _kv_cache_config(_config(hybrid=False), _gemma4_26b(), num_blocks=50)

    kv_cache_setup.allocate_spyre_kv_caches(config, {}, {}, torch.device("cpu"))

    assert [c[0] for c in recorded] == ["allocate"] * 30
    by_head = {c[2].head_size: c[2] for c in recorded}
    assert by_head[256].num_kv_heads == 8 and by_head[512].num_kv_heads == 2


def test_two_layers_of_one_group_at_one_offset_are_rejected(recorded):
    """Only layers of different groups may share bytes; two of one group would overwrite
    each other's KV."""
    config = _kv_cache_config(_config(hybrid=False), _gemma4_26b(), num_blocks=50)
    first, second = config.kv_cache_tensors
    second.offset = first.offset

    with pytest.raises(RuntimeError, match="KV cache group 0 already holds"):
        kv_cache_setup.allocate_spyre_kv_caches(config, {}, {}, torch.device("cpu"))


def test_a_mamba_hybrid_gets_state_tensors_beside_its_attention_pages(recorded):
    """Upstream places a Mamba page over attention bytes, Mamba groups first; Spyre keeps
    the states apart, so no attention page is viewed as a Mamba state or the reverse."""
    num_blocks = 10
    specs = _nemotron_h_4b()
    config = _kv_cache_config(_config(block_size=1024), specs, num_blocks)
    assert type(config.kv_cache_groups[0].kv_cache_spec) is MambaSpec

    kv_caches = kv_cache_setup.allocate_spyre_kv_caches(
        config, {}, {}, torch.device("cpu"), [1024] * len(config.kv_cache_groups)
    )

    assert kv_caches.keys() == specs.keys()
    mamba = [call for call in recorded if call[0] == "mamba"]
    attention = [call for call in recorded if call[0] != "mamba"]
    assert {call[1] for call in mamba} == {num_blocks}
    assert [call[0] for call in attention] == ["allocate"] * 4
    for name, cache in kv_caches.items():
        kind = "mamba" if name.endswith(".mixer") else "allocate"
        assert any(call[0] == kind and call[3] is cache for call in recorded)


def test_sizing_counts_one_k_v_pair_per_slot():
    """Upstream's bytes per block: five 1 MiB pages, as five K and five V tensors."""
    layout = _kv_block_layout(_config(), _gemma4_26b())

    assert layout is not None
    pool_bytes, tensor_bytes, _ = layout
    assert pool_bytes == 5 * MiB
    assert sorted(tensor_bytes) == [MiB // 2] * 10


def test_a_block_outer_kv_cache_layout_is_rejected_at_startup(monkeypatch):
    """The allocator reads a KVCacheTensor's block stride as one layer's page, so only a
    layer-outer layout resolves; a block-outer one would pack a block's layers together."""
    from vllm.v1.attention.backends.utils import (
        get_supported_kv_cache_layouts,
        resolve_kv_cache_layout,
    )

    from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionBackend

    supported = [
        [layout.name for layout in get_supported_kv_cache_layouts([SpyreAttentionBackend])]
    ]
    assert resolve_kv_cache_layout(VllmConfig(), supported).name == "LBNHC"

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "BLHNC")
    with pytest.raises(ValueError, match="VLLM_KV_CACHE_LAYOUT=BLHNC"):
        resolve_kv_cache_layout(VllmConfig(), supported)
