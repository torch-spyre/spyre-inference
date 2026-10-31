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

"""kv_cache.init: upstream's initialize_kv_cache with Spyre attention groups and pages."""

import os
import subprocess
import sys
import tempfile
import textwrap
import types

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu import model_runner as upstream_model_runner

from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl, SpyrePagedKVCache
from spyre_inference.v1.attention.backends.spyre_head_major_attn import (
    SpyreHeadMajorAttentionImpl,
)
from spyre_inference.v1.worker.spyre import attn_groups, kv_cache, triton_gate

from .fakes import make_fake_kv_cache_config

_NUM_LAYERS, _NUM_BLOCKS, _BLOCK, _HEAD = 2, 4, 64, 64


class _Layer:
    """A forward-context entry: an impl to pick the layout, and the layer-side bind."""

    bind_kv_cache = AttentionLayerBase.bind_kv_cache

    def __init__(self, impl_cls: type) -> None:
        self.impl = impl_cls.__new__(impl_cls)
        self.kv_cache: object = None
        self.kv_sharing_target_layer_name = None


def _runner(spyre_device: torch.device) -> types.SimpleNamespace:
    return types.SimpleNamespace(spyre_device=spyre_device, spyre_kv_caches={}, speculator=None)


def _run_init_kv_cache(monkeypatch, runner, impls, device) -> tuple[dict, dict[str, _Layer]]:
    """Drive the patched init_kv_cache the way upstream's initialize_kv_cache calls it."""
    forward_context = {f"layer.{i}": _Layer(impl) for i, impl in enumerate(impls)}
    monkeypatch.setattr(kv_cache, "get_shared_kv_cache_layers", lambda vllm_config: {})
    with kv_cache._patched(runner):
        returned = upstream_model_runner.init_kv_cache(
            forward_context,
            make_fake_kv_cache_config(_NUM_LAYERS, _NUM_BLOCKS, _BLOCK, 1, _HEAD),
            device,
            [_BLOCK],
            None,
            kv_cache_allocation_context=None,
            block_tables=None,
            attn_groups=(),
        )
    return returned, forward_context


@pytest.fixture
def fake_pages(monkeypatch: pytest.MonkeyPatch):
    """allocate_pages needs a card, so on CPU record its calls and hand back tagged pages."""
    calls: list[tuple[type, int, FullAttentionSpec]] = []

    def allocate(cls, num_blocks, spec, device):
        calls.append((cls, num_blocks, spec))
        return SpyrePagedKVCache(object(), object())

    for impl in (SpyreAttentionImpl, SpyreHeadMajorAttentionImpl):
        monkeypatch.setattr(impl, "allocate_pages", classmethod(allocate))
    return calls


@pytest.mark.parametrize(
    "impls",
    [
        pytest.param([SpyreAttentionImpl] * 2, id="token_major"),
        pytest.param([SpyreHeadMajorAttentionImpl] * 2, id="head_major"),
        pytest.param([SpyreAttentionImpl, SpyreHeadMajorAttentionImpl], id="mixed"),
    ],
)
def test_each_layer_gets_its_own_pages_in_its_backends_layout(
    monkeypatch, fake_pages, impls
) -> None:
    runner = _runner(torch.device("cpu"))

    returned, forward_context = _run_init_kv_cache(monkeypatch, runner, impls, "cpu")

    assert returned == {}, "upstream must get no caches it would filter by .device"
    assert [cls for cls, _, _ in fake_pages] == impls
    for _, num_blocks, spec in fake_pages:
        assert (num_blocks, spec.block_size, spec.head_size) == (_NUM_BLOCKS, _BLOCK, _HEAD)
    assert set(runner.spyre_kv_caches) == set(forward_context)
    assert len({id(c) for c in runner.spyre_kv_caches.values()}) == _NUM_LAYERS
    for name, layer in forward_context.items():
        assert layer.kv_cache is runner.spyre_kv_caches[name]


def test_a_layer_outside_the_forward_context_is_allocated_but_not_bound(
    monkeypatch, fake_pages
) -> None:
    """As upstream: e.g. a KV-sharing target or draft layer absent from the context."""
    runner = _runner(torch.device("cpu"))
    forward_context = {"layer.0": _Layer(SpyreAttentionImpl)}
    monkeypatch.setattr(kv_cache, "get_shared_kv_cache_layers", lambda vllm_config: {})
    with kv_cache._patched(runner):
        upstream_model_runner.init_kv_cache(
            forward_context,
            make_fake_kv_cache_config(_NUM_LAYERS, _NUM_BLOCKS, _BLOCK, 1, _HEAD),
            "cpu",
            [_BLOCK],
            None,
        )

    assert set(runner.spyre_kv_caches) == {"layer.0", "layer.1"}
    assert forward_context["layer.0"].kv_cache is runner.spyre_kv_caches["layer.0"]


def test_init_raises_when_the_swap_did_not_apply(monkeypatch) -> None:
    """A vLLM that allocates through another module must not fall back to CPU tensors."""
    monkeypatch.setattr(
        upstream_model_runner.GPUModelRunner, "initialize_kv_cache", lambda self, c, **k: None
    )
    config = make_fake_kv_cache_config(_NUM_LAYERS, _NUM_BLOCKS, _BLOCK, 1, _HEAD)

    with pytest.raises(RuntimeError, match="allocated no Spyre KV pages"):
        kv_cache.init(_runner(torch.device("cpu")), config)


def test_speculative_decoding_is_refused(monkeypatch) -> None:
    """The speculator calls init_attn_backend through its own import, past the swap."""
    called = []
    monkeypatch.setattr(
        upstream_model_runner.GPUModelRunner,
        "initialize_kv_cache",
        lambda self, c, **k: called.append(True),
    )
    runner = _runner(torch.device("cpu"))
    runner.speculator = object()

    with pytest.raises(NotImplementedError, match="Speculative decoding"):
        kv_cache.init(runner, make_fake_kv_cache_config())
    assert called == []


@pytest.mark.parametrize("is_profiling", [False, True])
def test_a_kv_connector_is_refused_outside_profiling(monkeypatch, is_profiling) -> None:
    """The connector would register the empty dict upstream gets back."""
    monkeypatch.setattr(kv_cache, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(
        upstream_model_runner.GPUModelRunner, "initialize_kv_cache", lambda self, c, **k: None
    )
    runner = _runner(torch.device("cpu"))
    empty = KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=[])

    if is_profiling:
        kv_cache.init(runner, empty, is_profiling=True)
    else:
        with pytest.raises(NotImplementedError, match="KV connectors"):
            kv_cache.init(runner, empty, is_profiling=False)


def test_patches_are_scoped_to_the_call(monkeypatch) -> None:
    before = (upstream_model_runner.init_attn_backend, upstream_model_runner.init_kv_cache)
    seen = []

    def initialize_kv_cache(self, kv_cache_config, **kwargs):
        seen.append(upstream_model_runner.init_attn_backend)
        raise RuntimeError("upstream failed")

    monkeypatch.setattr(
        upstream_model_runner.GPUModelRunner, "initialize_kv_cache", initialize_kv_cache
    )
    with pytest.raises(RuntimeError, match="upstream failed"):
        kv_cache.init(_runner(torch.device("cpu")), None)

    assert seen and seen[0] is not before[0], "init_attn_backend was not swapped in"
    assert (upstream_model_runner.init_attn_backend, upstream_model_runner.init_kv_cache) == before


def test_init_attn_backend_routes_through_the_window_split(monkeypatch) -> None:
    """attn_groups.init gets upstream's own init_attn_backend to wrap."""
    upstream = upstream_model_runner.init_attn_backend
    called = []
    monkeypatch.setattr(
        attn_groups, "init", lambda wrapped, *a, **k: called.append(wrapped) or ([], None, [])
    )
    with kv_cache._patched(_runner(torch.device("cpu"))):
        upstream_model_runner.init_attn_backend(None, None, None)
    assert called == [upstream]


_UPSTREAM_ATTRIBUTES = (
    "kv_cache_config",
    "attn_groups",
    "kernel_block_sizes",
    "block_tables",
    "pcp_manager",
    "ubatch_runner",
    "cudagraph_manager",
    "kv_caches",
    "kv_connector",
)


@pytest.mark.subprocess_isolated
@pytest.mark.skipif(
    not hasattr(triton_gate, "check"),
    reason="a V2 VllmConfig needs Triton until the Triton gate lets it past (#1354)",
)
def test_initialize_kv_cache_sets_every_upstream_attribute() -> None:
    """A real SpyreModelRunnerV2, in a fresh process for its distributed groups and
    device shims. A layer-free config keeps it off the card."""
    script = textwrap.dedent(f"""
        import torch
        from vllm.config import CacheConfig, ModelConfig, VllmConfig, set_current_vllm_config
        from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
        from vllm.v1.kv_cache_interface import KVCacheConfig
        from vllm.v1.worker.gpu.model_states.interface import ModelState

        from spyre_inference.v1.worker.spyre import device_shims
        from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2

        class State(ModelState):
            num_new_sampled_tokens_per_step = 1
            prepare_attn = prepare_dummy_inputs = prepare_inputs = prepare_inputs_embeds = None

            def __init__(self):
                pass

        device_shims.install()
        config = VllmConfig(
            model_config=ModelConfig(model="Qwen/Qwen3-0.6B", max_model_len=256),
            cache_config=CacheConfig(block_size=64),
        )
        with set_current_vllm_config(config):
            init_distributed_environment(1, 0, "file://{{store}}", 0, "gloo")
            ensure_model_parallel_initialized(1, 1)
            runner = SpyreModelRunnerV2(config, torch.device("cpu"))
            # What load_model, still a #1242 stub, would set.
            runner.model, runner.model_state, runner.decode_query_len = (
                torch.nn.Module(), State(), 1
            )
            # __init__ pre-sets some of these; drop them so each must come from the call.
            for name in {_UPSTREAM_ATTRIBUTES!r}:
                runner.__dict__.pop(name, None)
            empty = KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=[])
            runner.initialize_kv_cache(empty)
            print([n for n in {_UPSTREAM_ATTRIBUTES!r} if not hasattr(runner, n)])
    """)
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "-c", script.replace("{store}", os.path.join(tmp, "store"))],
            capture_output=True,
            text=True,
            env={**os.environ, "SPYRE_MODEL_RUNNER": "v2"},
            timeout=300,
        )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.split("\n")[-2] == "[]", result.stdout


@pytest.mark.spyre_device
@pytest.mark.parametrize(
    "impl",
    [
        pytest.param(SpyreAttentionImpl, id="token_major"),
        pytest.param(SpyreHeadMajorAttentionImpl, id="head_major"),
    ],
)
def test_pages_allocate_on_spyre_in_both_layouts(monkeypatch, impl) -> None:
    if not spyre_available():
        pytest.skip("Spyre device not available")
    runner = _runner(torch.device("spyre"))

    _, forward_context = _run_init_kv_cache(monkeypatch, runner, [impl] * 2, "cpu")

    want = (
        (_NUM_BLOCKS, 1, _BLOCK, _HEAD)
        if impl is SpyreHeadMajorAttentionImpl
        else (_NUM_BLOCKS, _BLOCK, 1, _HEAD)
    )
    for name, pages in runner.spyre_kv_caches.items():
        assert pages.k_pages.device.type == "spyre"
        assert tuple(pages.k_pages.shape) == tuple(pages.v_pages.shape) == want
        assert forward_context[name].kv_cache is pages
