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

"""Device-free tests for RoBERTa position offset and model-wrapper I/O."""

from __future__ import annotations

import types

import torch
import torch.nn as nn
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from spyre_inference.models.roberta import offset_roberta_position_ids
from spyre_inference.v1.worker import spyre_model_runner as mr


def test_offset_roberta_position_ids_runs_on_cpu(monkeypatch):
    seen_devices: list[object] = []

    def fake_convert(t, device=None, dtype=None):
        seen_devices.append(device)
        out = t
        if dtype is not None:
            out = out.to(dtype)
        if device is not None:
            # Stay on CPU in this unit test; only record the requested device.
            pass
        return out

    monkeypatch.setattr("spyre_inference.models.roberta.convert", fake_convert)

    pos = torch.tensor([0, 1, 2], dtype=torch.int64)
    out = offset_roberta_position_ids(pos, 1)
    # Round trips through the host, and lands back on the input's own device.
    assert seen_devices[0] == "cpu"
    assert seen_devices[1] == pos.device
    assert out.dtype == torch.int64
    torch.testing.assert_close(out, torch.tensor([2, 3, 4], dtype=torch.int64))


def test_offset_roberta_position_ids_is_an_opaque_op():
    """Whole-model compile needs the host round trip hidden behind one node: inlined,
    the CPU intermediate makes Inductor emit spyre::to_dtype_cpu, which has no CPU
    registration."""
    assert hasattr(torch.ops.spyre_inference, "roberta_offset_positions")
    pos = torch.tensor([0, 1, 2], dtype=torch.int64)
    torch.testing.assert_close(
        torch.ops.spyre_inference.roberta_offset_positions(pos, 1),
        torch.tensor([2, 3, 4], dtype=torch.int64),
    )


def test_wrapper_converts_ints_to_int64(monkeypatch):
    seen: list[torch.dtype | None] = []

    def fake_convert(t, device=None, dtype=None):
        seen.append(dtype)
        return t if dtype is None else t.to(dtype)

    monkeypatch.setattr(mr, "convert", fake_convert)

    class _Capture(nn.Module):
        def forward(self, input_ids=None, positions=None, **kwargs):
            return {"input_ids": input_ids, "positions": positions}

    wrapper = mr._SpyreModelWrapper(
        _Capture(),
        torch.device("cpu"),
        keep_outputs_on_device=True,
        model_dtype=torch.float16,
    )
    out = wrapper(
        input_ids=torch.tensor([1, 2], dtype=torch.int32),
        positions=torch.tensor([0, 1], dtype=torch.int32),
    )
    assert seen == [torch.int64, torch.int64]
    assert out["input_ids"].dtype == torch.int64
    assert out["positions"].dtype == torch.int64


def test_kv_sharing_attention_lookup_preserves_physical_specs(monkeypatch):
    owner = "layers.0.self_attn"
    sharing = "layers.1.self_attn"
    layer_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.float16,
    )
    uniform_spec = UniformTypeKVCacheSpecs.from_specs({owner: layer_spec})
    assert uniform_spec is not None
    group = KVCacheGroupSpec(layer_names=[owner, sharing], kv_cache_spec=uniform_spec)
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[KVCacheTensor(size=2 * layer_spec.page_size_bytes, shared_by=[owner])],
        kv_cache_groups=[group],
    )
    runner = mr.TorchSpyreModelRunner.__new__(mr.TorchSpyreModelRunner)
    runner.shared_kv_cache_layers = {sharing: owner}
    # `initialize_attn_backend` also splits groups by per-layer window after super();
    # no groups here makes that a no-op, leaving the spec resolution under test.
    runner.attn_groups = []
    runner.vllm_config = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(static_forward_context={})
    )
    seen = []

    def capture(_self, attn_config, is_profiling=False):
        seen.append(attn_config)

    monkeypatch.setattr(mr.GPUModelRunner, "initialize_attn_backend", capture)
    original_page_size = uniform_spec.page_size_bytes

    runner.initialize_attn_backend(config)

    resolved = seen[0].kv_cache_groups[0].kv_cache_spec
    assert resolved.kv_cache_specs[sharing] is layer_spec
    assert config.kv_cache_groups[0] is group
    assert uniform_spec.kv_cache_specs == {owner: layer_spec}
    assert uniform_spec.page_size_bytes == original_page_size
    assert config.kv_cache_tensors[0].shared_by == [owner]


def test_no_kv_sharing_passes_the_config_through_untouched(monkeypatch):
    """Without KV sharing the resolution step is skipped, not merely a no-op."""
    layer_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.float16,
    )
    owner = "layers.0.self_attn"
    uniform_spec = UniformTypeKVCacheSpecs.from_specs({owner: layer_spec})
    assert uniform_spec is not None
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[KVCacheTensor(size=2 * layer_spec.page_size_bytes, shared_by=[owner])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[owner], kv_cache_spec=uniform_spec)],
    )
    runner = mr.TorchSpyreModelRunner.__new__(mr.TorchSpyreModelRunner)
    runner.shared_kv_cache_layers = {}
    runner.attn_groups = []
    runner.vllm_config = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(static_forward_context={})
    )
    seen = []
    monkeypatch.setattr(
        mr.GPUModelRunner,
        "initialize_attn_backend",
        lambda _self, attn_config, is_profiling=False: seen.append(attn_config),
    )

    runner.initialize_attn_backend(config)

    assert seen[0] is config, "no KV sharing should hand super() the caller's own config"


def test_wrapper_casts_multimodal_floats_to_the_model_dtype(monkeypatch):
    """A bfloat16 checkpoint's pixel values must not be narrowed to float16 going in."""
    seen: list[torch.dtype | None] = []

    def fake_convert(t, device=None, dtype=None):
        seen.append(dtype)
        return t if dtype is None else t.to(dtype)

    monkeypatch.setattr(mr, "convert", fake_convert)

    class _Vision(nn.Module):
        def embed_multimodal(self, pixel_values=None, **kwargs):
            return pixel_values

    wrapper = mr._SpyreModelWrapper(_Vision(), torch.device("cpu"), model_dtype=torch.bfloat16)
    out = wrapper.embed_multimodal(pixel_values=torch.zeros(2, 3, dtype=torch.float32))

    assert seen == [torch.bfloat16]
    assert out.dtype == torch.bfloat16


def test_setattr_keeps_the_wrappers_own_state_off_the_model():
    """``__setattr__`` forwards to the wrapped model, so a write to one of the wrapper's
    own fields (``_model_dtype``) would land on the wrong object."""

    class _Plain(nn.Module):
        pass

    model = _Plain()
    wrapper = mr._SpyreModelWrapper(model, torch.device("cpu"), model_dtype=torch.float16)

    wrapper._model_dtype = torch.bfloat16
    assert wrapper._model_dtype == torch.bfloat16
    assert not hasattr(model, "_model_dtype")

    wrapper.some_model_flag = 7
    assert model.some_model_flag == 7
