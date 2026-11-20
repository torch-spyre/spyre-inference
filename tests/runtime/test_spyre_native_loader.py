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

"""Native model traversal and cache integration with the Spyre transfer backend."""

import copy

import pytest
import torch
import torch.nn as nn
from fastsafetensors.copier import spyre as _transfer_pipeline
from fastsafetensors.copier.spyre import TransferRecorder
from spyre_testing_plugin.pytest_plugin import spyre_available


class _Model(nn.Module):
    def __init__(self):
        super().__init__()

        def p(t):
            return nn.Parameter(t, requires_grad=False)

        self.linear = p(torch.randn(1024, 5376, dtype=torch.float16))
        self.norm = p(torch.randn(5376, dtype=torch.float16))
        self.unaligned = p(torch.randn(37, 100, dtype=torch.float16))
        self.transposed = p(torch.randn(256, 128, dtype=torch.float16).t())
        self.rank3 = p(torch.randn(4, 64, 192, dtype=torch.float16))
        self.sliced = p(torch.randn(64, 256, dtype=torch.float16)[:, :128])
        self.register_buffer("fp32", torch.randn(64, 64))
        self.register_buffer("int32", torch.arange(128, dtype=torch.int32))
        # Reachable twice: _apply visits it again after it has moved.
        self.shared = nn.Linear(64, 64, dtype=torch.float16).requires_grad_(False)
        self.alias = self.shared
        self.table = p(torch.randn(512, 128, dtype=torch.float16))

    def _apply(self, fn, recurse=True):
        # Like SpyreVocabParallelEmbedding: a gather table placed with its own layout.
        from spyre_inference.custom_ops.utils import place_row_gathered

        table = self._parameters.pop("table")
        super()._apply(fn, recurse)
        self._parameters["table"] = nn.Parameter(
            place_row_gathered(table.data, fn, "table"), requires_grad=False
        )
        return self


# Small limits force multiple batches and single-tensor oversized batches.
@pytest.mark.parametrize(
    "max_buffers, max_batch_bytes",
    [(1, 1 << 30), (3, 1 << 30), (64, 1 << 30), (4, 100_000)],
)
def test_matches_model_to(monkeypatch, max_buffers, max_batch_bytes):
    if not spyre_available():
        pytest.skip("Spyre device not available")
    monkeypatch.setattr(_transfer_pipeline, "_MAX_BUFFERS", max_buffers)
    monkeypatch.setattr(_transfer_pipeline, "_MAX_BATCH_BYTES", max_batch_bytes)

    device = torch.device("spyre")
    model = _Model()
    expected = {k: v.cpu() for k, v in copy.deepcopy(model).to(device).state_dict().items()}

    recorder = TransferRecorder(device)
    model._apply(recorder)
    recorder.fill()

    for name, t in model.state_dict().items():
        assert t.device.type == "spyre", name
        torch.testing.assert_close(t.cpu(), expected[name], rtol=0, atol=0, msg=name)


class _KeepsHostState(nn.Module):
    """Like Attention's CPU-pinned scales and RoPE's host caches: never moved."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([0.5, 2.0]))
        self.caches = {"a": torch.arange(6.0).reshape(2, 3)}
        self.width = 3

    def _apply(self, fn, recurse=True):
        return self


def _cache_model():
    model = _Model()
    model.host = _KeepsHostState()
    return model


def _all_tensors(model):
    import json

    tensors = {
        json.dumps([prefix, path]): t for prefix, path, t in _transfer_pipeline._tensor_paths(model)
    }
    return {path: t.detach().cpu() for path, t in tensors.items()}


def test_cache_restores_model_to_exactly(tmp_path):
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from spyre_inference.v1.worker.spyre_native_cache import restore_cache, write_cache

    device = torch.device("spyre")
    torch.manual_seed(0)
    model = _cache_model()
    expected = _all_tensors(copy.deepcopy(model).to(device))
    path = str(tmp_path / "model-rank0-abc")
    key = {"model": "test"}

    other_cache = tmp_path / "model-rank0-other-00001-of-00016.safetensors"
    other_cache.write_bytes(b"x")
    assert write_cache(model, device, path, key)
    del model
    assert sorted(p.name for p in tmp_path.glob("model-rank0-abc-*.safetensors")) == [
        f"model-rank0-abc-{i:05d}-of-00016.safetensors" for i in range(1, 17)
    ]
    assert other_cache.read_bytes() == b"x"

    with torch.device("meta"):
        restored = _cache_model()
    assert restore_cache(restored, device, path, key)

    got = _all_tensors(restored)
    assert got.keys() == expected.keys()
    for name, t in got.items():
        torch.testing.assert_close(t, expected[name], rtol=0, atol=0, msg=name)

    with torch.device("meta"):
        stale = _cache_model()
    assert not restore_cache(stale, device, path, {"model": "other"})

    with torch.device("meta"):
        changed = _cache_model()
    changed.host.width = 4
    assert restore_cache(changed, device, path, key)
    assert changed.host.width == 4


def test_spyre_native_is_a_registered_load_format():
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader import get_model_loader

    from spyre_inference.v1.worker.spyre_native_model_loader import SpyreNativeModelLoader

    loader = get_model_loader(LoadConfig(load_format="spyre_native"))
    assert isinstance(loader, SpyreNativeModelLoader)
    with pytest.raises(NotImplementedError, match="whole models"):
        loader.load_weights(nn.Module(), None)


def test_cache_miss_loads_safetensors_with_fastsafetensors(tmp_path):
    from types import SimpleNamespace

    from safetensors.torch import save_file
    from vllm.config import LoadConfig

    from spyre_inference.v1.worker.spyre_native_model_loader import SpyreNativeModelLoader

    loader = SpyreNativeModelLoader(LoadConfig(load_format="spyre_native"))
    st_dir, bin_dir = tmp_path / "st", tmp_path / "bin"
    st_dir.mkdir()
    bin_dir.mkdir()
    save_file({"w": torch.zeros(2)}, str(st_dir / "model.safetensors"))
    torch.save({"w": torch.zeros(2)}, str(bin_dir / "pytorch_model.bin"))

    def fmt(path):
        config = loader._host_config(SimpleNamespace(model=str(path), revision=None))
        assert config.device == "cpu"
        assert loader.load_config.load_format == "spyre_native"
        assert loader.load_config.device is None
        return config.load_format

    assert fmt(st_dir) == "fastsafetensors"
    assert fmt(bin_dir) == "auto"
