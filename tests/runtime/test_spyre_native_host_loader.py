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

"""Native cache source loading stays on CPU without changing vLLM's iterators."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fastsafetensors")


@pytest.fixture(autouse=True)
def _upstream_iterators_are_unchanged():
    from vllm.model_executor.model_loader import default_loader, weight_utils

    original = default_loader.fastsafetensors_weights_iterator
    original_utils = weight_utils.fastsafetensors_weights_iterator
    assert original.__module__ == weight_utils.__name__
    yield
    assert default_loader.fastsafetensors_weights_iterator is original
    assert weight_utils.fastsafetensors_weights_iterator is original_utils


def test_iterator_loads_onto_cpu(tmp_path):
    from safetensors.torch import save_file

    from spyre_inference.v1.worker import model_build

    tensors = {
        "a.weight": torch.randn(64, 32).half(),
        "b.bias": torch.arange(7, dtype=torch.int64),
    }
    files = []
    for i, (name, t) in enumerate(tensors.items()):
        files.append(str(tmp_path / f"model-{i}.safetensors"))
        save_file({name: t}, files[-1])
    got = dict(model_build._host_fastsafetensors_weights_iterator(files, use_tqdm_on_load=False))

    assert got.keys() == tensors.keys()
    for name, t in got.items():
        assert t.device.type == "cpu", name
        assert torch.equal(t, tensors[name]), name


def _load_on_rank(rank, init_file, files, expected):
    import torch.distributed as dist

    from spyre_inference import envs
    from spyre_inference.v1.worker import model_build

    envs.clear_env_cache()
    all_local = envs.SPYRE_NATIVE_HOST_ALL_LOCAL
    broadcasts = []
    broadcast = dist.broadcast

    def record_broadcast(*args, **kwargs):
        assert not all_local, "weights must be read locally, not broadcast"
        broadcasts.append(1)
        return broadcast(*args, **kwargs)

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        dist.broadcast = record_broadcast
        got = dict(
            model_build._host_fastsafetensors_weights_iterator(files, use_tqdm_on_load=False)
        )
        assert got.keys() == expected.keys()
        for name, t in got.items():
            assert t.device.type == "cpu", name
            assert torch.equal(t, expected[name]), name
        assert bool(broadcasts) is not all_local
    finally:
        dist.broadcast = broadcast
        dist.destroy_process_group()


@pytest.mark.parametrize("all_local_env", [None, "0", "1"])
def test_iterator_distributed_loading_modes(tmp_path, monkeypatch, all_local_env):
    from safetensors.torch import save_file

    if all_local_env is None:
        monkeypatch.delenv("SPYRE_NATIVE_HOST_ALL_LOCAL", raising=False)
    else:
        monkeypatch.setenv("SPYRE_NATIVE_HOST_ALL_LOCAL", all_local_env)
    tensors = {f"layer{i}.weight": torch.randn(16, 8).half() for i in range(4)}
    files = []
    for i, (name, t) in enumerate(tensors.items()):
        files.append(str(tmp_path / f"model-{i}.safetensors"))
        save_file({name: t}, files[-1])

    torch.multiprocessing.spawn(
        _load_on_rank, args=(str(tmp_path / "init"), files, tensors), nprocs=2
    )


def test_plugin_leaves_upstream_iterators_unchanged():
    from spyre_inference import register_ops

    register_ops()


def test_explicit_fastsafetensors_still_uses_upstream_loader(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader import default_loader, get_model_loader

    from spyre_inference.v1.worker import model_build

    expected = torch.ones(2)
    path = str(tmp_path / "model.safetensors")
    save_file({"w": expected}, path)
    calls = []

    def upstream(files, use_tqdm):
        calls.append(files)
        return iter([("w", expected)])

    def unexpected(*args, **kwargs):
        pytest.fail("explicit fastsafetensors must not use the private CPU iterator")

    with monkeypatch.context() as patch:
        patch.setattr(default_loader, "fastsafetensors_weights_iterator", upstream)
        patch.setattr(model_build, "_host_fastsafetensors_weights_iterator", unexpected)
        loader = get_model_loader(LoadConfig(load_format="fastsafetensors"))
        assert type(loader) is default_loader.DefaultModelLoader
        source = default_loader.DefaultModelLoader.Source(str(tmp_path), None)
        got = dict(loader._get_weights_iterator(source))
    assert got["w"] is expected
    assert calls == [[path]]


@pytest.mark.parametrize("stop", ["exhaust", "close", "error"])
@pytest.mark.parametrize(
    "all_local_env, expected_all_local", [(None, False), ("0", False), ("1", True)]
)
def test_iterator_closes_loader(monkeypatch, stop, all_local_env, expected_all_local):
    from fastsafetensors import parallel_loader

    from spyre_inference.v1.worker import model_build

    if all_local_env is None:
        monkeypatch.delenv("SPYRE_NATIVE_HOST_ALL_LOCAL", raising=False)
    else:
        monkeypatch.setenv("SPYRE_NATIVE_HOST_ALL_LOCAL", all_local_env)
    calls = []

    class Loader:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def iterate_weights(self):
            yield "w", torch.zeros(1)
            if stop == "error":
                raise RuntimeError("read failed")

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(parallel_loader, "ParallelLoader", Loader)
    iterator = model_build._host_fastsafetensors_weights_iterator(
        ["model-10.safetensors", "model-2.safetensors"], False
    )
    next(iterator)
    if stop == "close":
        iterator.close()
    elif stop == "error":
        with pytest.raises(RuntimeError, match="read failed"):
            list(iterator)
    else:
        assert list(iterator) == []
    assert calls[0]["device"] == "cpu"
    assert calls[0]["nogds"]
    assert calls[0]["all_local"] is expected_all_local
    assert calls[0]["hf_weights_files"] == ["model-2.safetensors", "model-10.safetensors"]
    assert calls[1:] == ["closed"]


def test_host_loader_preserves_sources_and_unused_shard_filter(tmp_path):
    from safetensors.torch import save_file
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    from spyre_inference.v1.worker.model_build import _HostModelLoader

    weights = {"weight": torch.arange(6.0).reshape(2, 3)}
    save_file(weights, str(tmp_path / "model-1.safetensors"))
    (tmp_path / "model-2.safetensors").write_bytes(b"must not be read")
    loader = _HostModelLoader(LoadConfig(load_format="fastsafetensors", use_tqdm_on_load=False))
    source = DefaultModelLoader.Source(
        str(tmp_path.parent),
        None,
        subfolder=tmp_path.name,
        prefix="decoder.",
        allow_patterns_overrides=["model-1.safetensors"],
    )
    got = dict(loader._get_weights_iterator(source))
    assert list(got) == ["decoder.weight"]
    torch.testing.assert_close(got["decoder.weight"], weights["weight"])
    assert loader.counter_before_loading_weights > 0

    (tmp_path / "model-2.safetensors").unlink()
    save_file({"unused": torch.zeros(1)}, str(tmp_path / "unused.safetensors"))
    source.allow_patterns_overrides = None
    source.is_unused_weight = lambda name: name == "unused"
    got = dict(loader._get_weights_iterator(source))
    assert list(got) == ["decoder.weight"]


def test_host_loader_preserves_mm_shard_filter(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    from spyre_inference.v1.worker import model_build

    path = str(tmp_path / "model.safetensors")
    save_file({"vision.weight": torch.ones(2)}, path)
    loader = model_build._HostModelLoader(LoadConfig(load_format="fastsafetensors"))
    loader._encoder_only_lm_prefixes = ("language_model.",)
    source = DefaultModelLoader.Source(str(tmp_path), None)
    calls = []

    def filter_files(files, *args, **kwargs):
        calls.append(files)
        return []

    monkeypatch.setattr(model_build, "filter_mm_encoder_only_safetensors_files", filter_files)
    with pytest.raises(RuntimeError, match="mm-encoder-only shard filter removed all"):
        list(loader._get_weights_iterator(source))
    assert calls == [[path]]


def test_host_loader_loads_primary_and_secondary_weights(tmp_path):
    from safetensors.torch import save_file
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    from spyre_inference.v1.worker.model_build import _HostModelLoader

    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    primary.mkdir()
    secondary.mkdir()
    save_file({"weight": torch.arange(6.0).reshape(2, 3)}, str(primary / "model.safetensors"))
    save_file({"bias": torch.arange(2.0)}, str(secondary / "model.safetensors"))

    class Model(torch.nn.Linear):
        secondary_weights = [DefaultModelLoader.Source(str(secondary), None)]

        def load_weights(self, weights):
            loaded = set()
            for name, tensor in weights:
                self.get_parameter(name).data.copy_(tensor)
                loaded.add(name)
            return loaded

    model = Model(3, 2, dtype=torch.float16)
    config = SimpleNamespace(
        model=str(primary), revision=None, quantization=None, is_moe=False, multimodal_config=None
    )
    loader = _HostModelLoader(LoadConfig(load_format="fastsafetensors", use_tqdm_on_load=False))
    loader.load_weights(model, config)
    torch.testing.assert_close(model.weight, torch.arange(6.0).reshape(2, 3).half())
    torch.testing.assert_close(model.bias, torch.arange(2.0).half())
    assert loader.counter_after_loading_weights >= loader.counter_before_loading_weights > 0


def test_native_build_uses_private_host_loader_and_restores_config(monkeypatch):
    from vllm.config import LoadConfig

    from spyre_inference.v1.worker import model_build

    original = LoadConfig(load_format="spyre_native")
    host_config = LoadConfig(load_format="fastsafetensors", device="cpu")
    config = SimpleNamespace(load_config=original)
    model = torch.nn.Module()
    loaders = []
    monkeypatch.setattr(model_build, "install_pad_loaders", lambda *args: None)
    monkeypatch.setattr(model_build, "fix_padded_model", lambda *args: None)
    monkeypatch.setattr(model_build.Attention, "_apply", model_build.Attention._apply)

    def load(self, vllm_config, model_config):
        assert vllm_config.load_config is host_config
        loaders.append(self)
        return model

    monkeypatch.setattr(model_build._HostModelLoader, "load_model", load)
    assert model_build.build_model(host_config, config, None) is model
    assert type(loaders[0]) is model_build._HostModelLoader
    assert config.load_config is original

    def fail(*args, **kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(model_build._HostModelLoader, "load_model", fail)
    with pytest.raises(RuntimeError, match="load failed"):
        model_build.build_model(host_config, config, None)
    assert config.load_config is original


def test_meta_build_does_not_read_source_weights(monkeypatch):
    from vllm.config import LoadConfig

    from spyre_inference.v1.worker import model_build

    original = LoadConfig(load_format="spyre_native")
    meta_config = LoadConfig(load_format="fastsafetensors", device="meta")
    config = SimpleNamespace(load_config=original)
    model_config = SimpleNamespace(dtype=torch.float16)
    monkeypatch.setattr(model_build, "install_pad_loaders", lambda *args: None)
    monkeypatch.setattr(model_build, "fix_padded_model", lambda *args: None)
    monkeypatch.setattr(model_build, "process_weights_after_loading", lambda *args: None)
    monkeypatch.setattr(model_build.Attention, "_apply", model_build.Attention._apply)
    monkeypatch.setattr(model_build, "initialize_model", lambda **kwargs: torch.nn.Linear(3, 2))

    def unexpected(*args, **kwargs):
        pytest.fail("meta build must not read source weights")

    monkeypatch.setattr(model_build, "_host_fastsafetensors_weights_iterator", unexpected)
    model = model_build.build_model(meta_config, config, model_config)
    assert model.weight.device.type == "meta"
    assert config.load_config is original


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_symbolic_host_copies_are_scoped_to_skeleton(dtype):
    from spyre_inference.v1.worker.model_build import _MetaHostCopies

    source = torch.arange(4.0)
    meta = torch.empty(4, device="meta")
    original_to = torch.Tensor.to
    with _MetaHostCopies():
        assert meta.to("cpu", dtype).is_meta
        assert meta.to(device="cpu", dtype=dtype).is_meta
        assert meta.cpu().is_meta
        torch.testing.assert_close(source.to("cpu", dtype), source.to(dtype))
    assert torch.Tensor.to is original_to
    with pytest.raises(NotImplementedError, match="meta"):
        meta.to("cpu")


def test_skeleton_constructs_transformers_rope_without_model_changes(monkeypatch):
    from vllm.config import LoadConfig

    from spyre_inference.transformers_backend import _SpyreRotaryEmbedding
    from spyre_inference.v1.worker import model_build

    def initialize(**kwargs):
        original = torch.nn.Module()
        original.register_buffer("inv_freq", torch.arange(4.0))
        return _SpyreRotaryEmbedding(original, 8, 64, torch.float16)

    monkeypatch.setattr(model_build, "initialize_model", initialize)
    monkeypatch.setattr(model_build, "process_weights_after_loading", lambda *args: None)
    loader = model_build.SkeletonLoader(LoadConfig(device="meta"))
    model = loader.load_model(SimpleNamespace(), SimpleNamespace(dtype=torch.float16))
    assert model._cpu_caches[None].is_meta
    assert not hasattr(model, "spyre_native_host_tensors")


def test_recorded_move_keeps_parameter_identity():
    from spyre_inference.v1.worker import model_load

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    impl = SimpleNamespace(_sinks_source=model.weight)
    calls = []

    def transfer(tensor):
        calls.append("transfer")
        return tensor + 1

    model_load.move_model_to_spyre(model, torch.device("cpu"), transfer=transfer)
    assert impl._sinks_source is model.weight
    torch.testing.assert_close(model.weight, torch.full((2,), 2.0))
    assert calls == ["transfer"]


def test_pt_fallback_still_uses_upstream_loader(tmp_path):
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    from spyre_inference.v1.worker.spyre_native_model_loader import SpyreNativeModelLoader

    expected = torch.arange(4.0)
    torch.save({"w": expected}, tmp_path / "pytorch_model.bin")
    native = SpyreNativeModelLoader(LoadConfig(load_format="spyre_native"))
    config = native._host_config(SimpleNamespace(model=str(tmp_path), revision=None))
    loader = get_model_loader(config)
    assert config.load_format == "auto" and config.device == "cpu"
    assert type(loader) is DefaultModelLoader
    got = dict(loader._get_weights_iterator(DefaultModelLoader.Source(str(tmp_path), None)))
    torch.testing.assert_close(got["w"], expected)


@pytest.mark.parametrize("mode", ["miss", "hit", "unusable", "unwritable"])
def test_native_cache_control_flow(tmp_path, monkeypatch, mode):
    from fastsafetensors.copier import spyre as transfer
    from safetensors.torch import save_file
    from vllm.config import LoadConfig

    from spyre_inference.v1.worker import spyre_native_cache as cache
    from spyre_inference.v1.worker import spyre_native_model_loader as native

    save_file({"w": torch.ones(2)}, str(tmp_path / "model.safetensors"))
    original = LoadConfig(load_format="spyre_native")
    config = SimpleNamespace(
        load_config=original,
        compilation_config=SimpleNamespace(static_forward_context={}, static_all_moe_layers=[]),
    )
    model_config = SimpleNamespace(model=str(tmp_path), revision=None)
    loader = native.SpyreNativeModelLoader(original)
    monkeypatch.setattr(native, "_cache_key", lambda *args: {"rank": 0})
    monkeypatch.setattr(cache, "cache_exists", lambda base: mode in ("hit", "unusable"))
    calls = []
    builds = []

    def build(load_config, *args):
        calls.append(("build", load_config.device))
        assert load_config.load_format == "fastsafetensors"
        model = torch.nn.Module()
        model.register_buffer("weight", torch.ones(2, dtype=torch.float16))
        builds.append(model)
        return model

    def write(model, *args):
        assert model is builds[-1]
        calls.append(("write",))
        return mode != "unwritable"

    def restore(model, *args):
        assert model is builds[-1]
        calls.append(("restore",))
        return mode != "unusable"

    class Recorder:
        def __init__(self, device):
            assert device == torch.device("cpu")

        def __call__(self, tensor):
            assert tensor is builds[-1].weight
            calls.append(("transfer",))
            return tensor

        def fill(self):
            calls.append(("fill",))
            return SimpleNamespace(summary=lambda: "backend transfer")

    monkeypatch.setattr(native, "build_model", build)
    monkeypatch.setattr(cache, "write_cache", write)
    monkeypatch.setattr(cache, "restore_cache", restore)
    monkeypatch.setattr(transfer, "TransferRecorder", Recorder)
    model = loader.load_model(config, model_config, device=torch.device("cpu"))
    assert model is builds[-1]
    expected = {
        "miss": [("build", "cpu"), ("write",), ("build", "meta"), ("restore",)],
        "hit": [("build", "meta"), ("restore",)],
        "unusable": [("build", "meta"), ("restore",), ("build", "cpu"), ("transfer",), ("fill",)],
        "unwritable": [("build", "cpu"), ("write",)],
    }[mode]
    assert calls == expected
    assert config.load_config is original
