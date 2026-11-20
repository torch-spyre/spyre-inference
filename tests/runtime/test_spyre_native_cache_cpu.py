# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""Cache metadata and ordinary tensor restoration, without Spyre hardware."""

import json
import sys
from dataclasses import dataclass
from types import ModuleType

import pytest
import torch
from fastsafetensors.copier import spyre as transfer
from fastsafetensors.copier.spyre import KEY_HOST, KEY_TRANSFERS, native_shard_metadata
from fastsafetensors.saver import ParallelSaver, save_sharded
from safetensors import safe_open
from safetensors.torch import save_file

from spyre_inference.v1.worker import spyre_native_cache as cache


def test_write_cache_uses_backend_native_writer(tmp_path, monkeypatch):
    from fastsafetensors.copier import spyre
    from fastsafetensors.copier.spyre import TransferRecord
    from fastsafetensors.native import NativeFormat, native_format, native_specs

    runtime = ModuleType("torch_spyre._C")
    calls = []

    def encode(work, num_threads):
        calls.append(num_threads)
        for buf, src, dst in work:
            assert len(buf) == dst.nbytes
            buf[:] = src.reshape(-1).view(torch.uint8).numpy().tobytes()

    runtime.convert_h2d_into = encode
    runtime.dma_bytes = lambda dst: dst.nbytes
    runtime.get_spyre_tensor_layout = lambda dst: {"test_layout": 1}
    layouts = ModuleType("torch_spyre.layout")
    layouts.layout_to_json = lambda layout: layout
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    monkeypatch.setitem(sys.modules, "torch_spyre.layout", layouts)

    class Recorder(spyre.TransferRecorder):
        def __call__(self, src):
            dst = torch.empty_like(src)
            self.records.append(TransferRecord("defer", src, dst))
            return dst

    def saver(**kwargs):
        assert kwargs["fill"] is spyre.native_fill
        assert kwargs["shard_metadata"] is spyre.native_shard_metadata
        return ParallelSaver(**kwargs)

    monkeypatch.setattr(cache, "TransferRecorder", Recorder)
    monkeypatch.setattr(cache, "ParallelSaver", saver)
    model = torch.nn.Module()
    expected = torch.arange(4, dtype=torch.float16)
    model.register_buffer("weight", expected.clone())
    base = str(tmp_path / "cache")
    key = {"model": "cpu-test", "rank": 0}
    assert cache.write_cache(model, torch.device("cpu"), base, key)
    paths = cache.shard_paths(base, cache._NUM_SHARDS)
    framework = cache.get_framework_op("pytorch")
    specs = {}
    for path in paths:
        meta = cache.SafeTensorsMetadata.from_file(path, framework)
        assert native_format(meta) == NativeFormat("spyre", "torch_spyre/1")
        specs.update(native_specs(meta))
        with safe_open(path, framework="pt") as checkpoint:
            names = checkpoint.keys()
            if "t0" in names:
                assert torch.equal(checkpoint.get_tensor("t0"), expected.view(torch.uint8))
    with safe_open(paths[0], framework="pt") as checkpoint:
        metadata = checkpoint.metadata()
    assert transfer.read_cache_metadata(metadata, key) is None
    assert {name for name in metadata if name.startswith("torch_spyre.cache.")} == {
        transfer.KEY_CACHE_FORMAT,
        transfer.KEY_CACHE_KEY,
        transfer.KEY_FP8,
    }
    assert not any(name.startswith("spyre_inference.") for name in metadata)
    records = json.loads(metadata[KEY_TRANSFERS])
    assert records[0]["kind"] == "raw" and records[0]["name"] == "t0"
    assert specs["t0"].layout == records[0]["layout"] == {"test_layout": 1}
    assert calls


@dataclass(frozen=True)
class _Padding:
    width: int
    pad: torch.Tensor


class _HostState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(0.5), persistent=False)
        self.table = {"empty": torch.empty(0)}
        pad = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float16)
        self.other_pad = pad.clone()
        self.vision_buffers = {"segments": _Padding(3, pad)}
        self.nested = [(_Padding(3, pad), {"offsets": torch.arange(3)})]
        self.strided = _Padding(2, torch.arange(6.0).reshape(2, 3).t())
        self.recursive = {"tensor": torch.tensor(7.0)}
        self.recursive["cycle"] = self.recursive

    def _apply(self, fn, recurse=True):
        return self


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.arange(6.0).reshape(2, 3).t())
        self.host = _HostState()


@pytest.fixture
def checkpoint(tmp_path):
    model = _Model()
    recorder = transfer.TransferRecorder(torch.device("cpu"))
    recorder.capture_host_state(model)
    recorder(model.weight)
    recorder.records[0].kind = "to"
    metadata, entries = recorder.checkpoint_metadata(model)
    base = str(tmp_path / "cache")
    key = {"model": "cpu-test"}
    envelope, shard0 = transfer.cache_metadata(key)
    paths = save_sharded(
        entries,
        base,
        num_shards=cache._NUM_SHARDS,
        shard_metadata=native_shard_metadata,
        metadata=envelope,
        shard0_metadata={**metadata, **shard0},
    )
    return model, base, key, paths


def test_restore_plain_cache_with_standard_readers(checkpoint, monkeypatch):
    model, base, key, _ = checkpoint
    runtime = ModuleType("torch_spyre._C")

    def unexpected(*args, **kwargs):
        pytest.fail("ordinary CPU restoration must not use Spyre primitives")

    runtime.dma_bytes = unexpected
    runtime.spyre_empty_with_layout = unexpected
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    with torch.device("meta"):
        restored = _Model()
    assert cache.restore_cache(restored, torch.device("cpu"), base, key)
    assert restored.weight.stride() == model.weight.stride()
    expected = {(module, json.dumps(path)): t for module, path, t in transfer._tensor_paths(model)}
    for module, path, t in transfer._tensor_paths(restored):
        assert t.device.type == "cpu"
        torch.testing.assert_close(t, expected[module, json.dumps(path)])
    assert restored.host.vision_buffers["segments"].pad is restored.host.nested[0][0].pad
    assert restored.host.recursive["cycle"] is restored.host.recursive
    assert restored.host.vision_buffers["segments"].pad.tolist() == [0.0, 0.0, 1.0]
    assert restored.host.nested[0][1]["offsets"].tolist() == [0, 1, 2]
    assert restored.host.recursive["tensor"].item() == 7


def test_metadata_paths_distinguish_dictionary_key_types():
    model = torch.nn.Module()
    model.table = {1: torch.tensor(1), "1": torch.tensor(2)}
    paths = [path for _, path, _ in transfer._tensor_paths(model)]
    assert [["attr", "table"], ["key", 1]] in paths
    assert [["attr", "table"], ["key", "1"]] in paths


def test_restore_does_not_compare_or_overwrite_plain_attributes(checkpoint):
    model, base, key, _ = checkpoint
    with torch.device("meta"):
        restored = _Model()
    restored.eval()
    restored.host.width = 4
    assert cache.restore_cache(restored, torch.device("cpu"), base, key)
    assert not restored.training and not restored.host.training
    assert restored.host.width == 4
    torch.testing.assert_close(restored.weight, model.weight)
    torch.testing.assert_close(restored.host.scale, model.host.scale)


def test_reject_nested_host_tensor_shape_mismatch(checkpoint, monkeypatch):
    _, base, key, _ = checkpoint
    runtime = ModuleType("torch_spyre._C")
    runtime.dma_bytes = lambda *args: pytest.fail("host shape check must not use DMA")
    runtime.spyre_empty_with_layout = runtime.dma_bytes
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    with torch.device("meta"):
        restored = _Model()
        restored.host.vision_buffers["segments"] = _Padding(2, torch.empty(2, dtype=torch.float16))
    assert not cache.restore_cache(restored, torch.device("cpu"), base, key)
    assert restored.host.vision_buffers["segments"].pad.device.type == "meta"


def test_restore_recorded_host_strides_from_contiguous_meta(checkpoint, monkeypatch):
    model, base, key, _ = checkpoint
    runtime = ModuleType("torch_spyre._C")
    runtime.dma_bytes = lambda *args: pytest.fail("ordinary host restoration must not use DMA")
    runtime.spyre_empty_with_layout = runtime.dma_bytes
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    with torch.device("meta"):
        restored = _Model()
        restored.host.strided = _Padding(2, torch.empty(3, 2))
    assert restored.host.strided.pad.stride() != model.host.strided.pad.stride()
    assert cache.restore_cache(restored, torch.device("cpu"), base, key)
    assert restored.host.strided.pad.stride() == model.host.strided.pad.stride()
    torch.testing.assert_close(restored.host.strided.pad, model.host.strided.pad)


@pytest.mark.parametrize("mismatch", ["key", "shard_set"])
def test_reject_cache_before_restoring_tensors(checkpoint, monkeypatch, mismatch):
    _, base, key, paths = checkpoint
    if mismatch == "key":
        key = {"model": "other"}
    else:
        # Repeating a valid shard must fail the existing shard-set validation.
        monkeypatch.setattr(cache, "shard_paths", lambda *args: [paths[0]] * len(paths))
    with torch.device("meta"):
        restored = _Model()
    assert not cache.restore_cache(restored, torch.device("cpu"), base, key)
    assert restored.weight.device.type == "meta"
    assert restored.host.scale.device.type == "meta"


def test_fp8_model_uses_normal_transfer(checkpoint, monkeypatch):
    model, base, key, _ = checkpoint
    monkeypatch.setattr(cache, "_fp8_layers", lambda model: [("linear", model, None)])
    fills = []
    monkeypatch.setattr(cache.TransferRecorder, "fill", lambda self: fills.append(self.records))
    runtime = ModuleType("torch_spyre._C")
    runtime.dma_bytes = lambda *args: pytest.fail("FP8 cache must not use raw DMA")
    runtime.get_spyre_tensor_layout = runtime.dma_bytes
    layouts = ModuleType("torch_spyre.layout")
    layouts.layout_to_json = runtime.dma_bytes
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    monkeypatch.setitem(sys.modules, "torch_spyre.layout", layouts)
    assert not cache.write_cache(model, torch.device("cpu"), base, key)
    assert len(fills) == 1
    with torch.device("meta"):
        restored = _Model()
    assert not cache.restore_cache(restored, torch.device("cpu"), base, key)
    assert restored.weight.device.type == "meta"


def test_unwritable_cache_fills_via_backend(tmp_path, monkeypatch):
    from fastsafetensors.copier.spyre import TransferRecorder

    assert cache.TransferRecorder is TransferRecorder
    runtime = ModuleType("torch_spyre._C")
    runtime.dma_bytes = lambda *args: pytest.fail("CPU tensors must not use DMA")
    runtime.get_spyre_tensor_layout = runtime.dma_bytes
    layouts = ModuleType("torch_spyre.layout")
    layouts.layout_to_json = runtime.dma_bytes
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    monkeypatch.setitem(sys.modules, "torch_spyre.layout", layouts)
    fills = []
    monkeypatch.setattr(TransferRecorder, "fill", lambda self: fills.append(self.records))

    def fail(**kwargs):
        raise OSError("cache is not writable")

    monkeypatch.setattr(cache, "ParallelSaver", fail)
    model = _Model()
    assert not cache.write_cache(model, torch.device("cpu"), str(tmp_path / "cache"), {})
    assert len(fills) == 1 and fills[0]


def test_reject_old_fp8_payload_before_restoring(checkpoint):
    _, base, key, paths = checkpoint
    with safe_open(paths[0], framework="pt") as shard:
        metadata = shard.metadata()
        names = shard.keys()
        tensors = {name: shard.get_tensor(name).clone() for name in names}
    metadata[transfer.KEY_FP8] = json.dumps([{"path": "linear", "tiles": []}])
    save_file(tensors, paths[0], metadata=metadata)
    with torch.device("meta"):
        restored = _Model()
    assert not cache.restore_cache(restored, torch.device("cpu"), base, key)
    assert restored.weight.device.type == "meta"


def test_old_application_owned_recipes_use_fallback(checkpoint):
    _, base, key, paths = checkpoint
    with safe_open(paths[0], framework="pt") as shard:
        metadata = shard.metadata()
        names = shard.keys()
        tensors = {name: shard.get_tensor(name).clone() for name in names}
    metadata["spyre_inference.host"] = metadata.pop(KEY_HOST)
    metadata["spyre_inference.records"] = metadata.pop(KEY_TRANSFERS)
    save_file(tensors, paths[0], metadata=metadata)
    with torch.device("meta"):
        model = _Model()
    assert not cache.restore_cache(model, torch.device("cpu"), base, key)
    assert model.weight.is_meta and model.host.scale.is_meta


@pytest.mark.parametrize("field", ["format", "key", "fp8"])
def test_old_application_owned_envelope_uses_fallback(checkpoint, field):
    _, base, key, paths = checkpoint
    with safe_open(paths[0], framework="pt") as shard:
        metadata = shard.metadata()
        names = shard.keys()
        tensors = {name: shard.get_tensor(name).clone() for name in names}
    new_key = {
        "format": transfer.KEY_CACHE_FORMAT,
        "key": transfer.KEY_CACHE_KEY,
        "fp8": transfer.KEY_FP8,
    }[field]
    metadata[f"spyre_inference.{field}"] = metadata.pop(new_key)
    save_file(tensors, paths[0], metadata=metadata)
    with torch.device("meta"):
        model = _Model()
    assert not cache.restore_cache(model, torch.device("cpu"), base, key)
    assert model.weight.is_meta and model.host.scale.is_meta


@pytest.mark.parametrize("mismatch", ["exhausted", "leftover", "shape"])
def test_backend_record_mismatch_rejects_model_cache(checkpoint, mismatch):
    _, base, key, paths = checkpoint
    with safe_open(paths[0], framework="pt") as shard:
        metadata = shard.metadata()
        names = shard.keys()
        tensors = {name: shard.get_tensor(name).clone() for name in names}
    records = json.loads(metadata[KEY_TRANSFERS])
    if mismatch == "exhausted":
        records = []
    elif mismatch == "leftover":
        records.append(dict(records[0]))
    else:
        records[0]["size"] = [2, 3]
    metadata[KEY_TRANSFERS] = json.dumps(records)
    save_file(tensors, paths[0], metadata=metadata)
    with torch.device("meta"):
        restored = _Model()
    assert not cache.restore_cache(restored, torch.device("cpu"), base, key)


def test_model_restore_uses_backend_replay_and_finish(checkpoint, monkeypatch):
    from fastsafetensors.copier.spyre import TransferRestorer

    assert cache.TransferRestorer is TransferRestorer
    _, base, key, _ = checkpoint
    calls = []

    class Restorer(TransferRestorer):
        def __call__(self, tensor):
            calls.append("restore")
            return super().__call__(tensor)

        def finish(self):
            calls.append("finish")
            return super().finish()

    monkeypatch.setattr(cache, "TransferRestorer", Restorer)
    with torch.device("meta"):
        restored = _Model()
    assert cache.restore_cache(restored, torch.device("cpu"), base, key)
    assert calls == ["restore", "finish"]


def test_host_writer_rejects_unmaterialized_nested_state(tmp_path):
    model = _Model()
    model.host.recursive["tensor"] = torch.empty(3, device="meta")
    base = str(tmp_path / "explicit")
    assert not cache.write_cache(model, torch.device("cpu"), base, {})
    assert not cache.cache_exists(base)


@pytest.mark.parametrize("change", ["missing", "shape", "split_alias", "merged_alias"])
def test_host_path_mismatch_rejected_before_any_swap(checkpoint, monkeypatch, change):
    _, base, key, _ = checkpoint
    with torch.device("meta"):
        model = _Model()
    pad = model.host.vision_buffers["segments"].pad
    if change == "missing":
        del model.host.table["empty"]
    elif change == "shape":
        model.host.other_pad = torch.empty(8, device="meta")
    elif change == "split_alias":
        model.host.nested[0] = (_Padding(3, torch.empty_like(pad)), model.host.nested[0][1])
    else:
        model.host.other_pad = pad
    monkeypatch.setattr(transfer, "_swap_in", lambda *args: pytest.fail("must validate first"))
    assert not cache.restore_cache(model, torch.device("cpu"), base, key)
    assert model.weight.is_meta


def test_remaining_registered_meta_buffer_is_not_silently_ignored(checkpoint):
    _, base, key, _ = checkpoint
    with torch.device("meta"):
        model = _Model()
        model.host.register_buffer("forgotten", torch.empty(3), persistent=False)
    assert not cache.restore_cache(model, torch.device("cpu"), base, key)


def test_host_swap_preserves_parameter_subclass_and_python_attributes():
    class HostParameter(torch.nn.Parameter):
        pass

    target = HostParameter(torch.empty(2, device="meta"), requires_grad=False)
    target.marker = object()
    alias, marker = target, target.marker
    transfer._swap_in(target, torch.tensor([0.5, 2.0]))
    assert target is alias and type(target) is HostParameter
    assert target.marker is marker
    torch.testing.assert_close(target, torch.tensor([0.5, 2.0]))


def test_attention_cpu_state_needs_no_model_adapter():
    from vllm.model_executor.layers.attention.attention import Attention

    model = Attention.__new__(Attention)
    torch.nn.Module.__init__(model)
    names = ("_q_scale", "_k_scale", "_v_scale", "_prob_scale", "_k_scale_cpu", "_v_scale_cpu")
    for name in names:
        setattr(model, name, torch.tensor(0.25))
    model.kv_cache = torch.empty(0)
    recorder = transfer.TransferRecorder(torch.device("cpu"))
    recorder.capture_host_state(model)
    metadata, _ = recorder.checkpoint_metadata(model)
    host = json.loads(metadata[KEY_HOST])
    assert {rec["path"][0][1] for rec in host} == set(names) | {"kv_cache"}


def test_rope_host_source_restored_before_deriving_transfer(tmp_path, monkeypatch):
    from spyre_inference.custom_ops.rotary_embedding import _SpyreRotaryMixin

    class Rope(_SpyreRotaryMixin, torch.nn.Module):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.register_buffer("cos_sin_cache", torch.arange(16.0).reshape(4, 4))
            self._padded_inner = 2
            self._rotation_cache = None
            self._device_rotation_cache = None
            self._rotate_half = None

    import spyre_inference.custom_ops.rotary_embedding as rotary

    monkeypatch.setattr(rotary, "place_row_gathered", lambda t, fn, label: fn(t))
    model = Rope()
    base = str(tmp_path / "rope")
    assert cache.write_cache(model, torch.device("cpu"), base, {})
    with torch.device("meta"):
        restored = Rope()
    assert cache.restore_cache(restored, torch.device("cpu"), base, {})
    assert not restored._rotation_cache.is_meta
    torch.testing.assert_close(restored._rotation_cache, model._rotation_cache)
    assert not hasattr(restored, "spyre_native_host_tensors")


def test_transformers_rope_meta_host_tables_round_trip(tmp_path):
    from spyre_inference.transformers_backend import _SpyreRotaryEmbedding
    from spyre_inference.v1.worker.model_build import _MetaHostCopies

    def model():
        original = torch.nn.Module()
        original.register_buffer("inv_freq", torch.arange(4.0))
        return _SpyreRotaryEmbedding(original, 8, 64, torch.float16)

    loaded = model()
    base = str(tmp_path / "hf-rope")
    assert cache.write_cache(loaded, torch.device("cpu"), base, {})
    with torch.device("meta"), _MetaHostCopies():
        restored = model()
    assert cache.restore_cache(restored, torch.device("cpu"), base, {})
    assert restored._cpu_caches[None].device.type == "cpu"
    torch.testing.assert_close(restored._caches[None], loaded._caches[None])


def test_vision_metadata_includes_unregistered_weight_and_padding():
    from types import SimpleNamespace

    from spyre_inference.models._vision_encoder import VisionBuffer
    from spyre_inference.models.gemma4_mm import SpyreGemma4ForConditionalGeneration
    from spyre_inference.models.granite4_vision import SpyreGranite4VisionForConditionalGeneration

    for cls in (SpyreGemma4ForConditionalGeneration, SpyreGranite4VisionForConditionalGeneration):
        model = cls.__new__(cls)
        torch.nn.Module.__init__(model)
        model.vision_buffers = {"segments": VisionBuffer(3, pad=torch.ones(3))}
        if cls is SpyreGemma4ForConditionalGeneration:
            model._spyre_vision_tail = SimpleNamespace(
                weight=torch.ones(2, 3), std_bias=torch.zeros(3), std_scale=torch.ones(3)
            )
            expected = {"weight", "std_bias", "std_scale", "pad"}
        else:
            model._packer = SimpleNamespace(image_newline=torch.ones(3))
            expected = {"image_newline", "pad"}
        recorder = transfer.TransferRecorder(torch.device("cpu"))
        recorder.capture_host_state(model)
        metadata, _ = recorder.checkpoint_metadata(model)
        host = json.loads(metadata[KEY_HOST])
        assert {rec["path"][-1][1] for rec in host} == expected
        assert not hasattr(model, "spyre_native_host_tensors")


def test_granite_restores_host_order_indices_from_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from spyre_inference.models.granite4_vision import (
        SpyreGranite4VisionForConditionalGeneration as Granite,
    )

    def model():
        result = Granite.__new__(Granite)
        torch.nn.Module.__init__(result)
        result.config = config
        result.spatial_projectors = torch.nn.ModuleList()
        result._init_spyre_vision()
        return result

    config = SimpleNamespace(
        vision_config=SimpleNamespace(image_size=16, patch_size=1, num_channels=64, hidden_size=64),
        text_config=SimpleNamespace(hidden_size=64),
        downsample_rate="4/8",
        deepstack_layer_map=[],
        spatial_target_layers=[1],
        vision_feature_select_strategy="full",
        image_grid_pinpoints=[[16, 16]],
    )
    monkeypatch.setattr(Granite, "_apply", torch.nn.Module._apply)
    loaded = model()
    recorder = transfer.TransferRecorder(torch.device("cpu"))
    recorder.capture_host_state(loaded)
    metadata, entries = recorder.checkpoint_metadata(loaded)
    base = str(tmp_path / "granite")
    paths = save_sharded(entries, base, num_shards=1, metadata=metadata)
    header = cache.SafeTensorsMetadata.from_file(paths[0], cache.get_framework_op("pytorch"))
    restorer = transfer.TransferRestorer.from_metadata(
        torch.device("cpu"), {name: header for name in header.tensors}, metadata
    )
    with torch.device("meta"):
        restored = model()
    assert restored._tile_order.is_meta
    restorer.restore_host_state(restored)
    assert restored._tile_order.device.type == restored._window_order.device.type == "cpu"
    assert sorted(restored._tile_order.tolist()) == list(range(256))


@pytest.mark.parametrize("head_size, proportional", [(64, False), (512, True)])
def test_real_decoder_rope_meta_round_trip(
    tmp_path, monkeypatch, default_vllm_config, head_size, proportional
):
    from unittest.mock import patch

    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT, get_rope

    import spyre_inference.custom_ops.rotary_embedding as rotary

    monkeypatch.setattr(rotary, "place_row_gathered", lambda t, fn, label: fn(t))

    class Recorder(cache.TransferRecorder):
        def __call__(self, tensor):
            result = super().__call__(tensor)
            # Emulate a device transfer: CPU->CPU would otherwise record a revisit,
            # which is not applicable to a meta skeleton's _rotate_half source.
            if self.records[-1].kind == "same":
                self.records[-1].kind = "to"
            return result

    monkeypatch.setattr(cache, "TransferRecorder", Recorder)
    params = (
        {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0}
        if proportional
        else None
    )

    def model(device):
        with patch.dict(_ROPE_DICT, clear=True), torch.device(device):
            return get_rope(
                head_size,
                max_position=32,
                is_neox_style=True,
                rope_parameters=params,
                dtype=torch.float16,
            )

    loaded = model("cpu")
    base = str(tmp_path / "decoder-rope")
    assert cache.write_cache(loaded, torch.device("cpu"), base, {})
    restored = model("meta")
    assert cache.restore_cache(restored, torch.device("cpu"), base, {})
    torch.testing.assert_close(restored.cos_sin_cache, loaded.cos_sin_cache)
    torch.testing.assert_close(restored._device_rotation_cache, loaded._device_rotation_cache)
    if head_size == 64:
        torch.testing.assert_close(restored._rotate_half, loaded._rotate_half)
