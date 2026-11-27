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

"""Unit tests for SpyreModelLoader and direct-to-device weight loading."""

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.config import LoadConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.parameter import BasevLLMParameter

import spyre_inference.model_loader as loader_mod
import spyre_inference.platform as plat_mod
from spyre_inference.model_loader import (
    SpyreModelLoader,
    _check_all_weights_loaded,
    _get_host_loaded_param_names,
    _hf_to_vllm_name,
    _iter_tp_sharded_weights,
    _param_name_for,
    _read_ahead,
    _resolve_module,
    _shard_plan,
)
from spyre_inference.platform import (
    TorchSpyrePlatform,
    _spyre_loader_supported,
)


def test_spyre_model_loader_registration():
    """Verify that --load-format spyre returns SpyreModelLoader."""
    load_config = LoadConfig(load_format="spyre")
    loader = get_model_loader(load_config)
    assert isinstance(loader, SpyreModelLoader)


def _make_vllm_config(
    tp_size=1,
    quantization=None,
    runner_type="generate",
    load_format="auto",
    transformers_backend=False,
    moe=False,
):
    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1,
        dtype=torch.float16,
        trust_remote_code=True,
        quantization=quantization,
    )
    if runner_type != "generate":
        model_config.runner_type = runner_type
    if transformers_backend:
        model_config.using_transformers_backend = lambda: True
    if moe:
        model_config.hf_text_config.num_experts = 8
    return VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(load_format=load_format),
        parallel_config=ParallelConfig(tensor_parallel_size=tp_size),
    )


def test_platform_defaults_load_format_to_spyre():
    """Positive control: a plain TP=1 text-generation config flips auto → spyre."""
    vllm_config = _make_vllm_config()
    assert vllm_config.load_config.load_format == "spyre"


def test_platform_defaults_load_format_to_spyre_tp2():
    """TP>1 is supported: auto resolves to spyre (linear weights shard on read)."""
    vllm_config = _make_vllm_config(tp_size=2)
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    assert vllm_config.load_config.load_format == "spyre"


def test_platform_defaults_load_format_to_spyre_multimodal():
    """Multimodal models are supported: auto resolves to spyre."""
    vllm_config = _make_vllm_config()
    vllm_config.model_config.multimodal_config = object()
    vllm_config.load_config.load_format = "auto"
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    assert vllm_config.load_config.load_format == "spyre"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantization": "fp8"},  # quantized
        {"runner_type": "pooling"},  # pooling
    ],
    ids=["quantized", "pooling"],
)
def test_platform_load_format_stays_auto(kwargs):
    """load_format stays 'auto' for configs SpyreModelLoader doesn't support."""
    vllm_config = _make_vllm_config(**kwargs)
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    assert vllm_config.load_config.load_format == "auto"


# ---------------------------------------------------------------------------
# _spyre_loader_supported unit tests
# ---------------------------------------------------------------------------


def _make_ns(
    tp_size=1,
    runner_type="generate",
    quantization=None,
    transformers_backend=False,
    moe=False,
    multimodal=False,
):
    """Build a minimal SimpleNamespace config for _spyre_loader_supported tests."""
    model_config = SimpleNamespace(
        runner_type=runner_type,
        quantization=quantization,
        hf_text_config=SimpleNamespace(num_experts=8) if moe else SimpleNamespace(),
        multimodal_config=object() if multimodal else None,
        is_multimodal_model=False,
    )
    if transformers_backend:
        model_config.using_transformers_backend = lambda: True
    parallel_config = SimpleNamespace(tensor_parallel_size=tp_size)
    return SimpleNamespace(model_config=model_config, parallel_config=parallel_config)


def test_spyre_loader_supported_accepts_valid_config():
    """Positive case: generate, unquantized, dense, non-transformers-backend is supported."""
    assert _spyre_loader_supported(_make_ns())


@pytest.mark.parametrize(
    "kwargs",
    [{"tp_size": 2}, {"tp_size": 4}, {"multimodal": True}, {"tp_size": 2, "multimodal": True}],
    ids=["tp2", "tp4", "multimodal", "tp2-multimodal"],
)
def test_spyre_loader_supported_accepts_tp_and_multimodal(kwargs):
    """TP > 1 and multimodal models are supported by the direct-to-device loader."""
    assert _spyre_loader_supported(_make_ns(**kwargs))


@pytest.mark.parametrize(
    ("kwargs", "description"),
    [
        ({"runner_type": "pooling"}, "pooling runner"),
        ({"runner_type": "unknown_future_runner"}, "unknown runner type"),
        ({"quantization": "fp8"}, "quantized"),
        ({"transformers_backend": True}, "transformers backend"),
        ({"moe": True}, "MoE"),
    ],
)
def test_spyre_loader_supported_rejects_unsupported(kwargs, description):
    """_spyre_loader_supported returns False for each excluded config class."""
    assert not _spyre_loader_supported(_make_ns(**kwargs)), description


def test_platform_load_format_stays_auto_transformers_backend():
    """load_format stays 'auto' for the transformers backend."""
    assert _make_vllm_config(transformers_backend=True).load_config.load_format == "auto"


def test_platform_load_format_stays_auto_moe():
    """load_format stays 'auto' for MoE models."""
    assert _make_vllm_config(moe=True).load_config.load_format == "auto"


def test_explicit_spyre_load_format_rejected_for_unsupported_config():
    """--load-format spyre passed explicitly raises for unsupported configs.

    VllmConfig is built normally (load_format="auto"), then load_format is
    patched to "spyre" directly on the load_config object to bypass pydantic
    validation (which only accepts registered formats at construction time).
    _spyre_loader_supported is mocked to False to simulate an unsupported config.
    """
    import unittest.mock as mock

    vllm_config = _make_vllm_config()
    vllm_config.load_config.load_format = "spyre"
    with (
        mock.patch.object(plat_mod, "_spyre_loader_supported", return_value=False),
        pytest.raises(ValueError, match="--load-format spyre is not supported"),
    ):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


def test_get_host_loaded_param_names(tp_group):
    """Verify VocabParallelEmbedding parameters stay host-loaded."""

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = VocabParallelEmbedding(10, 10)
            self.dense = nn.Linear(10, 10)

    model = DummyModel()
    host_names = _get_host_loaded_param_names(model)
    assert "embed.weight" in host_names
    assert "dense.weight" not in host_names


def test_check_all_weights_loaded_passes_when_complete():
    model = nn.Sequential(nn.Linear(2, 2))
    _check_all_weights_loaded(model, {"0.weight", "0.bias"})


def test_check_all_weights_loaded_raises_on_missing_param():
    """A parameter with no checkpoint tensor would keep its uninitialized placeholder."""
    model = nn.Sequential(nn.Linear(2, 2))
    with pytest.raises(ValueError, match="not initialized from checkpoint.*0.bias"):
        _check_all_weights_loaded(model, {"0.weight"})


def test_check_all_weights_loaded_skips_untracked_models():
    """A model whose load_weights returns None cannot be checked; it must not raise."""
    model = nn.Sequential(nn.Linear(2, 2))
    _check_all_weights_loaded(model, None)


@pytest.mark.parametrize("attr", ["_spyre_orig_head_dim", "_spyre_orig_intermediate_size"])
def test_spyre_loader_supported_rejects_padded(attr):
    ns = _make_ns()
    ns.model_config.hf_text_config = SimpleNamespace(**{attr: 16})
    assert not _spyre_loader_supported(ns)


def test_spyre_loader_supported_accepts_unpadded_text_config():
    ns = _make_ns()
    ns.model_config.hf_text_config = SimpleNamespace()
    assert _spyre_loader_supported(ns)


def test_gate_sees_padding_applied_by_the_hook(monkeypatch):
    """The hook must pad before gating, since check_and_update_config runs later."""
    vllm_config = _make_vllm_config()
    vllm_config.load_config.load_format = "auto"

    def fake_pad(cls, config):
        config.model_config.hf_text_config._spyre_orig_head_dim = 16

    monkeypatch.setattr(TorchSpyrePlatform, "_maybe_pad_head_dim", classmethod(fake_pad))
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    assert vllm_config.load_config.load_format == "auto"


def test_explicit_spyre_load_format_rejected_for_padded_config():
    vllm_config = _make_vllm_config()
    vllm_config.model_config.hf_text_config._spyre_orig_intermediate_size = 160
    vllm_config.load_config.load_format = "spyre"
    with pytest.raises(ValueError, match="--load-format spyre is not supported"):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


class _FakeMultimodal(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Linear(2, 2)
        self.vision_tower = nn.Linear(2, 2)
        self.projector = nn.Linear(2, 2)


_NON_TEXT = {"vision_tower.weight", "vision_tower.bias", "projector.weight", "projector.bias"}


def test_host_loaded_names_include_vision_tower(monkeypatch):
    from spyre_inference.custom_ops import text_backbone

    model = _FakeMultimodal()
    monkeypatch.setattr(
        text_backbone, "_language_backbones", lambda m: [("language_model", m.language_model)]
    )
    assert _get_host_loaded_param_names(model) == _NON_TEXT


def test_non_text_parameter_names_text_only_model(monkeypatch):
    from spyre_inference.custom_ops import text_backbone

    model = nn.Sequential(nn.Linear(2, 2))
    monkeypatch.setattr(text_backbone, "_language_backbones", lambda m: [("", m)])
    assert text_backbone.non_text_parameter_names(model) == set()


def test_non_text_parameter_names_unisolatable_backbone_is_conservative(monkeypatch):
    from spyre_inference.custom_ops import text_backbone

    model = _FakeMultimodal()
    monkeypatch.setattr(text_backbone, "_language_backbones", lambda m: None)
    assert text_backbone.non_text_parameter_names(model) == {n for n, _ in model.named_parameters()}


def test_load_model_patches_attention_before_building_the_model(monkeypatch):
    """SpyreModelLoader keeps Attention buffers on the host before any module exists."""
    import spyre_inference.model_loader as loader_module
    import spyre_inference.v1.worker.model_load as model_load

    calls = []
    monkeypatch.setattr(
        model_load, "keep_attention_buffers_on_host", lambda: calls.append("patched")
    )

    def stop(**kwargs):
        calls.append("init")
        raise RuntimeError("stop")

    monkeypatch.setattr(loader_module, "initialize_model", stop)
    monkeypatch.setattr(loader_module, "_has_safetensors_checkpoint", lambda *a: True)
    loader = SpyreModelLoader(LoadConfig(load_format="spyre"))
    with pytest.raises(RuntimeError, match="stop"):
        loader.load_model(SimpleNamespace(), SimpleNamespace(dtype=torch.float16))
    assert calls == ["patched", "init"]


class _MappedModel(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
        }
    )
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }


def test_hf_to_vllm_name_applies_the_mapper():
    model = _MappedModel()
    assert (
        _hf_to_vllm_name(model, "model.language_model.embed_tokens.weight")
        == "language_model.model.embed_tokens.weight"
    )


def test_hf_to_vllm_name_without_a_mapper_is_a_noop():
    assert (
        _hf_to_vllm_name(nn.Linear(2, 2), "model.embed_tokens.weight")
        == "model.embed_tokens.weight"
    )


def test_mistral_format_keys_are_untouched():
    """Mistral-format keys are renamed inside the model's own load_weights, not the mapper."""
    assert _param_name_for(_MappedModel(), "tok_embeddings.weight") == "tok_embeddings.weight"


def test_param_name_for_resolves_fused_layers():
    model = _MappedModel()
    key = "model.vision_tower.transformer.layers.0.attention.q_proj.weight"
    assert (
        _param_name_for(model, key) == "vision_tower.transformer.layers.0.attention.qkv_proj.weight"
    )
    key = "model.vision_tower.transformer.layers.0.feed_forward.up_proj.weight"
    assert (
        _param_name_for(model, key)
        == "vision_tower.transformer.layers.0.feed_forward.gate_up_proj.weight"
    )


def test_hf_to_vllm_name_matches_the_real_mistral3_mapper():
    from vllm.model_executor.models.mistral3 import Mistral3ForConditionalGeneration as M

    model = SimpleNamespace(hf_to_vllm_mapper=M.hf_to_vllm_mapper)
    assert (
        _hf_to_vllm_name(model, "model.language_model.embed_tokens.weight")
        == "language_model.model.embed_tokens.weight"
    )
    # The pre-4.52 layout already matches vLLM's names and must pass through unchanged.
    legacy = "language_model.model.embed_tokens.weight"
    assert _hf_to_vllm_name(model, legacy) == legacy


def test_hf_to_vllm_name_matches_the_real_pixtral_mapper():
    from vllm.model_executor.models.pixtral import PixtralForConditionalGeneration as P

    model = SimpleNamespace(hf_to_vllm_mapper=P.hf_to_vllm_mapper)
    assert (
        _hf_to_vllm_name(model, "model.language_model.embed_tokens.weight")
        == "language_model.model.embed_tokens.weight"
    )
    assert (
        _hf_to_vllm_name(model, "model.vision_tower.transformer.layers.0.attention.q_proj.weight")
        == "vision_encoder.transformer.layers.0.attention.q_proj.weight"
    )
    legacy = "language_model.model.embed_tokens.weight"
    assert _hf_to_vllm_name(model, legacy) == legacy


def test_spyre_loader_supported_rejects_speculative_decoding():
    """The draft model reuses the target's load_config; drafters need host tensors."""
    ns = _make_ns()
    ns.speculative_config = object()
    assert not _spyre_loader_supported(ns)


def test_spyre_loader_supported_accepts_no_speculative_config():
    ns = _make_ns()
    ns.speculative_config = None
    assert _spyre_loader_supported(ns)


def _loader_and_config(tmp_path):
    loader = SpyreModelLoader(LoadConfig(load_format="spyre"))
    config = SimpleNamespace(model=str(tmp_path), revision=None, dtype=torch.float16)
    return loader, config


def test_load_model_uses_default_loader_for_bin_only_checkpoints(tmp_path, monkeypatch):
    import spyre_inference.model_loader as loader_module

    (tmp_path / "pytorch_model.bin").write_bytes(b"")
    seen = {}

    class FakeDefaultLoader:
        def __init__(self, load_config):
            seen["load_format"] = load_config.load_format

        def load_model(self, vllm_config, model_config, prefix=""):
            seen["loaded"] = True
            return "model"

    monkeypatch.setattr(loader_module, "DefaultModelLoader", FakeDefaultLoader)
    loader, config = _loader_and_config(tmp_path)

    assert loader.load_model(SimpleNamespace(), config) == "model"
    assert seen == {"load_format": "auto", "loaded": True}
    assert loader.load_config.load_format == "spyre"  # the caller's config is untouched


def test_load_model_does_not_fall_back_when_safetensors_exist(tmp_path, monkeypatch):
    import spyre_inference.model_loader as loader_module

    (tmp_path / "model.safetensors").write_bytes(b"")

    def stop(**kwargs):
        raise RuntimeError("stop")

    monkeypatch.setattr(loader_module, "initialize_model", stop)
    loader, config = _loader_and_config(tmp_path)

    with pytest.raises(RuntimeError, match="stop"):
        loader.load_model(SimpleNamespace(), config)


_PHI3 = {"qkv_proj": ["qkv_proj"], "gate_up_proj": ["gate_up_proj"]}
_LLAMA = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}


def _bare(cls, **attrs):
    """A layer of ``cls`` without running its __init__ (no TP group or device needed)."""
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    layer.weight = nn.Parameter(torch.zeros(1))
    for name, value in attrs.items():
        setattr(layer, name, value)
    return layer


def _packed_model(mapping):
    model = nn.Module()
    model.packed_modules_mapping = mapping
    model.self_attn, model.mlp = nn.Module(), nn.Module()
    model.self_attn.qkv_proj = _bare(
        QKVParallelLinear, head_size=16, num_heads=2, num_kv_heads=2, num_kv_head_replicas=1
    )
    model.mlp.gate_up_proj = _bare(MergedColumnParallelLinear)
    return model


def test_resolve_module_reports_no_shard_for_a_fused_checkpoint_key():
    model = _packed_model(_PHI3)
    module, shard = _resolve_module(model, "self_attn.qkv_proj.weight")
    assert module is model.self_attn.qkv_proj and shard is None


@pytest.mark.parametrize("mapping", [_PHI3, _LLAMA, {}], ids=["phi3", "llama", "no-mapping"])
def test_fused_checkpoint_tensors_take_the_host_path(mapping):
    """A fused qkv / gate_up checkpoint tensor can't be pre-sharded as one contiguous block."""
    model = _packed_model(mapping)
    assert _shard_plan(model, "self_attn.qkv_proj.weight", (192, 64), 2, 1) is None
    assert _shard_plan(model, "mlp.gate_up_proj.weight", (256, 64), 2, 1) is None


def test_unfused_checkpoint_tensors_are_still_sharded_on_read():
    model = _packed_model(_LLAMA)
    q = _shard_plan(model, "self_attn.q_proj.weight", (64, 64), 2, 1)
    k = _shard_plan(model, "self_attn.k_proj.weight", (64, 64), 2, 1)
    gate = _shard_plan(model, "mlp.gate_proj.weight", (128, 64), 2, 1)
    assert q[1:] == (0, 32, 32) and k[1:] == (0, 32, 32) and gate[1:] == (0, 64, 64)


class _TwoLinearModel(nn.Module):
    """A normal linear layer plus one whose weight must stay on the host."""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(64, 64, bias=False, dtype=torch.float16)
        self.fallback = nn.Linear(64, 64, bias=False, dtype=torch.float16)

    def load_weights(self, weights):
        loaded = set()
        for name, tensor in weights:
            self.get_parameter(name).data.copy_(tensor)
            loaded.add(name)
        return loaded


def _reads(count, read=lambda i: i):
    return ((i, partial(read, i)) for i in range(count))


def _pool_threads():
    return [t for t in threading.enumerate() if t.name.startswith("spyre-load")]


def _pool(num_threads=4):
    return ThreadPoolExecutor(num_threads, thread_name_prefix="spyre-load")


@pytest.fixture
def host_only(monkeypatch):
    """The fake handles return host tensors, whatever an earlier test initialized."""
    if hasattr(torch, "spyre"):
        monkeypatch.setattr(torch.spyre, "is_initialized", lambda: False)


def test_read_ahead_keeps_order_when_reads_finish_out_of_order():
    last_read = threading.Event()

    def read(i):
        if i == 0:
            assert last_read.wait(30)
        if i == 3:
            last_read.set()
        return i * 10

    with _pool() as pool:
        assert list(_read_ahead(_reads(4, read), pool, 4)) == [(i, i * 10) for i in range(4)]


def test_read_ahead_bounds_the_reads_not_yet_consumed():
    issued = consumed = 0
    outstanding = []

    def reads():
        nonlocal issued
        for item in _reads(20):
            issued += 1
            outstanding.append(issued - consumed)
            yield item

    with _pool() as pool:
        for _ in _read_ahead(reads(), pool, 4):
            consumed += 1
    assert consumed == 20 and max(outstanding) == 4


def test_read_ahead_without_a_pool_reads_on_the_calling_thread_one_at_a_time():
    events = []

    def read(i):
        events.append(("read", i, threading.get_ident()))
        return i

    for tag, _ in _read_ahead(_reads(3, read), None, 1):
        events.append(("got", tag, threading.get_ident()))
    me = threading.get_ident()
    assert events == [(kind, i, me) for i in range(3) for kind in ("read", "got")]


def test_read_ahead_raises_a_failed_read_in_the_caller():
    def read(i):
        if i == 2:
            raise RuntimeError("bad read")
        return i

    got = []
    with _pool() as pool, pytest.raises(RuntimeError, match="bad read"):
        for tag, _ in _read_ahead(_reads(8, read), pool, 4):
            got.append(tag)
    assert got == [0, 1]


def _extra_config_loader(extra_config):
    return SpyreModelLoader(LoadConfig(load_format="spyre", model_loader_extra_config=extra_config))


@pytest.mark.parametrize(
    ("extra_config", "num_threads"),
    [
        ({}, 8),
        ({"num_threads": 4}, 4),
        ({"enable_multithread_load": False}, 1),
        ({"enable_multithread_load": False, "num_threads": 4}, 1),
        ({"enable_weights_track": True, "some_other_loaders_key": "x"}, 8),
    ],
    ids=["empty", "threads-only", "disabled", "disabled-4", "other-keys"],
)
def test_multithread_load_is_on_by_default_with_vllms_keys(extra_config, num_threads):
    assert _extra_config_loader(extra_config)._load_threads == num_threads


@pytest.mark.parametrize(
    "extra_config",
    [
        {"enable_multithread_load": "yes"},
        {"num_threads": 0},
        {"num_threads": "8"},
        {"num_threads": True},
    ],
    ids=["enable-str", "threads-zero", "threads-str", "threads-bool"],
)
def test_bad_multithread_load_values_are_rejected(extra_config):
    with pytest.raises(ValueError, match="enable_multithread_load|num_threads"):
        _extra_config_loader(extra_config)


@pytest.mark.parametrize(
    ("extra_config", "warned"),
    [
        ({"num_thread": 2}, ["num_thread"]),
        ({"enable_weights_track": True, "num_threads": 4}, ["enable_weights_track"]),
        ({"enable_multithread_load": True, "num_threads": 4}, None),
    ],
    ids=["typo", "weights-track", "known-keys"],
)
def test_unknown_extra_config_keys_are_ignored_with_a_warning(caplog, extra_config, warned):
    with caplog.at_level("WARNING", logger="spyre_inference.model_loader"):
        loader = _extra_config_loader(extra_config)

    messages = [r.getMessage() for r in caplog.records if "unknown" in r.getMessage()]
    assert messages == (
        [f"SpyreModelLoader: ignoring unknown model_loader_extra_config keys {warned}"]
        if warned
        else []
    )
    assert loader._load_threads == extra_config.get("num_threads", 8)


class _RecordingHandle:
    """A CPU safetensors handle that records which thread read each tensor."""

    def __init__(self, path, readers):
        self._handle = safe_open(path, framework="pt", device="cpu")
        self._readers = readers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._handle.__exit__(*exc)

    def offset_keys(self):
        return self._handle.offset_keys()

    def get_tensor(self, name):
        self._readers[name] = threading.current_thread().name
        return self._handle.get_tensor(name)

    def get_slice(self, name):
        self._readers[name] = threading.current_thread().name
        return self._handle.get_slice(name)


def _recorded_checkpoint(tmp_path, monkeypatch, weights):
    ckpt = tmp_path / "model.safetensors"
    save_file(weights, str(ckpt))
    readers: dict[str, str] = {}
    monkeypatch.setattr(
        "safetensors.safe_open", lambda path, **kwargs: _RecordingHandle(path, readers)
    )
    monkeypatch.setattr(
        "spyre_inference.model_loader._resolve_checkpoint_files",
        lambda model_config, load_config: (str(tmp_path), [str(ckpt)]),
    )
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    return readers


@pytest.mark.parametrize(
    ("extra_config", "torch_threads", "pooled"),
    [
        ({}, 8, True),
        ({"enable_multithread_load": False}, 8, False),
        ({}, 1, False),
    ],
    ids=["default", "disabled", "default-one-cpu"],
)
def test_get_all_weights_reads_on_the_pool_only_when_enabled(
    tmp_path, monkeypatch, tp_group, host_only, extra_config, torch_threads, pooled
):
    weights = {f"w{i}": torch.full((4,), float(i)) for i in range(6)}
    readers = _recorded_checkpoint(tmp_path, monkeypatch, weights)
    monkeypatch.setattr(torch, "get_num_threads", lambda: torch_threads)

    loader = _extra_config_loader(extra_config)
    loaded = dict(loader.get_all_weights(SimpleNamespace(dtype=torch.float16), nn.Module()))

    assert list(loaded) == list(weights)
    assert all(torch.equal(loaded[name], weights[name]) for name in weights)
    assert all(reader.startswith("spyre-load") == pooled for reader in readers.values())
    assert not _pool_threads()


def test_pool_is_capped_at_the_device_stream_count(tmp_path, monkeypatch, tp_group, host_only):
    _recorded_checkpoint(tmp_path, monkeypatch, {"w0": torch.zeros(4)})
    monkeypatch.setattr(torch, "get_num_threads", lambda: 64)
    widths = []
    monkeypatch.setattr(
        loader_mod,
        "ThreadPoolExecutor",
        lambda num_threads, **kwargs: (
            widths.append(num_threads) or ThreadPoolExecutor(num_threads, **kwargs)
        ),
    )

    loader = _extra_config_loader({"num_threads": 64})
    list(loader.get_all_weights(SimpleNamespace(dtype=torch.float16), nn.Module()))

    assert widths == [32]


def test_closing_get_all_weights_early_stops_the_pool(tmp_path, monkeypatch, tp_group, host_only):
    _recorded_checkpoint(tmp_path, monkeypatch, {f"w{i}": torch.zeros(4) for i in range(8)})
    loaded = _extra_config_loader({}).get_all_weights(
        SimpleNamespace(dtype=torch.float16), nn.Module()
    )
    assert next(loaded)[0] == "w0"
    assert _pool_threads()
    loaded.close()
    assert not _pool_threads()


def test_a_failing_weight_loader_stops_the_pool(tmp_path, monkeypatch, tp_group):
    _recorded_checkpoint(tmp_path, monkeypatch, {f"w{i}": torch.zeros(4) for i in range(8)})
    monkeypatch.delattr(torch, "spyre", raising=False)  # the fake handles return host tensors

    class Model(nn.Module):
        def load_weights(self, weights):
            next(iter(weights))
            raise RuntimeError("bad weight")

    with pytest.raises(RuntimeError, match="bad weight") as raised:
        _extra_config_loader({}).load_weights(Model(), SimpleNamespace(dtype=torch.float16))
    assert raised.traceback and not _pool_threads()


def test_pool_threads_take_their_own_stream_and_never_allocate(tmp_path, monkeypatch, tp_group):
    """On the card a pool thread only fills a tensor the loading thread allocated."""
    weights = {f"w{i}": torch.full((4,), float(i)) for i in range(16)}
    _recorded_checkpoint(tmp_path, monkeypatch, weights)
    spyre = SimpleNamespace(is_initialized=lambda: True, current_device=lambda: 3)
    monkeypatch.setattr(torch, "spyre", spyre, raising=False)
    streams, allocated, filled = [], [], []
    monkeypatch.setattr(
        loader_mod,
        "_use_own_stream",
        lambda device: streams.append((device, threading.get_ident())),
    )

    def alloc(read):
        allocated.append(threading.get_ident())
        return None if read.name == "w5" else read.name

    def fill(read, dst):
        filled.append(threading.get_ident())
        return read.f_cpu.get_tensor(dst)

    monkeypatch.setattr(loader_mod._CardRead, "alloc", alloc)
    monkeypatch.setattr(loader_mod._CardRead, "fill", fill)

    loaded = dict(
        _extra_config_loader({}).get_all_weights(SimpleNamespace(dtype=torch.float16), nn.Module())
    )

    me = threading.get_ident()
    assert list(loaded) == sorted(weights)
    assert all(torch.equal(loaded[name], weights[name]) for name in weights)
    assert allocated == [me] * 16
    assert len(filled) == 15 and me not in filled
    assert {device for device, _ in streams} == {3}
    idents = [ident for _, ident in streams]
    assert len(idents) == len(set(idents)) and set(filled) <= set(idents)


@pytest.mark.parametrize("num_threads", [1, 4])
def test_tp_sharded_reads_take_this_ranks_slice(tmp_path, monkeypatch, host_only, num_threads):
    weights = {f"w{i}": torch.arange(8.0).reshape(4, 2) + i for i in range(6)}
    ckpt = tmp_path / "model.safetensors"
    save_file(weights, str(ckpt))
    handle = _RecordingHandle(str(ckpt), {})
    params = {}
    for name in weights:
        params[name] = BasevLLMParameter(data=torch.empty(2, 2), weight_loader=lambda *args: None)
        params[name].tp_rank = 1
    monkeypatch.setattr(loader_mod, "_hf_to_vllm_name", lambda model, name: name)
    monkeypatch.setattr(
        loader_mod,
        "_shard_plan",
        lambda model, name, shape, tp_size, tp_rank: (params[name], 0, 2, 2),
    )

    seen = []
    with _pool(num_threads) if num_threads > 1 else nullcontext() as pool:
        sharded = _iter_tp_sharded_weights(
            nn.Module(),
            handle,
            handle,
            list(weights),
            set(),
            2,
            1,
            torch.float16,
            pool,
            num_threads,
        )
        for name, tensor in sharded:
            assert torch.equal(tensor, weights[name][2:4])
            assert params[name].tp_rank == 0
            seen.append(name)
    assert seen == list(weights)
    assert all(param.tp_rank == 1 for param in params.values())


@pytest.mark.spyre_device
@pytest.mark.parametrize(
    "extra_config", [{}, {"enable_multithread_load": False}], ids=["default", "serial"]
)
def test_loader_places_weights_on_the_right_device(tmp_path, monkeypatch, tp_group, extra_config):
    """A normal weight is streamed to Spyre, a host-loaded one stays on the CPU, and both
    hold the checkpoint's values."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    w_dense = torch.randn(64, 64, dtype=torch.float16)
    w_fallback = torch.randn(64, 64, dtype=torch.float16)
    ckpt = tmp_path / "model.safetensors"
    save_file({"dense.weight": w_dense, "fallback.weight": w_fallback}, str(ckpt))

    monkeypatch.setattr(
        "spyre_inference.model_loader._get_host_loaded_param_names",
        lambda model: {"fallback.weight"},
    )
    monkeypatch.setattr(
        "spyre_inference.model_loader._resolve_checkpoint_files",
        lambda model_config, load_config: (str(tmp_path), [str(ckpt)]),
    )

    model = _TwoLinearModel()
    # What SpyreModelLoader.load_model does before loading: a Spyre placeholder for every
    # parameter that is not host-loaded. Creating it also initializes the device.
    model.dense.weight = nn.Parameter(
        torch.empty(64, 64, dtype=torch.float16, device="spyre"), requires_grad=False
    )

    loader = _extra_config_loader(extra_config)
    loader.load_weights(model, SimpleNamespace(dtype=torch.float16))

    assert model.dense.weight.device.type == "spyre"
    assert model.fallback.weight.device.type == "cpu"
    assert torch.allclose(model.dense.weight.cpu(), w_dense, atol=1e-3, rtol=1e-3)
    assert torch.equal(model.fallback.weight, w_fallback)
