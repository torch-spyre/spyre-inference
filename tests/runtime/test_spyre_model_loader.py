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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.config import LoadConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.utils import WeightsMapper

from spyre_inference.model_loader import (
    SpyreModelLoader,
    _check_all_weights_loaded,
    _get_host_loaded_param_names,
    _hf_to_vllm_name,
    _param_name_for,
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

    import spyre_inference.platform as plat_mod

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


@pytest.mark.spyre_device
def test_loader_places_weights_on_the_right_device(tmp_path, monkeypatch, tp_group):
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

    loader = SpyreModelLoader(LoadConfig(load_format="spyre"))
    loader.load_weights(model, SimpleNamespace(dtype=torch.float16))

    assert model.dense.weight.device.type == "spyre"
    assert model.fallback.weight.device.type == "cpu"
    assert torch.allclose(model.dense.weight.cpu(), w_dense, atol=1e-3, rtol=1e-3)
    assert torch.equal(model.fallback.weight, w_fallback)
