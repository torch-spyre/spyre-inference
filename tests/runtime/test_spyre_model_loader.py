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
from vllm.config import LoadConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader import get_model_loader

from spyre_inference.model_loader import (
    SpyreModelLoader,
    _bind_weight,
    _get_host_loaded_param_names,
)
from spyre_inference.platform import TorchSpyrePlatform, _is_multimodal_config, _spyre_loader_supported


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
        class FakeMoEConfig:
            num_experts = 8
        model_config.hf_config = FakeMoEConfig()
    return VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(load_format=load_format),
        parallel_config=ParallelConfig(tensor_parallel_size=tp_size),
    )


def test_platform_defaults_load_format_to_spyre():
    """Positive control: a plain TP=1 text-generation config flips auto → spyre."""
    vllm_config = _make_vllm_config()
    assert vllm_config.load_config.load_format == "spyre"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tp_size": 2},           # TP > 1
        {"quantization": "fp8"},  # quantized
        {"runner_type": "pooling"},  # pooling
    ],
    ids=["tp>1", "quantized", "pooling"],
)
def test_platform_load_format_stays_auto(kwargs):
    """load_format stays 'auto' for configs SpyreModelLoader doesn't support."""
    vllm_config = _make_vllm_config(**kwargs)
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    assert vllm_config.load_config.load_format == "auto"


def test_platform_load_format_stays_auto_multimodal():
    """load_format stays 'auto' for multimodal models.

    Patches _is_multimodal_config (not _spyre_loader_supported) so the gate
    wiring is exercised, and builds VllmConfig inside the patch so
    apply_config_platform_defaults is intercepted during __init__ too.
    """
    import unittest.mock as mock
    import spyre_inference.platform as plat_mod

    with mock.patch.object(plat_mod, "_is_multimodal_config", return_value=True):
        vllm_config = _make_vllm_config()
    assert vllm_config.load_config.load_format == "auto"


# ---------------------------------------------------------------------------
# _is_multimodal_config unit tests
# ---------------------------------------------------------------------------


def test_is_multimodal_config_detects_multimodal():
    """Both vLLM-native multimodal attributes are recognised."""
    assert _is_multimodal_config(SimpleNamespace(multimodal_config=object()))
    assert _is_multimodal_config(SimpleNamespace(is_multimodal_model=True))


def test_is_multimodal_config_text_only():
    """Text-only configs are not treated as multimodal."""
    assert not _is_multimodal_config(SimpleNamespace())
    assert not _is_multimodal_config(
        SimpleNamespace(is_multimodal_model=False, multimodal_config=None)
    )


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
        hf_config=SimpleNamespace(num_experts=8) if moe else SimpleNamespace(),
        multimodal_config=object() if multimodal else None,
        is_multimodal_model=False,
    )
    if transformers_backend:
        model_config.using_transformers_backend = lambda: True
    parallel_config = SimpleNamespace(tensor_parallel_size=tp_size)
    return SimpleNamespace(model_config=model_config, parallel_config=parallel_config)


def test_spyre_loader_supported_accepts_valid_config():
    """Positive case: TP=1, generate, unquantized, dense, text-only is supported."""
    assert _spyre_loader_supported(_make_ns())


@pytest.mark.parametrize(
    ("kwargs", "description"),
    [
        ({"tp_size": 2}, "TP > 1"),
        ({"runner_type": "pooling"}, "pooling runner"),
        ({"runner_type": "unknown_future_runner"}, "unknown runner type"),
        ({"quantization": "fp8"}, "quantized"),
        ({"transformers_backend": True}, "transformers backend"),
        ({"moe": True}, "MoE"),
        ({"multimodal": True}, "multimodal"),
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
    with mock.patch.object(plat_mod, "_spyre_loader_supported", return_value=False):
        with pytest.raises(ValueError, match="--load-format spyre is not supported"):
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


def test_bind_weight_custom_loader():
    """Verify _bind_weight uses custom weight_loader when present."""

    class MockParam(nn.Parameter):
        def __new__(cls, data):
            return super().__new__(cls, data)

    calls = []

    def mock_loader(param, tensor):
        calls.append((param, tensor))

    class SubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = MockParam(torch.zeros(2, 2))
            self.weight.weight_loader = mock_loader

    class ParentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = SubModel()

    model = ParentModel()
    dummy_weight = torch.ones(2, 2)
    _bind_weight(model, "sub.weight", dummy_weight)

    assert len(calls) == 1
    assert calls[0][0] is model.sub.weight
    assert torch.equal(calls[0][1], dummy_weight)


def test_bind_weight_standard_param():
    """Verify _bind_weight updates parameter data directly."""

    class StandardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)

    model = StandardModel()
    new_data = torch.ones(2, 2) * 5.0
    _bind_weight(model, "linear.weight", new_data)

    assert torch.equal(model.linear.weight.data, new_data)
