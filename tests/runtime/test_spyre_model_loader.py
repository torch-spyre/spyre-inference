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
from spyre_inference.platform import TorchSpyrePlatform


def test_spyre_model_loader_registration():
    """Verify that --load-format spyre returns SpyreModelLoader."""
    load_config = LoadConfig(load_format="spyre")
    loader = get_model_loader(load_config)
    assert isinstance(loader, SpyreModelLoader)


def test_platform_defaults_load_format_to_spyre():
    """Verify that TorchSpyrePlatform defaults load_format 'auto' to 'spyre' for dense TP=1."""
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B",
            max_model_len=1,
            dtype=torch.float16,
            trust_remote_code=True,
        ),
        load_config=LoadConfig(load_format="auto"),
    )
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    assert vllm_config.load_config.load_format == "spyre"


@pytest.mark.parametrize(
    ("tp_size", "quantization", "transformers_backend", "is_moe"),
    [
        (2, None, False, False),
        (1, "fp8", False, False),
        (1, None, True, False),
        (1, None, False, True),
    ],
)
def test_platform_load_format_stays_auto(tp_size, quantization, transformers_backend, is_moe):
    """Verify load_format stays 'auto' for TP>1, quantized, transformers backend, or MoE."""
    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1,
        dtype=torch.float16,
        trust_remote_code=True,
        quantization=quantization,
    )
    if transformers_backend:
        model_config.using_transformers_backend = lambda: True
    if is_moe:
        class FakeMoEConfig:
            num_experts = 8

        model_config.hf_config = FakeMoEConfig()

    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(load_format="auto"),
        parallel_config=ParallelConfig(tensor_parallel_size=tp_size),
    )
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    assert vllm_config.load_config.load_format == "auto"


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
