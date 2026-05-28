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

"""pytest configuration for spyre-inference tests."""

import pytest
import torch


@pytest.fixture()
def default_vllm_config(monkeypatch):
    """Provide a default vLLM config for tests."""
    from vllm.config import DeviceConfig, VllmConfig, ModelConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig
    from vllm.platforms import PlatformEnum, current_platform
    from vllm.forward_context import set_forward_context
    from spyre_inference.custom_ops import register_all

    monkeypatch.setattr(type(current_platform), "_enum", PlatformEnum.OOT)

    # Explicitly register custom ops
    register_all()

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
        model_config=ModelConfig(dtype=torch.float16),
    )
    with set_current_vllm_config(config), set_forward_context(None, config):
        yield
