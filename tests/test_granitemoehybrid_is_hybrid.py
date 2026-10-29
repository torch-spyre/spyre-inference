# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

from spyre_inference import _patch_granitemoehybrid_is_hybrid


def test_patch_granitemoehybrid_is_hybrid():
    _patch_granitemoehybrid_is_hybrid()
    from vllm.config.model import ModelConfig

    # 1. Non-hybrid model
    cfg_non_hybrid = MagicMock(spec=ModelConfig)
    cfg_non_hybrid._model_info = SimpleNamespace(is_hybrid=False)
    cfg_non_hybrid.hf_config = SimpleNamespace(layer_types=None)
    assert ModelConfig.is_hybrid.fget(cfg_non_hybrid) is False

    # 2. Granite 4.0 Micro with 'full_attention' layer types
    cfg_micro = MagicMock(spec=ModelConfig)
    cfg_micro._model_info = SimpleNamespace(is_hybrid=True)
    cfg_micro.hf_config = SimpleNamespace(layer_types=["full_attention"] * 40)
    assert ModelConfig.is_hybrid.fget(cfg_micro) is False

    # 3. Model with mixed attention/full_attention
    cfg_mixed_attn = MagicMock(spec=ModelConfig)
    cfg_mixed_attn._model_info = SimpleNamespace(is_hybrid=True)
    cfg_mixed_attn.hf_config = SimpleNamespace(layer_types=["attention", "full_attention"])
    assert ModelConfig.is_hybrid.fget(cfg_mixed_attn) is False

    # 4. True hybrid model with Mamba / SSM layers
    cfg_true_hybrid = MagicMock(spec=ModelConfig)
    cfg_true_hybrid._model_info = SimpleNamespace(is_hybrid=True)
    cfg_true_hybrid.hf_config = SimpleNamespace(layer_types=["mamba", "full_attention"])
    assert ModelConfig.is_hybrid.fget(cfg_true_hybrid) is True
