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

from types import SimpleNamespace

import pytest
import torch

from spyre_inference.v1.spec_decode.checkpoint import validate_dflash_weights


@pytest.fixture
def checkpoint():
    config = SimpleNamespace(
        hidden_size=128,
        head_dim=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=256,
        num_hidden_layers=1,
        vocab_size=256,
        dflash_config={"target_layer_ids": [0, 2]},
    )
    shapes = {
        "fc.weight": (128, 256),
        "hidden_norm.weight": (128,),
        "norm.weight": (128,),
        "layers.0.self_attn.q_proj.weight": (128, 128),
        "layers.0.self_attn.k_proj.weight": (64, 128),
        "layers.0.self_attn.v_proj.weight": (64, 128),
        "layers.0.self_attn.o_proj.weight": (128, 128),
        "layers.0.self_attn.q_norm.weight": (64,),
        "layers.0.self_attn.k_norm.weight": (64,),
        "layers.0.mlp.gate_proj.weight": (256, 128),
        "layers.0.mlp.up_proj.weight": (256, 128),
        "layers.0.mlp.down_proj.weight": (128, 256),
        "layers.0.input_layernorm.weight": (128,),
        "layers.0.post_attention_layernorm.weight": (128,),
    }
    return config, {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}


def test_backbone_without_shared_embedding_and_head_is_complete(checkpoint):
    validate_dflash_weights(*checkpoint)


@pytest.mark.parametrize(
    "name", ["fc.weight", "layers.0.self_attn.k_proj.weight", "layers.0.mlp.up_proj.weight"]
)
def test_missing_weight_including_part_of_fused_parameter_fails(checkpoint, name):
    config, weights = checkpoint
    del weights[name]
    with pytest.raises(ValueError, match="missing="):
        validate_dflash_weights(config, weights)


def test_feature_width_mismatch_fails_before_loading(checkpoint):
    config, weights = checkpoint
    weights["fc.weight"] = torch.empty(128, 128, device="meta")
    with pytest.raises(ValueError, match="fc.weight shape"):
        validate_dflash_weights(config, weights)


@pytest.mark.parametrize(
    "name,value", [("target_hidden_size", 256), ("attention_bias", True), ("head_dim", None)]
)
def test_unpublished_backbone_layout_has_a_clear_scope_error(checkpoint, name, value):
    config, weights = checkpoint
    setattr(config, name, value)
    with pytest.raises(ValueError, match="Only published Qwen3 DFlash/XPress checkpoint layouts"):
        validate_dflash_weights(config, weights)
