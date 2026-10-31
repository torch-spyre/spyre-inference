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

# TEMPORARY compatibility for the vendored vllm-project/vllm#54448 XPress adaptation.
# Published weight contract: vllm/model_executor/models/qwen3_dflash.py at ce9467b757.
# Replace with upstream XPress loading on upgrade; retain the Spyre model overrides.
# Cleanup: https://github.com/torch-spyre/spyre-inference/issues/1417
# test_xpress_vendored.py flags native support.

"""Validate the published Qwen3 DFlash/XPress backbone weights."""

from collections.abc import Mapping

import torch


def validate_dflash_weights(config, weights: Mapping[str, torch.Tensor]) -> None:
    hidden = config.hidden_size
    head = getattr(config, "head_dim", None)
    scope = "Only published Qwen3 DFlash/XPress checkpoint layouts are supported"
    if (
        head is None
        or (getattr(config, "target_hidden_size", None) or hidden) != hidden
        or getattr(config, "attention_bias", False)
    ):
        raise ValueError(
            f"{scope}: require an explicit head_dim, equal target/draft hidden sizes, "
            "and attention_bias=False"
        )
    query = config.num_attention_heads * head
    kv = config.num_key_value_heads * head
    intermediate = config.intermediate_size
    expected = {
        "fc.weight": (hidden, hidden * len(config.dflash_config["target_layer_ids"])),
        "hidden_norm.weight": (hidden,),
        "norm.weight": (hidden,),
    }
    layer_shapes = {
        "self_attn.q_proj.weight": (query, hidden),
        "self_attn.k_proj.weight": (kv, hidden),
        "self_attn.v_proj.weight": (kv, hidden),
        "self_attn.o_proj.weight": (hidden, query),
        "self_attn.q_norm.weight": (head,),
        "self_attn.k_norm.weight": (head,),
        "mlp.gate_proj.weight": (intermediate, hidden),
        "mlp.up_proj.weight": (intermediate, hidden),
        "mlp.down_proj.weight": (hidden, intermediate),
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
    }
    for layer in range(config.num_hidden_layers):
        expected.update({f"layers.{layer}.{name}": shape for name, shape in layer_shapes.items()})
    optional = {"embed_tokens.weight", "lm_head.weight"}
    present = {name for name in weights if not name.startswith("xpress_head.")}
    missing = expected.keys() - present
    unexpected = present - expected.keys() - optional
    if missing or unexpected:
        raise ValueError(
            f"{scope}. Invalid DFlash backbone weights: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for name in present & optional:
        expected[name] = (config.vocab_size, hidden)
    for name, shape in expected.items():
        if tuple(weights[name].shape) != shape:
            raise ValueError(
                f"{scope}. Invalid {name} shape: expected {shape}, got {tuple(weights[name].shape)}"
            )
