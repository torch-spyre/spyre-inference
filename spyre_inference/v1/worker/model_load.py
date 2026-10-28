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

"""Model load steps shared by both model runners: head/MLP padding and the move to Spyre."""

from typing import Any

import torch
import torch.nn as nn
from vllm.config import ModelConfig
from vllm.model_executor.layers.attention.attention import Attention

from spyre_inference.custom_ops.bert_head_pad import install_bert_head_pad
from spyre_inference.custom_ops.conv import SpyreConv2d
from spyre_inference.custom_ops.head_pad import (
    fix_padded_attention_scale,
    fix_padded_qk_norm_eps,
    fix_padded_rope,
    install_head_pad_weight_loader,
    install_padded_head_dim,
    verify_padded_head_dim,
    verify_padded_qk_norm_weights,
)
from spyre_inference.custom_ops.mlp_pad import (
    install_mlp_pad_weight_loader,
    verify_padded_intermediate_size,
)


def install_pad_loaders(model_loader: Any, model_config: ModelConfig) -> None:
    """Install the head- and MLP-pad hooks the loader runs as weights stream in."""
    # Pad attention weights (q/k/v/o, and QK-norm) to the stick-aligned head_dim
    # as they stream in, when the platform overrode head_dim (e.g. head_size=64).
    # Must run before load_model builds+loads the (now 128-wide) params.
    text_config = model_config.hf_text_config
    install_padded_head_dim(model_config)
    install_bert_head_pad(model_config)
    install_head_pad_weight_loader(model_loader, text_config, model_config)
    install_mlp_pad_weight_loader(model_loader, text_config, model_config)


def fix_padded_model(model: nn.Module, model_config: ModelConfig) -> None:
    """Undo the padding's side effects on the CPU-loaded model."""
    # Restore original RoPE frequencies and attention scale, and compensate
    # QK-norm epsilon for the padded head_dim. All passes are scoped to the
    # padded text backbone (see custom_ops.text_backbone).
    text_config = model_config.hf_text_config
    verify_padded_head_dim(model, text_config, model_config)
    verify_padded_qk_norm_weights(model, text_config, model_config)
    verify_padded_intermediate_size(model, text_config, model_config)
    fix_padded_rope(model, text_config, model_config)
    fix_padded_attention_scale(model, text_config, model_config)
    fix_padded_qk_norm_eps(model, text_config, model_config)


def move_model_to_spyre(model: nn.Module, device: torch.device) -> None:
    """Move the layer weights to Spyre and place SpyreConv2d weights in their tiled layout."""
    # Keep Attention module buffers (_k_scale, _v_scale, etc.) on CPU.
    # Note: This _apply cannot reside in SpyreAttentionImpl, as it is not
    # an nn.Module, but just the attention implementation.
    Attention._apply = lambda self, fn, recurse=True: self  # ty: ignore[invalid-assignment]
    model.to(device=device)
    for module in model.modules():
        if isinstance(module, SpyreConv2d):
            module.process_weights_after_loading()
