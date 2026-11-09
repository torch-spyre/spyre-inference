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

"""Cache-free causal attention, shared by text embedders and CLIP."""

from copy import copy
from functools import lru_cache

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import AttentionBackend, AttentionType, subclass_attention_backend
from vllm.v1.attention.selector import get_attn_backend


@lru_cache
def _causal_encoder_backend(backend: type[AttentionBackend]) -> type[AttentionBackend]:
    class CausalEncoderOnlyBuilder(backend.get_builder_cls()):
        def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
            common_attn_metadata = copy(common_attn_metadata)
            common_attn_metadata.causal = True
            return super().build(common_prefix_len, common_attn_metadata, *args, **kwargs)

    return subclass_attention_backend(
        "CausalEncoderOnlyAttention_", backend, CausalEncoderOnlyBuilder
    )


class CausalEncoderOnlyAttention(Attention):
    """``EncoderOnlyAttention`` with the opposite builder: no KV cache, and the metadata is
    causal by this layer's say rather than by vLLM's ``causal=True`` default."""

    def __init__(self, num_heads: int, head_size: int, scale: float, **kwargs) -> None:
        backend = get_attn_backend(
            head_size, torch.get_default_dtype(), "auto", attn_type=AttentionType.ENCODER_ONLY
        )
        super().__init__(
            num_heads,
            head_size,
            scale,
            attn_backend=_causal_encoder_backend(backend),
            attn_type=AttentionType.ENCODER_ONLY,
            **kwargs,
        )
