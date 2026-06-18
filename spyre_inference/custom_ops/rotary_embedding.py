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

"""Spyre OOT replacement for RotaryEmbedding.

The CPU body is wrapped in a `direct_register_custom_op` so torch.compile does
NOT trace into RotaryEmbedding.forward_native.
"""

from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.base import (
    RotaryEmbedding,
    RotaryEmbeddingBase,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import get_layer, register_layer

logger = init_logger(__name__)


@RotaryEmbeddingBase.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(RotaryEmbedding):
    """OOT RotaryEmbedding: opaque CPU fallback wrapped as a custom op.

    Inductor sees one FallbackKernel returning (query, key); the entire
    rotary computation including index_select runs eagerly on CPU.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spyre_layer_name = register_layer(self, "spyre_rotary")

    def _apply(self, fn, recurse=True):
        # Keep cos_sin_cache (and any other buffers/params) on CPU. Spyre's
        # fp16 differs from CPU's bit-for-bit; round-tripping the cache
        # through Spyre corrupts it and produces wrong tokens.
        return self

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # infer_schema rejects Optional[Tensor] returns, so use an empty
        # tensor sentinel across the op boundary.
        key_in = key if key is not None else torch.empty(0, device=query.device, dtype=query.dtype)
        # cos_sin_cache is fetched inside the op via get_layer(layer_name);
        # torch.ops dispatcher signature is opaque to the type checker (resolves to `...`).
        out_q, out_k = torch.ops.vllm.spyre_rotary_cpu(
            positions,  # ty: ignore[invalid-argument-type]
            query,  # ty: ignore[invalid-argument-type]
            key_in,  # ty: ignore[invalid-argument-type]
            self._spyre_layer_name,  # ty: ignore[invalid-argument-type]
        )
        if key is None:
            return out_q, None
        return out_q, out_k


def _rotary_cpu_op_func(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # cos_sin_cache currently stays CPU permanently (see SpyreRotaryEmbedding._apply)
    layer = get_layer(layer_name)
    target_device = positions.device
    target_dtype = query.dtype

    cpu_positions = positions.to("cpu")
    cpu_query = query.to("cpu")
    cpu_key = key.to("cpu") if key.numel() > 0 else None

    out_q, out_k = RotaryEmbedding.forward_static(
        positions=cpu_positions,
        query=cpu_query,
        key=cpu_key,
        head_size=layer.head_size,
        rotary_dim=layer.rotary_dim,
        cos_sin_cache=layer.cos_sin_cache,
        is_neox_style=layer.is_neox_style,
    )

    out_q = out_q.to(device=target_device, dtype=target_dtype)
    if out_k is None:
        out_k = torch.empty(0, device=target_device, dtype=target_dtype)
    else:
        out_k = out_k.to(device=target_device, dtype=target_dtype)
    return out_q, out_k


def _rotary_cpu_op_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_q = torch.empty(query.shape, dtype=query.dtype, device=positions.device)
    out_k = torch.empty(key.shape, dtype=query.dtype, device=positions.device)
    return out_q, out_k


@lru_cache(maxsize=1)
def register():
    """Register the spyre_rotary_cpu custom op with vLLM."""
    direct_register_custom_op(
        op_name="spyre_rotary_cpu",
        op_func=_rotary_cpu_op_func,
        fake_impl=_rotary_cpu_op_fake,
        mutates_args=[],
        dispatch_key="CompositeExplicitAutograd",
    )
    logger.debug_once("Registered custom op: spyre_rotary_cpu")
