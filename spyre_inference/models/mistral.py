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

"""Mistral/Ministral adaptations: Llama-4 attention temperature scaling on Spyre.

The scale needs an int64->float convert and torch-spyre's typecast table has no int64
entry, so it runs on CPU behind an opaque op. Multiplying the rank-4 rope output by the
rank-2 scale makes ``SpyreTensorLayout`` raise "Incompatible host_size and dim_order",
so the scale is applied before rope.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from vllm.model_executor.models.llama import LlamaDecoderLayer
from vllm.model_executor.models.mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralForCausalLM,
)

from spyre_inference.custom_ops.utils import convert

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.models.llama import LlamaConfig


# Single-slot cache holding `(key, positions, scale)`. See `_llama4_attn_scale_op`.
_llama4_scale_cache: tuple | None = None


def _llama4_attn_scale_op(
    positions: torch.Tensor,
    beta: float,
    original_max_position_embeddings: int,
) -> torch.Tensor:
    """Upstream's ``_get_llama_4_attn_scale`` on CPU, reached through a stub because a
    custom op cannot take the module. Every layer in a step shares ``positions``, so the
    round trip is cached across all of them."""
    global _llama4_scale_cache
    from types import SimpleNamespace

    # No contents in the key: reading them is a D2H, and this runs once per layer.
    # Freshness comes from `reset_llama4_scale_cache` at the step boundary instead.
    key = (
        positions.data_ptr(),
        tuple(positions.shape),
        beta,
        original_max_position_embeddings,
        str(positions.device),
    )
    if _llama4_scale_cache is not None and _llama4_scale_cache[0] == key:
        return _llama4_scale_cache[2]

    stub = SimpleNamespace(
        llama_4_scaling_beta=beta,
        llama_4_scaling_original_max_position_embeddings=original_max_position_embeddings,
    )
    scaling = MistralAttention._get_llama_4_attn_scale(
        stub,  # ty: ignore[invalid-argument-type]
        positions.to("cpu"),
    )
    scaling = convert(scaling, device=positions.device, dtype=torch.float16)
    # Hold `positions`: a freed address could be reused within the step and hit.
    _llama4_scale_cache = (key, positions, scaling)
    return scaling


def reset_llama4_scale_cache() -> None:
    """Drop the scale cache; called once per model forward: inference tensors carry no
    version counter, so nothing but the contents marks a buffer rewritten in place."""
    global _llama4_scale_cache
    _llama4_scale_cache = None


def _llama4_attn_scale_fake(
    positions: torch.Tensor,
    beta: float,
    original_max_position_embeddings: int,
) -> torch.Tensor:
    return torch.empty((positions.shape[0], 1), dtype=torch.float16, device=positions.device)


@lru_cache(maxsize=1)
def _register_llama4_attn_scale_op() -> None:
    """Called from ``__init__`` rather than import, so ``current_platform`` is resolved."""
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="spyre_llama4_attn_scale",
        op_func=_llama4_attn_scale_op,
        fake_impl=_llama4_attn_scale_fake,
        dispatch_key=current_platform.dispatch_key,
    )


class SpyreMistralAttention(MistralAttention):
    """Llama-4 attention scaling computed on CPU and applied before rope."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.do_llama_4_scaling:
            _register_llama4_attn_scale_op()

    def _get_llama_4_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.spyre_llama4_attn_scale(
            positions,  # ty: ignore[invalid-argument-type]
            float(self.llama_4_scaling_beta),  # ty: ignore[invalid-argument-type]
            int(self.llama_4_scaling_original_max_position_embeddings),  # ty: ignore[invalid-argument-type]
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Upstream's forward with the scale applied before rope.

        Rope is a per-token linear rotation and the scale a per-token scalar, so
        ``R(s*q) == s*R(q)``. Copies upstream's body: diff it on every vLLM bump.
        """
        if not self.do_llama_4_scaling:
            return super().forward(positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = (q * self._get_llama_4_attn_scale(positions)).to(q.dtype)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class SpyreMistralDecoderLayer(MistralDecoderLayer):
    """Build ``SpyreMistralAttention`` instead of ``MistralAttention``.

    ``MistralDecoderLayer.__init__`` hardcodes ``attn_layer_type``, so this bypasses it
    and repeats its tail. Diff that tail on every vLLM bump.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
    ) -> None:
        LlamaDecoderLayer.__init__(
            self,
            vllm_config=vllm_config,
            prefix=prefix,
            config=config,
            attn_layer_type=SpyreMistralAttention,
        )

        self.layer_idx = int(prefix.split(sep=".")[-1])
        # A separate name, not `config`: these are dynamic attributes the declared
        # parameter type does not carry.
        hf_config: Any = config if config is not None else vllm_config.model_config.hf_config

        if getattr(hf_config, "ada_rms_norm_t_cond", False):
            from vllm.model_executor.layers.linear import (
                ColumnParallelLinear,
                RowParallelLinear,
            )

            self.ada_rms_norm_t_cond = nn.Sequential(
                ColumnParallelLinear(
                    input_size=hf_config.hidden_size,
                    output_size=hf_config.ada_rms_norm_t_cond_dim,
                    bias=False,
                    return_bias=False,
                ),
                nn.GELU(),
                RowParallelLinear(
                    input_size=hf_config.ada_rms_norm_t_cond_dim,
                    output_size=hf_config.hidden_size,
                    bias=False,
                    return_bias=False,
                ),
            )
        else:
            self.ada_rms_norm_t_cond = None


class SpyreMistralForCausalLM(MistralForCausalLM):
    """Ministral/Mistral with the Spyre attention-scale layer type."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = SpyreMistralDecoderLayer,
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
