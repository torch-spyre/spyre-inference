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

"""Experimental, per-engine HF-adapters preparation contract.

The provider prepares a loaded CPU HF backbone in place and supplies adapter
execution with explicit attention bindings. Generation and KV
caches belong to vLLM. Importing HF-adapters is deferred to the worker.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class AttentionBinding:
    layer_idx: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    scale: float
    bind: Callable[[nn.Module], None]
    is_causal: bool = True
    uses_cache: bool = True


class PagedAdapterAttention(nn.Module):
    """Translate adapter Q/K/V layouts into the existing paged attention call."""

    def __init__(self, attention: nn.Module):
        super().__init__()
        self.attn = attention

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, width = q.shape
        query = q.transpose(1, 2).reshape(batch * tokens, heads * width)
        key = k.transpose(1, 2).reshape(batch * tokens, -1)
        value = v.transpose(1, 2).reshape(batch * tokens, -1)
        output = self.attn(query, key, value)
        return output.reshape(batch, tokens, heads, -1).transpose(1, 2)


@dataclass(frozen=True)
class PreparedBody:
    """Adapter execution and compilation after attention binding/device placement.

    ``compile_body`` installs wrappers in place; it must not change geometry or
    attention registration. It is never called with enforce_eager=True.
    """

    head_size: int
    forward_backbone: Callable[..., torch.Tensor]
    attention_bindings: tuple[AttentionBinding, ...]
    compile_body: Callable[[nn.Module], None]
    # Absolute output scale for the retained LM head; None keeps the vLLM default.
    logits_scale: float | None = None


def uses_hf_adapters(config) -> bool:
    additional = getattr(config, "additional_config", None) or {}
    if not isinstance(additional, dict):
        return False
    mode = additional.get("spyre_model_preparation", "spyre")
    if mode not in ("spyre", "hf_adapters"):
        raise ValueError(f"Unknown spyre_model_preparation={mode!r}")
    if "spyre_hf_adapters_preparer" in additional:
        raise ValueError(
            "Remove spyre_hf_adapters_preparer; "
            "spyre_model_preparation='hf_adapters' selects the built-in implementation"
        )
    return mode == "hf_adapters"


def validate_hf_adapters_config(config: VllmConfig) -> None:
    if not uses_hf_adapters(config):
        return
    from spyre_inference import envs

    model = config.model_config
    if model is None or model.model_impl != "transformers":
        raise ValueError("hf_adapters preparation requires model_impl='transformers'")
    if (
        model.runner_type not in ("generate", "pooling")
        or model.is_multimodal_model
        or model.hf_config is not model.hf_text_config
    ):
        raise ValueError("hf_adapters preparation supports text generation and embedding only")
    parallel = config.parallel_config
    if (
        parallel.tensor_parallel_size,
        parallel.pipeline_parallel_size,
        parallel.data_parallel_size,
    ) != (1, 1, 1):
        raise ValueError("hf_adapters preparation currently requires TP=PP=DP=1")
    if model.quantization is not None or config.quant_config is not None:
        raise ValueError("hf_adapters preparation currently requires unquantized weights")
    if config.cache_config.cache_dtype != "auto":
        raise ValueError("hf_adapters preparation currently requires cache_dtype='auto'")
    if config.lora_config is not None or config.speculative_config is not None:
        raise ValueError("hf_adapters preparation does not support LoRA or speculative decoding")
    if envs.environment_variables["SPYRE_MODEL_RUNNER"]() != "v1":
        raise ValueError("hf_adapters preparation currently requires SPYRE_MODEL_RUNNER=v1")
    if config.load_config.load_format not in ("auto", "hf", "safetensors", "pt"):
        raise ValueError("hf_adapters preparation requires ordinary HF checkpoint loading")
    text = model.hf_text_config
    layer_types = getattr(text, "layer_types", None)
    full_attention_types = {"attention", "full_attention"}
    window = getattr(text, "sliding_window", None)
    if window is not None and window >= model.max_model_len:
        full_attention_types.add("sliding_attention")
    if (
        model.is_moe
        or model.use_mla
        or model.is_hybrid
        or getattr(text, "is_encoder_decoder", False)
        or model.model_arch_config.per_layer_overrides is not None
        or (layer_types and any(kind not in full_attention_types for kind in layer_types))
    ):
        raise ValueError(
            "hf_adapters preparation currently requires a dense full-attention backbone"
        )


def prepare_execution(model: nn.Module, *, vllm_config: VllmConfig) -> PreparedBody:
    """Adapt HF-adapters' engine-independent execution API to the serving contract."""
    # Optional dependency, installed only when HF-adapters execution is enabled.
    from hf_adapters.execution import prepare_adapter_execution  # ty: ignore[unresolved-import]

    validate_hf_adapters_config(vllm_config)
    prepared = prepare_adapter_execution(
        model,
        max_model_len=vllm_config.model_config.max_model_len,
        task="embed" if vllm_config.model_config.runner_type == "pooling" else "generate",
    )
    bindings = tuple(
        AttentionBinding(
            layer_idx=attention.spec.layer_idx,
            num_heads=attention.spec.num_heads,
            num_kv_heads=attention.spec.num_kv_heads,
            head_size=attention.spec.head_size,
            scale=attention.spec.scale,
            bind=attention.bind,
            is_causal=attention.spec.is_causal,
            uses_cache=attention.spec.uses_cache,
        )
        for attention in prepared.runtime.attentions
    )
    return PreparedBody(
        head_size=bindings[0].head_size,
        compile_body=prepared.runtime.compile_body,
        logits_scale=prepared.runtime.logits_scale,
        forward_backbone=prepared.forward,
        attention_bindings=bindings,
    )


def prepare_body(model: nn.Module, config: VllmConfig) -> PreparedBody:
    result = prepare_execution(model, vllm_config=config)
    if not isinstance(result, PreparedBody):
        raise TypeError("prepare_execution must return PreparedBody")
    if type(result.head_size) is not int or result.head_size <= 0 or result.head_size % 64:
        raise ValueError("Prepared attention head_size must be a positive multiple of 64")
    if not callable(result.compile_body):
        raise TypeError("PreparedBody.compile_body must be callable")
    if result.logits_scale is not None and (
        isinstance(result.logits_scale, bool)
        or not isinstance(result.logits_scale, (float, int))
        or not math.isfinite(result.logits_scale)
        or result.logits_scale <= 0
    ):
        raise ValueError("PreparedBody.logits_scale must be finite and positive or None")
    if not callable(result.forward_backbone):
        raise TypeError("PreparedBody.forward_backbone must be callable")
    cfg = config.model_config.hf_text_config
    if not isinstance(result.attention_bindings, tuple) or any(
        not isinstance(binding, AttentionBinding) for binding in result.attention_bindings
    ):
        raise TypeError("attention_bindings must be a tuple of AttentionBinding")
    if [binding.layer_idx for binding in result.attention_bindings] != list(
        range(cfg.num_hidden_layers)
    ):
        raise ValueError("Attention bindings must cover each layer exactly once, in order")
    for binding in result.attention_bindings:
        if (
            type(binding.is_causal) is not bool
            or type(binding.uses_cache) is not bool
            or binding.uses_cache != (config.model_config.runner_type == "generate")
            or (binding.uses_cache and not binding.is_causal)
        ):
            raise ValueError("Attention causality/cache semantics disagree with the runner")
        if (
            type(binding.layer_idx) is not int
            or type(binding.head_size) is not int
            or binding.head_size != result.head_size
            or type(binding.num_heads) is not int
            or binding.num_heads != cfg.num_attention_heads
            or type(binding.num_kv_heads) is not int
            or binding.num_kv_heads != getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        ):
            raise ValueError("Attention binding geometry disagrees with the prepared model")
        if (
            isinstance(binding.scale, bool)
            or not isinstance(binding.scale, (int, float))
            or not math.isfinite(binding.scale)
            or binding.scale <= 0
            or not callable(binding.bind)
        ):
            raise ValueError("Attention bindings require a positive finite scale and bind callable")
    return result
