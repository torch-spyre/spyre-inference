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

"""Spyre adaptation of vLLM's Transformers backend.

Upstream's fusers replace HF's linear/norm/GLU modules with vLLM layers, which the Spyre
OOT registrations pick up on their own. Two things are left to HF's module code:

* RoPE — there is no RoPE fuser, so HF's ``rotary_emb`` survives and derives cos/sin
  inside the forward from int64 ``position_ids``, a cast torch-spyre cannot lower.
  Replaced here with a precomputed rotation cache and a slice-free multiply-and-reduce
  rotation.
* Models shipping both ``config.json`` and ``params.json`` parse into a bare
  ``PretrainedConfig``, which HF cannot build a model from.
"""

from __future__ import annotations

import functools
import re
import sys
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.auto_factory import _get_model_class
from vllm.logger import init_logger
from vllm.model_executor.models.transformers import (
    TransformersEmbeddingModel,
    TransformersForCausalLM,
)

from spyre_inference.custom_ops.head_pad import original_head_dim
from spyre_inference.hf_adapters import (
    PagedAdapterAttention,
    PreparedBody,
    prepare_body,
    uses_hf_adapters,
    validate_hf_adapters_config,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _build_rotation_cache(
    inv_freq: torch.Tensor,
    scaling: float,
    max_position: int,
    padded_head_dim: int | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``[max_position, 2, 2, head_dim // 2]`` rotation matrices ``[[cos, -sin], [sin, cos]]``.

    Working from ``inv_freq``/``attention_scaling`` inherits whatever rope scaling the
    module being replaced had baked into them. *padded_head_dim* extends the cache with
    identity blocks, so a Q/K padded up to it passes its trailing dimensions through.
    """
    rope_half = inv_freq.shape[0]
    freqs = torch.outer(torch.arange(max_position, dtype=torch.float32), inv_freq)
    cos, sin = torch.cos(freqs) * scaling, torch.sin(freqs) * scaling
    rot = torch.stack([cos, -sin, sin, cos], dim=1).view(max_position, 2, 2, rope_half)

    if padded_head_dim is not None and padded_head_dim // 2 > rope_half:
        identity = torch.zeros(max_position, 2, 2, padded_head_dim // 2 - rope_half)
        identity[:, 0, 0, :] = 1.0
        identity[:, 1, 1, :] = 1.0
        rot = torch.cat([rot, identity], dim=-1)

    return rot.to(dtype)


def _apply_rope_matmul(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[B, H, L, D]`` by ``rot`` ``[B, L, 2, 2, D // 2]``.

    Multiply-and-reduce rather than HF's ``rotate_half`` cat: Spyre cannot restickify the
    halves that slicing a stick-aligned head_dim produces.
    """
    b, h, seq, head_dim = x.shape
    pairs = x.transpose(1, 2).reshape(b, seq, h, 2, head_dim // 2)
    out = rot.unsqueeze(2).mul(pairs.unsqueeze(-3)).sum(4, keepdim=True).flatten(3)
    return out.transpose(1, 2)


def _rope_frequencies(original: nn.Module) -> dict[str | None, tuple[torch.Tensor, float]]:
    """``{layer_type: (inv_freq, attention_scaling)}`` for the rope module being replaced.

    Models mixing global and sliding-window attention (Gemma 3, Olmo 3, ...) register one
    ``{layer_type}_inv_freq`` buffer per type and select between them on a third
    ``layer_type`` argument to ``forward``; a single rope is keyed under ``None``, its
    default.
    """
    layer_types = getattr(original, "layer_types", None)
    if not layer_types:
        scaling = float(getattr(original, "attention_scaling", 1.0))
        return {None: (original.get_buffer("inv_freq"), scaling)}

    freqs = {}
    for layer_type in layer_types:
        try:
            inv_freq = original.get_buffer(f"{layer_type}_inv_freq")
        except AttributeError:
            continue  # a layer type that does not rotate registers no buffer
        scaling = float(getattr(original, f"{layer_type}_attention_scaling", 1.0))
        freqs[layer_type] = (inv_freq, scaling)
    return freqs


class _SpyreRotaryEmbedding(nn.Module):
    """Drop-in for an HF rotary embedding, returning ``(rot, None)`` in place of
    ``(cos, sin)``; the patched ``apply_rotary_pos_emb`` ignores the second element.

    The cache covers ``max_position`` up front rather than growing on demand: sizing it
    from the batch's positions needs an ``.item()``, so a host sync per step and a
    data-dependent guard ``torch.compile`` cannot trace.
    """

    def __init__(
        self,
        original: nn.Module,
        max_position: int,
        padded_head_dim: int | None,
        dtype: torch.dtype,
    ):
        super().__init__()
        self._cpu_caches = {
            layer_type: _build_rotation_cache(
                inv_freq.to("cpu", torch.float32),
                scaling,
                max_position,
                padded_head_dim,
                dtype,
            )
            for layer_type, (inv_freq, scaling) in _rope_frequencies(original).items()
        }
        self._caches = self._cpu_caches

    def _apply(self, fn, recurse=True):
        # Prime the device caches when the model moves to Spyre, i.e. before compile, so only
        # the index_select is traced. They are not buffers (they are built after weight
        # loading) and there are no children, so super() has nothing to do.
        self._caches = {k: fn(v) for k, v in self._cpu_caches.items()}
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str | None = None):
        cache = self._caches[layer_type]
        rot = cache.index_select(0, position_ids.flatten())
        return rot.view(*position_ids.shape, *cache.shape[1:]), None


def _spyre_apply_rotary(q, k, rot, *args, **kwargs):
    """Rotate Q and K by the matrices ``_SpyreRotaryEmbedding`` returned."""
    return _apply_rope_matmul(q, rot), _apply_rope_matmul(k, rot)


def _rope_dispatch(original: Callable) -> Callable:
    """``apply_rotary_pos_emb`` replacement that hands stock HF calls to *original*.

    The patch lands on a modeling module in ``sys.modules`` and is never removed, so an HF
    model built later in the process has to keep working. Spyre's calls are the ones whose
    ``sin`` is the ``None`` standing in for ``_SpyreRotaryEmbedding``'s second return.
    """

    @functools.wraps(original)
    def apply_rotary_pos_emb(q, k, cos, sin=None, *args, **kwargs):
        if sin is None:
            return _spyre_apply_rotary(q, k, cos)
        return original(q, k, cos, sin, *args, **kwargs)

    apply_rotary_pos_emb._spyre_patched = True
    return apply_rotary_pos_emb


def _rope_at_original_head_dim(cfg, rope: nn.Module, orig_head_dim: int) -> nn.Module:
    """Rebuild *rope* at the pre-pad head_dim.

    HF derived ``inv_freq`` from the widened ``config.head_dim``, giving one frequency
    per padded pair instead of per real pair.
    """
    padded = cfg.head_dim
    cfg.head_dim = orig_head_dim
    try:
        return type(rope)(config=cfg)
    finally:
        cfg.head_dim = padded


class SpyreTransformersMixin:
    """Shared adapter preparation, attention binding and backbone execution."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        validate_hf_adapters_config(vllm_config)
        self._hf_adapters = uses_hf_adapters(vllm_config)
        self._prepared_body: PreparedBody | None = None
        self._fix_generic_config(vllm_config)
        self._max_position = vllm_config.model_config.max_model_len
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        logger.debug("Spyre Transformers backend ready: %s", type(self.model).__name__)

    def _patch_config(self):
        super()._patch_config()
        if self._hf_adapters and type(self.config) in AutoModel._model_mapping:
            model_class = _get_model_class(self.config, AutoModel._model_mapping)
            # Match vLLM's HF capability check; preserve its attention selection
            # unless the backbone needs standard HF attention for construction.
            if not (
                model_class.is_backend_compatible() or model_class._can_set_attn_implementation()
            ):
                self.text_config._attn_implementation = "eager"

    def _create_hf_to_vllm_mapper(self):
        super()._create_hf_to_vllm_mapper()
        if self._hf_adapters and self.config.model_type == "gpt2":
            # HF's legacy "attn.bias" mask pattern also matches real c_attn.bias
            # under vLLM's regex search. Match complete module path components.
            self.hf_to_vllm_mapper.orig_to_new_regex = {
                re.compile(rf"(?:^|\.){pattern.pattern}$") if target is None else pattern: target
                for pattern, target in self.hf_to_vllm_mapper.orig_to_new_regex.items()
            }
        if self._hf_adapters and self.config.model_type == "gpt_neo":
            # Old checkpoints persist causal-mask buffers absent from current HF.
            self.hf_to_vllm_mapper.orig_to_new_regex = {
                **self.hf_to_vllm_mapper.orig_to_new_regex,
                re.compile(r"(?:^|\.)attn\.attention\.(?:bias|masked_bias)$"): None,
            }

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self._hf_adapters and self._prepared_body is not None:
            raise NotImplementedError("Reloading a prepared HF-adapters body is not supported")
        result = super().load_weights(weights)
        if self._hf_adapters:
            # CausalMixin only ties VocabParallelEmbedding; the preserved HF embedding
            # needs copying through the LM head's loader before its packing pass.
            if self.model_config.runner_type == "generate" and self._get_tie_word_embeddings():
                embedding = self.model.get_input_embeddings().weight
                loader = getattr(self.lm_head.weight, "weight_loader", None)
                if not callable(loader):
                    raise TypeError("The retained vLLM LM head must provide a weight loader")
                loader(self.lm_head.weight, embedding)
                result.add("lm_head.weight")
        elif isinstance(self, TransformersForCausalLM):
            self._patch_rope()
        return result

    def recursive_replace(self):
        if not self._hf_adapters:
            super().recursive_replace()

    def _create_attention_instances(self, prefix: str = ""):
        if self._hf_adapters:
            self._attention_prefix = prefix
        else:
            super()._create_attention_instances(prefix)

    def prepare_hf_adapters(self) -> None:
        """Prepare loaded CPU weights, then register attention at the final geometry."""
        if not self._hf_adapters or self._prepared_body is not None:
            raise RuntimeError("HF-adapters preparation must run exactly once in adapter mode")
        cfg = self.text_config
        dimensions = (
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            cfg.hidden_size,
            cfg.vocab_size,
        )
        prepared = prepare_body(self.model, self.vllm_config)
        if dimensions != (
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            cfg.hidden_size,
            cfg.vocab_size,
        ):
            raise ValueError(
                "Preparation may pad head_size but must preserve layer/head counts, "
                "hidden_size and vocab_size"
            )
        if self.model.config is not self.config:
            raise ValueError("Preparation must retain the HF config object")
        self._bind_adapter_execution(prepared)
        self._prepared_body = prepared

    def _bind_adapter_execution(self, prepared: PreparedBody) -> None:
        self.text_config.head_dim = prepared.head_size
        self.model_config.model_arch_config.head_size = prepared.head_size
        if self.model_config.runner_type == "generate" and prepared.logits_scale is not None:
            self.logits_processor.scale = prepared.logits_scale
        self.adapter_attentions = nn.ModuleList()
        for binding in prepared.attention_bindings:
            attention_cls = self._adapter_attention_cls(binding)
            path = f"adapter_attentions.{binding.layer_idx}.attn"
            prefix = f"{self._attention_prefix}.{path}" if self._attention_prefix else path
            attention = attention_cls(
                num_heads=binding.num_heads,
                num_kv_heads=binding.num_kv_heads,
                head_size=binding.head_size,
                scale=binding.scale,
                cache_config=self.cache_config,
                quant_config=self.quant_config,
                prefix=prefix,
            )
            executor = PagedAdapterAttention(attention)
            self.adapter_attentions.append(executor)
            binding.bind(executor)
            attention.process_weights_after_loading(cast(torch.dtype, self.model_config.dtype))

    def _adapter_attention_cls(self, binding):
        if binding.uses_cache:
            return self._get_attn_cls()
        from vllm.model_executor.layers.attention import EncoderOnlyAttention

        from spyre_inference.encoder_attention import CausalEncoderOnlyAttention

        return CausalEncoderOnlyAttention if binding.is_causal else EncoderOnlyAttention

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs
    ):
        if not self._hf_adapters:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
        prepared = self._prepared_body
        if prepared is None:
            raise RuntimeError("HF-adapters body has not been prepared")
        if input_ids is None or inputs_embeds is not None or intermediate_tensors is not None:
            raise ValueError("Adapter execution currently requires token IDs and TP=PP=1")
        token_type_ids = kwargs.get("token_type_ids")
        if token_type_ids is not None:
            return prepared.forward_backbone(
                input_ids.unsqueeze(0),
                positions.unsqueeze(0),
                token_type_ids=token_type_ids.unsqueeze(0),
            )[0]
        return prepared.forward_backbone(input_ids.unsqueeze(0), positions.unsqueeze(0))[0]

    def compile_hf_adapters(self) -> None:
        if self._prepared_body is None:
            raise RuntimeError("HF-adapters body has not been prepared")
        self._prepared_body.compile_body(self.model)

    @staticmethod
    def _fix_generic_config(vllm_config: VllmConfig) -> None:
        """Re-resolve the bare PretrainedConfig that vLLM's Mistral parser produces for
        repos shipping both config.json and params.json, which AutoModel.from_config
        rejects, and force HF-format weight loading. ``--config-format hf`` skips it."""
        hf_config = vllm_config.model_config.hf_config
        if type(hf_config) is not PretrainedConfig:
            return

        model_id = vllm_config.model_config.hf_config_path or vllm_config.model_config.model
        try:
            resolved = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=vllm_config.model_config.trust_remote_code,
                revision=vllm_config.model_config.revision,
            )
        except Exception:
            logger.warning("AutoConfig re-resolve failed for %s", model_id, exc_info=True)
            return

        skip = {"model_type", "_name_or_path", "transformers_version", "auto_map", "architectures"}
        for key, val in hf_config.to_dict().items():
            if key not in skip and val is not None:
                setattr(resolved, key, val)

        vllm_config.model_config.hf_config = resolved
        vllm_config.model_config.hf_text_config = resolved.get_text_config()
        if vllm_config.load_config.load_format in ("auto", "mistral"):
            vllm_config.load_config.load_format = "hf"
        logger.debug(
            "Re-resolved config: %s (model_type=%s), load_format=hf",
            type(resolved).__name__,
            resolved.model_type,
        )

    def _patch_rope(self):
        """Swap HF's rotary embedding and ``apply_rotary_pos_emb`` for the Spyre ones.

        Partial rotary dimensions (e.g. Phi-3) are unsupported — the cache would cover
        only the rotated dims — but reach a shape mismatch here rather than a check:
        ``_maybe_pad_head_dim`` already rejects them whenever padding is needed.
        """
        # The text backbone holding rotary_emb; multimodal models nest it one level
        # deeper, at model.model.language_model, and carry the rope config on its own
        # config rather than the top-level one.
        inner = self.model.model if hasattr(self.model, "model") else self.model
        backbone = cast(nn.Module, getattr(inner, "language_model", inner))
        cfg = getattr(backbone, "config", self.model.config)

        # head_dim is already stick-aligned (the platform pads it, and the weight pass
        # pads Q/K interleaved to match), so the rotation only needs the pre-pad
        # frequencies identity-padded back out to the widened width.
        rope_source = backbone.get_submodule("rotary_emb")
        orig_head_dim = original_head_dim(cfg)
        padded_head_dim = None
        if orig_head_dim is not None:
            padded_head_dim = cfg.head_dim
            rope_source = _rope_at_original_head_dim(cfg, rope_source, orig_head_dim)

        spyre_rope = _SpyreRotaryEmbedding(
            rope_source,
            self._max_position,
            padded_head_dim,
            next(self.model.parameters()).dtype,
        )
        backbone.rotary_emb = spyre_rope

        patched_mods: set[int] = set()
        for name, module in self.model.named_modules():
            if module is spyre_rope:
                continue

            cls_name = module.__class__.__name__

            if cls_name.endswith("RotaryEmbedding"):
                parent_name, _, attr = name.rpartition(".")
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                setattr(parent, attr, spyre_rope)
                continue

            if "Attention" not in cls_name:
                continue

            if not hasattr(module, "rotary_emb"):
                module.rotary_emb = spyre_rope

            # apply_rotary_pos_emb is a module-level function in HF modeling files, so it
            # is patched once per modeling module rather than per layer.
            mod = sys.modules.get(type(module).__module__)
            if mod is None or id(mod) in patched_mods:
                continue
            existing = getattr(mod, "apply_rotary_pos_emb", None)
            if existing is None or getattr(existing, "_spyre_patched", False):
                continue
            mod.apply_rotary_pos_emb = _rope_dispatch(existing)
            patched_mods.add(id(mod))


class SpyreTransformersForCausalLM(SpyreTransformersMixin, TransformersForCausalLM):
    pass


class SpyreTransformersEmbeddingModel(SpyreTransformersMixin, TransformersEmbeddingModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if uses_hf_adapters(vllm_config):
            from vllm.transformers_utils.config import get_hf_file_to_dict

            config = vllm_config.model_config
            if config.pooler_config.seq_pooling_type not in ("CLS", "MEAN", "LAST"):
                raise ValueError("HF-adapters embedding supports CLS, MEAN or LAST pooling")
            modules = get_hf_file_to_dict("modules.json", config.model, config.revision)
            if modules is not None:
                kinds = [module["type"] for module in modules]
                expected = [
                    "sentence_transformers.models.Transformer",
                    "sentence_transformers.models.Pooling",
                ]
                if kinds not in (expected, [*expected, "sentence_transformers.models.Normalize"]):
                    raise ValueError(
                        "HF-adapters embedding supports "
                        "Transformer/Pooling/Normalize pipelines only"
                    )
                pooling = (
                    get_hf_file_to_dict(
                        f"{modules[1]['path']}/config.json", config.model, config.revision
                    )
                    or {}
                )
                modes = [
                    key
                    for key, value in pooling.items()
                    if key.startswith("pooling_mode_") and value is True
                ]
                if (
                    len(modes) != 1
                    or modes[0]
                    not in (
                        "pooling_mode_cls_token",
                        "pooling_mode_mean_tokens",
                        "pooling_mode_lasttoken",
                    )
                    or pooling.get("include_prompt", True) is not True
                ):
                    raise ValueError(
                        "HF-adapters requires one CLS/MEAN/LAST pooling mode "
                        "with include_prompt=True"
                    )
        super().__init__(vllm_config=vllm_config, prefix=prefix)


# vLLM's Transformers-backend detection compares these registered class names.
SpyreTransformersForCausalLM.__name__ = "TransformersForCausalLM"
SpyreTransformersEmbeddingModel.__name__ = "TransformersEmbeddingModel"
