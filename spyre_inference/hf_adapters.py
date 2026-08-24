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

"""Drop-in replacement for vLLM's TransformersForCausalLM using hf-adapters.

Registers as a drop-in replacement for vLLM's TransformersForCausalLM when
the Spyre platform is active.  vLLM's stock Transformers backend handles
model creation, weight loading, attention routing, KV cache, scheduling, and
forward execution.  Spyre OOT layers (SpyreRMSNorm, SpyreLinears, etc.)
are applied automatically at instantiation time.

Activated when ``model_impl="transformers"`` on the Spyre platform via
``register_hf_adapters()``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from hf_adapters.hf_common import (
    InvFreqShim,
    PrecomputedRotaryEmbedding,
    apply_rope_matmul,
    get_backbone,
)
from spyre_inference.custom_ops.head_pad import original_head_dim
from vllm.logger import init_logger
from vllm.model_executor.models.transformers import TransformersForCausalLM

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class _SpyreRotaryEmbedding(nn.Module):
    """Drop-in for HF RotaryEmbedding using the same approach followed by hf-adapters.

    Returns ``(rotation_matrices, None)`` matching HF's ``(cos, sin)`` API.
    The patched ``apply_rotary_pos_emb`` uses ``apply_rope_matmul`` with the
    rotation matrices and ignores the second element.
    """

    def __init__(self, pre):
        super().__init__()
        self._pre = pre

    def _apply(self, fn, recurse=True):
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        return self._pre(x, position_ids), None


def _spyre_apply_rotary(q, k, cos, sin=None, *args, **kwargs):
    """Matmul RoPE; ``cos`` carries [B, L, 2, 2, D/2] rotation matrices, ``sin`` unused."""
    return apply_rope_matmul(q, cos), apply_rope_matmul(k, cos)


_spyre_apply_rotary._spyre_patched = True


def _rope_at_original_head_dim(cfg, rope: nn.Module, orig_head_dim: int) -> InvFreqShim:
    """Rebuild ``inv_freq``/``attention_scaling`` at the pre-pad head_dim.

    HF derived them from the widened ``config.head_dim``, giving one frequency per
    padded pair instead of per real pair.
    """
    padded = cfg.head_dim
    cfg.head_dim = orig_head_dim
    try:
        ref = type(rope)(config=cfg)
    finally:
        cfg.head_dim = padded
    return InvFreqShim(ref.inv_freq, ref.attention_scaling)


class HfAdaptersForCausalLM(TransformersForCausalLM):
    """TransformersForCausalLM wrapper to use HF adapters."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self._fix_generic_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        logger.debug("HfAdaptersForCausalLM ready: %s", type(self.model).__name__)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights and patch rope."""
        result = super().load_weights(weights)
        self._patch_rope()
        return result

    @staticmethod
    def _fix_generic_config(vllm_config: VllmConfig) -> None:
        """Re-resolve generic PretrainedConfig produced by vLLM's
        config parser for some models where both config.json and params.json exists
        and force HF-format weight loading."""
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

    # TODO: Add support for models with fused QKV / gate_up projections
    # (e.g. Phi-3) by splitting them into separate modules with TP-aware
    # weight redistribution and partial-rotary dimension permutation.

    def _patch_rope(self):
        """Replace RoPE with matmul-based rotation.

        head_dim is already a 128-multiple (the platform pads it), so the rotation
        needs no expand/contract here — only the pre-pad frequencies, identity-padded
        back out to the padded width.
        """

        cfg = self.model.config
        orig_head_dim = original_head_dim(cfg)

        backbone = get_backbone(self.model)
        rope_source = backbone.rotary_emb
        padded_head_dim = None
        if orig_head_dim is not None:
            padded_head_dim = cfg.head_dim
            rope_source = _rope_at_original_head_dim(cfg, backbone.rotary_emb, orig_head_dim)

        spyre_rope = PrecomputedRotaryEmbedding(
            rope_source,
            padded_head_dim=padded_head_dim,
        )

        spyre_rope_emb = _SpyreRotaryEmbedding(spyre_rope)
        backbone.rotary_emb = spyre_rope_emb

        _own_ids = {id(m) for m in spyre_rope_emb.modules()}

        patched_mods: set[int] = set()
        for name, module in self.model.named_modules():
            if id(module) in _own_ids:
                continue

            cls_name = module.__class__.__name__

            if cls_name.endswith("RotaryEmbedding") and not isinstance(
                module, _SpyreRotaryEmbedding
            ):
                pname, _, attr = name.rpartition(".")
                parent = self.model.get_submodule(pname) if pname else self.model
                setattr(parent, attr, _SpyreRotaryEmbedding(spyre_rope))
                continue

            if "Attention" not in cls_name:
                continue

            if not hasattr(module, "rotary_emb"):
                module.rotary_emb = _SpyreRotaryEmbedding(spyre_rope)

            mod = sys.modules.get(type(module).__module__)
            if mod is None or id(mod) in patched_mods:
                continue
            existing = getattr(mod, "apply_rotary_pos_emb", None)
            if existing is None or getattr(existing, "_spyre_patched", False):
                continue
            mod.apply_rotary_pos_emb = _spyre_apply_rotary
            patched_mods.add(id(mod))


# vLLM's Transformers backend test checks ModelConfig.using_transformers_backend()
# compares _ModelInfo.architecture (set to model_cls.__name__) against "TransformersForCausalLM".
# Without this, the subclass name "HfAdaptersForCausalLM" causes that check to return False.
HfAdaptersForCausalLM.__name__ = "TransformersForCausalLM"
