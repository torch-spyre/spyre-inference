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
forward execution.  Spyre OOT layers (SpyreRMSNorm, SpyreSiluAndMul,
SpyreLinears, etc.) are applied automatically at instantiation time.

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
    BLOCK_SIZE,
    PrecomputedRotaryEmbedding,
    apply_rope_matmul,
    get_backbone,
)
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


def _qk_expand_matrix(orig_hd: int, padded_hd: int) -> torch.Tensor:
    """Interleaved expand matrix for Q/K (RoPE-compatible half-split)."""
    half, phalf = orig_hd // 2, padded_hd // 2
    m = torch.zeros(orig_hd, padded_hd)
    m[:half, :phalf] = torch.eye(half, phalf)
    m[half:, phalf:] = torch.eye(half, phalf)
    return m


def _make_spyre_apply_rotary(original_fn, qk_expand=None):
    """Replace apply_rotary_pos_emb with matmul-based RoPE.

    When *qk_expand* is provided (head_dim/2 is not stick-aligned), Q/K are
    temporarily padded into the stick-aligned dimension for the rotation,
    then contracted back to the original size.
    """
    qk_contract = qk_expand.t().contiguous() if qk_expand is not None else None
    _cached: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def wrapper(q, k, cos, sin=None, *args, **kwargs):
        if qk_expand is not None:
            dev = q.device
            if dev not in _cached:
                _cached[dev] = (
                    qk_expand.to(device=dev, dtype=q.dtype),
                    qk_contract.to(device=dev, dtype=q.dtype),
                )
            exp, con = _cached[dev]
            q = torch.matmul(q, exp)
            k = torch.matmul(k, exp)

        q, k = apply_rope_matmul(q, cos), apply_rope_matmul(k, cos)

        if qk_expand is not None:
            q = torch.matmul(q, con)
            k = torch.matmul(k, con)

        return q, k

    wrapper._spyre_patched = True  # type: ignore[attr-defined]
    return wrapper


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

        When head_dim/2 is not stick-aligned (not a multiple of BLOCK_SIZE),
        an expand/contract matrix pair is built and passed to the patched
        ``apply_rotary_pos_emb`` so that Q/K are temporarily padded into
        a stick-aligned dimension for the rotation, then contracted back.
        Attention and the KV cache keep using the original head_dim.
        """

        cfg = self.model.config
        orig_head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

        stick_aligned = ((orig_head_dim + 2 * BLOCK_SIZE - 1) // (2 * BLOCK_SIZE)) * (
            2 * BLOCK_SIZE
        )
        padded_head_dim = stick_aligned if stick_aligned > orig_head_dim else None

        qk_exp = (
            _qk_expand_matrix(orig_head_dim, padded_head_dim)
            if padded_head_dim is not None
            else None
        )

        backbone = get_backbone(self.model)
        spyre_rope = PrecomputedRotaryEmbedding(
            backbone.rotary_emb,
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
            orig = getattr(mod, "apply_rotary_pos_emb", None)
            if orig is None or getattr(orig, "_spyre_patched", False):
                continue
            mod.apply_rotary_pos_emb = _make_spyre_apply_rotary(orig, qk_exp)
            patched_mods.add(id(mod))


# vLLM's Transformers backend test checks ModelConfig.using_transformers_backend()
# compares _ModelInfo.architecture (set to model_cls.__name__) against "TransformersForCausalLM".
# Without this, the subclass name "HfAdaptersForCausalLM" causes that check to return False.
HfAdaptersForCausalLM.__name__ = "TransformersForCausalLM"
