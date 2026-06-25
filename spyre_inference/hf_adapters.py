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
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from hf_adapters.hf_common import PrecomputedRotaryEmbedding, apply_rope_matmul
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

    def __init__(self, original: nn.Module, head_dim: int):
        super().__init__()
        self._pre = PrecomputedRotaryEmbedding(original, padded_head_dim=head_dim)

    def _apply(self, fn, recurse=True):
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        return self._pre(x, position_ids), None


def _make_spyre_apply_rotary(original_fn):
    """Replace apply_rotary_pos_emb with the same approach followed by hf-adapters."""

    @torch.no_grad()
    def wrapper(q, k, cos, sin=None, *args, **kwargs):
        return apply_rope_matmul(q, cos), apply_rope_matmul(k, cos)

    wrapper._spyre_patched = True  # type: ignore[attr-defined]
    return wrapper


class HfAdaptersForCausalLM(TransformersForCausalLM):
    """TransformersForCausalLM wrapper to use HF adapters."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self._fix_generic_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._patch_rope()
        logger.debug("HfAdaptersForCausalLM ready: %s", type(self.model).__name__)

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

    def _patch_rope(self):
        """Replace RotaryEmbedding with the same approach followed by hf-adapters."""
        text_config = self.config.get_text_config()
        head_dim = getattr(text_config, "head_dim", None) or (
            text_config.hidden_size // text_config.num_attention_heads
        )

        for name, module in self.model.named_modules():
            if module.__class__.__name__.endswith("RotaryEmbedding") and not isinstance(
                module, _SpyreRotaryEmbedding
            ):
                pname, _, attr = name.rpartition(".")
                parent = self.model.get_submodule(pname) if pname else self.model
                setattr(parent, attr, _SpyreRotaryEmbedding(module, head_dim))

        patched: set[int] = set()
        for _, module in self.model.named_modules():
            cls = type(module)
            if "Attention" not in cls.__name__:
                continue
            mod = sys.modules.get(cls.__module__)
            if mod is None or id(mod) in patched:
                continue
            orig = getattr(mod, "apply_rotary_pos_emb", None)
            if orig is None or getattr(orig, "_spyre_patched", False):
                continue
            mod.apply_rotary_pos_emb = _make_spyre_apply_rotary(orig)
            patched.add(id(mod))


# vLLM's Transformers backend test checks ModelConfig.using_transformers_backend()
# compares _ModelInfo.architecture (set to model_cls.__name__) against "TransformersForCausalLM".
# Without this, the subclass name "HfAdaptersForCausalLM" causes that check to return False.
HfAdaptersForCausalLM.__name__ = "TransformersForCausalLM"
