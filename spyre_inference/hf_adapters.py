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

"""Spyre-safe drop-in for vLLM's TransformersForCausalLM.

Registers as a drop-in replacement for vLLM's TransformersForCausalLM when
the Spyre platform is active.  vLLM's stock Transformers backend handles
model creation, weight loading, attention routing, KV cache, scheduling,
and forward execution.  Spyre OOT layers (SpyreRMSNorm, SpyreSiluAndMul,
SpyreLinears, etc.) are applied automatically at instantiation time.

Activated when ``model_impl="transformers"`` on the Spyre platform.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from vllm.logger import init_logger
from vllm.model_executor.models.transformers import TransformersForCausalLM
from spyre_inference.custom_ops.utils import convert

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _cpu_safe_apply_rotary(original_fn):
    """Wrap apply_rotary_pos_emb so rotate_half slicing runs on CPU."""

    @torch.no_grad()
    def wrapper(q, k, cos, sin, *args, **kwargs):
        device = q.device
        if device.type == "cpu":
            return original_fn(q, k, cos, sin, *args, **kwargs)
        q_r, k_r = original_fn(
            convert(q, "cpu"),
            convert(k, "cpu"),
            convert(cos, "cpu"),
            convert(sin, "cpu"),
            *args,
            **kwargs,
        )
        return convert(q_r, device), convert(k_r, device)

    wrapper._spyre_patched = True  # type: ignore[attr-defined]
    return wrapper


class _CpuSafeRotaryEmbedding(nn.Module):
    """Wraps HF RotaryEmbedding to run RoPE computation on CPU."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original

    def _apply(self, fn, recurse=True):
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        device, dtype = x.device, x.dtype
        cpu_x = convert(x, "cpu") if device.type != "cpu" else x
        cpu_pos = (
            convert(position_ids, "cpu") if position_ids.device.type != "cpu" else position_ids
        )
        cos, sin = self.original(cpu_x, cpu_pos)
        return convert(cos, device, dtype), convert(sin, device, dtype)


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
        """Wrap RotaryEmbedding modules and patch apply_rotary_pos_emb."""
        for name, module in self.model.named_modules():
            if module.__class__.__name__.endswith("RotaryEmbedding") and not isinstance(
                module, _CpuSafeRotaryEmbedding
            ):
                pname, _, attr = name.rpartition(".")
                parent = self.model.get_submodule(pname) if pname else self.model
                setattr(parent, attr, _CpuSafeRotaryEmbedding(module))

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
            mod.apply_rotary_pos_emb = _cpu_safe_apply_rotary(orig)
            patched.add(id(mod))
