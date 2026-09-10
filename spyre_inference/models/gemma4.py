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

"""Spyre adaptations for vLLM's Gemma-4 model."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

from spyre_inference.moe import SpyreMoERecipe, configure_spyre_moe_layer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)

# Scalar buffers that ``Gemma4Model`` owns and ``Gemma4SelfDecoderLayers``
# re-exposes as plain attributes.
_ALIASED_SCALARS = (
    "normalizer",
    "embed_scale_per_layer",
    "per_layer_input_scale",
    "per_layer_projection_scale",
)


# Each "global" attribute vLLM's gemma-4 builder reads for full-attention layers,
# mapped to the per-layer attribute of the same role it is rebuilt from.
_GEMMA4_FULL_ATTENTION_ATTRS = {
    "global_head_dim": "head_dim",
    "num_global_key_value_heads": "num_key_value_heads",
}


def _gemma4_repair_head_dim_access(config: Any) -> None:
    """Let bare reads (config.head_dim, config.num_key_value_heads) return the
    sliding/global scalar, matching vLLM's sliding-layer path.

    This is the transformers>=5.16 heterogeneous-config repair
    (``force_text_backbone``'s docstring) applied in place. It's needed whether or
    not the checkpoint is multimodal: the ambiguous per-layer read lives in
    ``Gemma4TextConfig`` regardless of whether a sibling ``vision_config``/
    ``audio_config`` is also present, so a VLM checkpoint's nested ``text_config``
    needs the exact same fix as a plain text checkpoint's top-level config.
    """
    text_config = getattr(config, "text_config", config)
    for cfg in {id(config): config, id(text_config): text_config}.values():
        cfg.allow_global_per_layer_attribute_access = True
    per_layer = getattr(text_config, "per_layer_config", None)
    layer_types = getattr(text_config, "layer_types", None)
    if per_layer is not None and layer_types:
        full_idx = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
        for global_attr, src_attr in _GEMMA4_FULL_ATTENTION_ATTRS.items():
            if hasattr(text_config, global_attr) or not full_idx:
                continue
            values = {getattr(per_layer[i], src_attr) for i in full_idx}
            if len(values) == 1:
                setattr(text_config, global_attr, values.pop())


def _gemma4_text_backbone_override(config: Any) -> Any:
    # Module-level (not a closure) so it survives the pickle to EngineCore.
    config.architectures = ["Gemma4ForCausalLM"]
    _gemma4_repair_head_dim_access(config)
    return config


def _gemma4_multimodal_head_dim_override(config: Any) -> Any:
    """Repair the same heterogeneous-config head-dim access on a real multimodal
    Gemma4Config, without touching ``architectures`` -- vision/audio stay intact so
    vLLM resolves ``Gemma4ForConditionalGeneration`` normally. Needed because that
    class builds its nested text decoder via ``init_vllm_registered_model(hf_config=
    config.text_config, ...)``, which re-triggers the same bare ``head_dim`` read.
    """
    _gemma4_repair_head_dim_access(config)
    return config


# gemma-4 config model_types this fix applies to. Excludes the other gemma4_* types
# (unified, dspark, mtp, audio, vision): they have their own vLLM builders and must
# not be forced onto the text backbone.
_GEMMA4_TEXT_MODEL_TYPES = {"gemma4", "gemma4_text"}


def is_multimodal_gemma4(hf_config: Any) -> bool:
    """True for a gemma-4 config carrying a vision tower. Audio is out of scope."""
    if getattr(hf_config, "model_type", None) not in _GEMMA4_TEXT_MODEL_TYPES:
        return False
    return getattr(hf_config, "vision_config", None) is not None


def force_text_backbone(engine_args: EngineArgs) -> None:
    """Repair gemma-4's head-dim config; default a plain text checkpoint to its
    text-only backbone.

    transformers >=5.16 reclassifies gemma-4 as heterogeneous: a bare ``config.head_dim``
    read then raises (crashing vLLM's ``get_head_size``) and the ``global_*`` head/kv-head
    attributes vLLM needs are consumed into ``per_layer_config``. The override restores the
    <=5.14 view before ``ModelConfig`` is built; skipped when the user set ``hf_overrides``.

    A vision checkpoint gets the same head-dim repair but keeps its real
    ``architectures``; forcing ``Gemma4ForCausalLM`` there would strip the tower. An
    audio-only checkpoint is rejected rather than silently reduced to text.
    """
    if engine_args.hf_overrides:
        return
    from vllm.transformers_utils.config import get_config

    # Detect gemma-4 by config model_type, not the checkpoint name: derivatives such as
    # medgemma / translategemma carry a gemma4 config under an unrelated name. On any load
    # failure, defer to ModelConfig, which loads the same config and raises the real error.
    try:
        hf_config = get_config(
            engine_args.hf_config_path or engine_args.model,
            engine_args.trust_remote_code,
            engine_args.revision,
            engine_args.code_revision,
            engine_args.config_format,
            token=engine_args.hf_token,
        )
    except Exception:
        return
    if getattr(hf_config, "model_type", None) not in _GEMMA4_TEXT_MODEL_TYPES:
        return
    has_audio = getattr(hf_config, "audio_config", None) is not None
    # A multimodal Gemma4Config reports the same model_type as a text checkpoint, and
    # the head-dim repair below is unrelated to multimodality -- so repair it but keep
    # the real architectures, or the vision tower is stripped off.
    if is_multimodal_gemma4(hf_config):
        if has_audio:
            logger.warning(
                "gemma-4: this checkpoint has an audio tower, which is not supported on "
                "Spyre; image and text inputs work, audio input will fail."
            )
        engine_args.hf_overrides = _gemma4_multimodal_head_dim_override
        logger.info("gemma-4: repairing head-dim config access for a multimodal checkpoint.")
        return
    if has_audio:
        # Falling through would force the text-only backbone and silently drop the
        # tower, leaving a model that looks fine but ignores its audio inputs.
        raise NotImplementedError(
            "gemma-4 audio is not supported on Spyre, and this checkpoint has an audio "
            "tower but no vision tower. Use a text-only or vision checkpoint."
        )
    engine_args.hf_overrides = _gemma4_text_backbone_override
    logger.info("gemma-4: loading text-only backbone Gemma4ForCausalLM.")


def register_aliased_scalars(decoder: nn.Module) -> None:
    """Turn the self-decoder's aliased scalar attributes into buffers."""
    buffers = dict(decoder.named_buffers(recurse=False))
    for name in _ALIASED_SCALARS:
        scalar = getattr(decoder, name, None)
        if scalar is None or name in buffers:
            continue
        delattr(decoder, name)
        decoder.register_buffer(name, scalar, persistent=False)


def _fold_gemma4_expert_scale(down_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Fold Gemma's output scale into the source down-projection stack."""
    return down_weight * scale.detach().to(down_weight.dtype).view(-1, 1, 1)


def configure_gemma4_moe_layers(layers: Iterable[nn.Module]) -> None:
    """Register Gemma-4's full-softmax, scaled GELU expert recipe."""

    configured = 0
    for decoder in layers:
        moe = getattr(decoder, "moe", None)
        if moe is None:
            continue
        configure_spyre_moe_layer(
            moe.experts.routed_experts,
            SpyreMoERecipe(
                activation="gelu_tanh",
                routing="full_softmax",
                prepare_down_weight=partial(_fold_gemma4_expert_scale, scale=moe.per_expert_scale),
            ),
        )
        configured += 1
    if configured:
        logger.info("Spyre: configured %d Gemma-4 MoE layers.", configured)


class SpyreGemma4ForCausalLM(Gemma4ForCausalLM):
    """Gemma-4 on Spyre: device-resident scalars, and Spyre MoE expert dispatch.

    ``Gemma4SelfDecoderLayers`` holds four scalar buffers owned by ``Gemma4Model``
    as plain tensor attributes. ``model.to("spyre")`` rebinds the parent's buffers
    but leaves the aliases on CPU, so the compiled ``embed_input_ids`` feeds a 0-d
    CPU tensor into Inductor, which has no notion of a live CPU graph input.
    Re-registering the aliases restores the parent's stated intent (move with the
    model, interact with torch.compile) and needs no change to the embedding math:
    a device-side 0-d scalar lowers fine.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        register_aliased_scalars(self.model.self_decoder)
        configure_gemma4_moe_layers(self.model.layers)
