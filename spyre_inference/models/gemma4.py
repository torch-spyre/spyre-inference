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

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

if TYPE_CHECKING:
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


def _gemma4_text_backbone_override(config: Any) -> Any:
    # Module-level (not a closure) so it survives the pickle to EngineCore.
    config.architectures = ["Gemma4ForCausalLM"]
    text_config = getattr(config, "text_config", config)
    # Let bare reads (config.head_dim, config.num_key_value_heads) return the
    # sliding/global scalar, matching vLLM's sliding-layer path.
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
    return config


# gemma-4 config model_types this fix applies to. Excludes the other gemma4_* types
# (unified, dspark, mtp, audio, vision): they have their own vLLM builders and must
# not be forced onto the text backbone.
_GEMMA4_TEXT_MODEL_TYPES = {"gemma4", "gemma4_text"}


def force_text_backbone(engine_args: EngineArgs) -> None:
    """Default gemma-4 to its text-only backbone and repair its head-dim config.

    transformers >=5.16 reclassifies gemma-4 as heterogeneous: a bare ``config.head_dim``
    read then raises (crashing vLLM's ``get_head_size``) and the ``global_*`` head/kv-head
    attributes vLLM needs are consumed into ``per_layer_config``. The override restores the
    <=5.14 view before ``ModelConfig`` is built; skipped when the user set ``hf_overrides``.
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


class SpyreGemma4ForCausalLM(Gemma4ForCausalLM):
    """Gemma-4 with the self-decoder's aliased scalars registered as buffers.

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
