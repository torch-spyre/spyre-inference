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

import torch
from vllm.config import CompilationMode
from vllm.logger import init_logger
from vllm.model_executor.models.gemma4 import (
    Gemma4ForCausalLM,
    Gemma4Model,
    Gemma4SelfDecoderLayers,
)

from spyre_inference.custom_ops.lazy_compile import compile_when_outermost
from spyre_inference.models._retype import retype

if TYPE_CHECKING:
    from collections.abc import Callable

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


class SpyreGemma4SelfDecoderLayers(Gemma4SelfDecoderLayers):
    """Self-decoder without upstream's no-op PLE vocab-range mask."""

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """``get_per_layer_inputs`` without upstream's vocab-range mask.

        Spyre cannot lower a torch.bool result over an int32 operand, and the mask is a
        no-op whenever ``vocab_size_per_layer_input >= vocab_size``.
        """
        if self.embed_tokens_per_layer is None:
            return None
        if self.vocab_size_per_layer_input < self.config.vocab_size:
            return super().get_per_layer_inputs(input_ids)
        per_layer_embeds = self.embed_tokens_per_layer(input_ids) * self.embed_scale_per_layer
        return per_layer_embeds.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )


class _PerLayerRows(torch.Tensor):
    """Projected PLE whose ``[:, layer_idx, :]`` hands back a precomputed row.

    Upstream's backbone loop cuts each block's row with exactly that index, and the view
    it gets is at a nonzero storage offset, which a compiled block reads from offset 0
    (torch-spyre#3770). Carrying the rows lets that loop stand as written.
    """

    # No subclass propagation: only the instance the projection hands back carries rows.
    __torch_function__ = torch._C._disabled_torch_function_impl

    spyre_rows: tuple[torch.Tensor, ...]

    def __getitem__(self, index: Any) -> torch.Tensor:
        if (
            isinstance(index, tuple)
            and len(index) == 3
            and index[0] == index[2] == slice(None)
            and isinstance(index[1], int)
        ):
            return self.spyre_rows[index[1]]
        return torch.Tensor.__getitem__(self, index)


class SpyreGemma4Model(Gemma4Model):
    """Gemma-4 backbone cutting each block's PLE row outside the compiled block."""

    # ``compile_when_outermost`` reads these two. No ``__init__`` runs on a retyped
    # instance, so ``SpyreGemma4ForCausalLM`` assigns them.
    spyre_compile_enabled: bool
    spyre_compiled_kernel: Callable | None

    @compile_when_outermost
    def split_per_layer_inputs(self, ple: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Hand back every layer's PLE row, each in its own allocation.

        One graph, so the unavoidable copies cost one host launch per step instead of
        ``num_hidden_layers`` of them on an already host-bound forward.
        """
        ple_dim = self.hidden_size_per_layer_input
        return tuple(
            ple.narrow(1, layer_idx * ple_dim, ple_dim).clone()
            for layer_idx in range(len(self.layers))
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Upstream's projection, carrying the per-layer rows its loop will ask for."""
        ple = super().project_per_layer_inputs(inputs_embeds, per_layer_inputs)
        if ple is None:
            return None
        rows = ple.as_subclass(_PerLayerRows)
        rows.spyre_rows = self.split_per_layer_inputs(ple.reshape(ple.shape[0], -1))
        return rows


class SpyreGemma4ForCausalLM(Gemma4ForCausalLM):
    """Gemma-4 adapted for the Spyre compile path."""

    model: SpyreGemma4Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        retype(self.model, SpyreGemma4Model)
        retype(self.model.self_decoder, SpyreGemma4SelfDecoderLayers)
        # ``retype`` runs no ``__init__``, so set what ``compile_when_outermost`` reads.
        self.model.spyre_compile_enabled = (
            vllm_config.compilation_config.mode is not CompilationMode.NONE
        )
        self.model.spyre_compiled_kernel = None
        register_aliased_scalars(self.model.self_decoder)
