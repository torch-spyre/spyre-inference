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

"""Spyre adaptations for vLLM's Gemma-4 model.

Covers the dense 12B/31B variants and the E2B/E4B E-variants, whose per-layer embeddings
(PLE) need two changes to lower on Spyre. Their other distinguishing feature, KV-sharing,
needs nothing model-specific: the one vLLM KV-cache-group fix it wants is generic and
lives in ``TorchSpyreModelRunner``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.model_executor.models.gemma4 import (
    Gemma4DecoderLayer,
    Gemma4ForCausalLM,
    Gemma4Model,
    Gemma4SelfDecoderLayers,
)

if TYPE_CHECKING:
    import torch
    from torch import nn
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.sequence import IntermediateTensors

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


def _retype(module: nn.Module, upstream: type[nn.Module], spyre: type[nn.Module]) -> None:
    """Retype an already-built submodule to its Spyre subclass.

    ``Gemma4ForCausalLM`` and ``Gemma4Model`` name the classes they build, so there is no
    ``embedding_class``-style hook to pass a subclass through. The built instance is
    retyped instead: same ``__init__``, same parameters, same module tree — only the
    overridden methods differ. Checked rather than assumed, so an upstream rename fails
    loudly instead of silently running the unadapted forward.
    """
    if type(module) is not upstream:
        raise RuntimeError(
            f"expected {upstream.__name__}, got {type(module).__name__}; the Spyre "
            "gemma-4 adaptations need updating for this vLLM version."
        )
    module.__class__ = spyre


class SpyreGemma4DecoderLayer(Gemma4DecoderLayer):
    """Decoder layer that slices its own PLE row in-graph.

    ``SpyreGemma4Model.forward`` passes every layer's row packed into one tensor; slicing
    here is safe because the block argument itself starts at offset 0 (torch-spyre#3770).
    An argument already ``ple_dim`` wide was sliced by some other caller.
    """

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        per_layer_input: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ple_dim = self.hidden_size_per_layer_input
        if per_layer_input is not None and per_layer_input.shape[-1] != ple_dim:
            per_layer_input = per_layer_input.narrow(1, self.layer_idx * ple_dim, ple_dim)
        return super().forward(
            positions, hidden_states, residual, per_layer_input=per_layer_input, **kwargs
        )


class SpyreGemma4SelfDecoderLayers(Gemma4SelfDecoderLayers):
    """Self-decoder without upstream's no-op PLE vocab-range mask."""

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """``get_per_layer_inputs`` without upstream's vocab-range mask.

        The Spyre backend cannot lower a torch.bool result over an int32 operand, and the
        mask is a no-op whenever ``vocab_size_per_layer_input >= vocab_size``. Smaller PLE
        vocabs keep upstream's masked path.
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


class SpyreGemma4Model(Gemma4Model):
    """Gemma-4 backbone handing each block the whole PLE tensor at offset 0."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        """``forward`` for the plain single-rank text path; anything else goes upstream.

        Upstream slices ``per_layer_inputs[:, layer_idx, :]`` per layer; under per-block
        compile that view becomes a block argument, and a compiled kernel reads its
        arguments from offset 0, ignoring ``storage_offset`` (torch-spyre#3770). Each block
        is handed the whole offset-0 tensor and slices in-graph instead.

        ``residual=None`` each iteration is exact: ``Gemma4DecoderLayer.forward`` overwrites
        ``residual`` on entry and always returns ``None`` for it.
        """
        if (
            self.fast_prefill_enabled
            or input_ids is None
            or inputs_embeds is not None
            or intermediate_tensors is not None
            or per_layer_inputs is not None
            or self.aux_hidden_state_layers
            or self.start_layer != 0
            or self.end_layer != len(self.layers)
        ):
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                per_layer_inputs,
                **kwargs,
            )

        hidden_states = self.embed_input_ids(input_ids)
        ple = self.project_per_layer_inputs(hidden_states, self.get_per_layer_inputs(input_ids))
        if ple is not None:
            # Free (contiguous -> offset-0 view), and required: torch-spyre cannot build a
            # SpyreTensorLayout for the 3-D shape at a graph boundary ("Incompatible
            # host_size and dim_order").
            ple = ple.reshape(ple.shape[0], -1)
        for layer in self.layers:
            hidden_states, _ = layer(positions, hidden_states, None, per_layer_input=ple, **kwargs)
        return self.norm(hidden_states)


class SpyreGemma4ForCausalLM(Gemma4ForCausalLM):
    """Gemma-4 adapted for the Spyre compile path.

    Two adaptations, both retyped onto the built module tree:

    - The aliased scalars become buffers. ``Gemma4SelfDecoderLayers`` holds four scalar
      buffers owned by ``Gemma4Model`` as plain tensor attributes, so ``model.to("spyre")``
      rebinds the parent's buffers but leaves the aliases on CPU and the compiled
      ``embed_input_ids`` feeds a 0-d CPU tensor into Inductor, which has no notion of a
      live CPU graph input. Re-registering them restores the parent's stated intent (move
      with the model, interact with torch.compile) and needs no change to the embedding
      math: a device-side 0-d scalar lowers fine.
    - The PLE path (E2B/E4B) drops a mask Spyre cannot lower and moves the per-layer slice
      inside each compiled block; see the subclasses above.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _retype(self.model, Gemma4Model, SpyreGemma4Model)
        _retype(self.model.self_decoder, Gemma4SelfDecoderLayers, SpyreGemma4SelfDecoderLayers)
        # Also covers self_decoder.decoder_layers / cross_decoder.decoder_layers: both are
        # slices of this ModuleList, holding the same layer objects.
        for layer in self.model.layers:
            _retype(layer, Gemma4DecoderLayer, SpyreGemma4DecoderLayer)
        register_aliased_scalars(self.model.self_decoder)
