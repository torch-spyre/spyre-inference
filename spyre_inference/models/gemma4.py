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

from typing import TYPE_CHECKING, Any, TypeVar, cast

from vllm.config import CompilationMode
from vllm.logger import init_logger
from vllm.model_executor.models.gemma4 import (
    Gemma4ForCausalLM,
    Gemma4Model,
    Gemma4SelfDecoderLayers,
)

from spyre_inference.custom_ops.lazy_compile import compile_when_outermost

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from torch import nn
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.sequence import IntermediateTensors

_ModuleT = TypeVar("_ModuleT", bound="nn.Module")

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


def _retype(module: nn.Module, upstream: type[nn.Module], spyre: type[_ModuleT]) -> _ModuleT:
    """Retype an already-built submodule to its Spyre subclass, and hand it back.

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
    return cast("_ModuleT", module)


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
    """Gemma-4 backbone cutting each block's PLE row outside the compiled block."""

    # ``CompileOutermost``'s two fields, declared rather than set in an ``__init__`` that
    # retyping never runs; ``SpyreGemma4ForCausalLM`` fills them in.
    spyre_compile_enabled: bool
    spyre_compiled_kernel: Callable | None

    @compile_when_outermost
    def split_per_layer_inputs(self, ple: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Hand back every layer's PLE row, each in its own allocation.

        One graph rather than one eager ``clone`` per layer. The copies themselves are
        unavoidable — see ``forward`` — but eager ones cost a host launch each on a
        forward pass that is already host-bound, and there are ``num_hidden_layers`` of
        them per step. Compiled, they are one launch: measured ~2.4x cheaper than the
        eager form across every warmup bucket.
        """
        ple_dim = self.hidden_size_per_layer_input
        return tuple(
            ple.narrow(1, layer_idx * ple_dim, ple_dim).clone()
            for layer_idx in range(len(self.layers))
        )

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

        Upstream slices ``per_layer_inputs[:, layer_idx, :]`` per layer and hands the block
        that view. Two things stop it working here: a compiled kernel reads its arguments
        from offset 0, ignoring ``storage_offset`` (torch-spyre#3770), and torch-spyre
        cannot lay out the 3-D tensor at a graph boundary. So each row has to be a real
        allocation, cut from a 2-D view -- a view of any width is read as layer 0's row,
        and an in-graph ``index_select`` off the packed tensor does not lower at all
        ("no mechanism to resolve stick incompatibility").

        Cutting them here rather than inside the block is what keeps warmup affordable: a
        ``layer_idx``-derived offset inside ``forward`` is a graph constant, so every block
        would guard differently and compile its own artifact (35 for E2B, 42 for E4B)
        instead of all of them sharing one. ``split_per_layer_inputs`` then keeps the
        copies off the host critical path.

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
        rows = None
        if ple is not None:
            rows = self.split_per_layer_inputs(ple.reshape(ple.shape[0], -1))
        for layer_idx, layer in enumerate(self.layers):
            row = None if rows is None else rows[layer_idx]
            hidden_states, _ = layer(positions, hidden_states, None, per_layer_input=row, **kwargs)
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
    - The PLE path (E2B/E4B) drops a mask Spyre cannot lower and hands each block its own
      row as an offset-0 tensor; see the subclasses above.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        backbone = _retype(self.model, Gemma4Model, SpyreGemma4Model)
        _retype(self.model.self_decoder, Gemma4SelfDecoderLayers, SpyreGemma4SelfDecoderLayers)
        # What ``CompileOutermost.__init__`` would set. Retyping runs no ``__init__``, and
        # inheriting it would not help: its ``super().__init__()`` walks the *instance's*
        # MRO, which is the upstream backbone's.
        backbone.spyre_compile_enabled = (
            vllm_config.compilation_config.mode is not CompilationMode.NONE
        )
        backbone.spyre_compiled_kernel = None
        register_aliased_scalars(self.model.self_decoder)
