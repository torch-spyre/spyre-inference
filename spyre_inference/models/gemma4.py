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

Covers the dense 12B/31B variants and the E2B/E4B E-variants (per-layer embeddings and
KV-sharing). The KV-cache-group half of KV-sharing lives in ``TorchSpyreModelRunner``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)

# `Gemma4Model` registers these as buffers, then aliases them onto
# `Gemma4SelfDecoderLayers` as plain attributes, which do not follow `.to(device)`.
_ALIASED_SCALARS = (
    "normalizer",
    "embed_scale_per_layer",
    "per_layer_input_scale",
    "per_layer_projection_scale",
)

_orig_model_forward = None
_orig_layer_forward = None


def force_text_backbone(engine_args: EngineArgs) -> None:
    """Default gemma-4 to its text-only backbone (it ships multimodal).

    Sets ``hf_overrides["architectures"]`` so ``create_model_config`` resolves
    ``Gemma4ForCausalLM`` instead of the multimodal default. Skipped when the user
    already pinned an architecture (dict or callable ``hf_overrides``).
    """
    ov = engine_args.hf_overrides
    user_arch = callable(ov) or (isinstance(ov, dict) and "architectures" in ov)
    if "gemma-4" in (engine_args.model or "").lower() and not user_arch and isinstance(ov, dict):
        overrides = cast("dict[str, Any]", ov)
        overrides["architectures"] = ["Gemma4ForCausalLM"]
        logger.info("gemma-4: loading text-only backbone Gemma4ForCausalLM.")


def _register_aliased_scalars(self) -> None:
    """Re-register the parent's aliased scalars as buffers so they reach the device.

    Inductor has no notion of a live CPU graph input, so a stale 0-d CPU alias fails to
    lower where a device-side scalar is fine.
    """
    for name in _ALIASED_SCALARS:
        value = getattr(self, name, None)
        if not isinstance(value, torch.Tensor):
            # None on the dense variants (no PLE submodules).
            continue
        delattr(self, name)
        self.register_buffer(name, value, persistent=False)


def _spyre_get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
    """``get_per_layer_inputs`` without upstream's vocab-range mask.

    The Spyre backend cannot lower a torch.bool result over an int32 operand, and the mask
    is a no-op whenever ``vocab_size_per_layer_input >= vocab_size``. Smaller PLE vocabs
    keep upstream's masked path.
    """
    if self.embed_tokens_per_layer is None:
        return None
    per_layer_embeds = self.embed_tokens_per_layer(input_ids)
    per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer
    return per_layer_embeds.reshape(
        *input_ids.shape,
        self.config.num_hidden_layers,
        self.hidden_size_per_layer_input,
    )


def _spyre_model_forward(
    self,
    input_ids,
    positions,
    intermediate_tensors,
    inputs_embeds=None,
    per_layer_inputs=None,
    **kwargs,
):
    """``Gemma4Model.forward`` for the plain single-rank text path.

    Upstream slices ``per_layer_inputs[:, layer_idx, :]`` per layer; under per-block compile
    that view becomes a block argument, and a compiled kernel reads its arguments from
    offset 0, ignoring ``storage_offset`` (torch-spyre#3770). The block is handed the whole
    offset-0 tensor and slices in-graph instead.

    ``residual=None`` each iteration is exact: ``Gemma4DecoderLayer.forward`` overwrites
    ``residual`` on entry and always returns ``None`` for it.
    """
    if (
        self.fast_prefill_enabled
        or inputs_embeds is not None
        or intermediate_tensors is not None
        or per_layer_inputs is not None
        or getattr(self, "aux_hidden_state_layers", ())
        or self.start_layer != 0
        or self.end_layer != len(self.layers)
    ):
        return _orig_model_forward(
            self,
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
        # SpyreTensorLayout for the 3-D shape at a graph boundary ("Incompatible host_size
        # and dim_order").
        ple = ple.reshape(ple.shape[0], -1)
    for layer in self.layers:
        hidden_states, _ = layer(positions, hidden_states, None, per_layer_input=ple, **kwargs)
    return self.norm(hidden_states)


def _spyre_layer_forward(self, positions, hidden_states, residual, per_layer_input=None, **kwargs):
    """``Gemma4DecoderLayer.forward`` that slices its own PLE row in-graph.

    ``_spyre_model_forward`` passes every layer's row packed into one tensor; slicing here
    is safe because the block argument itself starts at offset 0 (torch-spyre#3770). An
    argument already ``ple_dim`` wide was sliced by some other caller.
    """
    ple_dim = self.hidden_size_per_layer_input
    if per_layer_input is not None and per_layer_input.shape[-1] != ple_dim:
        per_layer_input = per_layer_input.narrow(1, self.layer_idx * ple_dim, ple_dim)
    return _orig_layer_forward(
        self, positions, hidden_states, residual, per_layer_input=per_layer_input, **kwargs
    )


def install_spyre_patches() -> None:
    """Patch ``Gemma4SelfDecoderLayers`` for the Spyre compile path (idempotent)."""
    from vllm.model_executor.models.gemma4 import (
        Gemma4DecoderLayer,
        Gemma4Model,
        Gemma4SelfDecoderLayers,
    )

    if getattr(Gemma4SelfDecoderLayers, "_spyre_patched", False):
        return

    global _orig_model_forward, _orig_layer_forward
    _orig_model_forward = Gemma4Model.forward
    _orig_layer_forward = Gemma4DecoderLayer.forward
    Gemma4Model.forward = _spyre_model_forward  # ty: ignore[invalid-assignment]
    Gemma4DecoderLayer.forward = _spyre_layer_forward  # ty: ignore[invalid-assignment]

    orig_init = Gemma4SelfDecoderLayers.__init__
    orig_get_per_layer_inputs = Gemma4SelfDecoderLayers.get_per_layer_inputs

    def __init__(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        _register_aliased_scalars(self)

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if self.vocab_size_per_layer_input < self.config.vocab_size:
            return orig_get_per_layer_inputs(self, input_ids)
        return _spyre_get_per_layer_inputs(self, input_ids)

    Gemma4SelfDecoderLayers.__init__ = __init__  # ty: ignore[invalid-assignment]
    Gemma4SelfDecoderLayers.get_per_layer_inputs = get_per_layer_inputs  # ty: ignore[invalid-assignment]
    Gemma4SelfDecoderLayers._spyre_patched = True
    logger.info(
        "Spyre: Gemma-4 patched for the compile path (embedding/PLE scalars as buffers, "
        "no-op PLE vocab mask dropped, per-layer PLE row sliced inside each block)."
    )
