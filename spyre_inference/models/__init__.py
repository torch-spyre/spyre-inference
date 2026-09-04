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

"""Model-specific Spyre adaptations, per architecture.

Every architecture ``spyre_models()`` names is replaced by a subclass that lives
in the matching module here; ``_``-prefixed modules hold machinery shared between
them. Registration is lazy: nothing is imported until vLLM resolves the
architecture, so an architecture this deployment never serves costs nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs

# vLLM modules whose every architecture needs the same adaptation: these are the
# encoders carrying token_type_ids bit-packed into input_ids, and each unpacks them
# in forward, which does not lower on Spyre. Their Spyre subclasses pass the segment
# ids out of band instead (see models._token_type). Encoders that take the segment
# ids as a real argument (bert_with_rope) or have no segment embedding at all
# (modernbert) need nothing and are deliberately absent.
_ADAPTED_WHOLESALE = ("bert", "roberta")

# Architectures adapted individually, for reasons that reach no further.
_ADAPTED_INDIVIDUALLY: dict[str, str] = {
    # Gemma4ForConditionalGeneration needs no entry of its own: it builds its
    # language model through the registry, so it picks this one up.
    "Gemma4ForCausalLM": "spyre_inference.models.gemma4:SpyreGemma4ForCausalLM",
    # So that ``model_impl="transformers"`` picks up the Spyre RoPE adaptation.
    "TransformersForCausalLM": (
        "spyre_inference.transformers_backend:SpyreTransformersForCausalLM"
    ),
}


def spyre_models() -> dict[str, str]:
    """``{vLLM architecture: "module:SpyreClass"}`` for every adapted architecture.

    The encoders are derived from vLLM's own table rather than listed: thirteen
    architectures collapse onto six classes, all needing the one adaptation, and
    the Spyre subclass is ``Spyre`` + the class vLLM resolves the architecture to.
    Deriving them means a new bert/roberta architecture upstream is adapted
    instead of silently falling through to a ``forward`` that cannot compile.

    Reads the private ``_VLLM_MODELS`` because that is the pre-override table;
    ``ModelRegistry.models`` names the Spyre classes once registration has run.
    """
    from vllm.model_executor.models.registry import _VLLM_MODELS

    return {
        arch: f"spyre_inference.models.{module}:Spyre{cls_name}"
        for arch, (module, cls_name) in _VLLM_MODELS.items()
        if module in _ADAPTED_WHOLESALE
    } | _ADAPTED_INDIVIDUALLY


def register_models() -> None:
    """Point the Spyre-adapted architectures at their Spyre subclasses.

    ``register_model`` takes any architecture string and logs an override only at
    debug level, so a typo or an upstream rename in ``_ADAPTED_INDIVIDUALLY``
    would silently fall through to the unadapted vLLM class. The keys are checked
    against vLLM's registry first — a dict lookup, so nothing is imported and
    registration stays lazy.

    Raises:
        RuntimeError: if an adapted architecture is unknown to vLLM.
    """
    from vllm.model_executor.models import ModelRegistry

    models = spyre_models()
    known = ModelRegistry.get_supported_archs()
    unknown = sorted(arch for arch in models if arch not in known)
    if unknown:
        raise RuntimeError(
            f"vLLM does not register the architectures {unknown}; the Spyre "
            "model adaptations need updating for this vLLM version."
        )

    for arch, model_cls in models.items():
        ModelRegistry.register_model(arch, model_cls)


def apply_prelaunch_overrides(engine_args: EngineArgs) -> None:
    """Apply per-model EngineArgs overrides that must run before create_model_config
    builds the ModelConfig (e.g. text-only backbone selection)."""
    from spyre_inference.models import gemma4

    gemma4.force_text_backbone(engine_args)
