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
in the matching module here (the Transformers backend's lives in
``spyre_inference.transformers_backend``); ``_``-prefixed modules hold machinery shared
between them. Registration is lazy: nothing is imported until vLLM resolves the
architecture, so an architecture this deployment never serves costs nothing.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from vllm.engine.arg_utils import EngineArgs

# vLLM modules whose every architecture needs the same adaptation: these are the
# encoders carrying token_type_ids bit-packed into input_ids, and each unpacks them
# in forward, which does not lower on Spyre. Their Spyre subclasses pass the segment
# ids out of band instead (see models._token_type). Encoders that take the segment
# ids as a real argument (bert_with_rope) or have no segment embedding at all
# (modernbert) need nothing and are deliberately absent.
_ADAPTED_MODULES = ("bert", "roberta")

# Architectures adapted individually, for reasons that reach no further.
_ADAPTED_ARCHS: dict[str, str] = {
    "CLIPModel": "spyre_inference.models.clip:SpyreCLIPEmbeddingModel",
    # Vision tower only: upstream builds its language model through the registry, as
    # Gemma4ForCausalLM below.
    "Gemma4ForConditionalGeneration": (
        "spyre_inference.models.gemma4_mm:SpyreGemma4ForConditionalGeneration"
    ),
    "Gemma4ForCausalLM": "spyre_inference.models.gemma4:SpyreGemma4ForCausalLM",
    "Ministral3ForCausalLM": "spyre_inference.models.mistral:SpyreMistralForCausalLM",
    "MistralForCausalLM": "spyre_inference.models.mistral:SpyreMistralForCausalLM",
    # Mistral-format Pixtral checkpoints, e.g. Ministral-3 with config_format="mistral".
    "PixtralForConditionalGeneration": (
        "spyre_inference.models.pixtral:SpyrePixtralForConditionalGeneration"
    ),
    # So that ``model_impl="transformers"`` picks up the Spyre RoPE adaptation.
    "TransformersForCausalLM": (
        "spyre_inference.transformers_backend:SpyreTransformersForCausalLM"
    ),
}


def spyre_models() -> dict[str, str]:
    """``{vLLM architecture: "module:SpyreClass"}`` for every adapted architecture.

    The ``_ADAPTED_MODULES`` architectures are derived from vLLM's own table rather
    than listed manually. Members all need adaptation, and the Spyre subclass is
    ``Spyre`` + the class vLLM resolves the architecture to. If a new architecture
    is added upstream to one of those modules it will land here, and writing the
    subclass is still ours to do: ``tests/models/test_model_registration.py`` fails
    until that subclass exists.

    Reads the private ``_VLLM_MODELS`` because that is the pre-override table;
    ``ModelRegistry.models`` names the Spyre classes once registration has run.
    """
    from vllm.model_executor.models.registry import _VLLM_MODELS

    # Filtered before unpacking, so the (module, class) shape is only assumed for the
    # handful of architectures this reads.
    return {
        arch: f"spyre_inference.models.{spec[0]}:Spyre{spec[1]}"
        for arch, spec in _VLLM_MODELS.items()
        if spec[0] in _ADAPTED_MODULES
    } | _ADAPTED_ARCHS


def register_models() -> None:
    """Point the Spyre-adapted architectures at their Spyre subclasses.

    ``register_model`` takes any architecture string and logs an override only at
    debug level, so a typo or an upstream rename in ``_ADAPTED_ARCHS`` would
    silently fall through to the unadapted vLLM class. The keys are checked against
    vLLM's registry first — a dict lookup, so nothing is imported and registration
    stays lazy. Whether each *value* names a class that exists cannot be checked
    without importing it, which is what the tests do instead.

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


def check_spyre_model(model: torch.nn.Module, architecture: str) -> None:
    """Fail if ``architecture`` has a Spyre adaptation that ``model`` was not built as.

    vLLM caches the class it resolves per process, so an architecture it looks up
    before ``register_models`` has run keeps the unadapted vLLM class, and builds it
    without an error of its own. A worker looks a multimodal architecture up that early
    (for its multimodal cache), so this is what a ``VLLM_PLUGINS`` allowlist leaving out
    the general plugin produces.

    Raises:
        RuntimeError: if ``model`` is not an instance of the Spyre class.
    """
    target = spyre_models().get(architecture)
    if target is None:
        return
    module_name, _, cls_name = target.partition(":")
    spyre_cls = getattr(importlib.import_module(module_name), cls_name)
    if not isinstance(model, spyre_cls):
        raise RuntimeError(
            f"{architecture} was built as {type(model).__name__}, not {cls_name}: vLLM "
            "resolved it before the Spyre model adaptations were registered. Make sure "
            "VLLM_PLUGINS allows the spyre_inference general plugin."
        )


def apply_prelaunch_overrides(engine_args: EngineArgs) -> None:
    """Apply per-model EngineArgs overrides that must run before create_model_config
    builds the ModelConfig (e.g. text-only backbone selection).

    Also registers the Spyre architectures: a ``VLLM_PLUGINS`` allowlist without the ops
    plugin leaves that to the worker, too late for an in-process engine, which by then
    has cached the upstream class of a multimodal architecture such as CLIP.
    """
    from spyre_inference.models import clip, gemma4

    register_models()
    gemma4.force_text_backbone(engine_args)
    clip.force_disable_chunked_prefill(engine_args)
