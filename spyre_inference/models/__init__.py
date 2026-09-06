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

Each entry in ``SPYRE_MODELS`` replaces a vLLM architecture with a subclass that
lives in the matching module here; ``_``-prefixed modules hold machinery shared
between them. Registration is lazy: nothing is imported until vLLM resolves the
architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs

_BERT = "spyre_inference.models.bert"
_ROBERTA = "spyre_inference.models.roberta"

SPYRE_MODELS: dict[str, str] = {
    # Encoders: token_type_ids passed out of band (see models._token_type).
    "BertModel": f"{_BERT}:SpyreBertEmbeddingModel",
    "BertSpladeSparseEmbeddingModel": f"{_BERT}:SpyreBertSpladeSparseEmbeddingModel",
    "BertForSequenceClassification": f"{_BERT}:SpyreBertForSequenceClassification",
    "BertForTokenClassification": f"{_BERT}:SpyreBertForTokenClassification",
    "BertForMaskedLM": f"{_BERT}:SpyreBertForMaskedLM",
    "RobertaModel": f"{_ROBERTA}:SpyreRobertaEmbeddingModel",
    "RobertaForMaskedLM": f"{_ROBERTA}:SpyreRobertaEmbeddingModel",
    "XLMRobertaModel": f"{_ROBERTA}:SpyreRobertaEmbeddingModel",
    "RobertaForSequenceClassification": (f"{_ROBERTA}:SpyreRobertaForSequenceClassification"),
    "XLMRobertaForSequenceClassification": (f"{_ROBERTA}:SpyreRobertaForSequenceClassification"),
    "RobertaForTokenClassification": f"{_ROBERTA}:SpyreRobertaForTokenClassification",
    "XLMRobertaForTokenClassification": (f"{_ROBERTA}:SpyreRobertaForTokenClassification"),
    "BgeM3EmbeddingModel": f"{_ROBERTA}:SpyreBgeM3EmbeddingModel",
    # Decoders. Gemma4ForConditionalGeneration needs no entry of its own: it
    # builds its language model through the registry, so it picks this one up.
    "Gemma4ForCausalLM": "spyre_inference.models.gemma4:SpyreGemma4ForCausalLM",
    # So that ``model_impl="transformers"`` picks up the Spyre RoPE adaptation.
    "TransformersForCausalLM": (
        "spyre_inference.transformers_backend:SpyreTransformersForCausalLM"
    ),
}


def register_models() -> None:
    """Point the Spyre-adapted architectures at their Spyre subclasses.

    ``register_model`` takes any architecture string and logs an override only at
    debug level, so a typo or an upstream rename would silently fall through to
    the unadapted vLLM class. The keys are checked against vLLM's registry first
    — a dict lookup, so nothing is imported and registration stays lazy.

    Raises:
        RuntimeError: if an architecture in ``SPYRE_MODELS`` is unknown to vLLM.
    """
    from vllm.model_executor.models import ModelRegistry

    known = ModelRegistry.get_supported_archs()
    unknown = sorted(arch for arch in SPYRE_MODELS if arch not in known)
    if unknown:
        raise RuntimeError(
            f"vLLM does not register the architectures {unknown}; the Spyre "
            "model adaptations need updating for this vLLM version."
        )

    for arch, model_cls in SPYRE_MODELS.items():
        ModelRegistry.register_model(arch, model_cls)


def apply_prelaunch_overrides(engine_args: EngineArgs) -> None:
    """Apply per-model EngineArgs overrides that must run before create_model_config
    builds the ModelConfig (e.g. text-only backbone selection)."""
    from spyre_inference.models import gemma4

    gemma4.force_text_backbone(engine_args)
