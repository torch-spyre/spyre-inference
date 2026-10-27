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

"""The text backbone the config-level padding passes (head_dim, intermediate_size) apply to.

The platform pads ``hf_text_config``. For a single config that is the whole model; for a
composite config only the language backbone, whose checkpoint weights live under
``language_model.``. Every weight and module pass scopes itself through here, so they
cannot disagree on which part was padded.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

TEXT_WEIGHT_PREFIX = "language_model."

NamedModules = list[tuple[str, torch.nn.Module]]


def text_weight_prefix(model_config) -> str | None:
    """``TEXT_WEIGHT_PREFIX`` for a composite config, None (everything) for a single one."""
    if model_config is not None and model_config.hf_text_config is not model_config.hf_config:
        return TEXT_WEIGHT_PREFIX
    return None


def is_text_weight(name: str, prefix: str | None) -> bool:
    """Whether checkpoint weight *name* belongs to the padded text backbone."""
    return prefix is None or prefix in name


def _language_backbones(model: torch.nn.Module) -> NamedModules | None:
    """A multimodal model's language submodules; None if they cannot be isolated.

    Read from ``_mark_language_model``, which native and Transformers-backend models
    both call; ``get_language_model()`` is the fallback. A non-multimodal model (e.g.
    ``GEMMA4_TEXT_BACKBONE_OVERRIDE`` under a composite config) is its own backbone.
    """
    from vllm.model_executor.models.interfaces import supports_multimodal

    if not supports_multimodal(model):
        return [("", model)]
    names = [n for n in getattr(model, "_language_model_names", None) or () if n]
    if names:
        return [(n, model.get_submodule(n)) for n in names]
    try:
        language_model = model.get_language_model()
    except NotImplementedError:
        return None
    for name, module in model.named_modules():
        if module is language_model and module is not model:
            return [(name, module)]
    return None


def text_modules(
    model: torch.nn.Module,
    model_config,
    match: Callable[[str, torch.nn.Module], bool],
    what: str,
) -> NamedModules:
    """The ``(qualname, module)`` pairs of the padded text backbone that satisfy *match*.

    Also covers ``attention_instances``, the plain dict the Transformers backend keeps
    its (text-only) Attention layers in, which ``named_modules()`` never reaches.

    Raises:
        NotImplementedError: *match* hits modules of a composite model whose language
            backbone cannot be isolated, so they may belong to an unpadded tower.
    """
    if text_weight_prefix(model_config) is None:
        scopes: NamedModules | None = [("", model)]
    else:
        scopes = _language_backbones(model)
    instances = getattr(model, "attention_instances", None) or {}
    extra = [(f"attn.{i}", m) for i, m in instances.items()]
    if scopes is None:
        ambiguous = sorted(n for n, m in model.named_modules() if match(n, m))
        if ambiguous:
            raise NotImplementedError(
                f"Spyre padded the text backbone of {type(model).__name__}, but cannot "
                f"isolate its language model, so these {what} may belong to an unpadded "
                f"tower: {', '.join(ambiguous)}"
            )
        scopes = []
    found = {
        id(module): (name, module)
        for prefix, scope in scopes
        for name, module in scope.named_modules(prefix=prefix)
    }
    for name, module in extra:
        found.setdefault(id(module), (name, module))
    return [(n, m) for n, m in found.values() if match(n, m)]
