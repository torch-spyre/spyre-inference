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

"""Per-architecture workarounds for multimodal models on Spyre.

One module per architecture, each exposing `apply(model, device)`. These monkeypatch
upstream vLLM model code, unlike `custom_ops/`, which registers out-of-tree
implementations by layer class. Architectures migrated to a registered subclass on
`models._vision_encoder.SpyreVisionEncoderMixin` (native Pixtral) are not patched here.
"""

import torch
from vllm.model_executor.models.utils import StageMissingLayer

from . import (
    blip2,
    clip,
    granite4_vision,
    siglip,
)


def _tower(model: torch.nn.Module, *names: str) -> torch.nn.Module | None:
    """The first of `names` on `model`; None when absent or stage-skipped (every modality
    at 0: vLLM's `StageMissingLayer` on meta, forwards attribute reads, raises if called)."""
    for name in names:
        tower = getattr(model, name, None)
        # `is not None` rather than truthiness: an empty container module is falsy.
        if tower is not None:
            return None if isinstance(tower, StageMissingLayer) else tower
    return None


def apply_multimodal_patches(model: torch.nn.Module, device: torch.device) -> None:
    """Apply every Spyre workaround the loaded model's vision path needs.

    A no-op for text-only models. Call after weights are on the device but before
    compile, which wraps modules in `OptimizedModule` and breaks traversal.
    """
    from spyre_inference.models._vision_encoder import SpyreVisionEncoderMixin

    # Adapted by their registered architecture subclass, not by patches.
    if isinstance(model, SpyreVisionEncoderMixin):
        return

    vision_tower = _tower(model, "vision_tower")
    if vision_tower is not None:
        # Dispatch by tower class name: the `vision_tower` attribute is shared
        # across architectures, so it does not name the model family.
        tower_cls = type(vision_tower).__name__
        if tower_cls == "Gemma4VisionModel":
            # Lazy: it subclasses transformers' gemma-4 classes at import, and a move
            # upstream must not break models without this tower.
            from . import gemma4_vision

            gemma4_vision.apply(model, device)
        elif tower_cls == "SiglipVisionModel":
            # Granite4Vision uses a SigLIP tower. All three patches are
            # needed together and none apply to any other architecture.
            siglip.apply(model, device)
            granite4_vision.apply(model, device)
            blip2.apply(model, device)

    # CLIPEmbeddingModel: text_model/vision_model, not vision_encoder/vision_tower.
    # Gated on model_type, not just attribute presence: other architectures (e.g.
    # BLIP-2) also set `vision_model`, and clip.apply() assumes CLIP's specific
    # LayerNorm-based boundary norms.
    hf_config = getattr(model, "config", None)
    if (
        getattr(hf_config, "model_type", None) == "clip"
        and _tower(model, "vision_model") is not None
    ):
        clip.apply(model, device)
