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
implementations by layer class.
"""

import torch

from . import (
    blip2,
    clip,
    gemma4_vision,
    granite4_vision,
    pixtral,
    siglip,
)


def apply_multimodal_patches(model: torch.nn.Module, device: torch.device) -> None:
    """Apply every Spyre workaround the loaded model's vision path needs.

    A no-op for text-only models. Call after weights are on the device but before
    compile, which wraps modules in `OptimizedModule` and breaks traversal.
    """
    # Both spellings: mistral-format Pixtral names the tower `vision_encoder`, HF-format
    # Mistral3 and Gemma 4 `vision_tower`. Ungated, the patches rewrite vLLM's shared
    # module. `is not None` rather than `or`: an empty container module is falsy.
    vision_tower = getattr(model, "vision_tower", None)
    if vision_tower is None:
        vision_tower = getattr(model, "vision_encoder", None)
    if vision_tower is not None:
        # Dispatch by tower class name: the `vision_tower` / `vision_encoder`
        # attribute is shared across architectures, so we cannot infer the model
        # family from the attribute name alone.
        tower_cls = type(vision_tower).__name__
        if tower_cls == "Gemma4VisionModel":
            gemma4_vision.apply(model, device)
        elif tower_cls == "SiglipVisionModel":
            # Granite4Vision uses a SigLIP tower. All three patches are
            # needed together and none apply to any other architecture.
            siglip.apply(model, device)
            granite4_vision.apply(model, device)
            blip2.apply(model, device)
        else:
            # Pixtral (mistral-format, VisionTransformer) and Mistral3 HF-format
            # (PixtralHFVisionModel) both need the Pixtral patches. Any tower that
            # is not Gemma4 or SigLIP goes here; the patch functions are
            # individually guarded and no-op when the expected symbols are absent.
            pixtral.apply(model, device)

    # CLIPEmbeddingModel: text_model/vision_model, not vision_encoder/vision_tower.
    # Gated on model_type, not just attribute presence: other architectures (e.g.
    # BLIP-2) also set `vision_model`, and clip.apply() assumes CLIP's specific
    # LayerNorm-based boundary norms.
    hf_config = getattr(model, "config", None)
    if getattr(hf_config, "model_type", None) == "clip":
        clip.apply(model, device)
