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
    granite4_vision,
    pixtral,
    siglip,
)


def apply_multimodal_patches(model: torch.nn.Module, device: torch.device) -> None:
    """Apply every Spyre workaround the loaded model's vision path needs.

    A no-op for text-only models. Call after weights are on the device but before
    compile, which wraps modules in `OptimizedModule` and breaks traversal.
    """
    # Check if the model is a multimodal model (e.g. has vision_tower, vision_encoder,
    # vision_model, etc.)
    mm_attrs = (
        "vision_encoder",
        "vision_tower",
        "vision_model",
        "image_encoder",
        "visual",
        "multi_modal_projector",
    )
    if not any(hasattr(model, attr) for attr in mm_attrs):
        return

    pixtral.apply(model, device)
    siglip.apply(model, device)
    granite4_vision.apply(model, device)
    blip2.apply(model, device)
